"""Streamlit demo UI for the MARGE orchestrator."""

import asyncio
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from beeai_framework.backend.message import UserMessage
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from apps.orchestrator.agent import build_bundle, orchestrator_agent
from packages.schemas.patient import PatientRecord
from packages.llm_provider.client import build_chat_model_for_role
from packages.llm_provider.settings import Role, RoleConfig
from services.patient_data_mcp_server.sources._base import PatientSource
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


DEMO_PATIENT_HANDLE = "seed-001"

DIABETES_FEATURE_INFO = {
    "plas": {
        "label": "Blood sugar",
        "detail": "Recent fasting glucose, random glucose, or oral glucose tolerance test result.",
    },
    "mass": {
        "label": "BMI",
        "detail": "Height and weight are enough if BMI is not already known.",
    },
    "pres": {
        "label": "Blood pressure",
        "detail": "Recent diastolic blood pressure reading.",
    },
    "insu": {
        "label": "Insulin",
        "detail": "Recent 2-hour serum insulin result, if available.",
    },
    "skin": {
        "label": "Skinfold thickness",
        "detail": "Triceps skinfold thickness. This is often unavailable outside clinical records.",
    },
    "pedi": {
        "label": "Family history",
        "detail": "Family history of diabetes or a calculated diabetes pedigree score.",
    },
    "preg": {
        "label": "Pregnancy history",
        "detail": "Number of pregnancies, only when clinically relevant.",
    },
    "age": {
        "label": "Age",
        "detail": "Current age in years.",
    },
}

DIABETES_FEATURES = tuple(DIABETES_FEATURE_INFO.keys())
KEY_DIABETES_FEATURES = {"plas", "mass", "age"}


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _regex_feature_fallback(user_message: str) -> dict[str, Any]:
    lowered = user_message.lower()
    features: dict[str, Any] = {}

    age_match = re.search(r"(?:age|살|세|years?\s*old)[^\d]{0,12}(\d{1,3})", lowered)
    if not age_match:
        age_match = re.search(r"(\d{1,3})\s*(?:살|세|years?\s*old)", lowered)
    if age_match:
        features["age"] = float(age_match.group(1))

    glucose_match = re.search(
        r"(?:glucose|blood sugar|혈당|공복혈당|식후혈당)[^\d]{0,20}(\d+(?:\.\d+)?)",
        lowered,
    )
    if glucose_match:
        features["plas"] = float(glucose_match.group(1))

    bmi_match = re.search(r"(?:bmi|체질량|체질량지수)[^\d]{0,20}(\d+(?:\.\d+)?)", lowered)
    if bmi_match:
        features["mass"] = float(bmi_match.group(1))

    bp_match = re.search(
        r"(?:blood pressure|bp|혈압)[^\d]{0,12}(\d{2,3})\s*/\s*(\d{2,3})",
        lowered,
    )
    if bp_match:
        features["pres"] = float(bp_match.group(2))

    insulin_match = re.search(r"(?:insulin|인슐린)[^\d]{0,20}(\d+(?:\.\d+)?)", lowered)
    if insulin_match:
        features["insu"] = float(insulin_match.group(1))

    return features


async def extract_diabetes_features(user_message: str) -> dict[str, Any]:
    prompt = f"""
Extract diabetes model features from the user's message.
Return only a JSON object. Do not include markdown.

Schema:
{{
  "age": number or null,
  "sex": "female" or "male" or "other" or null,
  "features": {{
    "preg": number or null,
    "plas": number or null,
    "pres": number or null,
    "skin": number or null,
    "insu": number or null,
    "mass": number or null,
    "pedi": number or null,
    "age": number or null
  }},
  "notes": string
}}

Feature meanings:
- preg: number of pregnancies
- plas: plasma glucose / blood sugar in mg/dL
- pres: diastolic blood pressure in mm Hg
- skin: triceps skinfold thickness in mm
- insu: 2-hour serum insulin
- mass: BMI
- pedi: diabetes pedigree / family-history score
- age: age in years

If a value is not explicitly provided, use null.

User message:
{user_message}
""".strip()

    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    try:
        result = await llm.run([UserMessage(prompt)])
        raw = result.get_text_content() if hasattr(result, "get_text_content") else str(result)
        data = _extract_json_object(raw)
    except Exception:
        data = {}

    extracted_features = data.get("features") if isinstance(data.get("features"), dict) else {}
    fallback_features = _regex_feature_fallback(user_message)

    features: dict[str, Any] = {}
    for feature in DIABETES_FEATURES:
        value = extracted_features.get(feature, fallback_features.get(feature))
        features[feature] = _coerce_optional_float(value)

    age = _coerce_optional_float(data.get("age", features.get("age")))
    if age is not None:
        features["age"] = age

    return features


async def build_patient_record_from_input(user_message: str) -> PatientRecord:
    base_record = SqlitePatientSource().resolve(DEMO_PATIENT_HANDLE)
    diabetes_features = await extract_diabetes_features(user_message)
    merged_features = dict(base_record.features)
    merged_features.update(diabetes_features)

    extracted_age = _coerce_optional_float(diabetes_features.get("age"))
    return PatientRecord(
        handle=DEMO_PATIENT_HANDLE,
        age=int(extracted_age) if extracted_age is not None else base_record.age,
        sex=base_record.sex,
        features=merged_features,
        notes=(
            "Breast cancer screening features use the seeded demo record. "
            f"Diabetes features were extracted from user input: {user_message}"
        ),
    )


def _role_label(role: Role) -> str:
    cfg = RoleConfig.from_env(role)
    return f"{cfg.primary.provider.value}:{cfg.primary.model_id}"


class SinglePatientSource(PatientSource):
    def __init__(self, record: PatientRecord) -> None:
        self._record = record

    def resolve(self, handle: str) -> PatientRecord:
        if handle != self._record.handle:
            raise KeyError(f"Unknown patient handle: {handle}")
        return self._record

    def list_handles(self) -> list[str]:
        return [self._record.handle]


def _result_text(result: Any) -> str:
    structured = getattr(result, "output_structured", None)
    text = getattr(structured, "response", None)
    if text:
        return text

    answer = getattr(result, "answer", None)
    text = getattr(answer, "text", None)
    if text:
        return text

    return str(result)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _report_sections(text: str) -> dict[str, list[str]]:
    sections = {
        "Key Findings": [],
        "Recommended Follow-Up": [],
        "Clinical Note": [],
    }

    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if "recommend" in lowered or "follow-up" in lowered or "provide" in lowered:
            sections["Recommended Follow-Up"].append(sentence)
        elif "does not replace" in lowered or "supports clinical" in lowered:
            sections["Clinical Note"].append(sentence)
        else:
            sections["Key Findings"].append(sentence)

    return {title: items for title, items in sections.items() if items}


def _risk_tone(value: float) -> str:
    if value >= 85:
        return "high"
    if value >= 70:
        return "medium"
    return "low"


def _extract_metrics(text: str) -> list[dict[str, str]]:
    metrics = []

    breast_match = re.search(
        r"(?:breast|tumou?r|malignan\w*)[^.]{0,90}?(\d+(?:\.\d+)?)\s*%",
        text,
        flags=re.IGNORECASE,
    )
    if breast_match:
        value = float(breast_match.group(1))
        metrics.append(
            {
                "label": "Breast Screening",
                "value": f"{value:.1f}%",
                "caption": "high-risk flag",
                "tone": _risk_tone(value),
            }
        )

    diabetes_match = re.search(
        r"(?:diabetes|type-?2)[^.]{0,90}?(\d+(?:\.\d+)?)\s*%",
        text,
        flags=re.IGNORECASE,
    )
    if diabetes_match:
        value = float(diabetes_match.group(1))
        metrics.append(
            {
                "label": "Diabetes Risk",
                "value": f"{value:.1f}%",
                "caption": "elevated risk",
                "tone": _risk_tone(value),
            }
        )

    glucose_match = re.search(
        r"(?:glucose|plasma glucose)[^.]{0,40}?(\d+(?:\.\d+)?)\s*mg/dL",
        text,
        flags=re.IGNORECASE,
    )
    if glucose_match:
        metrics.append(
            {
                "label": "Glucose",
                "value": f"{float(glucose_match.group(1)):.0f} mg/dL",
                "caption": "lab value",
                "tone": "medium",
            }
        )

    return metrics


def _highlight_numbers(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(
        r"(\d+(?:\.\d+)?(?:\s?%|\s?mg/dL)?)",
        r"<span class='metric-inline'>\1</span>",
        escaped,
    )


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _missing_diabetes_fields(record: PatientRecord) -> list[str]:
    return [
        feature
        for feature in DIABETES_FEATURE_INFO
        if _is_missing(record.features.get(feature))
    ]


def _has_enough_diabetes_data(record: PatientRecord) -> bool:
    present = {
        feature
        for feature in KEY_DIABETES_FEATURES
        if not _is_missing(record.features.get(feature))
    }
    return len(present) >= 2


def _missing_report(record: PatientRecord, missing_fields: list[str]) -> str:
    known = [
        DIABETES_FEATURE_INFO[feature]["label"]
        for feature in DIABETES_FEATURES
        if not _is_missing(record.features.get(feature))
    ]
    known_text = ", ".join(known) if known else "no diabetes model values"
    return (
        "I need more health information before running the diabetes risk check. "
        f"Right now I found {known_text}. "
        "Please provide the missing values below if you have them. "
        "This system supports clinical judgement; it does not replace a clinician."
    )


def render_missing_info(missing_fields: list[str]) -> None:
    if not missing_fields:
        return

    st.markdown(
        (
            "<div class='missing-box'>"
            "<div class='missing-title'>More information needed</div>"
            "<div class='missing-subtitle'>"
            "The diabetes check is incomplete because these values are missing."
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    for feature in missing_fields:
        info = DIABETES_FEATURE_INFO[feature]
        st.markdown(
            (
                "<div class='missing-item'>"
                f"<div class='missing-label'>{info['label']}</div>"
                f"<div class='missing-detail'>{info['detail']}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_report(text: str, missing_fields: list[str] | None = None) -> None:
    sections = _report_sections(text)
    metrics = _extract_metrics(text)
    missing_fields = missing_fields or []

    st.markdown(
        """
        <style>
        .metric-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
            margin: 0.25rem 0 1rem;
        }
        .metric-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: rgba(15, 23, 42, 0.35);
        }
        .metric-label {
            color: #a7b0be;
            font-size: 0.82rem;
            margin-bottom: 0.15rem;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 750;
            line-height: 1.1;
        }
        .metric-caption {
            color: #a7b0be;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }
        .tone-high .metric-value, .metric-inline {
            color: #ff8a4c;
        }
        .tone-medium .metric-value {
            color: #facc15;
        }
        .tone-low .metric-value {
            color: #34d399;
        }
        .report-section {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }
        .report-title {
            font-weight: 750;
            margin-bottom: 0.4rem;
        }
        .report-section ul {
            margin-bottom: 0;
        }
        .missing-box {
            border: 1px solid rgba(250, 204, 21, 0.45);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.25rem 0 0.7rem;
            background: rgba(113, 63, 18, 0.16);
        }
        .missing-title {
            color: #facc15;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .missing-subtitle {
            color: #d6d3d1;
            font-size: 0.9rem;
        }
        .missing-item {
            display: grid;
            grid-template-columns: minmax(120px, 0.8fr) 1.6fr;
            gap: 0.75rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            padding: 0.65rem 0.1rem;
        }
        .missing-label {
            color: #f8fafc;
            font-weight: 700;
        }
        .missing-detail {
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Patient Report")
    if metrics:
        cards = "".join(
            (
                f"<div class='metric-card tone-{metric['tone']}'>"
                f"<div class='metric-label'>{metric['label']}</div>"
                f"<div class='metric-value'>{metric['value']}</div>"
                f"<div class='metric-caption'>{metric['caption']}</div>"
                "</div>"
            )
            for metric in metrics
        )
        st.markdown(f"<div class='metric-row'>{cards}</div>", unsafe_allow_html=True)

    for title, items in sections.items():
        body = "".join(f"<li>{_highlight_numbers(item)}</li>" for item in items)
        st.markdown(
            (
                "<div class='report-section'>"
                f"<div class='report-title'>{title}</div>"
                f"<ul>{body}</ul>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    render_missing_info(missing_fields)


async def _run_demo_analysis(user_message: str) -> tuple[str, list[str], list[str], dict[str, Any]]:
    record = await build_patient_record_from_input(user_message)
    missing_fields = _missing_diabetes_fields(record)

    if not _has_enough_diabetes_data(record):
        return _missing_report(record, missing_fields), [], missing_fields, record.features

    prompt = (
        f"Analyse patient `{DEMO_PATIENT_HANDLE}`. "
        "The breast cancer screening measurements come from the seeded demo record; "
        "the diabetes values come from the user's message. "
        "Use the available ML tools to assess their risk profile, then consult "
        "the medical expert to validate. Produce a final report for a non-technical "
        "patient-facing demo. Start with the plain-language conclusion, include "
        "confidence percentages only for models that were actually evaluated, "
        "key measured values, and avoid ML jargon. If diabetes values are missing, "
        "say the diabetes check needs more information instead of giving a risk score. "
        f"User note for this demo: {user_message}"
    )

    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    bundle = build_bundle(patient_source=SinglePatientSource(record))

    async with orchestrator_agent(bundle=bundle, llm=llm) as agent:
        result = await agent.run(prompt)

    return _result_text(result), list(bundle.enforcer.trajectory), missing_fields, record.features


def run_demo_analysis(user_message: str) -> tuple[str, list[str], list[str], dict[str, Any]]:
    return asyncio.run(_run_demo_analysis(user_message))


st.set_page_config(page_title="MARGE Demo", page_icon="M", layout="centered")

st.title("MARGE Demo")

with st.sidebar:
    st.caption("LLM")
    st.code(_role_label(Role.ORCHESTRATOR), language=None)
    st.caption("Patient")
    st.code(DEMO_PATIENT_HANDLE, language=None)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()
    show_debug_state = st.checkbox("Show session debug")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Tell me your age and any diabetes-related values you know.",
            "trajectory": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("report"):
            render_report(message["content"], message.get("missing_fields"))
        else:
            st.markdown(message["content"])

if show_debug_state:
    with st.sidebar.expander("Session state", expanded=True):
        st.json(st.session_state.get("messages", []))

user_input = st.chat_input("Message")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "trajectory": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("Running", expanded=False):
            try:
                response, trajectory, missing_fields, extracted_features = run_demo_analysis(user_input)
            except Exception as exc:  # noqa: BLE001 - surface demo failures in UI
                response = f"Run failed: `{type(exc).__name__}: {exc}`"
                trajectory = []
                missing_fields = []
                extracted_features = {}
                st.error(response)
            else:
                render_report(response, missing_fields)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "trajectory": trajectory,
            "report": True,
            "missing_fields": missing_fields,
            "extracted_features": extracted_features,
        }
    )

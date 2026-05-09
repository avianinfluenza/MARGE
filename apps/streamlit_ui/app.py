"""Streamlit demo UI for the MARGE orchestrator."""

import asyncio
from datetime import datetime, timezone
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from beeai_framework.backend.message import UserMessage
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

CHAT_LOG_PATH = ROOT / "logs" / "streamlit_chat.jsonl"

from packages.schemas.patient import PatientRecord
from packages.schemas.prediction import Prediction
from packages.llm_provider.client import build_chat_model_for_role
from packages.llm_provider.settings import Role, RoleConfig
from services.medical_expert_agent.agent import StubMedicalExpert
from services.ml_mcp_server.registry import discover_models
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


DEMO_PATIENT_HANDLE = "seed-001"


@dataclass(frozen=True)
class ModelFeatureSpec:
    name: str
    label: str
    detail: str
    aliases: list[str]


@dataclass(frozen=True)
class ModelRunPlan:
    name: str
    label: str
    model: Any
    features: list[ModelFeatureSpec]
    missing: list[ModelFeatureSpec]
    runnable: bool


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


def _model_label(model_name: str) -> str:
    label = model_name
    for prefix in ("predict_",):
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label.replace("_", " ").title()


def _feature_label(feature_name: str, field_info: Any) -> str:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    label = extra.get("label")
    if label:
        return str(label)
    return feature_name.replace("_", " ").title()


def _feature_detail(feature_name: str, field_info: Any) -> str:
    description = getattr(field_info, "description", None)
    if description:
        return str(description)
    return f"Value for {feature_name}."


def _feature_aliases(feature_name: str, field_info: Any) -> list[str]:
    extra = getattr(field_info, "json_schema_extra", None) or {}
    aliases = extra.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    return [str(alias) for alias in aliases if alias]


def _model_feature_specs(model: Any) -> list[ModelFeatureSpec]:
    specs = []
    for feature_name, field_info in model.input_schema.model_fields.items():
        specs.append(
            ModelFeatureSpec(
                name=feature_name,
                label=_feature_label(feature_name, field_info),
                detail=_feature_detail(feature_name, field_info),
                aliases=_feature_aliases(feature_name, field_info),
            )
        )
    return specs


def _all_dynamic_feature_specs(models: list[Any]) -> dict[str, ModelFeatureSpec]:
    specs: dict[str, ModelFeatureSpec] = {}
    for model in models:
        for spec in _model_feature_specs(model):
            specs.setdefault(spec.name, spec)
    return specs


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _regex_feature_fallback(
    user_message: str,
    feature_specs: dict[str, ModelFeatureSpec],
) -> dict[str, Any]:
    lowered = user_message.lower()
    features: dict[str, Any] = {}

    for feature_name, spec in feature_specs.items():
        terms = [feature_name, spec.label, *spec.aliases]
        for term in sorted({term.lower() for term in terms if term}, key=len, reverse=True):
            escaped_term = re.escape(term)
            match = re.search(
                rf"{escaped_term}[^\d-]{{0,24}}(-?\d+(?:\.\d+)?)",
                lowered,
            )
            if not match and feature_name == "age":
                match = re.search(r"(\d{1,3})\s*(?:살|세|years?\s*old)", lowered)
            if match:
                features[feature_name] = float(match.group(1))
                break

    return features


async def extract_features_from_message(
    user_message: str,
    feature_specs: dict[str, ModelFeatureSpec],
) -> dict[str, Any]:
    feature_schema = "\n".join(
        f'- "{name}": number or null. {spec.detail}'
        for name, spec in sorted(feature_specs.items())
    )
    alias_text = "\n".join(
        f"- {name}: {', '.join(spec.aliases)}"
        for name, spec in sorted(feature_specs.items())
        if spec.aliases
    )
    prompt = f"""
Extract clinical model features from the user's message.
Return only a JSON object. Do not include markdown.

Schema:
{{
  "age": number or null,
  "sex": "female" or "male" or "other" or null,
  "features": {{ ... }},
  "notes": string
}}

The "features" object must include exactly these keys:
{feature_schema}

Useful aliases:
{alias_text or "- No aliases provided."}

If a value is not explicitly provided, use null.
If two features are aliases for the same value, fill both keys. For example,
if the user provides blood sugar, fill both plas and glucose when both exist.

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
    fallback_features = _regex_feature_fallback(user_message, feature_specs)

    features: dict[str, Any] = {}
    for feature in feature_specs:
        value = extracted_features.get(feature, fallback_features.get(feature))
        features[feature] = _coerce_optional_float(value)

    age = _coerce_optional_float(data.get("age", features.get("age")))
    if age is not None:
        features["age"] = age

    return features


async def build_patient_record_from_input(user_message: str, models: list[Any]) -> PatientRecord:
    base_record = SqlitePatientSource().resolve(DEMO_PATIENT_HANDLE)
    dynamic_feature_specs = _all_dynamic_feature_specs(models)
    extracted_features = await extract_features_from_message(user_message, dynamic_feature_specs)
    merged_features = dict(base_record.features)
    seeded_feature_names = {
        spec.name
        for model in models
        if model.name == "predict_breast_cancer_malignancy"
        for spec in _model_feature_specs(model)
    }
    for feature_name, value in extracted_features.items():
        if (
            value is None
            and feature_name in seeded_feature_names
            and not _is_missing(merged_features.get(feature_name))
        ):
            continue
        merged_features[feature_name] = value

    extracted_age = _coerce_optional_float(extracted_features.get("age"))
    return PatientRecord(
        handle=DEMO_PATIENT_HANDLE,
        age=int(extracted_age) if extracted_age is not None else base_record.age,
        sex=base_record.sex,
        features=merged_features,
        notes=(
            "Breast cancer screening features use the seeded demo record. "
            f"Additional features were extracted from user input: {user_message}"
        ),
    )


def _role_label(role: Role) -> str:
    cfg = RoleConfig.from_env(role)
    return f"{cfg.primary.provider.value}:{cfg.primary.model_id}"


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


def _build_model_plans(record: PatientRecord, models: list[Any]) -> list[ModelRunPlan]:
    plans = []
    for model in models:
        specs = _model_feature_specs(model)
        missing = [spec for spec in specs if _is_missing(record.features.get(spec.name))]
        present_count = len(specs) - len(missing)
        min_required = max(1, math.ceil(len(specs) * 0.5))
        runnable = present_count >= min_required
        plans.append(
            ModelRunPlan(
                name=model.name,
                label=_model_label(model.name),
                model=model,
                features=specs,
                missing=missing,
                runnable=runnable,
            )
        )
    return plans


def _missing_report(plans: list[ModelRunPlan]) -> str:
    runnable_labels = [plan.label for plan in plans if plan.runnable]
    known_text = ", ".join(runnable_labels) if runnable_labels else "no runnable models"
    return (
        "I need more health information before running the available checks. "
        f"Right now I found {known_text}. "
        "Please provide the missing values below if you have them. "
        "This system supports clinical judgement; it does not replace a clinician."
    )


def _prediction_input_for_model(record: PatientRecord, model: Any) -> Any:
    data = {
        field_name: record.features.get(field_name)
        for field_name in model.input_schema.model_fields
    }
    return model.input_schema(**data)


def _confidence_percent(prediction: Prediction) -> str:
    return f"{prediction.confidence * 100:.1f}%"


def _top_feature_text(prediction: Prediction, limit: int = 3) -> str:
    items = []
    for score in prediction.xai_scores[:limit]:
        value = score.feature_value
        if value is None:
            items.append(score.feature_name)
        else:
            items.append(f"{score.feature_name} {value}")
    return ", ".join(items)


def _build_demo_report(predictions: list[Prediction], plans: list[ModelRunPlan]) -> str:
    by_name = {prediction.model_name: prediction for prediction in predictions}
    sentences: list[str] = []

    breast = by_name.get("predict_breast_cancer_malignancy")
    if breast:
        if breast.predicted_class == "malignant":
            sentences.append(
                "The breast screening model flagged a high-risk result "
                f"with {_confidence_percent(breast)} confidence."
            )
        else:
            sentences.append(
                "The breast screening model did not flag a high-risk result "
                f"with {_confidence_percent(breast)} confidence."
            )

    diabetes = by_name.get("predict_diabetes_risk")
    if diabetes:
        if diabetes.predicted_class == "diabetic_risk":
            sentences.append(
                "The diabetes model flagged elevated risk "
                f"with {_confidence_percent(diabetes)} confidence."
            )
        else:
            sentences.append(
                "The diabetes model did not flag elevated risk "
                f"with {_confidence_percent(diabetes)} confidence."
            )

        feature_text = _top_feature_text(diabetes)
        if feature_text:
            sentences.append(f"The most important diabetes values were {feature_text}.")

    for prediction in predictions:
        if prediction.model_name in {"predict_breast_cancer_malignancy", "predict_diabetes_risk"}:
            continue
        label = _model_label(prediction.model_name)
        sentences.append(
            f"The {label} model predicted {prediction.predicted_class} "
            f"with {_confidence_percent(prediction)} confidence."
        )

    missing_labels = [
        f"{plan.label}: {', '.join(spec.label for spec in plan.missing[:3])}"
        for plan in plans
        if plan.missing
    ]
    if missing_labels:
        sentences.append(f"For a more complete check, please provide: {'; '.join(missing_labels)}.")

    findings = {
        prediction.model_name: prediction.model_dump(mode="json")
        for prediction in predictions
    }
    StubMedicalExpert().consult(
        question="Summarize the model findings for a patient-facing demo.",
        findings=findings,
    )
    sentences.append("Please follow up with your doctor to interpret these results.")
    sentences.append("This system supports clinical judgement; it does not replace a clinician.")
    return " ".join(sentences)


def _run_model_pipeline(record: PatientRecord, plans: list[ModelRunPlan]) -> tuple[str, list[str]]:
    predictions: list[Prediction] = []
    trajectory = ["get_patient_history"]

    for plan in plans:
        if not plan.runnable:
            trajectory.append(f"{plan.name}: skipped_missing_features")
            continue
        try:
            inputs = _prediction_input_for_model(record, plan.model)
            predictions.append(plan.model.predict(inputs))
            trajectory.append(plan.name)
        except Exception as exc:
            trajectory.append(f"{plan.name}: failed ({type(exc).__name__})")

    if not predictions:
        return _missing_report(plans), trajectory

    trajectory.append("consult_medical_expert")
    trajectory.append("final_report")
    return _build_demo_report(predictions, plans), trajectory


def render_missing_info(plans: list[ModelRunPlan] | None) -> None:
    plans = plans or []
    plans_with_missing = [plan for plan in plans if plan.missing]
    if not plans_with_missing:
        return

    st.markdown(
        (
            "<div class='missing-box'>"
            "<div class='missing-title'>More information needed</div>"
            "<div class='missing-subtitle'>"
            "Some model checks are incomplete because these values are missing."
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    for plan in plans_with_missing:
        st.markdown(f"**{plan.label}**")
        for spec in plan.missing:
            st.markdown(
                (
                    "<div class='missing-item'>"
                    f"<div class='missing-label'>{spec.label}</div>"
                    f"<div class='missing-detail'>{spec.detail}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def _plans_from_state(raw_plans: list[dict[str, Any]] | None) -> list[ModelRunPlan]:
    plans = []
    for item in raw_plans or []:
        missing = [
            ModelFeatureSpec(
                name=feature["name"],
                label=feature["label"],
                detail=feature["detail"],
                aliases=feature.get("aliases", []),
            )
            for feature in item.get("missing", [])
        ]
        plans.append(
            ModelRunPlan(
                name=item["name"],
                label=item["label"],
                model=None,
                features=[],
                missing=missing,
                runnable=item.get("runnable", False),
            )
        )
    return plans


def _plans_to_state(plans: list[ModelRunPlan]) -> list[dict[str, Any]]:
    return [
        {
            "name": plan.name,
            "label": plan.label,
            "runnable": plan.runnable,
            "missing": [
                {
                    "name": spec.name,
                    "label": spec.label,
                    "detail": spec.detail,
                    "aliases": spec.aliases,
                }
                for spec in plan.missing
            ],
        }
        for plan in plans
    ]


def _append_chat_log(
    user_input: str,
    response: str,
    trajectory: list[str],
    model_plans: list[ModelRunPlan],
    extracted_features: dict[str, Any],
    error: str | None = None,
) -> None:
    CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "assistant_response": response,
        "trajectory": trajectory,
        "models": _plans_to_state(model_plans),
        "extracted_features": extracted_features,
        "error": error,
    }
    with CHAT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def render_report(text: str, plans: list[ModelRunPlan] | None = None) -> None:
    sections = _report_sections(text)
    metrics = _extract_metrics(text)
    plans = plans or []

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

    render_missing_info(plans)


async def _run_demo_analysis(user_message: str) -> tuple[str, list[str], list[ModelRunPlan], dict[str, Any]]:
    models = discover_models()
    record = await build_patient_record_from_input(user_message, models)
    plans = _build_model_plans(record, models)
    response, trajectory = _run_model_pipeline(record, plans)
    return response, trajectory, plans, record.features


def run_demo_analysis(user_message: str) -> tuple[str, list[str], list[ModelRunPlan], dict[str, Any]]:
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
            render_report(message["content"], _plans_from_state(message.get("model_plans")))
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
            error = None
            try:
                response, trajectory, model_plans, extracted_features = run_demo_analysis(user_input)
            except Exception as exc:  # noqa: BLE001 - surface demo failures in UI
                response = f"Run failed: `{type(exc).__name__}: {exc}`"
                trajectory = []
                model_plans = []
                extracted_features = {}
                error = f"{type(exc).__name__}: {exc}"
                st.error(response)
            else:
                render_report(response, model_plans)
            finally:
                _append_chat_log(
                    user_input=user_input,
                    response=response,
                    trajectory=trajectory,
                    model_plans=model_plans,
                    extracted_features=extracted_features,
                    error=error,
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "trajectory": trajectory,
            "report": True,
            "model_plans": _plans_to_state(model_plans),
            "extracted_features": extracted_features,
        }
    )

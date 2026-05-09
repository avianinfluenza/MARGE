"""Streamlit demo UI for the MARGE orchestrator."""

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from apps.orchestrator.agent import build_bundle, orchestrator_agent
from packages.llm_provider.client import build_chat_model_for_role
from packages.llm_provider.settings import Role, RoleConfig


DEMO_PATIENT_HANDLE = "seed-001"


def _role_label(role: Role) -> str:
    cfg = RoleConfig.from_env(role)
    return f"{cfg.primary.provider.value}:{cfg.primary.model_id}"


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


def render_report(text: str) -> None:
    sections = _report_sections(text)
    metrics = _extract_metrics(text)

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


async def _run_demo_analysis(user_message: str) -> tuple[str, list[str]]:
    prompt = (
        f"Analyse seed patient `{DEMO_PATIENT_HANDLE}`. "
        "Use the available ML tools to assess their risk profile, then consult "
        "the medical expert to validate. Produce a final report for a non-technical "
        "patient-facing demo. Start with the plain-language conclusion, include "
        "the breast screening confidence percentage, the diabetes risk confidence "
        "percentage, key measured values, and avoid ML jargon. "
        f"User note for this demo: {user_message}"
    )

    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    bundle = build_bundle()

    async with orchestrator_agent(bundle=bundle, llm=llm) as agent:
        result = await agent.run(prompt)

    return _result_text(result), list(bundle.enforcer.trajectory)


def run_demo_analysis(user_message: str) -> tuple[str, list[str]]:
    return asyncio.run(_run_demo_analysis(user_message))


st.set_page_config(page_title="MARGE Demo", page_icon="M", layout="centered")

st.title("MARGE Demo")

with st.sidebar:
    st.caption("LLM")
    st.code(_role_label(Role.ORCHESTRATOR), language=None)
    st.caption("Patient")
    st.code(DEMO_PATIENT_HANDLE, language=None)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"Ready for `{DEMO_PATIENT_HANDLE}`.",
            "trajectory": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("report"):
            render_report(message["content"])
        else:
            st.markdown(message["content"])

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
                response, trajectory = run_demo_analysis(user_input)
            except Exception as exc:  # noqa: BLE001 - surface demo failures in UI
                response = f"Run failed: `{type(exc).__name__}: {exc}`"
                trajectory = []
                st.error(response)
            else:
                render_report(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "trajectory": trajectory,
            "report": True,
        }
    )

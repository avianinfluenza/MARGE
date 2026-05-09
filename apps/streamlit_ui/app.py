"""Streamlit demo UI for the MARGE orchestrator."""

import asyncio
from datetime import datetime, timezone
import json
import queue as _queue
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from beeai_framework.backend.message import UserMessage
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = ROOT / "sessions"
CHAT_LOG_PATH = ROOT / "logs" / "streamlit_chat.jsonl"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from apps.orchestrator.agent import build_bundle, orchestrator_agent
from packages.llm_provider.client import build_chat_model_for_role
from packages.llm_provider.settings import Role, RoleConfig
from services.patient_data_mcp_server.sources.csv_ingest import ingest_csv, init_empty_db


# ---------------------------------------------------------------------------
# Session DB management
# ---------------------------------------------------------------------------

def _get_or_create_session_db() -> Path:
    """Return the session-scoped SQLite path, creating an empty DB if needed."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:8]

    db_path = SESSIONS_DIR / f"{st.session_state['session_id']}.db"
    if not db_path.exists():
        init_empty_db(db_path)
        st.session_state["patients"] = []
        st.session_state["current_patient"] = None
    return db_path


def _reset_session() -> None:
    for key in ("session_id", "patients", "current_patient", "messages", "csv_file_id"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _report_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
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
        metrics.append({"label": "Breast Screening", "value": f"{value:.1f}%",
                         "caption": "high-risk flag", "tone": _risk_tone(value)})
    diabetes_match = re.search(
        r"(?:diabetes|type-?2)[^.]{0,90}?(\d+(?:\.\d+)?)\s*%",
        text,
        flags=re.IGNORECASE,
    )
    if diabetes_match:
        value = float(diabetes_match.group(1))
        metrics.append({"label": "Diabetes Risk", "value": f"{value:.1f}%",
                         "caption": "elevated risk", "tone": _risk_tone(value)})
    glucose_match = re.search(
        r"(?:glucose|plasma glucose)[^.]{0,40}?(\d+(?:\.\d+)?)\s*mg/dL",
        text,
        flags=re.IGNORECASE,
    )
    if glucose_match:
        metrics.append({"label": "Glucose", "value": f"{float(glucose_match.group(1)):.0f} mg/dL",
                         "caption": "lab value", "tone": "medium"})
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

    st.markdown("""
        <style>
        .metric-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
                      gap:.75rem; margin:.25rem 0 1rem; }
        .metric-card { border:1px solid rgba(148,163,184,.28); border-radius:8px;
                       padding:.75rem .85rem; background:rgba(15,23,42,.35); }
        .metric-label { color:#a7b0be; font-size:.82rem; margin-bottom:.15rem; }
        .metric-value { font-size:1.45rem; font-weight:750; line-height:1.1; }
        .metric-caption { color:#a7b0be; font-size:.78rem; margin-top:.2rem; }
        .tone-high .metric-value, .metric-inline { color:#ff8a4c; }
        .tone-medium .metric-value { color:#facc15; }
        .tone-low .metric-value { color:#34d399; }
        .report-section { border:1px solid rgba(148,163,184,.28); border-radius:8px;
                          padding:.85rem 1rem; margin-bottom:.75rem; }
        .report-title { font-weight:750; margin-bottom:.4rem; }
        .report-section ul { margin-bottom:0; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### Patient Report")
    if metrics:
        cards = "".join(
            f"<div class='metric-card tone-{m['tone']}'>"
            f"<div class='metric-label'>{m['label']}</div>"
            f"<div class='metric-value'>{m['value']}</div>"
            f"<div class='metric-caption'>{m['caption']}</div></div>"
            for m in metrics
        )
        st.markdown(f"<div class='metric-row'>{cards}</div>", unsafe_allow_html=True)

    for title, items in sections.items():
        body = "".join(f"<li>{_highlight_numbers(item)}</li>" for item in items)
        st.markdown(
            f"<div class='report-section'><div class='report-title'>{title}</div>"
            f"<ul>{body}</ul></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Chat log
# ---------------------------------------------------------------------------

def _append_chat_log(
    session_id: str,
    patient_handle: str,
    user_input: str,
    response: str,
    trajectory: list[str],
    error: str | None = None,
) -> None:
    CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "patient_handle": patient_handle,
        "user_input": user_input,
        "assistant_response": response,
        "trajectory": trajectory,
        "error": error,
    }
    with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Orchestrator runner
# ---------------------------------------------------------------------------

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


def _role_label(role: Role) -> str:
    cfg = RoleConfig.from_env(role)
    return f"{cfg.primary.provider.value}:{cfg.primary.model_id}"


def stream_analysis(
    user_message: str,
    patient_handle: str | None,
    db_path: Path,
    state: dict,
) -> Any:
    """Return a generator that streams tool-call lines then the final response.

    Populates `state` with 'response', 'trajectory', and 'error' when done.
    Designed for use with st.write_stream().
    """
    eq: _queue.Queue = _queue.Queue()

    async def _run() -> None:
        if patient_handle:
            prompt = (
                f"Current patient handle: `{patient_handle}`. "
                "If the user's message contains new clinical values (glucose, BMI, blood pressure, "
                "age, insulin, pregnancy count, family history score, etc.), call `update_patient` "
                "to persist them before running the ML tools. "
                f"User message: {user_message}"
            )
            patient_db_path: Path | None = db_path
        else:
            prompt = user_message
            patient_db_path = None

        llm = build_chat_model_for_role(Role.ORCHESTRATOR)
        bundle = build_bundle()

        # Intercept enforcer.record so every tool call streams to the queue
        _orig_record = bundle.enforcer.record
        def _streaming_record(tool_name: str) -> None:
            _orig_record(tool_name)
            eq.put(("tool", tool_name))
        bundle.enforcer.record = _streaming_record

        try:
            async with orchestrator_agent(bundle=bundle, llm=llm, patient_db_path=patient_db_path) as agent:
                result = await agent.run(prompt)
            state["response"] = _result_text(result)
            state["trajectory"] = list(bundle.enforcer.trajectory)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["response"] = f"Run failed: `{state['error']}`"
            state["trajectory"] = list(bundle.enforcer.trajectory)
        finally:
            eq.put(None)

    threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()

    def _generator():
        while True:
            item = eq.get(timeout=180)
            if item is None:
                break
            _, tool_name = item
            yield f"🔧 `{tool_name}`\n\n"
        yield state.get("response", "")

    return _generator()


# ---------------------------------------------------------------------------
# Streamlit UI — only runs inside the Streamlit runtime
# ---------------------------------------------------------------------------

def _app_main() -> None:
    st.set_page_config(page_title="MARGE Demo", page_icon="M", layout="centered")
    st.title("MARGE Demo")

    db_path = _get_or_create_session_db()

    # --- Sidebar ---
    with st.sidebar:
        st.caption("LLM")
        st.code(_role_label(Role.ORCHESTRATOR), language=None)

        st.markdown("---")
        st.markdown("**Patient Data**")

        uploaded_csv = st.file_uploader("Upload patient CSV", type="csv")
        if uploaded_csv is not None:
            file_id = f"{uploaded_csv.name}_{uploaded_csv.size}"
            if st.session_state.get("csv_file_id") != file_id:
                with st.spinner("Importing patients…"):
                    handles = ingest_csv(uploaded_csv.read(), db_path)
                st.session_state["patients"] = handles
                st.session_state["current_patient"] = handles[0]
                st.session_state["csv_file_id"] = file_id
                st.session_state.pop("messages", None)
                st.success(f"Loaded {len(handles)} patient(s).")

        patients: list[str] = st.session_state.get("patients", [])
        if not patients:
            st.caption("No patients loaded. Upload a CSV to start.")
            st.session_state["current_patient"] = None
        elif len(patients) == 1:
            st.caption(f"Patient: `{patients[0]}`")
            st.session_state["current_patient"] = patients[0]
        else:
            current = st.selectbox(
                "Active patient",
                patients,
                index=patients.index(st.session_state.get("current_patient", patients[0])),
            )
            if current != st.session_state.get("current_patient"):
                st.session_state["current_patient"] = current
                st.session_state.pop("messages", None)

        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()
        if st.button("Reset session", use_container_width=True):
            _reset_session()
            st.rerun()
        if st.checkbox("Show session debug"):
            with st.expander("Session state", expanded=True):
                st.json(st.session_state.get("messages", []))

    # --- Chat ---
    current_patient = st.session_state.get("current_patient")
    no_patients = not current_patient

    if "messages" not in st.session_state:
        welcome = (
            "Upload a patient CSV from the sidebar to get started."
            if no_patients else
            "Tell me about the patient, or share any clinical values you have "
            "(age, blood sugar, BMI, blood pressure, etc.) and I'll run the analysis."
        )
        st.session_state.messages = [{"role": "assistant", "content": welcome, "trajectory": []}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            traj = message.get("trajectory") or []
            for tool in traj:
                st.markdown(f"🔧 `{tool}`")
            st.markdown(message["content"])

    user_input = st.chat_input("Message")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "trajectory": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            stream_state: dict = {"response": "", "trajectory": [], "error": None}
            st.write_stream(stream_analysis(user_input, current_patient, db_path, stream_state))

        response = stream_state["response"]
        trajectory = stream_state["trajectory"]
        error_msg = stream_state["error"]

        _append_chat_log(
            session_id=st.session_state.get("session_id", "unknown"),
            patient_handle=current_patient,
            user_input=user_input,
            response=response,
            trajectory=trajectory,
            error=error_msg,
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "trajectory": trajectory}
        )


# Guard: only execute UI when running inside the Streamlit runtime.
# This prevents module-level Streamlit calls from firing during pytest import.
def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if _in_streamlit():
    _app_main()

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
    events: list[dict] | None = None,
) -> None:
    """Append a turn record to the JSONL chat log.

    `events` is a structured trace of everything that happened during the
    turn (LLM reasoning text, tool calls, tool outputs) for debug / replay.
    """
    CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "patient_handle": patient_handle,
        "user_input": user_input,
        "assistant_response": response,
        "trajectory": trajectory,
        "events": events or [],
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


def _extract_token_text(token_event_data: Any) -> str:
    """Extract the delta text from a BeeAI ChatModelNewTokenEvent.

    Each new_token event delivers a ChatModelOutput whose `output` list
    contains a single AssistantMessage. The message's text content is the
    delta token (not accumulated).
    """
    out = getattr(token_event_data, "value", None)
    if out is None or not getattr(out, "output", None):
        return ""
    msgs = out.output
    if not msgs:
        return ""
    msg = msgs[0]
    if hasattr(msg, "get_text_content"):
        try:
            return msg.get_text_content() or ""
        except Exception:
            return ""
    return str(msg)


def stream_analysis(
    user_message: str,
    patient_handle: str | None,
    db_path: Path,
    state: dict,
) -> Any:
    """Generator for st.write_stream that streams LLM tokens + tool events live.

    Yields, in real time:
    - LLM reasoning text (token by token from `new_token` emitter event)
    - Tool call markers ("🔧 `tool_name`(...args)") on each tool start
    - Tool output snippets on each tool success/error

    Populates `state` with:
    - 'response': final user-facing text
    - 'trajectory': enforcer-recorded tool sequence
    - 'events': structured list of every event in this turn (for JSONL log)
    - 'error': exception string if the run failed
    """
    eq: _queue.Queue = _queue.Queue()
    state.setdefault("events", [])
    events: list[dict] = state["events"]

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

        # Hook 1: enforcer record (existing behaviour) — also queue tool name
        _orig_record = bundle.enforcer.record
        def _streaming_record(tool_name: str) -> None:
            _orig_record(tool_name)
            eq.put(("tool_call", {"name": tool_name, "input": None}))
        bundle.enforcer.record = _streaming_record

        # Hook 2: LLM token streaming
        def _on_llm_event(data: Any, event: Any) -> None:
            if getattr(event, "name", "") != "new_token":
                return
            chunk = _extract_token_text(data)
            if chunk:
                eq.put(("token", chunk))

        try:
            llm.emitter.match("*", _on_llm_event)
        except Exception:
            pass  # if emitter API differs, silently skip — tool streaming still works

        try:
            async with orchestrator_agent(bundle=bundle, llm=llm, patient_db_path=patient_db_path) as agent:
                # Hook 3: per-tool start/success/error (input + output capture)
                for tool in agent._tools:
                    def make_recorder(tool_name: str):
                        pending: dict = {}

                        def on_evt(data: Any, event: Any) -> None:
                            name = getattr(event, "name", "")
                            if name == "start":
                                pending["input"] = getattr(data, "input", None)
                            elif name == "success":
                                output = getattr(data, "output", None)
                                output_repr: Any = None
                                if output is not None:
                                    if hasattr(output, "to_json_safe"):
                                        try:
                                            output_repr = output.to_json_safe()
                                        except Exception:
                                            output_repr = str(output)[:1500]
                                    else:
                                        output_repr = str(output)[:1500]
                                eq.put((
                                    "tool_output",
                                    {"name": tool_name, "input": pending.get("input"),
                                     "output": output_repr, "success": True},
                                ))
                                pending.clear()
                            elif name == "error":
                                err = getattr(data, "error", None)
                                eq.put((
                                    "tool_output",
                                    {"name": tool_name, "input": pending.get("input"),
                                     "error": repr(err)[:500], "success": False},
                                ))
                                pending.clear()

                        return on_evt

                    try:
                        tool.emitter.match("*", make_recorder(tool.name))
                    except Exception:
                        pass

                result = await agent.run(prompt, stream=True)
            state["response"] = _result_text(result)
            state["trajectory"] = list(bundle.enforcer.trajectory)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["response"] = f"Run failed: `{state['error']}`"
            state["trajectory"] = list(bundle.enforcer.trajectory)
        finally:
            eq.put(None)

    threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()

    def _to_jsonable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(x) for x in obj]
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump(mode="json")
            except Exception:
                pass
        return repr(obj)[:500]

    def _generator():
        while True:
            item = eq.get(timeout=300)
            if item is None:
                break
            kind, payload = item
            if kind == "token":
                events.append({"kind": "reasoning_token", "text": payload})
                yield payload  # typewriter effect
            elif kind == "tool_call":
                events.append({"kind": "tool_call", **_to_jsonable(payload)})
                yield f"\n\n🔧 `{payload['name']}`\n\n"
            elif kind == "tool_output":
                events.append({"kind": "tool_output", **_to_jsonable(payload)})
                # Brief inline echo (truncated) so the user sees tool finished
                out_str = json.dumps(payload.get("output"), ensure_ascii=False, default=str)
                snippet = (out_str[:140] + "…") if len(out_str) > 140 else out_str
                marker = "↳" if payload.get("success") else "⚠"
                yield f"  {marker} `{snippet}`\n\n"
        # Don't double-yield the final response — tokens already streamed it.

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
            stream_state: dict = {"response": "", "trajectory": [], "error": None, "events": []}
            st.write_stream(stream_analysis(user_input, current_patient, db_path, stream_state))

        response = stream_state["response"]
        trajectory = stream_state["trajectory"]
        error_msg = stream_state["error"]
        turn_events = stream_state.get("events", [])

        _append_chat_log(
            session_id=st.session_state.get("session_id", "unknown"),
            patient_handle=current_patient,
            user_input=user_input,
            response=response,
            trajectory=trajectory,
            error=error_msg,
            events=turn_events,
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

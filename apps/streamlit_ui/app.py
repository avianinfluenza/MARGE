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
LOGS_DIR = ROOT / "logs"
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
    for key in (
        "session_id", "patients", "current_patient", "messages",
        "csv_file_id", "orch_memory", "expert",
    ):
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


def render_clinical_report_card(payload: dict) -> None:
    """Render a structured clinical_report payload (new schema).

    payload shape: {summary, recommendation, confidence, evidence: [...],
                    expert_quote, safety_note}
    """
    summary = payload.get("summary", "")
    recommendation = payload.get("recommendation", "")
    confidence = payload.get("confidence", "medium")
    evidence = payload.get("evidence") or []
    expert_quote = payload.get("expert_quote")
    safety = payload.get("safety_note", "")

    tone_color = {"low": "#facc15", "medium": "#facc15", "high": "#34d399"}.get(confidence, "#a7b0be")

    st.markdown(f"""
        <style>
        .cr-card {{ border:1px solid rgba(148,163,184,.32); border-radius:10px;
                    padding:1rem 1.1rem; margin:.4rem 0 .6rem;
                    background:rgba(15,23,42,.45); }}
        .cr-title {{ font-weight:750; font-size:1.05rem; margin-bottom:.4rem; }}
        .cr-conf {{ display:inline-block; padding:.1rem .55rem; border-radius:999px;
                    font-size:.75rem; font-weight:650; margin-left:.4rem;
                    background:rgba(148,163,184,.18); color:{tone_color}; }}
        .cr-section {{ margin-top:.6rem; }}
        .cr-h {{ font-weight:650; color:#cbd5e1; font-size:.85rem; margin-bottom:.15rem; }}
        .cr-quote {{ border-left:3px solid #94a3b8; padding-left:.7rem; color:#cbd5e1;
                     font-style:italic; margin:.3rem 0; }}
        .cr-evid {{ font-size:.86rem; padding:.45rem .6rem;
                    border:1px solid rgba(148,163,184,.22); border-radius:6px;
                    margin:.25rem 0; background:rgba(15,23,42,.32); }}
        .cr-safety {{ font-size:.78rem; color:#a7b0be; margin-top:.7rem;
                      padding-top:.5rem; border-top:1px dashed rgba(148,163,184,.25); }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='cr-card'>"
                f"<div class='cr-title'>Clinical report"
                f"<span class='cr-conf'>{confidence.upper()}</span></div>"
                f"<div class='cr-section'><div class='cr-h'>Summary</div>{summary}</div>"
                f"<div class='cr-section'><div class='cr-h'>Recommendation</div>{recommendation}</div>",
                unsafe_allow_html=True)

    if evidence:
        evid_html = "".join(
            f"<div class='cr-evid'><b>{e.get('model','?')}</b> → "
            f"{e.get('predicted_class','?')} (conf {e.get('confidence',0):.2f})</div>"
            for e in evidence
        )
        st.markdown(f"<div class='cr-section'><div class='cr-h'>ML evidence</div>{evid_html}</div>",
                    unsafe_allow_html=True)

    if expert_quote:
        st.markdown(f"<div class='cr-section'><div class='cr-h'>Expert insight</div>"
                    f"<div class='cr-quote'>{expert_quote}</div></div>", unsafe_allow_html=True)

    if safety:
        st.markdown(f"<div class='cr-safety'>{safety}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)


def render_abstain_card(payload: dict) -> None:
    reason = payload.get("reason", "")
    fallback = payload.get("fallback_recommendation", "")
    st.warning(f"**Cannot reliably advise**\n\n{reason}\n\n**Suggested next step:** {fallback}")


def render_request_more_info_card(payload: dict) -> None:
    rationale = payload.get("rationale", "")
    needed = payload.get("needed") or []
    st.info(f"**Need more info**\n\n{rationale}")
    if needed:
        rows = "\n".join(
            f"- **{n.get('name')}** ({n.get('field_type','text')}"
            f"{', ' + n.get('unit') if n.get('unit') else ''}): {n.get('why','')}"
            for n in needed
        )
        st.markdown(rows)


def _terminal_payload_from_events(events: list[dict]) -> tuple[str | None, dict | None]:
    """Find the last terminal tool call in this turn's events; return (name, input).

    Only structured terminals (clinical_report / abstain / request_more_info)
    have UI cards. Casual chat lives in the streamed natural-language content
    and produces no terminal record.
    """
    terminals = {"clinical_report", "abstain", "request_more_info"}
    for e in reversed(events):
        if (
            e.get("kind") == "tool_call"
            and e.get("name") in terminals
            and e.get("agent", "orchestrator") == "orchestrator"
        ):
            # The structured payload may live on either tool_call or tool_output.
            inp = e.get("input")
            if inp:
                return e["name"], inp
            # Fall back to the matching tool_output (which carries input)
            for f in reversed(events):
                if (f.get("kind") == "tool_output" and f.get("name") == e["name"]
                        and f.get("input")):
                    return e["name"], f["input"]
            return e["name"], None
    return None, None


def render_terminal_card(events: list[dict]) -> None:
    name, payload = _terminal_payload_from_events(events)
    if not name or not payload:
        return
    if name == "clinical_report":
        render_clinical_report_card(payload)
    elif name == "abstain":
        render_abstain_card(payload)
    elif name == "request_more_info":
        render_request_more_info_card(payload)


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

def _chat_log_path_for_session(session_id: str) -> Path:
    """One JSONL file per chat session — appended to across all turns of that
    session, but isolated from other sessions and prior runs.

    File: logs/streamlit_chat_<session_id>.jsonl
    """
    return LOGS_DIR / f"streamlit_chat_{session_id}.jsonl"


def _append_chat_log(
    session_id: str,
    patient_handle: str,
    user_input: str,
    response: str,
    trajectory: list[str],
    error: str | None = None,
    events: list[dict] | None = None,
) -> None:
    """Append a turn record to this session's JSONL chat log.

    `events` is a structured trace of everything that happened during the
    turn (LLM reasoning text, tool calls, tool outputs) for debug / replay.
    """
    log_path = _chat_log_path_for_session(session_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    with log_path.open("a", encoding="utf-8") as f:
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


def _get_or_create_orch_memory():
    """Persist BeeAI conversation memory across user turns (Streamlit reruns).

    The orchestrator's UnconstrainedMemory carries the user prompt, all tool
    calls, and assistant messages. Reusing the same instance across turns is
    what makes multi-turn data gathering ("now share HbA1c") feel natural.
    """
    if "orch_memory" not in st.session_state:
        from beeai_framework.memory import UnconstrainedMemory
        st.session_state["orch_memory"] = UnconstrainedMemory()
    return st.session_state["orch_memory"]


def _get_or_create_expert():
    """Persist the real MedicalExpertAgent across user turns.

    The expert has its own LLM (Role.MEDICAL_EXPERT — Featherless Kimi K2.5
    by default) and its own conversation memory, separate from the
    orchestrator's. The orchestrator passes findings as clinical values;
    the expert reasons in clinical terms. Same instance is reused so the
    expert remembers prior consultations within a session.
    """
    if "expert" not in st.session_state:
        from services.medical_expert_agent.agent import MedicalExpertAgent
        st.session_state["expert"] = MedicalExpertAgent.from_env()
    return st.session_state["expert"]


_TOOL_PROGRESS_FALLBACK = {
    "consult_medical_expert":      "🩺 Consulting the medical expert…",
    "search_medical_web":          "🔎 Searching medical web sources…",
    "predict_diabetes_risk":       "📊 Running the diabetes risk model…",
    "predict_breast_cancer_malignancy": "📊 Running the breast-tumor malignancy model…",
    "get_patient":                 "📁 Fetching patient record…",
    "list_patients":               "📁 Listing patients…",
    "update_patient":              "📁 Updating patient record…",
    "request_more_info":           "❓ Requesting additional information…",
    "clinical_report":             "📝 Drafting the clinical report…",
    "abstain":                     "⚠ Explaining the scope boundary…",
}


def _tool_progress_message(tool_name: str, agent: str | None = None) -> str:
    message = _TOOL_PROGRESS_FALLBACK.get(
        tool_name, f"🔧 **{tool_name}** _running…_"
    )
    if agent == "expert":
        return f"Expert tool: {message}"
    return message


def _log(kind: str, payload: Any) -> None:
    """Print a one-line trace event to stderr so /tmp/streamlit.log shows
    real-time progress while a turn is still mid-flight (jsonl only flushes
    after the whole turn ends)."""
    ts = datetime.now().strftime("%H:%M:%S")
    snippet = json.dumps(payload, ensure_ascii=False, default=str)[:300]
    print(f"[{ts}] [trace] {kind}: {snippet}", file=sys.stderr, flush=True)


def render_events_timeline(events: list[dict]) -> None:
    """Render a complete per-turn trace inside an expander.

    Groups consecutive reasoning_token deltas into a single reasoning
    block, then interleaves tool_call (with input) and tool_output (with
    output / error) in execution order. Always available for chat
    history messages so users can see exactly what the LLM thought and
    what each tool returned without leaving the UI.
    """
    if not events:
        return

    blocks: list[tuple[str, Any]] = []
    cur_text: list[str] = []
    for e in events:
        kind = e.get("kind")
        if kind == "reasoning_token":
            cur_text.append(e.get("text", ""))
        elif kind in ("tool_call", "tool_output"):
            if cur_text:
                blocks.append(("reasoning", "".join(cur_text)))
                cur_text = []
            blocks.append((kind, e))
    if cur_text:
        blocks.append(("reasoning", "".join(cur_text)))

    n_tools = sum(1 for k, _ in blocks if k == "tool_call")
    n_errs = sum(
        1 for k, p in blocks
        if k == "tool_output" and isinstance(p, dict) and not p.get("success", True)
    )
    n_reason = sum(1 for k, _ in blocks if k == "reasoning")
    label_parts = []
    if n_reason: label_parts.append(f"{n_reason} reasoning step(s)")
    if n_tools:  label_parts.append(f"{n_tools} tool call(s)")
    if n_errs:   label_parts.append(f"{n_errs} error(s)")
    label = "Trace · " + (", ".join(label_parts) if label_parts else "no events")

    with st.expander(label, expanded=bool(n_errs)):
        for i, (kind, payload) in enumerate(blocks):
            if kind == "reasoning":
                if payload.strip():
                    st.markdown(f"**[{i:02d}] 🧠 reasoning**")
                    st.markdown(
                        f"<div style='border-left:3px solid #94a3b8; padding-left:.7rem; "
                        f"color:#cbd5e1; white-space:pre-wrap; font-size:.88rem;'>"
                        f"{payload}</div>",
                        unsafe_allow_html=True,
                    )
            elif kind == "tool_call":
                inp = payload.get("input")
                inp_text = (
                    json.dumps(inp, indent=2, ensure_ascii=False, default=str)
                    if inp else "(no input)"
                )
                agent_name = payload.get("agent", "orchestrator")
                st.markdown(
                    f"**[{i:02d}] → call** `{agent_name}.{payload.get('name')}`"
                )
                st.code(inp_text, language="json")
            elif kind == "tool_output":
                ok = payload.get("success", True)
                tag = "✓ ok" if ok else "✗ ERROR"
                pl = payload.get("output") if ok else payload.get("error")
                pl_text = (
                    json.dumps(pl, indent=2, ensure_ascii=False, default=str)
                    if isinstance(pl, (dict, list))
                    else str(pl)
                )
                agent_name = payload.get("agent", "orchestrator")
                st.markdown(
                    f"**[{i:02d}] ← return** `{agent_name}.{payload.get('name')}` · _{tag}_"
                )
                st.code(pl_text, language="json" if ok else "text")


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

    # Capture the per-session memory and expert in the main Streamlit thread
    # before launching the worker thread (st.session_state access from the
    # worker thread is unreliable).
    orch_memory = _get_or_create_orch_memory()
    expert = _get_or_create_expert()

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
        bundle = build_bundle(expert=expert)

        _log("turn_start", {"prompt": prompt[:120]})

        # Hook 1: enforcer record (existing behaviour) — also queue tool name
        _orig_record = bundle.enforcer.record
        def _streaming_record(tool_name: str) -> None:
            _orig_record(tool_name)
            eq.put((
                "tool_call",
                {"agent": "orchestrator", "name": tool_name, "input": None},
            ))
        bundle.enforcer.record = _streaming_record

        if hasattr(expert, "set_event_sink"):
            expert.set_event_sink(lambda event: eq.put(("expert_event", event)))

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
            async with orchestrator_agent(
                bundle=bundle, llm=llm, patient_db_path=patient_db_path, memory=orch_memory,
            ) as agent:
                # Hook 3: per-tool start/success/error (input + output capture)
                for tool in agent._tools:
                    def make_recorder(tool_name: str):
                        pending: dict = {}

                        def on_evt(data: Any, event: Any) -> None:
                            name = getattr(event, "name", "")
                            if name == "start":
                                pending["input"] = getattr(data, "input", None)
                                _log("tool_start", {"name": tool_name,
                                                     "input": pending["input"]})
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
                                _log("tool_ok", {"name": tool_name,
                                                  "output": output_repr})
                                eq.put((
                                    "tool_output",
                                    {"agent": "orchestrator", "name": tool_name,
                                     "input": pending.get("input"),
                                     "output": output_repr, "success": True},
                                ))
                                pending.clear()
                            elif name == "error":
                                err = getattr(data, "error", None)
                                err_chain = []
                                cur = err
                                for _ in range(6):
                                    if cur is None: break
                                    err_chain.append(f"{type(cur).__name__}: {str(cur)[:200]}")
                                    cur = getattr(cur, "__cause__", None)
                                err_str = " ⇠ ".join(err_chain) if err_chain else repr(err)[:500]
                                _log("tool_err", {"name": tool_name, "error": err_str})
                                eq.put((
                                    "tool_output",
                                    {"agent": "orchestrator", "name": tool_name,
                                     "input": pending.get("input"),
                                     "error": err_str, "success": False},
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
            # Walk the __cause__ chain so the user sees the real underlying
            # error (e.g., LiteLLM RateLimitError, OpenAIError 5xx, ...)
            chain = []
            cur: BaseException | None = exc
            for _ in range(6):
                if cur is None:
                    break
                chain.append(f"{type(cur).__name__}: {str(cur)[:200]}")
                cur = cur.__cause__
            state["error"] = " ⇠ ".join(chain) if chain else f"{type(exc).__name__}: {exc}"
            state["response"] = f"Run failed:\n```\n{state['error']}\n```"
            state["trajectory"] = list(bundle.enforcer.trajectory)
        finally:
            if hasattr(expert, "set_event_sink"):
                expert.set_event_sink(None)
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
        # Streamlit's st.write_stream consumes the first yielded chunk for
        # type detection — empty strings are silently dropped so the actual
        # first character of the reply still ends up being the probe target.
        # Yield a zero-width space (real char, invisible) so detection runs
        # on it instead of on the user-visible content.
        yield "​"
        streamed_text_chars = 0
        while True:
            item = eq.get(timeout=300)
            if item is None:
                break
            kind, payload = item
            if kind == "token":
                events.append({"kind": "reasoning_token", "text": payload})
                streamed_text_chars += len(payload)
                yield payload  # typewriter effect
            elif kind == "tool_call":
                events.append({"kind": "tool_call", **_to_jsonable(payload)})
                yield (
                    f"\n\n{_tool_progress_message(payload['name'], payload.get('agent'))}"
                    "\n\n"
                )
            elif kind == "tool_output":
                events.append({"kind": "tool_output", **_to_jsonable(payload)})
                if payload.get("success"):
                    out_str = json.dumps(payload.get("output"), ensure_ascii=False,
                                         default=str)
                    snippet = (out_str[:200] + "…") if len(out_str) > 200 else out_str
                    yield f"  ✓ `{snippet}`\n\n"
                else:
                    err = payload.get("error", "(unknown)")
                    yield f"\n\n  ❌ **{payload.get('name')} failed:** `{err}`\n\n"
            elif kind == "expert_event":
                event = _to_jsonable(payload)
                events.append(event)
                _log("expert_tool_event", event)
                if event.get("kind") == "tool_call":
                    yield (
                        f"\n\n{_tool_progress_message(event['name'], event.get('agent'))}"
                        "\n\n"
                    )
                elif event.get("kind") == "tool_output" and event.get("success", True):
                    out_str = json.dumps(event.get("output"), ensure_ascii=False, default=str)
                    snippet = (out_str[:200] + "…") if len(out_str) > 200 else out_str
                    yield f"  ✓ `{snippet}`\n\n"
                elif event.get("kind") == "tool_output" and not event.get("success", True):
                    err = event.get("error", "(unknown)")
                    yield f"\n\n  ❌ **{event.get('name')} failed:** `{err}`\n\n"
        # Fallback: BeeAI's `new_token` emitter is silent on some OpenAI-compat
        # providers (Featherless, etc.), so token streaming may produce nothing
        # even on a successful run. If the stream emitted no chat text, render
        # the captured final response so the user actually sees the reply.
        final = state.get("response") or ""
        if streamed_text_chars == 0 and final:
            yield final

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
            if message.get("events"):
                render_terminal_card(message["events"])
                render_events_timeline(message["events"])

    user_input = st.chat_input("Message")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "trajectory": []})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            stream_state: dict = {"response": "", "trajectory": [], "error": None, "events": []}
            st.write_stream(stream_analysis(user_input, current_patient, db_path, stream_state))
            # After the stream completes, render any structured terminal card
            render_terminal_card(stream_state.get("events", []))
            if stream_state.get("error"):
                st.error(f"❌ {stream_state['error']}")
            render_events_timeline(stream_state.get("events", []))

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
            {"role": "assistant", "content": response, "trajectory": trajectory,
             "events": turn_events}
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

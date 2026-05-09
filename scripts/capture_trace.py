"""Run the orchestrator end-to-end and emit a structured trace.

Produces two artifacts:
- docs/traces/trace_<timestamp>.json — machine-readable
- docs/traces/trace_<timestamp>.md   — human-readable demo-ready chat log

Subscribes to BeeAI tool emitters to capture each tool call's input + output
in real time, then renders both files at the end.

Run: `uv run python scripts/capture_trace.py`
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from apps.orchestrator.agent import build_bundle, orchestrator_agent
from packages.llm_provider.client import build_chat_model_for_role
from packages.llm_provider.settings import Role, RoleConfig

PROMPT = (
    "Analyse seed patient `seed-001`. "
    "Follow the MARGE workflow: consult the medical expert first, run relevant "
    "ML tools with sufficient tabular data, consult the expert again with the "
    "ML results, then produce a final report."
)

OUT_DIR = Path(__file__).parent.parent / "docs" / "traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _shorten(obj, max_chars: int = 200):
    """Truncate long string payloads for terse markdown view."""
    if isinstance(obj, str) and len(obj) > max_chars:
        return obj[:max_chars] + "…"
    return obj


def _to_jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return repr(obj)[:500]


async def main() -> None:
    started_at = datetime.utcnow().isoformat(timespec="seconds")

    orch_cfg = RoleConfig.from_env(Role.ORCHESTRATOR)
    expert_cfg = RoleConfig.from_env(Role.MEDICAL_EXPERT)

    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    bundle = build_bundle()

    captured: list[dict] = []

    def make_recorder(tool_name: str, tool_kind: str):
        pending: dict = {}

        def on_event(data, event):
            name = getattr(event, "name", None)
            if name == "start":
                pending["input"] = _to_jsonable(getattr(data, "input", None))
            elif name == "success":
                output = getattr(data, "output", None)
                # JSONToolOutput / StringToolOutput
                output_repr = None
                if output is not None:
                    if hasattr(output, "to_json_safe"):
                        try:
                            output_repr = output.to_json_safe()
                        except Exception:
                            output_repr = repr(output)[:500]
                    else:
                        output_repr = str(output)[:1000]
                captured.append({
                    "tool": tool_name,
                    "tool_kind": tool_kind,
                    "input": pending.get("input"),
                    "output": output_repr,
                    "status": "success",
                })
                pending.clear()
            elif name == "error":
                err = getattr(data, "error", None)
                captured.append({
                    "tool": tool_name,
                    "tool_kind": tool_kind,
                    "input": pending.get("input"),
                    "status": "error",
                    "error": repr(err)[:500],
                })
                pending.clear()

        return on_event

    async with orchestrator_agent(bundle=bundle, llm=llm) as agent:
        for t in agent._tools:
            kind = "mcp" if t.__class__.__name__ == "MCPTool" else "local"
            t.emitter.match("*", make_recorder(t.name, kind))

        result = await agent.run(PROMPT)

    answer_text = ""
    if hasattr(result, "answer"):
        answer_text = getattr(result.answer, "text", "") or ""
    if not answer_text and hasattr(result, "output_structured"):
        outs = getattr(result, "output_structured", None)
        if outs and hasattr(outs, "response"):
            answer_text = outs.response

    finished_at = datetime.utcnow().isoformat(timespec="seconds")

    trace = {
        "trace_id": f"trace_{started_at.replace(':', '-')}",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "environment": {
            "orchestrator_llm": f"{orch_cfg.primary.provider.value}:{orch_cfg.primary.model_id}",
            "fallback_llm": (
                f"{orch_cfg.fallback.provider.value}:{orch_cfg.fallback.model_id}"
                if orch_cfg.fallback else None
            ),
            "medical_expert": f"{expert_cfg.primary.provider.value}:{expert_cfg.primary.model_id}",
        },
        "user_prompt": PROMPT,
        "iterations": [
            {"iteration": i + 1, **step} for i, step in enumerate(captured)
        ],
        "trajectory_recorded_by_enforcer": list(bundle.enforcer.trajectory),
        "final_report_reached": bundle.enforcer.has_called("final_report"),
        "final_response_to_user": answer_text,
    }

    stamp = started_at.replace(":", "-")
    json_path = OUT_DIR / f"trace_{stamp}.json"
    md_path = OUT_DIR / f"trace_{stamp}.md"

    json_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
    md_path.write_text(_render_markdown(trace))

    print(f"\n[✓] JSON  → {json_path.relative_to(Path.cwd())}")
    print(f"[✓] MD    → {md_path.relative_to(Path.cwd())}")


def _render_markdown(trace: dict) -> str:
    lines = [
        f"# MARGE — Captured Trace ({trace['trace_id']})\n",
        f"- started: `{trace['started_at_utc']} UTC`",
        f"- orchestrator LLM: `{trace['environment']['orchestrator_llm']}`",
        f"- fallback LLM: `{trace['environment']['fallback_llm']}`",
        f"- medical expert: `{trace['environment']['medical_expert']}`",
        "",
        "## 👤 User",
        f"\n> {trace['user_prompt']}",
        "",
        "## 🤖 Orchestrator",
    ]
    for step in trace["iterations"]:
        n = step["iteration"]
        marker = " ★ TERMINAL" if step.get("tool") == "final_report" else ""
        lines += [
            f"\n### iter {n} — `{step.get('tool')}` *[{step.get('tool_kind')}]*{marker}",
            "```json",
            json.dumps(step.get("input"), indent=2, ensure_ascii=False)[:1500],
            "```",
        ]
        if step.get("status") == "success":
            out = step.get("output")
            if isinstance(out, (dict, list)):
                lines += [
                    "\n**↳ output**",
                    "```json",
                    json.dumps(out, indent=2, ensure_ascii=False)[:1500],
                    "```",
                ]
            else:
                lines += [f"\n**↳ output**: `{str(out)[:300]}`"]
        else:
            lines += [f"\n**↳ ERROR**: `{step.get('error')}`"]

    lines += [
        "\n---\n",
        "## 🩺 Final response to user",
        f"\n> {trace['final_response_to_user']}",
        "",
        "## 🧾 Verification",
        f"\n- trajectory: `{' → '.join(trace['trajectory_recorded_by_enforcer'])}`",
        f"- final_report reached: **{trace['final_report_reached']}**",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())

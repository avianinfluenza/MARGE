"""Capture the exact HTTP request BeeAI/litellm sends to NVIDIA NIM.

Hooks httpx.AsyncClient.send to log every outgoing request body. We then
trigger the agent loop until the 2nd-iteration NIM call (the one that
hangs) is dispatched. The captured body lets us:

1. Replay it directly with curl (separates BeeAI bug vs NIM bug).
2. Compare iteration 1 (works) vs iteration 2 (hangs) to find the schema
   difference responsible.

Output: prints each request to stdout, then aborts after iteration 2
starts (we don't actually need to wait for it to hang).
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import httpx

_CAPTURED: list[dict] = []
_orig_send = httpx.AsyncClient.send


async def _logging_send(self, request, **kwargs):
    """Replacement for httpx.AsyncClient.send that logs each outgoing request."""
    body = None
    try:
        body = json.loads(request.content.decode())
    except Exception:
        body = request.content.decode(errors="replace") if request.content else None

    captured = {
        "iteration": len(_CAPTURED) + 1,
        "method": request.method,
        "url": str(request.url),
        "headers": {k: "***" if k.lower() in ("authorization",) else v for k, v in request.headers.items()},
        "body": body,
    }
    _CAPTURED.append(captured)
    print(f"\n[capture] >>> request #{captured['iteration']} {request.method} {request.url}", flush=True)
    if isinstance(body, dict):
        # Show only the diff-relevant fields
        msgs = body.get("messages", [])
        print(f"[capture]   messages: {len(msgs)} (roles={[m.get('role') for m in msgs]})", flush=True)
        if "tools" in body:
            print(f"[capture]   tools: {len(body['tools'])}", flush=True)
        # Save full body to file for replay
        out_path = Path("/tmp") / f"nim_req_{captured['iteration']}.json"
        out_path.write_text(json.dumps(body, indent=2, default=str))
        print(f"[capture]   full body saved to {out_path}", flush=True)
    return await _orig_send(self, request, **kwargs)


httpx.AsyncClient.send = _logging_send


async def main() -> None:
    from apps.orchestrator.agent import build_bundle, build_orchestrator_agent
    from packages.llm_provider.client import build_chat_model_for_role
    from packages.llm_provider.settings import Role

    print("[diag] building agent on NVIDIA NIM (Qwen3.5-397B)...", flush=True)
    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=llm)

    # Force-stop after iteration 2 starts so we don't sit through the hang
    iteration_started = [0]
    stop_event = asyncio.Event()

    def on_iter_start(data, event):
        iteration_started[0] += 1
        print(f"\n[diag] === iteration {iteration_started[0]} start ===", flush=True)
        if iteration_started[0] >= 2:
            # We've captured iteration 2's outgoing request — stop
            stop_event.set()

    agent.emitter.match("agent.requirement.start", on_iter_start)

    prompt = (
        "Get patient seed-001, consult the medical expert for pre-ML context, "
        "then call predict_diabetes_risk if the data is sufficient, then consult "
        "the medical expert again with the ML result."
    )
    print(f"[diag] prompt: {prompt}\n", flush=True)

    # agent.run returns a Run object; wrap it in a coroutine
    async def _run_agent():
        return await agent.run(prompt)

    run_task = asyncio.create_task(_run_agent())
    stop_task = asyncio.create_task(stop_event.wait())

    # Wait for either the agent to finish OR for iteration 2 to start
    done, pending = await asyncio.wait(
        [run_task, stop_task],
        timeout=60,
        return_when=asyncio.FIRST_COMPLETED,
    )
    # Give stop_event a moment to propagate the next request
    await asyncio.sleep(2)

    for t in pending:
        t.cancel()

    print(f"\n[diag] captured {len(_CAPTURED)} HTTP request(s)", flush=True)
    print(f"[diag] iterations observed: {iteration_started[0]}", flush=True)
    for c in _CAPTURED:
        print(f"  req #{c['iteration']}: {c['method']} {c['url']} body→/tmp/nim_req_{c['iteration']}.json", flush=True)
    print("\n[diag] DONE. Inspect /tmp/nim_req_*.json to compare iteration 1 vs 2.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

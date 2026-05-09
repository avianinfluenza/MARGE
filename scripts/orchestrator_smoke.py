"""Manual end-to-end smoke for the orchestrator with a live LLM.

Loads provider keys from .env, builds the orchestrator with the role-aware
LLM (primary + fallback), and runs one happy-path analysis.

Setup (once):
    cp .env.example .env  # then paste your API keys
    uv run python -m packages.ml_training.train_breast_cancer
    uv run python -m packages.ml_training.train_diabetes

Run:
    uv run python scripts/orchestrator_smoke.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env vars at import time.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from apps.orchestrator.agent import build_bundle, build_orchestrator_agent
from packages.llm_provider.client import build_chat_model_for_role
from packages.llm_provider.settings import Role, RoleConfig


_PROMPT = (
    "Analyse seed patient `seed-001`. "
    "Use the available ML tools to assess their risk profile, then consult "
    "the medical expert to validate. Produce a final report."
)


def _describe(cfg: RoleConfig) -> str:
    primary = f"{cfg.primary.provider.value}:{cfg.primary.model_id}"
    if cfg.fallback:
        return f"{primary}  (fallback: {cfg.fallback.provider.value}:{cfg.fallback.model_id})"
    return primary


async def main() -> None:
    print("=" * 72)
    print(" MARGE orchestrator end-to-end smoke (LIVE LLM)")
    print("=" * 72)

    orch_cfg = RoleConfig.from_env(Role.ORCHESTRATOR)
    expert_cfg = RoleConfig.from_env(Role.MEDICAL_EXPERT)
    print(f"\nOrchestrator LLM   : {_describe(orch_cfg)}")
    print(f"Medical-expert LLM : {_describe(expert_cfg)}  [stub for this slice]")

    llm = build_chat_model_for_role(Role.ORCHESTRATOR)
    bundle = build_bundle()
    agent = await build_orchestrator_agent(bundle=bundle, llm=llm)

    print("\nRunning agent...\n" + "-" * 72)
    result = await agent.run(_PROMPT)

    print("\n" + "=" * 72)
    print(" Result")
    print("=" * 72)
    answer = getattr(result, "answer", None)
    text = getattr(answer, "text", str(result))
    print(text)

    print("\n" + "=" * 72)
    print(" Trajectory")
    print("=" * 72)
    for i, step in enumerate(bundle.enforcer.trajectory, 1):
        print(f"  {i:2d}. {step}")

    final_called = bundle.enforcer.has_called("final_report")
    print(f"\n[ok] final_report reached: {final_called}")


if __name__ == "__main__":
    asyncio.run(main())

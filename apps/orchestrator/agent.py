"""BeeAI orchestrator assembly.

This module wires together:
- ProtocolEnforcer (middleware that records every tool call)
- Local tools (consult_expert, patient_history, final_report, abstain, ask_user_back)
- ML tools (discovered dynamically via the FastMCP `ml-models` server)
- An LLM backend through `packages.llm_provider.client`

It is intentionally thin: every domain rule and every clinical behaviour is
implemented and unit-tested in the modules above. This file is just the
glue layer that hands the assembled tool surface to BeeAI.

NOTE: This module is **not unit-tested** — exercising it requires a live
LLM. Use `scripts/orchestrator_smoke.py` (next slice) for manual end-to-end
verification with a real Anthropic / watsonx key.
"""

from dataclasses import dataclass
from pathlib import Path

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.ask_user_back import make_ask_user_back
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources._base import PatientSource
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


@dataclass
class OrchestratorBundle:
    """All the deterministic pieces of the orchestrator, ready to be handed to BeeAI.

    The BeeAI assembly itself is deferred to the next slice — once the
    BeeAI Python API is verified end-to-end against a live LLM.
    """

    enforcer: ProtocolEnforcer
    system_prompt: str
    local_tools: dict[str, object]
    patient_source: PatientSource


def build_bundle(
    patient_source: PatientSource | None = None,
) -> OrchestratorBundle:
    """Build the orchestrator's deterministic dependencies.

    The BeeAI agent (LLM + tool dispatch loop) is constructed elsewhere
    using this bundle. ML tools are discovered separately via the FastMCP
    `ml-models` server and added to the BeeAI tool list at agent build time.
    """
    enforcer = ProtocolEnforcer()
    expert = StubMedicalExpert()
    source = patient_source or SqlitePatientSource()

    local_tools = {
        "get_patient_history": make_patient_history(source, enforcer),
        "consult_medical_expert": make_consult_expert(expert, enforcer),
        "final_report": make_final_report(enforcer),
        "abstain": make_abstain(enforcer),
        "ask_user_back": make_ask_user_back(enforcer),
    }

    return OrchestratorBundle(
        enforcer=enforcer,
        system_prompt=_SYSTEM_PROMPT_PATH.read_text(),
        local_tools=local_tools,
        patient_source=source,
    )

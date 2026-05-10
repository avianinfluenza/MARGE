"""BeeAI orchestrator assembly.

Two layers:

`build_bundle()` returns deterministic dependencies (enforcer, local tools,
system prompt). No LLM or DB dependency — tests reach for this directly.

`orchestrator_agent(bundle, llm, patient_db_path, memory)` is an async
context manager:
- Opens in-process MCP clients for both the ML-models server and the
  patient-data server (backed by the session SQLite DB).
- Discovers tools from both MCP sessions.
- Wires the MARGE protocol Requirement (final_answer gating + ordering).
- Yields a fully wired `RequirementAgent`.
- Closes both MCP sessions on exit.

Memory: the caller may pass a `BaseMemory` instance to persist conversation
history across user turns (Streamlit reuses one per session). Defaults to
fresh `UnconstrainedMemory` when omitted.

Usage:
    async with orchestrator_agent(bundle, llm, patient_db_path=db,
                                   memory=session_memory) as agent:
        result = await agent.run("Analyse patient csv-42.")
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.requirements.marge_protocol import (
    build_marge_protocol_requirement,
)
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.clinical_report import make_clinical_report
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.request_more_info import make_request_more_info
from services.medical_expert_agent.agent import StubMedicalExpert

if TYPE_CHECKING:
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.backend.chat import ChatModel
    from beeai_framework.memory.base_memory import BaseMemory

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


@dataclass
class OrchestratorBundle:
    """Deterministic dependencies of the orchestrator (no LLM, no DB)."""

    enforcer: ProtocolEnforcer
    system_prompt: str
    local_tools: dict[str, object]


def build_bundle(expert=None) -> OrchestratorBundle:
    """Build the orchestrator's deterministic dependencies.

    Args:
        expert: Optional MedicalExpert implementation. Defaults to
            `StubMedicalExpert` (deterministic test stub). Production
            use should pass the live BeeAI sub-agent expert.
    """
    enforcer = ProtocolEnforcer()
    if expert is None:
        expert = StubMedicalExpert()

    local_tools = {
        "consult_medical_expert": make_consult_expert(expert, enforcer),
        "request_more_info": make_request_more_info(enforcer),
        "clinical_report": make_clinical_report(enforcer),
        "abstain": make_abstain(enforcer),
    }

    return OrchestratorBundle(
        enforcer=enforcer,
        system_prompt=_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        local_tools=local_tools,
    )


@asynccontextmanager
async def orchestrator_agent(
    bundle: OrchestratorBundle,
    llm: "ChatModel",
    patient_db_path: Path | None = None,
    memory: "BaseMemory | None" = None,
) -> AsyncIterator["RequirementAgent"]:
    """Build and yield a fully wired RequirementAgent.

    Opens in-process MCP sessions for the ML server and the patient-data
    server. Both sessions stay alive for the duration of agent.run() —
    closing either early causes MCPTool to raise ToolError.

    Args:
        bundle: Deterministic dependencies from `build_bundle()`.
        llm: Chat model instance.
        patient_db_path: Path to the session SQLite DB. If None, the patient
            MCP server is not attached (ML-only mode).
        memory: Optional `BaseMemory` to persist conversation across turns.
            Defaults to a fresh `UnconstrainedMemory`.
    """
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.memory import UnconstrainedMemory
    from beeai_framework.tools.mcp import MCPTool
    from fastmcp import Client

    from apps.orchestrator.tools._adapter import local_tools_as_beeai
    from services.ml_mcp_server.server import build_server
    from services.patient_data_mcp_server.server import build_patient_server

    local_tools = local_tools_as_beeai(bundle)
    ml_server = build_server()

    if memory is None:
        memory = UnconstrainedMemory()

    def _make_recorder(tool_name: str):
        def _record(data, event) -> None:
            if getattr(event, "name", None) == "start":
                bundle.enforcer.record(tool_name)
        return _record

    async with Client(ml_server) as ml_client:
        ml_tools = await MCPTool.from_client(ml_client.session)
        for t in ml_tools:
            t.emitter.match("*", _make_recorder(t.name))

        if patient_db_path is not None:
            patient_server = build_patient_server(patient_db_path)
            async with Client(patient_server) as patient_client:
                patient_tools = await MCPTool.from_client(patient_client.session)
                for t in patient_tools:
                    t.emitter.match("*", _make_recorder(t.name))

                agent = RequirementAgent(
                    llm=llm,
                    memory=memory,
                    tools=[*local_tools, *ml_tools, *patient_tools],
                    requirements=[build_marge_protocol_requirement()],
                    name="MARGE Orchestrator",
                    description=(
                        "Clinical ML head researcher: orchestrates ML tools, "
                        "manages patient data, and consults a medical expert."
                    ),
                    instructions=bundle.system_prompt,
                    final_answer_as_tool=False,
                )
                yield agent
        else:
            agent = RequirementAgent(
                llm=llm,
                memory=memory,
                tools=[*local_tools, *ml_tools],
                requirements=[build_marge_protocol_requirement()],
                name="MARGE Orchestrator",
                description=(
                    "Clinical ML head researcher: orchestrates ML tools "
                    "and consults a medical expert."
                ),
                instructions=bundle.system_prompt,
                final_answer_as_tool=False,
            )
            yield agent

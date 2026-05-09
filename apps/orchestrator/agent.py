"""BeeAI orchestrator assembly.

Two layers:

`build_bundle()` returns deterministic dependencies (enforcer, local tools,
system prompt). No LLM or DB dependency — tests reach for this directly.

`orchestrator_agent(bundle, llm, patient_db_path)` is an async context manager:
- Opens in-process MCP clients for both the ML-models server and the
  patient-data server (backed by the session SQLite DB).
- Discovers tools from both MCP sessions.
- Yields a fully wired `RequirementAgent`.
- Closes both MCP sessions on exit.

Usage:
    async with orchestrator_agent(bundle, llm, patient_db_path=db) as agent:
        result = await agent.run("Analyse patient csv-42.")
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.requirements.marge_protocol import (
    build_marge_protocol_requirements,
)
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from services.medical_expert_agent.agent import (
    MedicalExpert,
    build_medical_expert_agent,
)
from services.patient_data_mcp_server.sources._base import PatientSource
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource

if TYPE_CHECKING:
    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.backend.chat import ChatModel

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


@dataclass
class OrchestratorBundle:
    """Deterministic dependencies of the orchestrator (no LLM, no DB)."""

    enforcer: ProtocolEnforcer
    system_prompt: str
    local_tools: dict[str, object]


def build_bundle(
    patient_source: PatientSource | None = None,
    expert: MedicalExpert | None = None,
) -> OrchestratorBundle:
    """Build the orchestrator's deterministic dependencies."""
    enforcer = ProtocolEnforcer()
    expert = expert or build_medical_expert_agent()
    source = patient_source or SqlitePatientSource()

    local_tools = {
        "consult_medical_expert": make_consult_expert(expert, enforcer),
        "final_report": make_final_report(enforcer),
    }

    return OrchestratorBundle(
        enforcer=enforcer,
        system_prompt=_SYSTEM_PROMPT_PATH.read_text(),
        local_tools=local_tools,
    )


@asynccontextmanager
async def orchestrator_agent(
    bundle: OrchestratorBundle,
    llm: "ChatModel",
    patient_db_path: Path | None = None,
) -> AsyncIterator["RequirementAgent"]:
    """Build and yield a fully wired RequirementAgent.

    Opens in-process MCP sessions for the ML server and the patient-data
    server. Both sessions stay alive for the duration of agent.run() —
    closing either early causes MCPTool to raise ToolError.

    Args:
        bundle: Deterministic dependencies from `build_bundle()`.
        llm: Chat model instance.
        patient_db_path: Path to the session SQLite DB. If None, the patient
            MCP server is not attached (ML-only mode; patient tools unavailable).
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

    async with Client(mcp_server) as mcp_client:
        from beeai_framework.tools.mcp import MCPTool

        ml_tools = await MCPTool.from_client(mcp_client.session)

        agent = RequirementAgent(
            llm=llm,
            memory=UnconstrainedMemory(),
            tools=[*local_tools, *ml_tools],
            requirements=build_marge_protocol_requirements(t.name for t in ml_tools),
            name="MARGE Orchestrator",
            description=(
                "Clinical ML head researcher: orchestrates ML tools and a "
                "medical expert sub-agent."
            ),
            instructions=bundle.system_prompt,
            # Disable BeeAI's automatic FinalAnswerTool — our `final_report`
            # is the gated terminal tool from architecture.md §2 and must
            # be the only path to a user answer.
            final_answer_as_tool=False,
        )

        # Hook the enforcer onto MCP tools only. Local tools already
        # record themselves via the factory closures (test-covered).
        # ML predictions go through MCP, so without this hook the
        # finalize gate would never see a `predict_*` call and would
        # incorrectly block final_report.
        def _make_recorder(tool_name: str):
            def _record(data, event) -> None:
                # Fire only on the start event for each tool; using "*" so
                # it matches across BeeAI's emitter namespacing.
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

                agent = RequirementAgent(
                    llm=llm,
                    memory=UnconstrainedMemory(),
                    tools=[*local_tools, *ml_tools, *patient_tools],
                    requirements=[],  # MARGE protocol requirement temporarily disabled
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
                memory=UnconstrainedMemory(),
                tools=[*local_tools, *ml_tools],
                requirements=[],  # MARGE protocol requirement temporarily disabled
                name="MARGE Orchestrator",
                description=(
                    "Clinical ML head researcher: orchestrates ML tools "
                    "and consults a medical expert."
                ),
                instructions=bundle.system_prompt,
                final_answer_as_tool=False,
            )
            yield agent

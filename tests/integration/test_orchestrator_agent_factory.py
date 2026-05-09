"""Integration: assemble the BeeAI RequirementAgent via the orchestrator_agent
context manager.

Discovers ML tools via MCP (live in-process server), converts the bundle's
local tools to BeeAI Tools, and yields a fully wired `RequirementAgent`
with the system prompt and memory configured.

We pass a fake ChatModel (no real LLM call) — these tests only verify that
the agent has the expected tool surface and configuration. Live LLM calls
happen in `scripts/orchestrator_smoke.py`.
"""

import pytest

from apps.orchestrator.agent import build_bundle, orchestrator_agent


class _FakeChatModel:
    """Stand-in for BeeAI ChatModel — never invoked, just stored on the agent."""

    model_id = "fake-model"
    provider_id = "fake"
    parameters = None
    tool_choice_support = ["auto", "single"]
    middlewares = ()


@pytest.mark.asyncio
async def test_yields_requirement_agent():
    from beeai_framework.agents.requirement import RequirementAgent

    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        assert isinstance(agent, RequirementAgent)


@pytest.mark.asyncio
async def test_agent_has_local_tools_plus_ml_tools():
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
    expected_local = {"consult_medical_expert", "final_report"}
    expected_ml = {"predict_breast_cancer_malignancy", "predict_diabetes_risk"}
    assert expected_local <= tool_names
    assert expected_ml <= tool_names


@pytest.mark.asyncio
async def test_agent_has_four_tools_without_patient_db():
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        # 2 local + 2 MCP-discovered ML = 4 (no patient_db_path)
        assert len(agent._tools) == 4


@pytest.mark.asyncio
async def test_agent_has_patient_tools_when_db_provided(tmp_path):
    from services.patient_data_mcp_server.sources.csv_ingest import seed_demo_db

    db = tmp_path / "test.db"
    seed_demo_db(db)
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel(), patient_db_path=db) as agent:
        tool_names = {t.name for t in agent._tools}
    assert {"list_patients", "get_patient", "update_patient"} <= tool_names


@pytest.mark.skip(reason="MARGE protocol requirement temporarily disabled")
@pytest.mark.asyncio
async def test_agent_has_marge_protocol_requirement():
    from apps.orchestrator.requirements.marge_protocol import (
        build_marge_protocol_requirement,
    )

    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        sources = [getattr(r, "source", None) for r in agent._requirements]
        assert "final_report" in sources


@pytest.mark.asyncio
async def test_agent_uses_provided_llm():
    fake = _FakeChatModel()
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=fake) as agent:
        assert agent._llm is fake


@pytest.mark.asyncio
async def test_mcp_tool_invocations_are_recorded_in_enforcer():
    """ML calls go through MCP, not local closures — emitter hook records them."""
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        ml_tool = next(
            t for t in agent._tools if t.name == "predict_diabetes_risk"
        )
        sample = {
            "preg": 6.0, "plas": 148.0, "pres": 72.0, "skin": 35.0,
            "insu": 0.0, "mass": 33.6, "pedi": 0.627, "age": 50.0,
        }
        await ml_tool.run({"inputs": sample})

    assert bundle.enforcer.has_called("predict_diabetes_risk")


@pytest.mark.asyncio
async def test_final_answer_as_tool_disabled():
    """BeeAI's auto-final-answer tool must be off — our final_report is the
    only path to a user answer (architecture.md §2)."""
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        tool_names = {t.name for t in agent._tools}
        # final_answer is the BeeAI auto-tool name when final_answer_as_tool=True.
        assert "final_answer" not in tool_names


@pytest.mark.asyncio
async def test_system_prompt_marker_present_in_bundle():
    """The system prompt is fed to BeeAI's `instructions` parameter and merged
    into RequirementAgent's prompt template at run time. We verify that the
    bundle (the source of truth) carries the role marker — BeeAI's wiring of
    `instructions` into its template is its own responsibility."""
    bundle = build_bundle()
    async with orchestrator_agent(bundle=bundle, llm=_FakeChatModel()) as agent:
        assert "ML Orchestrator" in bundle.system_prompt
        assert agent.meta.name == "MARGE Orchestrator"

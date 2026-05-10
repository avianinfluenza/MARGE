"""Integration: protocol-flow walkthrough (no LLM, no agent loop).

Simulates the orchestrator's tool-call sequence (recording each step in
the ProtocolEnforcer). Verifies that the trajectory ends up with the
correct shape for each scenario.

In the hybrid pattern there is no chat-as-tool wrapper — natural-language
chat lives in the LLM's `content` field and produces no enforcer record.
The four tool actions tracked here are: `consult_medical_expert`,
`predict_*` (simulated), `clinical_report`, `request_more_info`,
`abstain`.

The actual gate enforcement is now LLM-side via `MARGEProtocolRequirement`;
the enforcer's role here is trajectory recording for logs/UI. The
`consult_medical_expert` tool factory is async (so it can await a real
LLM-backed expert), so these tests are async too.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.clinical_report import make_clinical_report
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.request_more_info import make_request_more_info
from services.medical_expert_agent.agent import StubMedicalExpert


@pytest.fixture
def deps():
    enforcer = ProtocolEnforcer()
    return {
        "enforcer": enforcer,
        "consult": make_consult_expert(StubMedicalExpert(), enforcer),
        "request": make_request_more_info(enforcer),
        "report": make_clinical_report(enforcer),
        "abstain": make_abstain(enforcer),
    }


@pytest.mark.asyncio
async def test_happy_path_expert_then_ml_then_report(deps):
    await deps["consult"](question="Differential for given symptoms?", findings={"sx": "polydipsia"})
    deps["enforcer"].record("predict_diabetes_risk")  # ML simulated
    await deps["consult"](question="ML score interpretation?", findings={"score": 0.85})
    deps["report"](
        summary="High diabetes risk.",
        recommendation="Refer to PCP.",
        confidence="high",
    )

    assert deps["enforcer"].trajectory == (
        "consult_medical_expert",
        "predict_diabetes_risk",
        "consult_medical_expert",
        "clinical_report",
    )


@pytest.mark.asyncio
async def test_scope_mismatch_path_ends_in_abstain(deps):
    await deps["consult"](question="Differential for headache + fatigue", findings={})
    deps["abstain"](reason="No relevant ML predictor for these symptoms.")

    traj = deps["enforcer"].trajectory
    assert "abstain" in traj
    assert "consult_medical_expert" in traj


@pytest.mark.asyncio
async def test_request_more_info_path_terminates_without_ml(deps):
    deps["request"](
        needed=[{"name": "HbA1c", "why": "confirm diabetes range"}],
        rationale="HbA1c materially shifts diabetes risk.",
    )

    traj = deps["enforcer"].trajectory
    assert traj == ("request_more_info",)


@pytest.mark.asyncio
async def test_natural_language_only_turn_records_no_tool_calls(deps):
    """A casual-chat turn (greeting / smalltalk) makes no tool calls; the
    LLM's natural-language `content` is the entire reply. The enforcer
    trajectory stays empty."""
    # No tool invocation simulated — purely natural-language turn.
    assert deps["enforcer"].trajectory == ()


@pytest.mark.asyncio
async def test_multiple_consult_calls_within_one_turn(deps):
    await deps["consult"](question="A", findings={})
    await deps["consult"](question="B", findings={})
    deps["request"](needed=[], rationale="x")

    traj = deps["enforcer"].trajectory
    assert traj.count("consult_medical_expert") == 2
    assert traj[-1] == "request_more_info"

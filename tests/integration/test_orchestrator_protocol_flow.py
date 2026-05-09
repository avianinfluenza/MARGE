"""Integration: end-to-end protocol flow without LLM.

Simulates the orchestrator's tool-call sequence (recording each step in the
ProtocolEnforcer) and verifies that the protocol-enforcement layer correctly
gates final_report. This mirrors what BeeAI will do at runtime, minus the
LLM-driven decision making.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import (
    ProtocolEnforcer,
    ProtocolViolation,
)
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


@pytest.fixture
def deps():
    enforcer = ProtocolEnforcer()
    return {
        "enforcer": enforcer,
        "history": make_patient_history(SqlitePatientSource(), enforcer),
        "consult": make_consult_expert(StubMedicalExpert(), enforcer),
        "final": make_final_report(enforcer),
    }


def test_happy_path(deps):
    """patient -> ML (simulated) -> expert -> final_report succeeds."""
    patient = deps["history"](handle="seed-001")
    assert patient.handle == "seed-001"

    # ML call is simulated by directly recording it (in production this is
    # done by the BeeAI middleware whenever any MCP tool is invoked).
    deps["enforcer"].record("predict_breast_cancer_malignancy")

    expert_response = deps["consult"](
        question="Given the prediction, what is the recommendation?",
        findings={"prediction": "malignant", "confidence": 0.989},
    )
    assert expert_response.citations

    result = deps["final"](
        summary="High-confidence malignancy prediction confirmed by expert review.",
        recommendation="Refer for biopsy and imaging.",
        confidence_note="Both ML and expert in agreement.",
    )
    assert result["summary"]


def test_blocked_when_skipping_ml(deps):
    """consult expert only, then try to finalize — must fail."""
    deps["consult"](question="?", findings={})
    with pytest.raises(ProtocolViolation, match="ML model"):
        deps["final"](summary="x", recommendation="y")


def test_blocked_when_skipping_expert(deps):
    """ML only, no expert — must fail."""
    deps["enforcer"].record("predict_diabetes_risk")
    with pytest.raises(ProtocolViolation, match="expert"):
        deps["final"](summary="x", recommendation="y")


def test_trajectory_records_full_sequence(deps):
    deps["history"](handle="seed-001")
    deps["enforcer"].record("predict_breast_cancer_malignancy")
    deps["consult"](question="?", findings={})
    deps["final"](summary="x", recommendation="y")

    # All tool calls are recorded for audit, including the terminal final_report.
    assert deps["enforcer"].trajectory == (
        "get_patient_history",
        "predict_breast_cancer_malignancy",
        "consult_medical_expert",
        "final_report",
    )

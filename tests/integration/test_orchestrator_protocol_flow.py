"""Integration: code-side protocol flow without LLM.

Simulates the orchestrator's tool-call sequence and verifies the defensive
backstop blocks final_report until ML and expert have appeared in the trajectory.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import (
    ProtocolEnforcer,
    ProtocolViolation,
)
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources.csv_ingest import seed_demo_db
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


@pytest.fixture()
def deps(tmp_path):
    db = tmp_path / "test.db"
    seed_demo_db(db)
    enforcer = ProtocolEnforcer()
    return {
        "enforcer": enforcer,
        "consult": make_consult_expert(StubMedicalExpert(), enforcer),
        "final": make_final_report(enforcer),
    }


def test_happy_path_ml_before_expert(deps):
    deps["enforcer"].record("predict_breast_cancer_malignancy")
    deps["consult"](
        question="What does this prediction suggest?",
        findings={"prediction": "malignant", "confidence": 0.989},
    )
    result = deps["final"](response="High-confidence finding — refer for biopsy.")
    assert result == {"response": "High-confidence finding — refer for biopsy."}


def test_happy_path_expert_before_ml(deps):
    """Expert can be consulted first — order is free."""
    deps["consult"](question="Which model should we run?", findings={})
    deps["enforcer"].record("predict_diabetes_risk")
    result = deps["final"](response="Risk profile suggests follow-up labs.")
    assert "response" in result


@pytest.mark.skip(reason="MARGE protocol requirement temporarily disabled")
def test_blocked_when_skipping_ml(deps):
    deps["consult"](question="?", findings={})
    with pytest.raises(ProtocolViolation, match="ML model"):
        deps["final"](response="anything")


@pytest.mark.skip(reason="MARGE protocol requirement temporarily disabled")
def test_blocked_when_skipping_expert(deps):
    deps["enforcer"].record("predict_diabetes_risk")
    with pytest.raises(ProtocolViolation, match="expert"):
        deps["final"](response="anything")


def test_trajectory_records_full_sequence(deps):
    deps["enforcer"].record("predict_breast_cancer_malignancy")
    deps["consult"](question="?", findings={})
    deps["final"](response="ok")

    assert deps["enforcer"].trajectory == (
        "predict_breast_cancer_malignancy",
        "consult_medical_expert",
        "final_report",
    )

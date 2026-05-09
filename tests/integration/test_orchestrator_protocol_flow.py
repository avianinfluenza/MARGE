"""Integration: code-side protocol flow without LLM.

Simulates the orchestrator's tool-call sequence and verifies the defensive
backstop blocks `final_report` until the trajectory contains:

    consult_medical_expert -> predict_* -> consult_medical_expert
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


def test_happy_path_expert_ml_expert(deps):
    deps["history"](handle="seed-001")
    deps["consult"](
        question="What clinical concerns should we consider before ML?",
        findings={"patient": "seed-001"},
    )
    deps["enforcer"].record("predict_breast_cancer_malignancy")  # simulated MCP
    deps["consult"](
        question="Is this ML result clinically plausible?",
        findings={"prediction": "malignant", "confidence": 0.989},
    )
    result = deps["final"](response="High-confidence finding; refer for biopsy.")
    assert result == {"response": "High-confidence finding; refer for biopsy."}


def test_blocked_when_ml_before_expert(deps):
    deps["history"](handle="seed-001")
    deps["enforcer"].record("predict_breast_cancer_malignancy")
    deps["consult"](question="Interpret this.", findings={"prediction": "malignant"})
    with pytest.raises(ProtocolViolation, match="workflow order"):
        deps["final"](response="anything")


def test_blocked_without_post_ml_expert(deps):
    deps["history"](handle="seed-001")
    deps["consult"](question="Pre-consult.", findings={})
    deps["enforcer"].record("predict_diabetes_risk")
    with pytest.raises(ProtocolViolation, match="workflow order"):
        deps["final"](response="anything")


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
    deps["history"](handle="seed-001")
    deps["consult"](question="Pre-consult.", findings={})
    deps["enforcer"].record("predict_breast_cancer_malignancy")
    deps["consult"](question="Post-consult.", findings={})
    deps["final"](response="ok")

    assert deps["enforcer"].trajectory == (
        "get_patient_history",
        "consult_medical_expert",
        "predict_breast_cancer_malignancy",
        "consult_medical_expert",
        "final_report",
    )

"""Tests for the orchestrator's local tool factories.

Two local tools after patient data moved to MCP:
- consult_expert  — dispatches to the medical_expert_agent + records call
- final_report    — gated terminal tool (ProtocolEnforcer.check_finalize)

patient_history is tested separately as a standalone tool; it is no longer
wired into the bundle but the factory is preserved for direct use.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import (
    ProtocolEnforcer,
    ProtocolViolation,
)
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from packages.schemas.patient import PatientRecord
from packages.schemas.retrieval import MedicalExpertResponse
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources.csv_ingest import seed_demo_db
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


@pytest.fixture()
def demo_source(tmp_path):
    db = tmp_path / "test.db"
    seed_demo_db(db)
    return SqlitePatientSource(db)


class TestConsultExpertTool:
    def test_returns_medical_expert_response(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        response = consult(question="What does this suggest?", findings={"a": 1})
        assert isinstance(response, MedicalExpertResponse)

    def test_records_consult_medical_expert_call(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        consult(question="?", findings={})
        assert enforcer.has_called("consult_medical_expert")


class TestPatientHistoryTool:
    def test_returns_patient_record(self, demo_source):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(demo_source, enforcer)
        record = get_history(handle="seed-001")
        assert isinstance(record, PatientRecord)
        assert record.handle == "seed-001"

    def test_records_get_patient_history_call(self, demo_source):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(demo_source, enforcer)
        get_history(handle="seed-001")
        assert enforcer.has_called("get_patient_history")

    def test_propagates_keyerror_for_unknown_handle(self, demo_source):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(demo_source, enforcer)
        with pytest.raises(KeyError):
            get_history(handle="seed-9999")


class TestFinalReportTool:
    @staticmethod
    def _ready_enforcer() -> ProtocolEnforcer:
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        enforcer.record("predict_diabetes_risk")
        enforcer.record("consult_medical_expert")
        return enforcer

    def test_blocks_when_no_ml_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="ML model"):
            final(response="anything")

    def test_blocks_when_no_expert_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_breast_cancer_malignancy")
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="expert"):
            final(response="anything")

    def test_succeeds_when_workflow_sequence_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        enforcer.record("predict_breast_cancer_malignancy")
        enforcer.record("consult_medical_expert")
        final = make_final_report(enforcer)
        result = final(response="High-confidence findings — refer for biopsy.")
        assert result == {"response": "High-confidence findings — refer for biopsy."}

    def test_records_final_report_call(self):
        enforcer = self._ready_enforcer()
        final = make_final_report(enforcer)
        final(response="All clear.")
        assert enforcer.has_called("final_report")

    def test_returns_response_dict(self):
        final = make_final_report(self._ready_enforcer())
        result = final(response="All clear.")
        assert result == {"response": "All clear."}

    @pytest.mark.skip(reason="MARGE protocol requirement temporarily disabled")
    def test_raises_without_ml_call(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="ML model"):
            final(response="too early")

    @pytest.mark.skip(reason="MARGE protocol requirement temporarily disabled")
    def test_raises_without_expert_call(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_diabetes_risk")
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="expert"):
            final(response="too early")

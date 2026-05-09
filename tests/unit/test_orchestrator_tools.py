"""Tests for the orchestrator's local tool factories.

These tools are thin wrappers that:
- consult_expert    — dispatches to the medical_expert_agent + records call
- patient_history   — dispatches to a PatientSource + records call
- final_report      — gated by ProtocolEnforcer.check_finalize()
- abstain           — escape hatch, always allowed, records call
- ask_user_back     — escape hatch, always allowed, records call
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import (
    ProtocolEnforcer,
    ProtocolViolation,
)
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.ask_user_back import make_ask_user_back
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.final_report import make_final_report
from apps.orchestrator.tools.patient_history import make_patient_history
from packages.schemas.patient import PatientRecord
from packages.schemas.retrieval import MedicalExpertResponse
from services.medical_expert_agent.agent import StubMedicalExpert
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


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
    def test_returns_patient_record(self):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(SqlitePatientSource(), enforcer)
        record = get_history(handle="seed-001")
        assert isinstance(record, PatientRecord)
        assert record.handle == "seed-001"

    def test_records_get_patient_history_call(self):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(SqlitePatientSource(), enforcer)
        get_history(handle="seed-001")
        assert enforcer.has_called("get_patient_history")

    def test_propagates_keyerror_for_unknown_handle(self):
        enforcer = ProtocolEnforcer()
        get_history = make_patient_history(SqlitePatientSource(), enforcer)
        with pytest.raises(KeyError):
            get_history(handle="seed-9999")


class TestFinalReportTool:
    def test_blocks_when_no_ml_called(self):
        enforcer = ProtocolEnforcer()
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="ML model"):
            final(summary="ok", recommendation="ok")

    def test_blocks_when_no_expert_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_breast_cancer_malignancy")
        final = make_final_report(enforcer)
        with pytest.raises(ProtocolViolation, match="expert"):
            final(summary="ok", recommendation="ok")

    def test_succeeds_when_both_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_breast_cancer_malignancy")
        enforcer.record("consult_medical_expert")
        final = make_final_report(enforcer)
        result = final(summary="findings", recommendation="seek imaging")
        assert result["summary"] == "findings"
        assert result["recommendation"] == "seek imaging"


class TestAbstainTool:
    def test_always_allowed(self):
        enforcer = ProtocolEnforcer()
        abstain = make_abstain(enforcer)
        result = abstain(reason="insufficient evidence")
        assert result["abstained"] is True
        assert result["reason"] == "insufficient evidence"

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        abstain = make_abstain(enforcer)
        abstain(reason="x")
        assert enforcer.has_called("abstain")


class TestAskUserBackTool:
    def test_always_allowed(self):
        enforcer = ProtocolEnforcer()
        ask = make_ask_user_back(enforcer)
        result = ask(missing_info=["recent labs", "family history"])
        assert result["asking_user_back"] is True
        assert result["missing_info"] == ["recent labs", "family history"]

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        ask = make_ask_user_back(enforcer)
        ask(missing_info=["x"])
        assert enforcer.has_called("ask_user_back")

"""Tests for the orchestrator's local tool factories.

Five local tools after the agent_fix refactor:
- update_user        — non-terminal mid-flow message; records call
- consult_expert     — sub-agent invocation; records call
- request_more_info  — terminal, free; records call
- clinical_report    — terminal, gated by Requirement; records call (no in-tool gate)
- abstain            — terminal, gated by Requirement; records call

Gating is now enforced LLM-side by `MARGEProtocolRequirement` (tested in
test_marge_requirement.py). The tool factories themselves only record
the call into the enforcer for trajectory logging.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from apps.orchestrator.tools.abstain import make_abstain
from apps.orchestrator.tools.clinical_report import make_clinical_report
from apps.orchestrator.tools.consult_expert import make_consult_expert
from apps.orchestrator.tools.request_more_info import make_request_more_info
from apps.orchestrator.tools.update_user import make_update_user
from packages.schemas.retrieval import MedicalExpertResponse
from services.medical_expert_agent.agent import StubMedicalExpert


class TestUpdateUserTool:
    def test_returns_text(self):
        enforcer = ProtocolEnforcer()
        upd = make_update_user(enforcer)
        result = upd(text="Now consulting the expert…")
        assert result == {"text": "Now consulting the expert…"}

    def test_records_update_user_call(self):
        enforcer = ProtocolEnforcer()
        upd = make_update_user(enforcer)
        upd(text="hi")
        assert enforcer.has_called("update_user")


class TestConsultExpertTool:
    @pytest.mark.asyncio
    async def test_returns_medical_expert_response(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        response = await consult(question="What does this suggest?", findings={"a": 1})
        assert isinstance(response, MedicalExpertResponse)

    @pytest.mark.asyncio
    async def test_records_consult_medical_expert_call(self):
        enforcer = ProtocolEnforcer()
        consult = make_consult_expert(StubMedicalExpert(), enforcer)
        await consult(question="?", findings={})
        assert enforcer.has_called("consult_medical_expert")


class TestRequestMoreInfoTool:
    def test_returns_structured_payload(self):
        enforcer = ProtocolEnforcer()
        ask = make_request_more_info(enforcer)
        out = ask(
            needed=[{"name": "HbA1c", "why": "diabetes confirm",
                     "field_type": "number", "unit": "%"}],
            rationale="Refines diabetes risk",
        )
        assert out["needs_more_info"] is True
        assert out["needed"][0]["name"] == "HbA1c"
        assert out["rationale"] == "Refines diabetes risk"

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        ask = make_request_more_info(enforcer)
        ask(needed=[], rationale="x")
        assert enforcer.has_called("request_more_info")


class TestClinicalReportTool:
    def test_returns_structured_report(self):
        enforcer = ProtocolEnforcer()
        report = make_clinical_report(enforcer)
        out = report(
            summary="High diabetes risk.",
            recommendation="See PCP for HbA1c repeat.",
            confidence="high",
            evidence=[{"model": "predict_diabetes_risk",
                       "predicted_class": "diabetic_risk",
                       "confidence": 0.85, "top_features": []}],
            expert_quote="HbA1c 6.5% meets ADA criteria.",
        )
        assert out["summary"] == "High diabetes risk."
        assert out["confidence"] == "high"
        assert out["evidence"][0]["model"] == "predict_diabetes_risk"
        assert "clinician" in out["safety_note"]

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        report = make_clinical_report(enforcer)
        report(summary="x", recommendation="y", confidence="medium")
        assert enforcer.has_called("clinical_report")


class TestAbstainTool:
    def test_returns_abstention_payload(self):
        enforcer = ProtocolEnforcer()
        abst = make_abstain(enforcer)
        out = abst(reason="Symptoms outside ML scope.")
        assert out["abstained"] is True
        assert "scope" in out["reason"]
        assert out["fallback_recommendation"]

    def test_records_call(self):
        enforcer = ProtocolEnforcer()
        abst = make_abstain(enforcer)
        abst(reason="x")
        assert enforcer.has_called("abstain")

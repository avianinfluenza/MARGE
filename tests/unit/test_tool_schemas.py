"""Tests for the per-tool Pydantic input schemas (4 local tools).

Casual chat is natural-language content with no tool call (hybrid
pattern — see apps/orchestrator/system_prompt.md), so there is no
update_user / conversational_reply schema to verify.
"""

import pytest
from pydantic import ValidationError

from apps.orchestrator.tools import (
    abstain as ab_mod,
    clinical_report as cr_mod,
    consult_expert as ce_mod,
    request_more_info as rmi_mod,
)


class TestConsultExpertSchema:
    def test_constants_exposed(self):
        assert ce_mod.TOOL_NAME == "consult_medical_expert"
        assert ce_mod.TOOL_DESCRIPTION

    def test_accepts_question_and_findings(self):
        obj = ce_mod.ToolInput(question="why?", findings={"a": 1})
        assert obj.question == "why?"
        assert obj.findings == {"a": 1}

    def test_findings_defaults_to_empty(self):
        obj = ce_mod.ToolInput(question="?")
        assert obj.findings == {}


class TestRequestMoreInfoSchema:
    def test_constants_exposed(self):
        assert rmi_mod.TOOL_NAME == "request_more_info"
        assert rmi_mod.TOOL_DESCRIPTION

    def test_accepts_needed_and_rationale(self):
        obj = rmi_mod.ToolInput(
            needed=[{"name": "HbA1c", "why": "confirm diabetes"}],
            rationale="HbA1c clarifies the diabetes risk estimate.",
        )
        assert obj.needed[0].name == "HbA1c"
        assert obj.needed[0].field_type == "text"  # default

    def test_rejects_missing_rationale(self):
        with pytest.raises(ValidationError):
            rmi_mod.ToolInput(needed=[])


class TestClinicalReportSchema:
    def test_constants_exposed(self):
        assert cr_mod.TOOL_NAME == "clinical_report"
        assert cr_mod.TOOL_DESCRIPTION

    def test_accepts_full_payload(self):
        obj = cr_mod.ToolInput(
            summary="High diabetes risk.",
            recommendation="Refer for confirmation.",
            confidence="high",
            evidence=[{
                "model": "predict_diabetes_risk",
                "predicted_class": "diabetic_risk",
                "confidence": 0.85,
                "top_features": [],
            }],
            expert_quote="HbA1c at threshold.",
        )
        assert obj.confidence == "high"
        assert obj.evidence[0].model == "predict_diabetes_risk"

    def test_confidence_must_be_one_of_three(self):
        with pytest.raises(ValidationError):
            cr_mod.ToolInput(summary="s", recommendation="r", confidence="absolute")


class TestAbstainSchema:
    def test_constants_exposed(self):
        assert ab_mod.TOOL_NAME == "abstain"
        assert ab_mod.TOOL_DESCRIPTION

    def test_accepts_reason(self):
        obj = ab_mod.ToolInput(reason="Models conflict.")
        assert obj.reason == "Models conflict."
        assert obj.fallback_recommendation  # default present

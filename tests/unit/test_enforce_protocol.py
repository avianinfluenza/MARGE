"""Unit tests for the ProtocolEnforcer code-side defensive backstop.

The main enforcement is LLM-side via MARGE Requirements. This middleware is
the defensive backstop that raises if `final_report` is invoked without the
required expert -> ML -> expert trajectory.
"""

import pytest

from apps.orchestrator.middleware.enforce_protocol import (
    ProtocolEnforcer,
    ProtocolViolation,
)


class TestRecording:
    def test_records_tool_calls(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_breast_cancer_malignancy")
        enforcer.record("consult_medical_expert")
        assert enforcer.has_called("predict_breast_cancer_malignancy")
        assert enforcer.has_called("consult_medical_expert")
        assert not enforcer.has_called("predict_diabetes_risk")

    def test_no_calls_initially(self):
        enforcer = ProtocolEnforcer()
        assert not enforcer.has_called("anything")
        assert not enforcer.can_finalize()


class TestFinalizeGate:
    def test_blocks_finalize_when_nothing_called(self):
        enforcer = ProtocolEnforcer()
        with pytest.raises(ProtocolViolation, match="medical expert"):
            enforcer.check_finalize()

    def test_blocks_finalize_when_only_ml_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_breast_cancer_malignancy")
        with pytest.raises(ProtocolViolation, match="medical expert"):
            enforcer.check_finalize()

    def test_blocks_finalize_when_only_expert_called(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        with pytest.raises(ProtocolViolation, match="ML model"):
            enforcer.check_finalize()

    def test_blocks_finalize_without_post_ml_expert(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        enforcer.record("predict_breast_cancer_malignancy")
        with pytest.raises(ProtocolViolation, match="workflow order"):
            enforcer.check_finalize()

    def test_blocks_finalize_when_ml_precedes_first_expert(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("predict_breast_cancer_malignancy")
        enforcer.record("consult_medical_expert")
        with pytest.raises(ProtocolViolation, match="workflow order"):
            enforcer.check_finalize()

    def test_allows_finalize_when_expert_ml_expert_sequence_exists(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        enforcer.record("predict_breast_cancer_malignancy")
        enforcer.record("consult_medical_expert")
        assert enforcer.can_finalize()
        enforcer.check_finalize()

    def test_any_predict_prefix_satisfies_ml_requirement(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("consult_medical_expert")
        enforcer.record("predict_diabetes_risk")
        enforcer.record("consult_medical_expert")
        enforcer.check_finalize()


class TestConfigurability:
    def test_custom_ml_prefix(self):
        enforcer = ProtocolEnforcer(
            ml_tool_prefixes=("ml_",),
            expert_tool_names=("consult_medical_expert",),
        )
        enforcer.record("consult_medical_expert")
        enforcer.record("ml_breast_cancer")
        enforcer.record("consult_medical_expert")
        enforcer.check_finalize()

    def test_custom_expert_name(self):
        enforcer = ProtocolEnforcer(
            ml_tool_prefixes=("predict_",),
            expert_tool_names=("ask_doctor",),
        )
        enforcer.record("ask_doctor")
        enforcer.record("predict_x")
        enforcer.record("ask_doctor")
        enforcer.check_finalize()


class TestTrajectory:
    def test_trajectory_preserves_order(self):
        enforcer = ProtocolEnforcer()
        enforcer.record("a")
        enforcer.record("b")
        enforcer.record("a")
        assert enforcer.trajectory == ("a", "b", "a")

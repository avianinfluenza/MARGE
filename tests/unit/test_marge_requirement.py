"""Tests for disabled BeeAI Requirements helpers.

The preferred workflow used to be enforced at the BeeAI requirement layer:

    consult_medical_expert -> predict_* -> consult_medical_expert -> final_report

Runtime gating is currently disabled so no-data and missing-info turns can
call `final_report` without forcing an ML prediction.
"""

from dataclasses import dataclass

from apps.orchestrator.requirements.marge_protocol import (
    build_marge_protocol_requirement,
    build_marge_protocol_requirements,
    has_any_ml_prediction,
    has_consulted_expert,
    has_expert_ml_expert_sequence,
    has_post_ml_expert_consult,
    has_pre_ml_expert_consult,
)


@dataclass
class _StubTool:
    name: str


@dataclass
class _StubStep:
    tool: object | None
    error: object | None = None


@dataclass
class _StubState:
    steps: list[_StubStep]


def _state(*tool_names: str, error_at: set[int] | None = None) -> _StubState:
    error_at = error_at or set()
    steps = [
        _StubStep(tool=_StubTool(name=n), error=Exception("x") if i in error_at else None)
        for i, n in enumerate(tool_names)
    ]
    return _StubState(steps=steps)


class TestBasicChecks:
    def test_has_any_ml_prediction(self):
        assert has_any_ml_prediction(_state("predict_diabetes_risk"))
        assert not has_any_ml_prediction(_state("consult_medical_expert"))

    def test_has_consulted_expert(self):
        assert has_consulted_expert(_state("consult_medical_expert"))
        assert not has_consulted_expert(_state("predict_diabetes_risk"))

    def test_ignores_failed_steps(self):
        assert not has_any_ml_prediction(_state("predict_diabetes_risk", error_at={0}))
        assert not has_consulted_expert(_state("consult_medical_expert", error_at={0}))


class TestWorkflowSequence:
    def test_expert_then_ml_has_pre_ml_consult(self):
        assert has_pre_ml_expert_consult(
            _state("consult_medical_expert", "predict_diabetes_risk")
        )

    def test_ml_then_expert_has_no_pre_ml_consult(self):
        assert not has_pre_ml_expert_consult(
            _state("predict_diabetes_risk", "consult_medical_expert")
        )

    def test_ml_then_expert_has_post_ml_consult(self):
        assert has_post_ml_expert_consult(
            _state("predict_diabetes_risk", "consult_medical_expert")
        )

    def test_expert_then_ml_without_second_expert_is_incomplete(self):
        assert not has_expert_ml_expert_sequence(
            _state("consult_medical_expert", "predict_diabetes_risk")
        )

    def test_ml_then_expert_without_valid_preconsult_is_incomplete(self):
        assert not has_expert_ml_expert_sequence(
            _state("predict_diabetes_risk", "consult_medical_expert")
        )

    def test_expert_ml_expert_sequence_satisfies_workflow(self):
        assert has_expert_ml_expert_sequence(
            _state(
                "get_patient_history",
                "consult_medical_expert",
                "predict_diabetes_risk",
                "consult_medical_expert",
            )
        )

    def test_failed_post_consult_does_not_satisfy_workflow(self):
        assert not has_expert_ml_expert_sequence(
            _state(
                "consult_medical_expert",
                "predict_diabetes_risk",
                "consult_medical_expert",
                error_at={2},
            )
        )


class TestRequirementBuilder:
    def test_build_terminal_requirement_is_disabled(self):
        assert build_marge_protocol_requirement() is None

    def test_build_all_requirements_returns_empty_list(self):
        reqs = build_marge_protocol_requirements(
            ["predict_diabetes_risk", "not_a_predictor"]
        )
        assert reqs == []

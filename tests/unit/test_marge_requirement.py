"""Tests for the BeeAI Requirement that encodes architecture.md §2.

The single requirement gates `final_report` so:
- It is only `allowed` after at least one predict_* tool call AND at least
  one consult_medical_expert call have appeared in the trajectory.
- The agent's stop is `prevent`-ed until `final_report` has been called
  at least once.

Order between predict_* and consult_medical_expert is intentionally
unconstrained — the orchestrator can talk to the expert before, after,
or interleaved with ML calls.
"""

from dataclasses import dataclass

import pytest

from apps.orchestrator.requirements.marge_protocol import (
    has_any_ml_prediction,
    has_consulted_expert,
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


class TestHasAnyMLPrediction:
    def test_empty_trajectory_returns_false(self):
        assert not has_any_ml_prediction(_state())

    def test_returns_true_for_predict_breast_cancer(self):
        assert has_any_ml_prediction(_state("predict_breast_cancer_malignancy"))

    def test_returns_true_for_predict_diabetes(self):
        assert has_any_ml_prediction(_state("predict_diabetes_risk"))

    def test_ignores_non_predict_tools(self):
        assert not has_any_ml_prediction(
            _state("get_patient", "consult_medical_expert")
        )

    def test_ignores_failed_steps(self):
        assert not has_any_ml_prediction(
            _state("predict_breast_cancer_malignancy", error_at={0})
        )


class TestHasConsultedExpert:
    def test_empty_trajectory_returns_false(self):
        assert not has_consulted_expert(_state())

    def test_returns_true_when_called(self):
        assert has_consulted_expert(_state("consult_medical_expert"))

    def test_does_not_match_other_tools(self):
        assert not has_consulted_expert(
            _state("predict_breast_cancer_malignancy", "get_patient")
        )

    def test_ignores_failed_step(self):
        assert not has_consulted_expert(
            _state("consult_medical_expert", error_at={0})
        )


class TestOrderIsFree:
    """ML and expert may be called in any order — only presence in trajectory matters."""

    def test_expert_then_ml_satisfies_both(self):
        s = _state("consult_medical_expert", "predict_diabetes_risk")
        assert has_consulted_expert(s) and has_any_ml_prediction(s)

    def test_ml_then_expert_satisfies_both(self):
        s = _state("predict_diabetes_risk", "consult_medical_expert")
        assert has_consulted_expert(s) and has_any_ml_prediction(s)

    def test_interleaved_satisfies_both(self):
        s = _state(
            "consult_medical_expert",
            "predict_breast_cancer_malignancy",
            "consult_medical_expert",
            "predict_diabetes_risk",
        )
        assert has_consulted_expert(s) and has_any_ml_prediction(s)


class TestRequirementBuilder:
    """Verify the ConditionalRequirement we wire into the agent."""

    def test_build_requirement_returns_conditional_requirement(self):
        from beeai_framework.agents.requirement.requirements.conditional import (
            ConditionalRequirement,
        )
        from apps.orchestrator.requirements.marge_protocol import (
            build_marge_protocol_requirement,
        )

        req = build_marge_protocol_requirement()
        assert isinstance(req, ConditionalRequirement)

    def test_build_requirement_targets_final_report(self):
        from apps.orchestrator.requirements.marge_protocol import (
            build_marge_protocol_requirement,
        )

        req = build_marge_protocol_requirement()
        # ConditionalRequirement stores the target as `source` after init
        assert req.source == "final_report"

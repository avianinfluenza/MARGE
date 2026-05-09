"""Disabled BeeAI Requirements for the MARGE workflow.

The previous runtime gate enforced:

    consult_medical_expert -> predict_* -> consult_medical_expert -> final_report

That made no-data, missing-info, and information-only turns fail because the
agent could not call `final_report` until an ML tool succeeded. For now, the
BeeAI ConditionalRequirements are intentionally disabled; the workflow remains
prompt guidance rather than a hard tool-availability gate.
"""

from collections.abc import Iterable
from typing import Any


_TERMINAL_TOOL_NAME = "final_report"
_ML_PREDICTION_PREFIX = "predict_"
_EXPERT_TOOL_NAME = "consult_medical_expert"


def _successful_tool_names(state: Any) -> list[str]:
    return [
        s.tool.name
        for s in state.steps
        if s.tool is not None and not s.error and getattr(s.tool, "name", None)
    ]


def has_any_ml_prediction(state: Any) -> bool:
    """True if any predict_* tool has succeeded in the trajectory."""
    return any(name.startswith(_ML_PREDICTION_PREFIX) for name in _successful_tool_names(state))


def has_consulted_expert(state: Any) -> bool:
    """True if consult_medical_expert has succeeded in the trajectory."""
    return _EXPERT_TOOL_NAME in _successful_tool_names(state)


def has_pre_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded before a successful ML prediction."""
    seen_expert = False
    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            seen_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_expert:
            return True
    return False


def has_post_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded after a successful ML prediction."""
    seen_ml = False
    for name in _successful_tool_names(state):
        if name.startswith(_ML_PREDICTION_PREFIX):
            seen_ml = True
        elif name == _EXPERT_TOOL_NAME and seen_ml:
            return True
    return False


def has_expert_ml_expert_sequence(state: Any) -> bool:
    """True if the successful trajectory contains expert -> ML -> expert."""
    seen_pre_expert = False
    seen_ml_after_pre_expert = False

    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            if seen_ml_after_pre_expert:
                return True
            seen_pre_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_pre_expert:
            seen_ml_after_pre_expert = True

    return False


def build_marge_protocol_requirement() -> None:
    """Return no terminal BeeAI requirement while runtime gating is disabled."""
    return None


def build_marge_protocol_requirements(
    ml_tool_names: Iterable[str] = (),
) -> list[Any]:
    """Return no BeeAI requirements for the current tool surface."""
    _ = tuple(ml_tool_names)
    return []

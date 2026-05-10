"""Local tool: abstain — terminal for "we cannot reliably advise".

Use when:
- ML predictions conflict irresolvably and the medical expert cannot reconcile.
- The medical expert flags the user's symptoms as outside the scope of any
  available predict_* tool ("scope mismatch" path).
- Data quality is too poor for a reliable analytical conclusion.

Gating: requires at least one consult_medical_expert in the trajectory
(enforced by MARGEProtocolRequirement). The orchestrator must have at
least *tried* to consult the expert before declining.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "abstain"
TOOL_DESCRIPTION = (
    "Terminal: decline to provide a clinical recommendation, with a structured "
    "rationale and a fallback action for the user. Use when ML predictions "
    "conflict irresolvably, when the expert flags data as unreliable, or when "
    "the user's concern is outside the scope of available ML predictors. "
    "Requires at least one consult_medical_expert in the trajectory."
)


class ToolInput(BaseModel):
    reason: str = Field(
        description=(
            "Concrete reason this turn cannot give a reliable recommendation "
            "(e.g., 'Expert review indicates the symptoms do not map to any "
            "available ML predictor', or 'Diabetes and breast-screening models "
            "both returned low-confidence outputs that the expert could not "
            "reconcile')."
        )
    )
    fallback_recommendation: str = Field(
        default="Please consult a qualified clinician for evaluation.",
        description="What the user should do instead — concrete next step.",
    )


def make_abstain(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def abstain(
        reason: str,
        fallback_recommendation: str = "Please consult a qualified clinician for evaluation.",
    ) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)
        return {
            "abstained": True,
            "reason": reason,
            "fallback_recommendation": fallback_recommendation,
        }

    abstain.__doc__ = TOOL_DESCRIPTION
    return abstain

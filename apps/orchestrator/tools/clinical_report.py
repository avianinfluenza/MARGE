"""Local tool: clinical_report — structured terminal for confident answers.

Replaces the previous free-text `final_report`. The orchestrator passes a
structured payload that the Streamlit UI renders as a clinical report card
(summary + recommendation + per-ML evidence + expert quote + citations).

LLM-side gating is handled by `MARGEProtocolRequirement` (clinical_report
is hidden from the agent's tool list until at least one predict_* and one
consult_medical_expert call have succeeded).
"""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "clinical_report"
TOOL_DESCRIPTION = (
    "Emit the FINAL structured clinical report to the user — use ONLY when "
    "you have ML evidence AND an expert clinical interpretation that together "
    "support a concrete recommendation. The framework hides this tool until "
    "consult_medical_expert and at least one predict_* tool have succeeded."
)


class MLEvidence(BaseModel):
    model: str = Field(description="Tool name, e.g., 'predict_diabetes_risk'")
    predicted_class: str = Field(description="The class the model returned")
    confidence: float = Field(description="Model confidence, 0–1")
    top_features: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top SHAP contributors: [{feature, contribution, value}, ...]",
    )


class ToolInput(BaseModel):
    summary: str = Field(
        description="One-paragraph plain-language summary of the case for the user."
    )
    recommendation: str = Field(
        description="Concrete next step (e.g., 'see a primary care clinician for fasting glucose confirmation')."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in the recommendation given ML + expert agreement."
    )
    evidence: list[MLEvidence] = Field(
        default_factory=list,
        description="Per-ML-model evidence summaries — what each predict_* returned and why.",
    )
    expert_quote: str | None = Field(
        default=None,
        description="Key clinical reasoning from the medical expert (paraphrased OK).",
    )
    safety_note: str = Field(
        default="This system supports clinical judgement; it does not replace a clinician.",
        description="Always-included safety reminder. Override only if you have a stronger phrasing.",
    )


def make_clinical_report(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def clinical_report(
        summary: str,
        recommendation: str,
        confidence: str,
        evidence: list[dict[str, Any]] | None = None,
        expert_quote: str | None = None,
        safety_note: str = "This system supports clinical judgement; it does not replace a clinician.",
    ) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)
        return {
            "summary": summary,
            "recommendation": recommendation,
            "confidence": confidence,
            "evidence": evidence or [],
            "expert_quote": expert_quote,
            "safety_note": safety_note,
        }

    clinical_report.__doc__ = TOOL_DESCRIPTION
    return clinical_report

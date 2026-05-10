"""Local tool: request_more_info — terminal asking the user for specific data.

Use when one or two missing data points would meaningfully shift the
analysis (e.g., "HbA1c", "current medications", "family history of breast
cancer"). The orchestrator should NOT use this for vague "tell me more about
yourself" prompts — be specific.

This is a terminal because it ends the agent loop and waits for the user's
next message. It is NOT gated — the orchestrator may request more info at
any point in the conversation, including before any expert consultation.
"""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "request_more_info"
TOOL_DESCRIPTION = (
    "Terminal: ask the user for specific additional information that would "
    "meaningfully change the analysis. Use a structured `needed` list — each "
    "item names exactly what to collect and explains why. Always allowed "
    "(no protocol prerequisites)."
)


class NeededField(BaseModel):
    name: str = Field(description="Short name of the field (e.g., 'HbA1c', 'family_history_diabetes').")
    why: str = Field(description="One-line clinical reason this field would shift the analysis.")
    field_type: Literal["number", "text", "category", "yes_no"] = Field(
        default="text",
        description="Hint to the UI for input rendering.",
    )
    unit: str | None = Field(
        default=None,
        description="Optional unit if numeric (e.g., 'mg/dL', 'kg/m^2').",
    )


class ToolInput(BaseModel):
    needed: list[NeededField] = Field(
        description="Specific items to ask the user for. Keep the list short (1–4 items).",
    )
    rationale: str = Field(
        description=(
            "One- or two-sentence framing for the user explaining why these "
            "specific data points matter for the analysis."
        )
    )


def make_request_more_info(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def request_more_info(
        needed: list[dict[str, Any]],
        rationale: str,
    ) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)
        return {
            "needs_more_info": True,
            "needed": needed,
            "rationale": rationale,
        }

    request_more_info.__doc__ = TOOL_DESCRIPTION
    return request_more_info

"""Local tool: conversational_reply — terminal for casual / non-analytical chat.

Use this when:
- The user is greeting, thanking, or making smalltalk.
- You're answering a clarification question that doesn't require ML analysis
  (e.g., "what can you do?", "is this confidential?").
- You're closing the conversation politely.

This ends the turn naturally without invoking any clinical structure (no
recommendation card, no info request form). It is freely allowed (no
protocol prerequisites).

The three other terminals remain for analytical paths:
- `clinical_report`     — confident analytical conclusion (gated: ML + expert)
- `abstain`             — unable to advise (gated: expert)
- `request_more_info`   — need specific data points to proceed (free)

Use `conversational_reply` ONLY when no clinical analysis is happening or
warranted. If the user has shared symptoms or asked you to analyze
something, the analytical terminals are the correct exit.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "conversational_reply"
TOOL_DESCRIPTION = (
    "Terminal: end the turn with a casual conversational reply (no clinical "
    "structure). Use for greetings, smalltalk, capability questions, and "
    "polite closes. Always allowed. Do NOT use for analytical answers — "
    "use clinical_report, abstain, or request_more_info instead when the "
    "user has shared symptoms or asked for analysis."
)


class ToolInput(BaseModel):
    text: str = Field(
        description=(
            "Conversational reply to the user. Plain natural language, "
            "warm tone. This message ends the turn — finish what you want "
            "to say in one shot."
        )
    )


def make_conversational_reply(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def conversational_reply(text: str) -> dict[str, Any]:
        enforcer.record(TOOL_NAME)
        return {"reply": text}

    conversational_reply.__doc__ = TOOL_DESCRIPTION
    return conversational_reply

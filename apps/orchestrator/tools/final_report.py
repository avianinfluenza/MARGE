"""Local tool: emit the final response to the user.

The single terminal tool. The orchestrator writes a natural-language
`response` here, covering positive recommendations, abstentions
("cannot reliably advise; please see a doctor"), or follow-up questions
("please provide HbA1c, recent mammogram, family history").

BeeAI workflow requirements are temporarily disabled, so this tool is available
for both ML-grounded reports and missing-information responses. The
`ProtocolEnforcer` still records the call trajectory for logs and debugging.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

TOOL_NAME = "final_report"
TOOL_DESCRIPTION = (
    "Emit the final response to the user. THIS IS THE ONLY PATH TO A USER-FACING "
    "ANSWER. The `response` field accepts free-form natural language; use it for "
    "the recommendation, an abstention ('I cannot give reliable guidance; please "
    "see a doctor'), or a follow-up question ('please provide X, Y, Z'). Report "
    "ML scores only when a predict_* tool actually ran successfully."
)


class ToolInput(BaseModel):
    response: str = Field(
        description=(
            "Natural-language reply to the user. Covers all three response modes: "
            "(a) clinical recommendation when ML and expert agree, "
            "(b) explicit abstention with referral, "
            "(c) follow-up question requesting additional information."
        )
    )


def make_final_report(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def final_report(response: str) -> dict[str, Any]:
        # enforcer.check_finalize()  # temporarily disabled with MARGE protocol requirement
        enforcer.record(TOOL_NAME)
        return {"response": response}

    final_report.__doc__ = TOOL_DESCRIPTION
    return final_report

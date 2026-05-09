"""Local tool: abstain — decline to answer, recommending professional consultation.

Always allowed (no protocol prerequisites). The orchestrator should prefer
this when ML predictions conflict irresolvably or the medical expert flags
the data as unreliable.
"""

from collections.abc import Callable

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

_TOOL_NAME = "abstain"


def make_abstain(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, str | bool]]:
    def abstain(reason: str) -> dict[str, str | bool]:
        """Decline to give a clinical recommendation.

        Args:
            reason: Why a reliable answer cannot be given (e.g.,
                    "ML predictions conflict and expert flagged data quality").
        """
        enforcer.check_can_abstain()
        enforcer.record(_TOOL_NAME)
        return {"abstained": True, "reason": reason}

    return abstain

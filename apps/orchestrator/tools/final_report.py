"""Local tool: emit the final report to the user.

This is the only path to a user-facing answer. The ProtocolEnforcer gates
it: if neither an ML tool nor the medical expert have been called yet, the
call raises ProtocolViolation. See architecture.md §2 for the constraint.
"""

from collections.abc import Callable
from typing import Any

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

_TOOL_NAME = "final_report"


def make_final_report(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def final_report(
        summary: str,
        recommendation: str,
        confidence_note: str | None = None,
    ) -> dict[str, Any]:
        """Emit the final clinical report to the user.

        Fails with ProtocolViolation unless at least one ML prediction tool
        AND consult_medical_expert have been called in the current trajectory.

        Args:
            summary: One-paragraph summary of the case and findings.
            recommendation: Action the user (clinician) should consider.
            confidence_note: Optional note about confidence / agreement.
        """
        enforcer.check_finalize()
        enforcer.record(_TOOL_NAME)
        return {
            "summary": summary,
            "recommendation": recommendation,
            "confidence_note": confidence_note,
        }

    return final_report

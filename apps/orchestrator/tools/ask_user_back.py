"""Local tool: ask the user for additional information before answering.

Always allowed (no protocol prerequisites). The orchestrator should prefer
this when one missing feature would meaningfully shift the ML predictions —
i.e., where the information gain is high.
"""

from collections.abc import Callable
from typing import Any

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer

_TOOL_NAME = "ask_user_back"


def make_ask_user_back(enforcer: ProtocolEnforcer) -> Callable[..., dict[str, Any]]:
    def ask_user_back(missing_info: list[str]) -> dict[str, Any]:
        """Request additional information from the user before answering.

        Args:
            missing_info: List of items to request (e.g., recent labs,
                          family history, current medications).
        """
        enforcer.check_can_ask_user_back()
        enforcer.record(_TOOL_NAME)
        return {"asking_user_back": True, "missing_info": missing_info}

    return ask_user_back

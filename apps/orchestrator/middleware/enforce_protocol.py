"""Protocol-enforcement middleware for the orchestrator.

Implements the structural rule from architecture.md §2:
- `final_report` is blocked unless at least one ML tool AND one
  `consult_medical_expert` call appear in the trajectory.
- `abstain` and `ask_user_back` are escape hatches and may be called
  without these prerequisites.

This is the orchestrator's structural guarantee that it will never produce
medical advice without first consulting an ML model and the medical expert.
The rule lives in code, not in a system prompt — it cannot be jailbroken.

Wired into BeeAI as a per-tool-call hook: every tool invocation passes
through `record(name)`; the gated tools call `check_finalize()` before
producing output.
"""

from collections.abc import Iterable


class ProtocolViolation(Exception):
    """Raised when a tool is called in violation of the orchestration protocol."""


class ProtocolEnforcer:
    """Tracks tool calls and gates terminal tools by precondition checks."""

    def __init__(
        self,
        ml_tool_prefixes: Iterable[str] = ("predict_",),
        expert_tool_names: Iterable[str] = ("consult_medical_expert",),
    ) -> None:
        self._ml_prefixes = tuple(ml_tool_prefixes)
        self._expert_names = tuple(expert_tool_names)
        self._calls: list[str] = []

    def record(self, tool_name: str) -> None:
        self._calls.append(tool_name)

    @property
    def trajectory(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def has_called(self, name: str) -> bool:
        return name in self._calls

    def _ml_called(self) -> bool:
        return any(c.startswith(p) for c in self._calls for p in self._ml_prefixes)

    def _expert_called(self) -> bool:
        return any(c in self._expert_names for c in self._calls)

    def can_finalize(self) -> bool:
        return self._ml_called() and self._expert_called()

    def check_finalize(self) -> None:
        if not self._ml_called():
            raise ProtocolViolation(
                "Cannot call final_report: no ML model has been consulted. "
                "Call at least one ML prediction tool first, then consult the "
                "medical expert before finalising."
            )
        if not self._expert_called():
            raise ProtocolViolation(
                "Cannot call final_report: medical expert has not been consulted. "
                "Call consult_medical_expert with the ML findings before finalising."
            )

    def check_can_abstain(self) -> None:
        return  # always allowed

    def check_can_ask_user_back(self) -> None:
        return  # always allowed

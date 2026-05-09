"""Protocol-enforcement middleware (defense in depth).

Architecture.md §2's structural rule is enforced two ways:

- LLM-side (primary): `MARGEProtocolRequirement` keeps `final_report`
  disallowed in BeeAI's tool listing until ML and expert have been
  consulted. The agent literally cannot pick the tool too early.
- Code-side (this file, defensive): if `final_report` is somehow invoked
  outside the agent loop, `check_finalize()` still raises.

`final_report` is now the only terminal tool. Abstention and follow-up
questions are expressed as natural-language `response` text inside
`final_report`, not as separate tools.
"""

from collections.abc import Iterable


class ProtocolViolation(Exception):
    """Raised when `final_report` is invoked without preconditions."""


class ProtocolEnforcer:
    """Tracks tool calls and gates `final_report` by precondition check."""

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

    def _is_ml_call(self, name: str) -> bool:
        return any(name.startswith(p) for p in self._ml_prefixes)

    def _is_expert_call(self, name: str) -> bool:
        return name in self._expert_names

    def _ml_called(self) -> bool:
        return any(self._is_ml_call(c) for c in self._calls)

    def _expert_called(self) -> bool:
        return any(self._is_expert_call(c) for c in self._calls)

    def _has_expert_ml_expert_sequence(self) -> bool:
        seen_pre_expert = False
        seen_ml_after_pre_expert = False

        for call in self._calls:
            if self._is_expert_call(call):
                if seen_ml_after_pre_expert:
                    return True
                seen_pre_expert = True
            elif self._is_ml_call(call) and seen_pre_expert:
                seen_ml_after_pre_expert = True

        return False

    def can_finalize(self) -> bool:
        return self._has_expert_ml_expert_sequence()

    def check_finalize(self) -> None:
        if not self._expert_called():
            raise ProtocolViolation(
                "Cannot call final_report: medical expert has not been consulted. "
                "Call consult_medical_expert before ML and again after ML findings."
            )
        if not self._ml_called():
            raise ProtocolViolation(
                "Cannot call final_report: no ML model has been consulted. "
                "Call at least one relevant ML prediction tool before finalising."
            )
        if not self._has_expert_ml_expert_sequence():
            raise ProtocolViolation(
                "Cannot call final_report: workflow order is incomplete. "
                "Required sequence is consult_medical_expert -> predict_* -> "
                "consult_medical_expert."
            )

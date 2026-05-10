"""MARGE Protocol Requirement (custom BeeAI Requirement).

Encodes architecture.md §2 plus the role-aware extensions agreed in the
agent_fix slice. Single requirement, four rules:

  A. predict_* tools are disallowed until consult_medical_expert was
     called successfully at least once. The expert decides which conditions
     to consider; the orchestrator translates that into specific ML calls.
  B. clinical_report (terminal) needs at least one predict_* AND at least
     one consult_medical_expert in the trajectory.
  C. abstain (terminal) needs at least one consult_medical_expert. This
     covers the "expert says nothing in our ML scope is suspected" path.
  D. request_more_info (terminal) is always allowed — the orchestrator may
     ask the user for additional data at any point.
  E. The agent cannot terminate (prevent_stop=True on every terminal) until
     at least one of {clinical_report, abstain, request_more_info} has been
     called. The intermediate `update_user` tool is unrestricted and never
     terminates the loop.

Order between predict_* and consult_medical_expert beyond rule A is
intentionally unconstrained — the orchestrator can re-consult the expert
after running ML, run more ML after expert validation, etc.

Helper functions `has_any_ml_prediction` and `has_consulted_expert` are
exposed for tests and for ad-hoc inspection of an agent state.
"""

from typing import Any

from beeai_framework.agents.requirement.requirements.requirement import (
    Requirement,
    Rule,
    run_with_context,
)


_ML_PREDICTION_PREFIX = "predict_"
_EXPERT_TOOL_NAME = "consult_medical_expert"


def _successful_tool_names(state: Any) -> list[str]:
    return [
        s.tool.name
        for s in state.steps
        if s.tool is not None and not s.error and getattr(s.tool, "name", None)
    ]


def has_any_ml_prediction(state: Any) -> bool:
    """True if any predict_* tool has succeeded in the trajectory."""
    return any(
        name.startswith(_ML_PREDICTION_PREFIX)
        for name in _successful_tool_names(state)
    )


def has_consulted_expert(state: Any) -> bool:
    """True if consult_medical_expert has succeeded in the trajectory."""
    return _EXPERT_TOOL_NAME in _successful_tool_names(state)


class MARGEProtocolRequirement(Requirement):
    """Single Requirement encoding the four MARGE protocol rules (A–E above)."""

    TERMINALS = frozenset(
        {"clinical_report", "abstain", "request_more_info", "conversational_reply"}
    )

    def __init__(self) -> None:
        super().__init__()
        # BeeAI's Requirement base treats `name` as a required attribute —
        # the framework reads it (via `to_safe_word(name)`) when creating
        # the per-requirement emitter group. Set it explicitly here.
        self.name = "marge_protocol"

    @property
    def priority(self) -> int:
        return 50

    async def init(self, *, tools, ctx) -> None:
        await super().init(tools=tools, ctx=ctx)
        self._predict_tools = [
            t for t in tools if t.name.startswith(_ML_PREDICTION_PREFIX)
        ]
        self._terminal_tools = [t for t in tools if t.name in self.TERMINALS]

    @run_with_context
    async def run(self, state: Any, context: Any) -> list[Rule]:  # noqa: ARG002
        return self._compute_rules(state)

    def _compute_rules(self, state: Any) -> list[Rule]:
        """Pure-sync rule evaluation. Public for tests."""
        called = _successful_tool_names(state)
        has_expert = _EXPERT_TOOL_NAME in called
        has_ml = any(n.startswith(_ML_PREDICTION_PREFIX) for n in called)
        has_terminal = any(n in self.TERMINALS for n in called)

        rules: list[Rule] = []

        # Rule A: predict_* gated on expert
        for tool in getattr(self, "_predict_tools", []):
            rules.append(
                Rule(
                    target=tool.name,
                    allowed=has_expert,
                    prevent_stop=False,
                    hidden=False,
                    forced=False,
                    reason=None
                    if has_expert
                    else (
                        "Consult the medical expert first — they decide which "
                        "clinical concerns warrant ML-based screening, then "
                        "you map those to the available predict_* models."
                    ),
                )
            )

        # Rules B / C / D + E (terminals)
        for tool in getattr(self, "_terminal_tools", []):
            if tool.name == "clinical_report":
                allowed = has_ml and has_expert
                reason = (
                    None
                    if allowed
                    else (
                        "clinical_report needs at least one predict_* run AND "
                        "one consult_medical_expert in the trajectory."
                    )
                )
            elif tool.name == "abstain":
                allowed = has_expert
                reason = (
                    None
                    if allowed
                    else (
                        "abstain may only be used after consulting the medical "
                        "expert at least once."
                    )
                )
            else:  # request_more_info / conversational_reply — free
                allowed = True
                reason = None

            rules.append(
                Rule(
                    target=tool.name,
                    allowed=allowed,
                    # Rule E: every terminal blocks stop until *any* terminal
                    # has been called once.
                    prevent_stop=not has_terminal,
                    hidden=False,
                    forced=False,
                    reason=reason,
                )
            )

        return rules


def build_marge_protocol_requirement() -> MARGEProtocolRequirement:
    """Factory used by `apps/orchestrator/agent.py`."""
    return MARGEProtocolRequirement()

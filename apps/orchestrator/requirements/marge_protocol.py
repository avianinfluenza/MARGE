"""MARGE Protocol Requirement (custom BeeAI Requirement).

Hybrid pattern (post-`agent_fix` refactor): casual chat is plain
natural-language `content` from the LLM and ends the turn with **no tool
call**. Only structured outputs that the UI must render as cards are
expressed as terminal tools — the surface is now four:

  A. predict_* tools are disallowed until consult_medical_expert was
     called successfully at least once. The expert decides which conditions
     to consider; the orchestrator translates that into specific ML calls.
  B. clinical_report (terminal) needs at least one predict_* AND at least
     one consult_medical_expert in the trajectory.
  C. abstain (terminal) needs at least one consult_medical_expert. This
     covers the "expert says nothing in our ML scope is suspected" path.
  D. request_more_info (terminal) is always allowed — the orchestrator may
     ask the user for additional data at any point.

There is no `prevent_stop` rule any more. A natural-language reply with
no tool call is a perfectly valid turn ending (a chatbot saying "hi
back"). The prompt steers the model toward picking the right structured
terminal when one is appropriate.

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


def has_pre_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded before a successful ML prediction."""
    seen_expert = False
    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            seen_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_expert:
            return True
    return False


def has_post_ml_expert_consult(state: Any) -> bool:
    """True if an expert consult succeeded after a successful ML prediction."""
    seen_ml = False
    for name in _successful_tool_names(state):
        if name.startswith(_ML_PREDICTION_PREFIX):
            seen_ml = True
        elif name == _EXPERT_TOOL_NAME and seen_ml:
            return True
    return False


def has_expert_ml_expert_sequence(state: Any) -> bool:
    """True if the successful trajectory contains expert -> ML -> expert."""
    seen_pre_expert = False
    seen_ml_after_pre_expert = False

    for name in _successful_tool_names(state):
        if name == _EXPERT_TOOL_NAME:
            if seen_ml_after_pre_expert:
                return True
            seen_pre_expert = True
        elif name.startswith(_ML_PREDICTION_PREFIX) and seen_pre_expert:
            seen_ml_after_pre_expert = True

    return False


class MARGEProtocolRequirement(Requirement):
    """Single Requirement encoding the four MARGE protocol rules (A–D above)."""

    TERMINALS = frozenset({"clinical_report", "abstain", "request_more_info"})

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

        # Rules B / C / D (terminals — gating only, no prevent_stop)
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
            else:  # request_more_info — free
                allowed = True
                reason = None

            rules.append(
                Rule(
                    target=tool.name,
                    allowed=allowed,
                    # Natural-language replies (no tool call) are a valid
                    # turn ending; never block stop.
                    prevent_stop=False,
                    hidden=False,
                    forced=False,
                    reason=reason,
                )
            )

        return rules


def build_marge_protocol_requirement() -> MARGEProtocolRequirement:
    """Factory used by `apps/orchestrator/agent.py`."""
    return MARGEProtocolRequirement()

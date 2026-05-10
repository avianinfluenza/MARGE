"""Local tool: consult the medical expert sub-agent.

Async-aware: the expert may be a sync `StubMedicalExpert` (tests) or an
async real `MedicalExpertAgent` (live). The closure detects which and
awaits if needed.
"""

import inspect
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from packages.schemas.retrieval import MedicalExpertResponse

TOOL_NAME = "consult_medical_expert"
TOOL_DESCRIPTION = (
    "Consult the medical expert sub-agent for clinical reasoning. Pass a focused "
    "clinical question and a `findings` summary (clinical context, lab values, "
    "or — once available — ML results expressed as raw clinical numbers, never "
    "as 'model X says ...'). Returns reasoning + citations."
)


class ToolInput(BaseModel):
    question: str = Field(description="The clinical question to ask the expert.")
    findings: dict[str, Any] = Field(
        default_factory=dict,
        description="Clinical context (symptoms, lab values, ML results as raw clinical numbers).",
    )


class _MedicalExpert(Protocol):
    def consult(self, question: str, findings: dict[str, Any]) -> Any:  # may be sync or async
        ...


def make_consult_expert(
    expert: _MedicalExpert,
    enforcer: ProtocolEnforcer,
) -> Callable[..., Any]:
    """Build the consult_medical_expert tool bound to a specific expert + enforcer.

    The returned callable is async — the BeeAI tool adapter awaits it. Sync
    `expert.consult` (e.g., StubMedicalExpert) returns a value directly;
    async `expert.consult` returns a coroutine that we await.
    """

    async def consult_medical_expert(
        question: str, findings: dict[str, Any]
    ) -> MedicalExpertResponse:
        enforcer.record(TOOL_NAME)
        result = expert.consult(question=question, findings=findings)
        if inspect.iscoroutine(result):
            result = await result
        return result

    consult_medical_expert.__doc__ = TOOL_DESCRIPTION
    return consult_medical_expert

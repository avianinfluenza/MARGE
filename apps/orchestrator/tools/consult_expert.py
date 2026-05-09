"""Local tool: consult the medical expert sub-agent."""

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from packages.schemas.retrieval import MedicalExpertResponse

TOOL_NAME = "consult_medical_expert"
TOOL_DESCRIPTION = (
    "Consult the medical expert sub-agent for clinical reasoning. Pass a focused "
    "clinical question and a `findings` summary (ML predictions, patient context) "
    "the expert should reason over. Returns reasoning + citations. Required before "
    "any final report."
)


class ToolInput(BaseModel):
    question: str = Field(description="The clinical question to ask the expert.")
    findings: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of ML predictions and patient context the expert should consider.",
    )


class MedicalExpert(Protocol):
    def consult(self, question: str, findings: dict[str, Any]) -> Any:
        """Return or await a MedicalExpertResponse."""


def make_consult_expert(
    expert: MedicalExpert,
    enforcer: ProtocolEnforcer,
) -> Callable[..., MedicalExpertResponse | Any]:
    """Build the consult_medical_expert tool bound to a specific expert + enforcer."""

    def consult_medical_expert(
        question: str, findings: dict[str, Any]
    ) -> MedicalExpertResponse | Any:
        enforcer.record(TOOL_NAME)
        return expert.consult(question=question, findings=findings)

    consult_medical_expert.__doc__ = TOOL_DESCRIPTION
    return consult_medical_expert

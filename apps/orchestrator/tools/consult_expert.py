"""Local tool: consult the medical expert sub-agent."""

from collections.abc import Callable
from typing import Any

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from packages.schemas.retrieval import MedicalExpertResponse
from services.medical_expert_agent.agent import StubMedicalExpert

_TOOL_NAME = "consult_medical_expert"


def make_consult_expert(
    expert: StubMedicalExpert,
    enforcer: ProtocolEnforcer,
) -> Callable[..., MedicalExpertResponse]:
    """Build the consult_medical_expert tool bound to a specific expert + enforcer."""

    def consult_medical_expert(
        question: str, findings: dict[str, Any]
    ) -> MedicalExpertResponse:
        """Consult the medical expert sub-agent for clinical reasoning.

        Args:
            question: The clinical question to ask.
            findings: Summary of ML predictions and patient context the
                      expert should reason over.

        Returns:
            MedicalExpertResponse with reasoning and citations.
        """
        enforcer.record(_TOOL_NAME)
        return expert.consult(question=question, findings=findings)

    return consult_medical_expert

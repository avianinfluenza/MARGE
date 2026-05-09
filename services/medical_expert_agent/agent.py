"""Stub medical_expert agent.

Returns a deterministic `MedicalExpertResponse` for use in early integration
tests of the orchestrator. The real BeeAI sub-agent — backed by Granite or
Claude, with `search_local_kb` / `search_web` tools and the `enforce_citation`
middleware — replaces this in the next slice.

The stub is intentionally schema-conformant: it returns at least one
Citation so that a real `enforce_citation` middleware would also accept it.
"""

from typing import Any

from packages.schemas.retrieval import Citation, MedicalExpertResponse, RetrievedDocument

_STUB_DOC = RetrievedDocument(
    title="WHO clinical guideline (stub)",
    snippet=(
        "Stub citation used during early development. The orchestrator's "
        "downstream consumers can rely on the same shape they will see when "
        "the real medical_expert sub-agent is wired up."
    ),
    source_url="https://stub.example.org/who-guideline",
    retrieval_source="local_kb",
)

_STUB_REASONING = (
    "Based on the supplied findings (stub response): the patient's profile "
    "warrants further targeted screening. Clinical judgement should "
    "incorporate the ML findings, the patient's history, and standard "
    "guideline-driven thresholds. This is a stub response — replace with the "
    "real medical_expert sub-agent before any clinical use."
)


class StubMedicalExpert:
    """Returns a fixed MedicalExpertResponse without invoking any LLM."""

    def consult(self, question: str, findings: dict[str, Any]) -> MedicalExpertResponse:
        return MedicalExpertResponse(
            reasoning=_STUB_REASONING,
            citations=[Citation(document=_STUB_DOC, supporting_quote=None)],
        )

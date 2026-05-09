"""Tests for StubMedicalExpert.

The stub returns a fixed MedicalExpertResponse for use in early integration
tests. It must conform to the schema (so `enforce_citation` middleware would
be satisfied) and must include at least one Citation.
"""

from packages.schemas.retrieval import Citation, MedicalExpertResponse
from services.medical_expert_agent.agent import StubMedicalExpert


class TestStubMedicalExpert:
    def test_returns_medical_expert_response(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="any question", findings={})
        assert isinstance(response, MedicalExpertResponse)

    def test_response_has_non_empty_reasoning(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert response.reasoning
        assert len(response.reasoning) > 20

    def test_response_has_at_least_one_citation(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert len(response.citations) >= 1
        assert isinstance(response.citations[0], Citation)

    def test_citation_has_source_url(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert response.citations[0].document.source_url

    def test_not_abstained_by_default(self):
        expert = StubMedicalExpert()
        response = expert.consult(question="?", findings={})
        assert response.abstained is False

    def test_response_independent_of_input(self):
        """Stub returns the same response regardless of input — that's the point."""
        expert = StubMedicalExpert()
        a = expert.consult(question="q1", findings={"x": 1})
        b = expert.consult(question="q2", findings={"y": 2})
        assert a.reasoning == b.reasoning

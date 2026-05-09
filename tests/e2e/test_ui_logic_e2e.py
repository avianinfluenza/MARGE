"""E2E: Streamlit UI pure-Python logic — no browser required.

Tests the stateless helper functions in apps/streamlit_ui/app.py:
metric parsing, report section splitting, number highlighting, and
result text extraction.

These run without launching a Streamlit server or a real LLM.
Note: feature extraction helpers were removed in the patient-MCP refactor;
      those responsibilities now live inside the orchestrator agent itself.
"""

import pytest

from apps.streamlit_ui.app import (
    _extract_metrics,
    _highlight_numbers,
    _report_sections,
    _result_text,
    _risk_tone,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# _risk_tone
# ---------------------------------------------------------------------------

class TestRiskTone:
    def test_high_risk(self):
        assert _risk_tone(90.0) == "high"

    def test_medium_risk(self):
        assert _risk_tone(75.0) == "medium"

    def test_low_risk(self):
        assert _risk_tone(50.0) == "low"

    def test_boundary_85_is_high(self):
        assert _risk_tone(85.0) == "high"

    def test_boundary_70_is_medium(self):
        assert _risk_tone(70.0) == "medium"


# ---------------------------------------------------------------------------
# _extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_extracts_breast_cancer_metric(self):
        text = "The breast malignancy probability is 72%."
        metrics = _extract_metrics(text)
        labels = [m["label"] for m in metrics]
        assert "Breast Screening" in labels

    def test_extracts_diabetes_metric(self):
        text = "Diabetes risk is estimated at 65%."
        metrics = _extract_metrics(text)
        labels = [m["label"] for m in metrics]
        assert "Diabetes Risk" in labels

    def test_extracts_glucose_metric(self):
        text = "Plasma glucose: 148 mg/dL"
        metrics = _extract_metrics(text)
        labels = [m["label"] for m in metrics]
        assert "Glucose" in labels

    def test_returns_empty_for_plain_text(self):
        assert _extract_metrics("No medical values here.") == []


# ---------------------------------------------------------------------------
# _report_sections
# ---------------------------------------------------------------------------

class TestReportSections:
    def test_recommendation_goes_to_recommended_followup(self):
        text = "We recommend a follow-up in 3 months. The result looks normal."
        sections = _report_sections(text)
        assert "Recommended Follow-Up" in sections

    def test_clinical_disclaimer_goes_to_clinical_note(self):
        text = "This does not replace a clinical decision. It supports clinical judgement."
        sections = _report_sections(text)
        assert "Clinical Note" in sections

    def test_general_text_goes_to_key_findings(self):
        text = "The patient shows elevated glucose levels."
        sections = _report_sections(text)
        assert "Key Findings" in sections

    def test_empty_sections_not_included(self):
        text = "Normal results observed."
        sections = _report_sections(text)
        assert "Recommended Follow-Up" not in sections
        assert "Clinical Note" not in sections


# ---------------------------------------------------------------------------
# _highlight_numbers
# ---------------------------------------------------------------------------

class TestHighlightNumbers:
    def test_wraps_number_in_span(self):
        result = _highlight_numbers("Value is 42.5%")
        assert "<span class='metric-inline'>42.5%</span>" in result

    def test_escapes_html_entities(self):
        result = _highlight_numbers("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_splits_on_period(self):
        parts = _split_sentences("First sentence. Second sentence.")
        assert len(parts) == 2

    def test_splits_on_exclamation(self):
        parts = _split_sentences("Alert! Be careful.")
        assert len(parts) == 2

    def test_empty_string_returns_empty_list(self):
        assert _split_sentences("") == []


# ---------------------------------------------------------------------------
# _result_text
# ---------------------------------------------------------------------------

class TestResultText:
    def test_returns_str_for_plain_object(self):
        class FakeResult:
            def __str__(self):
                return "plain text"

        assert _result_text(FakeResult()) == "plain text"

    def test_prefers_output_structured_response(self):
        class Structured:
            response = "structured response"

        class FakeResult:
            output_structured = Structured()

        assert _result_text(FakeResult()) == "structured response"

    def test_falls_back_to_answer_text(self):
        class Answer:
            text = "answer text"

        class FakeResult:
            output_structured = None
            answer = Answer()

        assert _result_text(FakeResult()) == "answer text"

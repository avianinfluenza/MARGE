"""Tests for MARGEProtocolRequirement (custom BeeAI Requirement).

Encodes four protocol rules (no `prevent_stop` — natural-language replies
with no tool call are valid turn endings under the hybrid pattern):
  A. predict_* tools are disallowed until consult_medical_expert was called
     successfully at least once.
  B. clinical_report (terminal) is disallowed until BOTH at least one predict_*
     and at least one consult_medical_expert have succeeded.
  C. abstain (terminal) is disallowed until at least one consult_medical_expert
     has succeeded.
  D. request_more_info (terminal) is always allowed.

There is no chat-as-tool wrapper: casual chat is the LLM's natural-language
`content` and produces no tool record. ML model calls (predict_*) and
consult_medical_expert may be called in any order beyond rule A; multiple
calls are fine.
"""

from dataclasses import dataclass

from apps.orchestrator.requirements.marge_protocol import (
    MARGEProtocolRequirement,
    has_any_ml_prediction,
    has_consulted_expert,
    has_expert_ml_expert_sequence,
    has_post_ml_expert_consult,
    has_pre_ml_expert_consult,
)


# --------------------------- helpers ---------------------------

@dataclass
class _StubTool:
    name: str


@dataclass
class _StubStep:
    tool: object | None
    error: object | None = None


@dataclass
class _StubState:
    steps: list[_StubStep]


def _state(*tool_names: str, error_at: set[int] | None = None) -> _StubState:
    error_at = error_at or set()
    return _StubState(steps=[
        _StubStep(tool=_StubTool(name=n), error=Exception("x") if i in error_at else None)
        for i, n in enumerate(tool_names)
    ])


def _make_tool(name: str) -> _StubTool:
    return _StubTool(name=name)


def _build_req() -> MARGEProtocolRequirement:
    """Construct + manually init the requirement with the standard MARGE tool set.

    We bypass the async `init` (which expects a RunContext) and prep the
    private attributes directly — the test scope is the rule logic, not
    BeeAI's lifecycle plumbing.
    """
    req = MARGEProtocolRequirement()
    tools = [
        _make_tool("get_patient_history"),
        _make_tool("consult_medical_expert"),
        _make_tool("predict_breast_cancer_malignancy"),
        _make_tool("predict_diabetes_risk"),
        _make_tool("clinical_report"),
        _make_tool("abstain"),
        _make_tool("request_more_info"),
    ]
    req._predict_tools = [t for t in tools if t.name.startswith("predict_")]
    req._terminal_tools = [t for t in tools if t.name in MARGEProtocolRequirement.TERMINALS]
    return req


def _rules_by_target(req: MARGEProtocolRequirement, state: _StubState) -> dict[str, object]:
    return {r.target: r for r in req._compute_rules(state)}


# --------------------------- helper functions ---------------------------

class TestHasAnyMLPrediction:
    def test_empty(self):
        assert not has_any_ml_prediction(_state())

    def test_breast_cancer_predictor(self):
        assert has_any_ml_prediction(_state("predict_breast_cancer_malignancy"))

    def test_diabetes_predictor(self):
        assert has_any_ml_prediction(_state("predict_diabetes_risk"))

    def test_ignores_non_predict(self):
        assert not has_any_ml_prediction(_state("consult_medical_expert", "request_more_info"))

    def test_ignores_failed(self):
        assert not has_any_ml_prediction(
            _state("predict_breast_cancer_malignancy", error_at={0})
        )


class TestHasConsultedExpert:
    def test_empty(self):
        assert not has_consulted_expert(_state())

    def test_basic(self):
        assert has_consulted_expert(_state("consult_medical_expert"))

    def test_other_tools_dont_count(self):
        assert not has_consulted_expert(_state("predict_diabetes_risk", "request_more_info"))

    def test_ignores_failed(self):
        assert not has_consulted_expert(
            _state("consult_medical_expert", error_at={0})
        )


# --------------------------- Rule A: predict_* gated on expert ---------------------------

class TestPredictGatedOnExpert:
    def test_predict_disallowed_when_expert_not_called(self):
        req = _build_req()
        rules = _rules_by_target(req, _state("get_patient_history"))
        for name in ("predict_breast_cancer_malignancy", "predict_diabetes_risk"):
            assert rules[name].allowed is False
            assert rules[name].reason

    def test_predict_allowed_after_expert(self):
        req = _build_req()
        rules = _rules_by_target(req, _state("consult_medical_expert"))
        for name in ("predict_breast_cancer_malignancy", "predict_diabetes_risk"):
            assert rules[name].allowed is True

    def test_predict_allowed_after_expert_then_more_consults(self):
        req = _build_req()
        rules = _rules_by_target(
            req, _state("consult_medical_expert", "consult_medical_expert")
        )
        for name in ("predict_breast_cancer_malignancy", "predict_diabetes_risk"):
            assert rules[name].allowed is True


# --------------------------- Rule B: clinical_report needs ML + expert ---------------------------

class TestClinicalReportGate:
    def test_disallowed_when_neither(self):
        req = _build_req()
        r = _rules_by_target(req, _state())["clinical_report"]
        assert not r.allowed

    def test_disallowed_when_only_expert(self):
        req = _build_req()
        r = _rules_by_target(req, _state("consult_medical_expert"))["clinical_report"]
        assert not r.allowed

    def test_disallowed_when_only_ml(self):
        # (Should never happen given Rule A, but the rule is an AND-of-both.)
        req = _build_req()
        r = _rules_by_target(req, _state("predict_diabetes_risk"))["clinical_report"]
        assert not r.allowed

    def test_allowed_when_both_called(self):
        req = _build_req()
        s = _state("consult_medical_expert", "predict_diabetes_risk")
        r = _rules_by_target(req, s)["clinical_report"]
        assert r.allowed


# --------------------------- Rule C: abstain needs expert ---------------------------

class TestAbstainGate:
    def test_disallowed_with_no_expert(self):
        req = _build_req()
        r = _rules_by_target(req, _state("get_patient_history"))["abstain"]
        assert not r.allowed

    def test_allowed_after_expert(self):
        req = _build_req()
        r = _rules_by_target(req, _state("consult_medical_expert"))["abstain"]
        assert r.allowed

    def test_allowed_after_expert_without_ml(self):
        # abstain after expert-only consult is the "scope mismatch" case
        # (orchestrator probed, expert said no relevant ML maps to symptoms).
        req = _build_req()
        r = _rules_by_target(req, _state("consult_medical_expert"))["abstain"]
        assert r.allowed


# --------------------------- Rule D: request_more_info free ---------------------------

class TestRequestMoreInfoIsFree:
    def test_allowed_at_start(self):
        req = _build_req()
        r = _rules_by_target(req, _state())["request_more_info"]
        assert r.allowed

    def test_allowed_after_anything(self):
        req = _build_req()
        s = _state("consult_medical_expert", "predict_diabetes_risk")
        r = _rules_by_target(req, s)["request_more_info"]
        assert r.allowed


# --------------------------- Stop is never blocked ---------------------------

class TestPreventStopAlwaysFalse:
    """Hybrid pattern: a natural-language reply with no tool call is a valid
    turn ending. The Requirement never sets prevent_stop=True on any rule."""

    def test_at_start(self):
        req = _build_req()
        rules = _rules_by_target(req, _state())
        for name in ("clinical_report", "abstain", "request_more_info",
                     "predict_breast_cancer_malignancy", "predict_diabetes_risk"):
            assert rules[name].prevent_stop is False

    def test_after_consult_only(self):
        req = _build_req()
        rules = _rules_by_target(req, _state("consult_medical_expert"))
        for name in ("clinical_report", "abstain", "request_more_info"):
            assert rules[name].prevent_stop is False

    def test_after_terminal(self):
        req = _build_req()
        s = _state("consult_medical_expert", "predict_diabetes_risk", "clinical_report")
        rules = _rules_by_target(req, s)
        for name in ("clinical_report", "abstain", "request_more_info"):
            assert rules[name].prevent_stop is False


# --------------------------- Order freedom (sanity) ---------------------------

class TestOrderingFreedom:
    def test_expert_then_ml_then_terminal(self):
        req = _build_req()
        s = _state(
            "consult_medical_expert",
            "predict_diabetes_risk",
            "consult_medical_expert",
            "clinical_report",
        )
        rules = _rules_by_target(req, s)
        assert rules["clinical_report"].allowed
        assert rules["clinical_report"].prevent_stop is False

    def test_multiple_predicts_one_expert(self):
        req = _build_req()
        s = _state(
            "consult_medical_expert",
            "predict_diabetes_risk",
            "predict_breast_cancer_malignancy",
        )
        rules = _rules_by_target(req, s)
        assert rules["predict_diabetes_risk"].allowed
        assert rules["predict_breast_cancer_malignancy"].allowed
        assert rules["clinical_report"].allowed

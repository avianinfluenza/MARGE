"""E2E: ML pipeline — registry discovery → model load → predict → SHAP scores.

Tests the full in-process pipeline for both models that changed in this iteration:
- diabetes_xgb (was CatBoost, now DynamicMLAgent/XGBoost)
- breast_cancer_xgb (refactored to DynamicMLAgent)

No real MCP transport; calls predict() directly via the MLModel interface.
Requires trained artifacts in services/ml_mcp_server/artifacts/.
"""

import math

import pytest

from packages.schemas.prediction import Prediction, XAIScore
from services.ml_mcp_server.registry import discover_models


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def models():
    return {m.name: m for m in discover_models()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_discovers_both_models(self, models):
        assert "predict_breast_cancer_malignancy" in models
        assert "predict_diabetes_risk" in models

    def test_no_extra_public_models(self, models):
        """Only non-underscore model files are registered."""
        expected = {"predict_breast_cancer_malignancy", "predict_diabetes_risk"}
        assert set(models.keys()) == expected

    def test_all_models_comply_with_mlmodel_abc(self, models):
        from services.ml_mcp_server.models._base import MLModel

        for name, model in models.items():
            assert isinstance(model, MLModel), f"{name} is not an MLModel subclass"


# ---------------------------------------------------------------------------
# Shared contract: every registered model
# ---------------------------------------------------------------------------

class TestMLModelContract:
    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_sample_inputs_matches_schema(self, models, model_name):
        model = models[model_name]
        sample = model.sample_inputs()
        schema = model.input_schema
        parsed = schema(**sample)
        assert parsed is not None

    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_predict_returns_prediction(self, models, model_name):
        model = models[model_name]
        inputs = model.input_schema(**model.sample_inputs())
        result = model.predict(inputs)
        assert isinstance(result, Prediction)

    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_prediction_has_valid_confidence(self, models, model_name):
        model = models[model_name]
        inputs = model.input_schema(**model.sample_inputs())
        result = model.predict(inputs)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_prediction_has_xai_scores(self, models, model_name):
        model = models[model_name]
        inputs = model.input_schema(**model.sample_inputs())
        result = model.predict(inputs)
        assert len(result.xai_scores) > 0
        assert all(isinstance(s, XAIScore) for s in result.xai_scores)

    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_xai_scores_have_no_nan(self, models, model_name):
        model = models[model_name]
        inputs = model.input_schema(**model.sample_inputs())
        result = model.predict(inputs)
        for score in result.xai_scores:
            assert not math.isnan(score.contribution), f"NaN contribution in {score.feature_name}"

    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_xai_scores_ordered_by_abs_contribution(self, models, model_name):
        model = models[model_name]
        inputs = model.input_schema(**model.sample_inputs())
        result = model.predict(inputs)
        scores = result.xai_scores
        if len(scores) >= 2:
            for i in range(len(scores) - 1):
                assert abs(scores[i].contribution) >= abs(scores[i + 1].contribution)

    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_prediction_class_probabilities_sum_to_one(self, models, model_name):
        model = models[model_name]
        inputs = model.input_schema(**model.sample_inputs())
        result = model.predict(inputs)
        if result.class_probabilities:
            total = sum(result.class_probabilities.values())
            assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}"

    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_metadata_model_name_matches_tool_name(self, models, model_name):
        model = models[model_name]
        assert model.metadata.name == model_name


# ---------------------------------------------------------------------------
# Diabetes model specifics (DynamicMLAgent refactor)
# ---------------------------------------------------------------------------

class TestDiabetesModel:
    def test_expected_output_classes(self, models):
        m = models["predict_diabetes_risk"]
        inputs = m.input_schema(**m.sample_inputs())
        result = m.predict(inputs)
        assert result.predicted_class in {"low_risk", "diabetic_risk"}

    def test_high_risk_sample_predicts_diabetic(self, models):
        """Pima Indians row 0 (preg=6, glucose=148, ...) is a known positive case."""
        m = models["predict_diabetes_risk"]
        inputs = m.input_schema(**m.sample_inputs())
        result = m.predict(inputs)
        assert result.predicted_class == "diabetic_risk"

    def test_schema_has_all_eight_features(self, models):
        m = models["predict_diabetes_risk"]
        fields = set(m.input_schema.model_fields.keys())
        expected = {"preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"}
        assert expected == fields

    def test_predict_with_partial_inputs(self, models):
        """DynamicMLAgent treats missing features as NaN — should not crash."""
        m = models["predict_diabetes_risk"]
        partial = {"plas": 148.0, "mass": 33.6, "age": 50.0}
        inputs = m.input_schema(**partial)
        result = m.predict(inputs)
        assert isinstance(result, Prediction)

    def test_version_reflects_factory_refactor(self, models):
        m = models["predict_diabetes_risk"]
        assert "factory" in m.metadata.version.lower()


# ---------------------------------------------------------------------------
# Breast cancer model specifics (DynamicMLAgent refactor)
# ---------------------------------------------------------------------------

class TestBreastCancerModel:
    def test_expected_output_classes(self, models):
        m = models["predict_breast_cancer_malignancy"]
        inputs = m.input_schema(**m.sample_inputs())
        result = m.predict(inputs)
        assert result.predicted_class in {"malignant", "benign"}

    def test_schema_has_30_features(self, models):
        m = models["predict_breast_cancer_malignancy"]
        assert len(m.input_schema.model_fields) == 30

    def test_sample_first_row_predicts_malignant(self, models):
        """sklearn breast cancer row 0 is a known malignant case."""
        m = models["predict_breast_cancer_malignancy"]
        inputs = m.input_schema(**m.sample_inputs())
        result = m.predict(inputs)
        assert result.predicted_class == "malignant"

    def test_version_reflects_factory_refactor(self, models):
        m = models["predict_breast_cancer_malignancy"]
        assert "factory" in m.metadata.version.lower()


# ---------------------------------------------------------------------------
# Serialisation — NaN must become JSON null (MCP transport contract)
# ---------------------------------------------------------------------------

class TestPredictionSerialisation:
    @pytest.mark.parametrize("model_name", [
        "predict_diabetes_risk",
        "predict_breast_cancer_malignancy",
    ])
    def test_json_roundtrip_has_no_nan(self, models, model_name):
        import json

        from pydantic import TypeAdapter

        model = models[model_name]
        inputs = model.input_schema(**model.sample_inputs())
        result = model.predict(inputs)
        dumped = TypeAdapter(Prediction).dump_python(result, mode="json")
        json_text = json.dumps(dumped)
        assert "NaN" not in json_text, "NaN leaked into JSON output (breaks MCP transport)"

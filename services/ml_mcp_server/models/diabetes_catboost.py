"""CatBoost classifier on the Pima Indians Diabetes Database (UCI / OpenML).

Binary task: tested_positive (1) vs tested_negative (0) for diabetes from
8 metabolic and demographic features.

Train: `python -m packages.ml_training.train_diabetes`

This is the second drop-in model — the registry auto-discovers it without
any change to `server.py` or to the orchestrator's tool list.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field, create_model

from packages.schemas.prediction import ModelMetadata, Prediction, XAIScore
from services.ml_mcp_server.explainers.shap_wrapper import compute_shap_values
from services.ml_mcp_server.models._base import MLModel

_ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "diabetes_catboost.joblib"

# Pima Indians Diabetes feature codes -> human-readable descriptions.
_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "preg": "Number of pregnancies",
    "plas": "Plasma glucose concentration (oral glucose tolerance test, 2-hour, mg/dL)",
    "pres": "Diastolic blood pressure (mm Hg)",
    "skin": "Triceps skin fold thickness (mm)",
    "insu": "2-hour serum insulin (mu U/ml)",
    "mass": "Body mass index (kg / m^2)",
    "pedi": "Diabetes pedigree function (genetic predisposition score)",
    "age": "Age in years",
}


class DiabetesCATBoost(MLModel):
    """CatBoost binary classifier for type-2 diabetes risk."""

    @property
    def name(self) -> str:
        return "predict_diabetes_risk"

    def __init__(self) -> None:
        if not _ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact missing: {_ARTIFACT_PATH}. "
                f"Run: python -m packages.ml_training.train_diabetes"
            )
        bundle = joblib.load(_ARTIFACT_PATH)
        self._model = bundle["model"]
        self._test_accuracy: float = bundle["test_accuracy"]
        self._test_f1: float | None = bundle.get("test_f1")
        self._feature_names: list[str] = bundle["feature_names"]
        self._sample: dict[str, float] = bundle["sample_inputs"]

        # Build the input schema from feature names recorded at training time.
        self._input_cls: type[BaseModel] = create_model(
            "DiabetesInputs",
            **{
                name: (
                    float | None,
                    Field(description=_FEATURE_DESCRIPTIONS.get(name, name)),
                )
                for name in self._feature_names
            },
        )

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            version="0.1.0",
            description=(
                "Binary classifier for type-2 diabetes risk from 8 metabolic and "
                "demographic features (pregnancies, plasma glucose, blood pressure, "
                "triceps skin fold, serum insulin, BMI, genetic pedigree function, "
                "age). Trained on the Pima Indians Diabetes Database (UCI / OpenML, "
                "n=768)."
            ),
            trained_on="OpenML 'diabetes' v1 (Pima Indians, n=768)",
            test_accuracy=self._test_accuracy,
            test_f1=self._test_f1,
            feature_count=len(self._feature_names),
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return self._input_cls

    def sample_inputs(self) -> dict[str, Any]:
        return dict(self._sample)

    def predict(self, inputs: BaseModel) -> Prediction:
        data = inputs.model_dump()
        X = np.array(
            [[np.nan if data[name] is None else data[name] for name in self._feature_names]],
            dtype=float,
        )   

        proba = self._model.predict_proba(X)[0]
        # CatBoost / sklearn convention: positive class index = 1.
        p_positive = float(proba[1])
        p_negative = float(proba[0])
        predicted_class = "diabetic_risk" if p_positive >= 0.5 else "low_risk"
        confidence = max(p_positive, p_negative)

        shap_vals = compute_shap_values(self._model, X)
        all_scores = [
            XAIScore(
                feature_name=self._feature_names[i],
                contribution=float(shap_vals[0, i]),
                feature_value=float(X[0, i]),
            )
            for i in range(len(self._feature_names))
        ]
        all_scores.sort(key=lambda s: abs(s.contribution), reverse=True)

        return Prediction(
            model_name=self.name,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities={"diabetic_risk": p_positive, "low_risk": p_negative},
            xai_scores=all_scores[:5],
            metadata=self.metadata,
        )

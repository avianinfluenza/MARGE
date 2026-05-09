"""GBDT ensemble classifier on the Pima Indians Diabetes Database (UCI / OpenML).

Binary task: tested_positive (1) vs tested_negative (0) for diabetes from
8 metabolic and demographic features.
Refactored to utilize the generalized DynamicMLAgent Factory pattern.
"""

from pathlib import Path
from typing import Any

# Assuming ml_agent_factory.py is accessible
from ._agent_factory import AgentConfig, DynamicMLAgent

# --- 1. Define Paths and Constants ---
_ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "diabetes_dynamic.joblib"

_FEATURE_NAMES = [
    "preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"
]

_FEATURE_METADATA = {
    "preg": {
        "label": "Pregnancy history",
        "detail": "Number of pregnancies, only when clinically relevant.",
        "aliases": ["pregnancies", "pregnancy count", "임신 횟수", "임신"],
    },
    "plas": {
        "label": "Blood sugar",
        "detail": "Recent fasting glucose, random glucose, or oral glucose tolerance test result.",
        "aliases": ["plasma glucose", "blood sugar", "glucose", "혈당", "공복혈당", "식후혈당"],
    },
    "pres": {
        "label": "Blood pressure",
        "detail": "Recent diastolic blood pressure reading.",
        "aliases": ["diastolic blood pressure", "blood pressure", "bp", "혈압"],
    },
    "skin": {
        "label": "Skinfold thickness",
        "detail": "Triceps skinfold thickness. This is often unavailable outside clinical records.",
        "aliases": ["skinfold", "skin thickness", "피부 두께", "피부두께"],
    },
    "insu": {
        "label": "Insulin",
        "detail": "Recent 2-hour serum insulin result, if available.",
        "aliases": ["insulin", "인슐린"],
    },
    "mass": {
        "label": "BMI",
        "detail": "Height and weight are enough if BMI is not already known.",
        "aliases": ["bmi", "body mass index", "체질량", "체질량지수"],
    },
    "pedi": {
        "label": "Family history",
        "detail": "Family history of diabetes or a calculated diabetes pedigree score.",
        "aliases": ["diabetes pedigree", "family history", "가족력", "유전"],
    },
    "age": {
        "label": "Age",
        "detail": "Current age in years.",
        "aliases": ["age", "나이", "살", "세"],
    },
}

# --- 2. Define the Agent Configuration ---
_DIABETES_CONFIG = AgentConfig(
    agent_name="predict_diabetes_risk",
    description=(
        "Binary classifier for type-2 diabetes risk from 8 metabolic and "
        "demographic features (pregnancies, plasma glucose, blood pressure, "
        "triceps skin fold, serum insulin, BMI, genetic pedigree function, "
        "age). Trained dynamically via Agent Factory."
    ),
    version="0.2.0-factory",
    artifact_path=_ARTIFACT_PATH,
    feature_names=_FEATURE_NAMES,
    feature_metadata=_FEATURE_METADATA,
    target_classes=["low_risk", "diabetic_risk"],  # 0 = low risk, 1 = diabetic risk
    trained_on_desc="OpenML 'diabetes' v1 (Pima Indians, n=768) - Dynamic Ensemble",
    n_splits=5
)

# --- 3. Create the Server-Compatible Model Class ---
class DiabetesClassifier(DynamicMLAgent):
    """Dynamic binary classifier for type-2 diabetes risk."""

    def __init__(self) -> None:
        # Initialize the generalized factory agent
        super().__init__(_DIABETES_CONFIG)

        if not self.is_trained:
            from sklearn.datasets import fetch_openml
            import pandas as pd
            
            # Fetch the Pima Indians Diabetes database
            dataset = fetch_openml(name='diabetes', version=1, as_frame=True)
            X_df = dataset.data
            y_series = dataset.target.map({'tested_negative': 0, 'tested_positive': 1})
            
            self.train(X_df, y_series)

    def sample_inputs(self) -> dict[str, Any]:
        """Sample high-risk case from the Pima Indians dataset."""
        # Providing a static dictionary representation of a true positive case
        return {
            "preg": 6.0,
            "plas": 148.0,
            "pres": 72.0,
            "skin": 35.0,
            "insu": 0.0,
            "mass": 33.6,
            "pedi": 0.627,
            "age": 50.0
        }

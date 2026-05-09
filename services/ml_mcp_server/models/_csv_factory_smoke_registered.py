"""CSV factory smoke model registered with the ML MCP server."""

from pathlib import Path
from typing import Any

from ._agent_factory import AgentConfig, DynamicMLAgent

_ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "csv_factory_smoke_model.joblib"

_FEATURE_METADATA = {
    "risk_score": {
        "label": "Risk score",
        "detail": "A numeric risk score from the source CSV or upstream risk calculator.",
        "aliases": ["risk score", "risk_score", "위험점수", "리스크"],
    },
    "glucose": {
        "label": "Glucose",
        "detail": "Recent fasting glucose, random glucose, or oral glucose tolerance test result.",
        "aliases": ["glucose", "blood sugar", "혈당", "공복혈당", "식후혈당"],
    },
    "bmi": {
        "label": "BMI",
        "detail": "Height and weight are enough if BMI is not already known.",
        "aliases": ["bmi", "body mass index", "체질량", "체질량지수"],
    },
    "age": {
        "label": "Age",
        "detail": "Current age in years.",
        "aliases": ["age", "나이", "살", "세"],
    },
}

_CSV_SMOKE_CONFIG = AgentConfig(
    agent_name="predict_csv_factory_smoke",
    description=(
        "Automated ensemble classifier built from demo_patients.csv. "
        "Uses risk_score, glucose, BMI, and age to predict has_disease."
    ),
    version="1.0.0-auto",
    artifact_path=_ARTIFACT_PATH,
    feature_names=["risk_score", "glucose", "bmi", "age"],
    feature_metadata=_FEATURE_METADATA,
    target_classes=["Class_0", "Class_1"],
    trained_on_desc="Local CSV: demo_patients.csv (n=10)",
    n_splits=5,
)


class CSVFactorySmokeModel(DynamicMLAgent):
    """Drop-in MCP model generated from the CSV factory smoke dataset."""

    def __init__(self) -> None:
        super().__init__(_CSV_SMOKE_CONFIG)
        self._sample_inputs = self.sample_inputs()

    def sample_inputs(self) -> dict[str, Any]:
        return {
            "risk_score": 0.82,
            "glucose": 168.0,
            "bmi": 35.8,
            "age": 60.0,
        }

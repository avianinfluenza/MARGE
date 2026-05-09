"""XGBoost classifier on the sklearn breast cancer (Wisconsin Diagnostic) dataset.

Binary task: malignant (0) vs benign (1) from 30 fine-needle aspiration features.
Refactored to utilize the generalized DynamicMLAgent Factory pattern.
"""

from pathlib import Path
from typing import Any
from sklearn.datasets import load_breast_cancer

# Assuming ml_agent_factory.py is in the same directory or accessible module
from ._agent_factory import AgentConfig, DynamicMLAgent

# --- 1. Load Dataset & Constants ---
_DATASET = load_breast_cancer()
_FEATURE_NAMES: list[str] = list(_DATASET.feature_names)
_ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "breast_cancer_xgb_dynamic.joblib"

# --- 2. Define the Agent Configuration ---
_CANCER_CONFIG = AgentConfig(
    agent_name="predict_breast_cancer_malignancy",
    description=(
        "Binary classifier for breast tumor malignancy from 30 fine-needle "
        "aspiration features (cell nucleus measurements: radius, texture, "
        "perimeter, area, smoothness, compactness, concavity, etc., aggregated "
        "as mean / standard error / worst). Trained dynamically via Agent Factory."
    ),
    version="0.4.0-factory",
    artifact_path=_ARTIFACT_PATH,
    feature_names=_FEATURE_NAMES,
    target_classes=["malignant", "benign"], # sklearn convention: 0=malignant, 1=benign
    trained_on_desc="sklearn.datasets.load_breast_cancer (Wisconsin Diagnostic, n=569)",
    n_splits=5
)

# --- 3. Create the Server-Compatible Model Class ---
class BreastCancerXGB(DynamicMLAgent):
    """XGBoost binary classifier for breast tumor malignancy."""

    def __init__(self) -> None:
        # Initialize the generalized factory agent with our specific config
        super().__init__(_CANCER_CONFIG)
        
        if not self.is_trained:
            import pandas as pd
            X_df = pd.DataFrame(_DATASET.data, columns=self.config.feature_names)
            y_series = pd.Series(_DATASET.target)
            self.train(X_df, y_series)

    def sample_inputs(self) -> dict[str, Any]:
        """First row of the dataset — a real malignant case."""
        # self._sanitize is inherited from DynamicMLAgent
        return {
            self._sanitize(name): float(val)
            for name, val in zip(self.config.feature_names, _DATASET.data[0], strict=False)
        }
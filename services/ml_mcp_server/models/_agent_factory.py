"""Generalized ML Agent Factory for tree-based ensemble models.

Provides a factory to instantiate dynamic ML agents with "Init-or-Train" 
lifecycles, K-Fold ensembling, and integrated SHAP explainability.
"""

from pathlib import Path
from typing import Any, Optional, Dict, List, Type
from dataclasses import dataclass, field

import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pydantic import BaseModel, Field, create_model
from sklearn.model_selection import KFold

# Assuming these schemas are available in your project structure:
from packages.schemas.prediction import ModelMetadata, Prediction, XAIScore
from services.ml_mcp_server.models._base import MLModel

@dataclass
class AgentConfig:
    """Configuration required to instantiate a DynamicMLAgent."""
    agent_name: str
    description: str
    version: str
    artifact_path: Path
    feature_names: List[str]
    target_classes: List[str]  # e.g., ["malignant", "benign"] (index 0 is negative, 1 is positive class)
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary:logistic',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'eval_metric': 'logloss'
    })
    n_splits: int = 5
    trained_on_desc: str = "Dynamic Internal Dataset"
    feature_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SHAPExplainer:
    """Handles SHAP calculations using pandas DataFrames as internal reference data."""
    def __init__(self, models: List[xgb.XGBClassifier], background_data: pd.DataFrame):
        self.models = models
        self.background_data = background_data
        self.explainers = [
            shap.TreeExplainer(model, self.background_data) for model in self.models
        ]
        
    def explain(self, X_external: pd.DataFrame) -> np.ndarray:
        all_shap_values = []
        for explainer in self.explainers:
            shap_values = explainer.shap_values(X_external)
            # For binary classification, extract positive class if returned as a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            all_shap_values.append(shap_values)
        return np.mean(all_shap_values, axis=0)


class EnsembleWrapper:
    """General wrapper for GBDT K-fold training, ensembling, and SHAP interpretation."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.models: List[xgb.XGBClassifier] = []
        self.explainer: Optional[SHAPExplainer] = None
        self.feature_names: List[str] = []

    def train_with_cv(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        self.feature_names = X.columns.tolist()
        self.models = [] 
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**self.params)
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            self.models.append(model)

        self.explainer = SHAPExplainer(self.models, X)

    def predict_ensemble_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        return np.mean(preds, axis=1)

    def get_explanations(self, X_external: pd.DataFrame) -> np.ndarray:
        if not self.explainer:
            raise ValueError("Model must be trained before generating explanations.")
        return self.explainer.explain(X_external)


class DynamicMLAgent(MLModel):
    """A generalized, persistent ML Agent capable of dynamic schema generation and training."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._input_schema = self._build_dynamic_schema()
        self._sample_inputs: dict[str, float] = {}
        
        if self.config.artifact_path.exists():
            print(f"[{self.name}] Found existing artifact. Loading state...")
            self.wrapper = joblib.load(self.config.artifact_path)
            self.is_trained = True
        else:
            print(f"[{self.name}] No artifact found. Server requires training call.")
            self.wrapper = EnsembleWrapper(self.config.model_params)
            self.is_trained = False

    @property
    def name(self) -> str:
        return self.config.agent_name

    def _sanitize(self, name: str) -> str:
        return name.replace(" ", "_")

    def _build_dynamic_schema(self) -> Type[BaseModel]:
        """Dynamically creates a Pydantic model for input validation based on features."""
        fields = {}
        for name in self.config.feature_names:
            sanitized_name = self._sanitize(name)
            metadata = (
                self.config.feature_metadata.get(sanitized_name)
                or self.config.feature_metadata.get(name)
                or {}
            )
            description = metadata.get("detail") or metadata.get("description")
            if not description:
                description = f"Feature measurement: {name}."
            fields[sanitized_name] = (
                Optional[float],
                Field(
                    default=None,
                    description=description,
                    json_schema_extra={
                        "label": metadata.get("label"),
                        "aliases": metadata.get("aliases", []),
                    },
                ),
            )
        return create_model(f"{self.name}Inputs", **fields)

    @property
    def input_schema(self) -> Type[BaseModel]:
        return self._input_schema

    def sample_inputs(self) -> dict[str, Any]:
        return dict(self._sample_inputs)

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            version=self.config.version,
            description=self.config.description,
            trained_on=self.config.trained_on_desc,
            test_accuracy=0.0,
            feature_count=len(self.config.feature_names),
        )

    def train(self, X_internal: pd.DataFrame, y_internal: pd.Series) -> None:
        """Trains the model via CV, sets up SHAP, and serializes the agent's state."""
        print(f"[{self.name}] Training ensemble model on {len(X_internal)} records...")
        
        # Verify columns match the expected configuration
        missing_cols = set(self.config.feature_names) - set(X_internal.columns)
        if missing_cols:
            raise ValueError(f"Training data is missing configured features: {missing_cols}")

        # Train and set state
        self.wrapper.train_with_cv(
            X_internal[self.config.feature_names], 
            y_internal, 
            n_splits=self.config.n_splits
        )
        self.is_trained = True
        
        # Serialize to disk
        self.config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.wrapper, self.config.artifact_path)
        print(f"[{self.name}] Training complete. Artifact saved to {self.config.artifact_path}")

    def predict(self, inputs: BaseModel | dict) -> Prediction:
        if not self.is_trained:
            raise RuntimeError(f"[{self.name}] Agent not trained. Provide internal data to `train()`.")

        data = inputs if isinstance(inputs, dict) else inputs.model_dump()
        
        # Construct single-row pandas DataFrame maintaining exact column order
        row_dict = {
            name: data.get(self._sanitize(name)) if data.get(self._sanitize(name)) is not None else np.nan
            for name in self.config.feature_names
        }
        X_df = pd.DataFrame([row_dict])

        # Prediction logic
        p_class_1 = float(self.wrapper.predict_ensemble_proba(X_df)[0])
        p_class_0 = 1.0 - p_class_1
        
        class_0_name, class_1_name = self.config.target_classes
        predicted_class = class_1_name if p_class_1 >= 0.5 else class_0_name
        confidence = max(p_class_1, p_class_0)

        # Explainability
        shap_vals = self.wrapper.get_explanations(X_df)[0]
        
        all_scores = [
            XAIScore(
                feature_name=self.config.feature_names[i],
                contribution=float(shap_vals[i]),
                feature_value=None if pd.isna(X_df.iloc[0, i]) else float(X_df.iloc[0, i]),
            )
            for i in range(len(self.config.feature_names))
        ]
        all_scores.sort(key=lambda s: abs(s.contribution), reverse=True)

        return Prediction(
            model_name=self.name,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities={class_0_name: p_class_0, class_1_name: p_class_1},
            xai_scores=all_scores[:5], 
            metadata=self.metadata,
        )


def create_ml_agent(config: AgentConfig) -> DynamicMLAgent:
    """Factory function to build and return an ML Agent from a configuration."""
    return DynamicMLAgent(config)


# =====================================================================
# EXAMPLE USAGE: Recreating the Breast Cancer classifier dynamically
# =====================================================================
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    
    # 1. Prep Data
    dataset = load_breast_cancer()
    feature_names = list(dataset.feature_names)
    X_df = pd.DataFrame(dataset.data, columns=feature_names)
    y_series = pd.Series(dataset.target)

    # 2. Define Configuration
    cancer_config = AgentConfig(
        agent_name="breast_cancer_agent",
        description="Dynamic Breast Cancer Classifier",
        version="1.0.0",
        artifact_path=Path("./artifacts/dynamic_breast_cancer.joblib"),
        feature_names=feature_names,
        # In sklearn, 0=malignant, 1=benign
        target_classes=["malignant", "benign"], 
        n_splits=5
    )

    # 3. Instantiate via Factory
    agent = create_ml_agent(cancer_config)

    # 4. Train (if not already trained/loaded)
    if not agent.is_trained:
        agent.train(X_df, y_series)

    # 5. Predict using Pydantic Schema
    # Grab the first row to test
    sample_data = {
        agent._sanitize(k): float(v) 
        for k, v in zip(feature_names, dataset.data[0])
    }
    InputSchema = agent.input_schema
    mock_input = InputSchema(**sample_data)

    result = agent.predict(mock_input)
    
    print("\n--- PREDICTION RESULT ---")
    print(f"Predicted Class: {result.predicted_class} (Confidence: {result.confidence:.2f})")
    print("\nTop SHAP Drivers:")
    for score in result.xai_scores:
        print(f"  {score.feature_name}: {score.contribution:.4f} (Value: {score.feature_value:.2f})")


def ingest_csv_and_build_agent(
    csv_path: str | Path,
    target_column: str,
    agent_name: str,
    artifact_filename: str
) -> DynamicMLAgent:
    """CSV Ingestion and Agent Builder Script.

    Reads a pandas-compatible CSV, automatically configures the ML Agent, 
    trains the K-Fold ensemble, and saves the joblib artifact for the server.
    """
    
    csv_path = Path(csv_path)
    print(f"--- Starting Agent Build Process for: {agent_name} ---")
    
    # 1. Load the Dataset
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")
    
    print(f"Loading data from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    # [ACTION REQUIRED] 2: Data Preprocessing & Cleaning
    # XGBoost requires numeric features. If your CSV has string categories or 
    # datetime columns, you MUST encode or drop them here before passing to the factory.
    # Example: df = pd.get_dummies(df, drop_first=True)
    # df = df.dropna()  # Or impute missing values
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {df.columns.tolist()}")

    # 2. Split Features and Target
    y_series = df[target_column]
    X_df = df.drop(columns=[target_column])
    
    feature_names = X_df.columns.tolist()
    
    # [ACTION REQUIRED] 3: Target Class Mapping
    # If your target is already 0/1, you just need to name the classes.
    # If your target is string labels (e.g., "Healthy", "Sick"), map them to 0 and 1 here.
    # XGBoost binary classification strictly expects targets in {0, 1}.
    unique_classes = sorted(y_series.unique().tolist())
    if set(unique_classes) != {0, 1}:
        print(f"WARNING: Target classes are {unique_classes}. Ensure they are mapped to 0 and 1.")
        # Example: y_series = y_series.map({"Healthy": 0, "Sick": 1})
        # unique_classes = ["Healthy", "Sick"]
    
    # For the config, we map 0 -> target_classes[0], 1 -> target_classes[1]
    target_class_names = [f"Class_{cls}" for cls in unique_classes] 

    # 3. Define Output Paths
    # [ACTION REQUIRED] 4: Set your desired artifact output directory
    artifact_dir = Path("./artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / artifact_filename

    # 4. Generate the Configuration
    config = AgentConfig(
        agent_name=agent_name,
        description=f"Automated ensemble classifier built from {csv_path.name}.",
        version="1.0.0-auto",
        artifact_path=artifact_path,
        feature_names=feature_names,
        target_classes=target_class_names, # Update these to human-readable names if preferred
        trained_on_desc=f"Local CSV: {csv_path.name} (n={len(df)})",
        n_splits=5
        # [ACTION REQUIRED] 5: If you need custom XGBoost params (e.g., higher depth, scale_pos_weight for imbalance), 
        # add them here via `model_params={'objective': 'binary:logistic', ...}`
    )

    # 5. Instantiate and Train the Agent
    print(f"Configuring agent with {len(feature_names)} features.")
    agent = create_ml_agent(config)
    
    print("Initiating training pipeline...")
    agent.train(X_df, y_series)
    
    # 6. Verification Test
    print("\n--- Verifying Inference Pipeline ---")
    InputSchema = agent.input_schema
    
    # Extract the first row of the training data as a dictionary to test Pydantic validation
    sample_data = {
        agent._sanitize(k): float(v) 
        for k, v in zip(feature_names, X_df.iloc[0].values)
    }
    agent._sample_inputs = sample_data
    
    # Generate the Python wrapper class!
    generate_agent_class_file(
        output_py_path=f".{agent_name}_agent.py", # Where your server looks for models
        class_name="CustomClinicalAgent",                 # Name of the Python class
        agent_name=agent_name,
        artifact_filename=artifact_filename,
        feature_names=feature_names,
        target_classes=target_class_names,
        sample_data_dict=sample_data
    )

    return agent

def generate_agent_class_file(
    output_py_path: str | Path,
    class_name: str,
    agent_name: str,
    artifact_filename: str,
    feature_names: list[str],
    target_classes: list[str],
    sample_data_dict: dict[str, float]
):
    """Generates the drop-in Python wrapper class for the MCP server."""
    
    # Format the sample dictionary nicely for the Python file
    formatted_sample = json.dumps(sample_data_dict, indent=8).replace('}', '        }')
    
    config_var_name = f"_{class_name.upper()}_CONFIG"
    
    python_code = f'''"""Autogenerated ML Agent wrapper for {class_name}.

This file was generated by the CSV Builder Script and is ready 
to be discovered by the ML-Agent MCP server.
"""

from pathlib import Path
from typing import Any

# Ensure this import matches your server's module structure
from ._agent_factory import AgentConfig, DynamicMLAgent

# --- 1. Paths & Configuration ---
_ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "{artifact_filename}"

{config_var_name} = AgentConfig(
    agent_name="{agent_name}",
    description="Automated ensemble classifier generated from local data.",
    version="1.0.0-auto",
    artifact_path=_ARTIFACT_PATH,
    feature_names={feature_names},
    target_classes={target_classes},
    trained_on_desc="Local Data via Automated Builder",
    n_splits=5
)

# --- 2. MCP Server Agent Class ---
class {class_name}(DynamicMLAgent):
    """Dynamic binary classifier for {agent_name}."""

    def __init__(self) -> None:
        # Initialize the generalized factory agent with our specific config
        super().__init__({config_var_name})

    def sample_inputs(self) -> dict[str, Any]:
        """A representative sample case extracted during the build process."""
        return {formatted_sample}
'''

    # Write the generated code to the target .py file
    output_py_path = Path(output_py_path)
    output_py_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_py_path, "w") as f:
        f.write(python_code)
        
    print(f"[Code Generator] Successfully wrote server class to: {output_py_path}")

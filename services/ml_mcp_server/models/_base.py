"""Base class for niche clinical ML models exposed via MCP.

A subclass implements:
- `name`            — MCP tool name
- `metadata`        — what the orchestrator sees about the model
- `input_schema`    — Pydantic class describing required input features
- `predict(inputs)` — returns a `Prediction` including SHAP-style XAI scores
- `sample_inputs()` — one realistic input dict for tests / UI defaults / eval

The MCP server's registry imports every module in this folder, finds every
MLModel subclass, instantiates it, and exposes it as a tool. Adding a new
clinical predictor is therefore a single-file change.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from packages.schemas.prediction import ModelMetadata, Prediction


class MLModel(ABC):
    """Contract for a registered ML model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The MCP tool name (e.g., 'predict_breast_cancer_malignancy')."""

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Self-description: dataset, accuracy, what this model knows."""

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Pydantic class for input validation. Becomes the MCP tool's argument schema."""

    @abstractmethod
    def predict(self, inputs: BaseModel) -> Prediction:
        """Run the model and return a Prediction with XAI scores."""

    @abstractmethod
    def sample_inputs(self) -> dict[str, Any]:
        """Return one realistic input dict that conforms to `input_schema`.

        Used by the smoke test, UI default-fill, and eval scenarios. Must
        return a dict whose keys match the input schema's field names.
        """

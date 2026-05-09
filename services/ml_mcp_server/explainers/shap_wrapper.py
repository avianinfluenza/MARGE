"""Shared SHAP utility for tree-based MLModel implementations.

Wraps `shap.TreeExplainer` for XGBoost / CATBoost / RandomForest. Models that
need a different explainer (e.g., kernel SHAP for non-tree models) can call
shap directly.
"""

from typing import Any

import numpy as np
import shap


def compute_shap_values(model: Any, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a tree model.

    Returns an array shaped (n_samples, n_features) — for binary classifiers
    these are values for the positive class (so positive contribution pushes
    the prediction toward class 1).

    Handles three layouts returned by `shap.TreeExplainer.shap_values`:
    - XGBoost binary:    np.ndarray of shape (n_samples, n_features)
    - sklearn classifiers: list per class — pick the positive-class array
    - CatBoost binary:   np.ndarray of shape (n_samples, n_features + 1)
                         where the last column is the bias / expected value
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    arr = np.asarray(values)
    n_features = X.shape[1]
    if arr.shape[-1] == n_features + 1:
        # Trim CatBoost's trailing bias column.
        arr = arr[..., :n_features]
    return arr

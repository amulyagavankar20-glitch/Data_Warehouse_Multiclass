"""Hybrid supervised classifier with meta-features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModelWrapper


class HybridClassifier(BaseModelWrapper):
    """RandomForest classifier augmented with anomaly meta-features.

    Features: original biomarkers + iso_score + ae_mse (meta-features from
    unsupervised anomaly detection models).

    Args:
        n_estimators: Number of trees in the forest.
        class_weight: Class weighting strategy.
        random_state: Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> None:
        params = {
            "n_estimators": n_estimators,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        super().__init__(name="hybrid_classifier", params=params)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> HybridClassifier:
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.model = RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            class_weight=self.params["class_weight"],
            random_state=self.params["random_state"],
            n_jobs=-1,
        )
        self.model.fit(X_array, y)
        self._is_fitted = True
        self._feature_names: list[str] = (
            list(X.columns) if isinstance(X, pd.DataFrame) else []
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_array)

    def get_feature_importance(self) -> pd.DataFrame | None:
        """Return feature importances as a DataFrame."""
        if not self._is_fitted or not self._feature_names:
            return None
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

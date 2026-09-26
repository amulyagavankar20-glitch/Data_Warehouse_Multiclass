"""Isolation Forest anomaly detection model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.base import BaseModelWrapper


class IsolationForestModel(BaseModelWrapper):
    """Isolation Forest for unsupervised anomaly detection.

    Trained on a healthy (normal) cohort to learn the normal
    physiological distribution. Returns anomaly scores where
    lower scores indicate higher anomaly.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        params = {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
        }
        super().__init__(name="isolation_forest", params=params)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> IsolationForestModel:
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        self.model = IsolationForest(
            n_estimators=self.params["n_estimators"],
            contamination=self.params["contamination"],
            random_state=self.params["random_state"],
            n_jobs=-1,
        )
        self.model.fit(X_values)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_values)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return anomaly decision scores. Lower = more anomalous."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.decision_function(X_values)

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return anomaly scores for samples."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.score_samples(X_values)

    def get_anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Get iso_score: lower means more anomalous."""
        return self.decision_function(X)

"""Isolation Forest anomaly detection model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.base import BaseModelWrapper


class IsolationForestModel(BaseModelWrapper):
    """Isolation Forest fitted on the healthy (``N``) training cohort.

    Lower decision scores indicate larger deviation from a healthy profile.
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

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None
    ) -> IsolationForestModel:
        self.model = IsolationForest(**self.params, n_jobs=-1)
        self.model.fit(self._values(X))
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ``-1`` for anomalies and ``1`` for inliers."""
        self._check_fitted()
        return np.asarray(self.model.predict(self._values(X)))

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.model.decision_function(self._values(X)))

    def get_anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.decision_function(X)

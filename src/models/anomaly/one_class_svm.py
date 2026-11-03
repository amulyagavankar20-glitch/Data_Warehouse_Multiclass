"""One-Class SVM anomaly detection model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from src.models.base import BaseModelWrapper


class OneClassSVMModel(BaseModelWrapper):
    """RBF One-Class SVM fitted on the healthy (``N``) training cohort.

    Kernel SVM training is quadratic in sample count, so the cohort is randomly
    subsampled to ``max_train_samples`` rows.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        nu: float = 0.05,
        max_train_samples: int = 10_000,
        random_state: int = 42,
    ) -> None:
        params = {"kernel": kernel, "gamma": gamma, "nu": nu}
        super().__init__(name="one_class_svm", params=params)
        self.max_train_samples = max_train_samples
        self.random_state = random_state

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None
    ) -> OneClassSVMModel:
        values = self._values(X)
        if len(values) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(values), self.max_train_samples, replace=False)
            values = values[idx]
        self.model = OneClassSVM(**self.params)
        self.model.fit(values)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ``-1`` for anomalies and ``1`` for inliers."""
        self._check_fitted()
        return np.asarray(self.model.predict(self._values(X)))

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Signed distance to the boundary; negative means anomalous."""
        self._check_fitted()
        return np.asarray(self.model.decision_function(self._values(X)))

    def get_anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.decision_function(X)

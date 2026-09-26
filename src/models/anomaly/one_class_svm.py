"""One-Class SVM anomaly detection model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from src.models.base import BaseModelWrapper


class OneClassSVMModel(BaseModelWrapper):
    """One-Class SVM for unsupervised anomaly detection.

    Trained on a healthy (normal) cohort to learn the normal
    physiological distribution boundary.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        nu: float = 0.05,
    ) -> None:
        params = {"kernel": kernel, "gamma": gamma, "nu": nu}
        super().__init__(name="one_class_svm", params=params)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> OneClassSVMModel:
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        self.model = OneClassSVM(
            kernel=self.params["kernel"],
            gamma=self.params["gamma"],
            nu=self.params["nu"],
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
        """Return signed anomaly scores. Negative = anomaly."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.decision_function(X_values)

    def get_anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Get signed anomaly scores. Negative = anomaly."""
        return self.decision_function(X)

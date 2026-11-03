"""Abstract base class for model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers."""

    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params = params or {}
        self.model: Any = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> BaseModelWrapper:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_params(self) -> dict[str, Any]:
        return self.params

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet")

    @staticmethod
    def _values(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

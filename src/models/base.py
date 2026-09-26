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
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return prediction probabilities if supported."""
        return None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_params(self) -> dict[str, Any]:
        return self.params

    def set_params(self, **params: Any) -> BaseModelWrapper:
        self.params.update(params)
        return self

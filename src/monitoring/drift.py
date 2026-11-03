"""Population Stability Index (PSI) drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.features.engineering import add_clinical_interactions

N_BINS = 10
EPS = 1e-6


def monitored_columns() -> list[str]:
    return [*settings.BASE_NUMERIC_COLUMNS, *settings.ENGINEERED_COLUMNS]


def build_reference(X_raw: pd.DataFrame, n_bins: int = N_BINS) -> dict[str, Any]:
    """Capture training-fold decile edges and bin proportions per feature."""
    X = add_clinical_interactions(X_raw)
    reference: dict[str, Any] = {}
    for col in monitored_columns():
        values = X[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            continue
        reference[col] = {
            "edges": edges.tolist(),
            "proportions": _bin_proportions(values, edges).tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return reference


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    inner = edges[1:-1]
    idx = np.searchsorted(inner, values, side="right")
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    return np.asarray(counts / max(counts.sum(), 1.0))


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass
class DriftReport:
    psi_by_feature: dict[str, float]
    threshold: float

    @property
    def drifted_features(self) -> list[str]:
        return sorted(k for k, v in self.psi_by_feature.items() if v >= self.threshold)

    @property
    def max_psi(self) -> float:
        return max(self.psi_by_feature.values(), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "max_psi": self.max_psi,
            "drifted_features": self.drifted_features,
            "psi": self.psi_by_feature,
        }


def compute_drift(
    reference: dict[str, Any],
    X_raw: pd.DataFrame,
    threshold: float | None = None,
) -> DriftReport:
    """PSI of a new raw batch against the training reference.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, >= 0.25 major shift.
    """
    X = add_clinical_interactions(X_raw)
    scores: dict[str, float] = {}
    for col, ref in reference.items():
        if col not in X.columns:
            continue
        edges = np.asarray(ref["edges"])
        actual = _bin_proportions(X[col].to_numpy(dtype=float), edges)
        scores[col] = psi(np.asarray(ref["proportions"]), actual)
    return DriftReport(
        psi_by_feature=scores,
        threshold=settings.psi_alert_threshold if threshold is None else threshold,
    )

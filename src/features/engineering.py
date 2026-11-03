"""Leakage-safe clinical feature pipeline shared by training and serving."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.config import settings

IMPUTATION_ATOL = 1e-5


def add_clinical_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise clinical ratios (stateless, safe to compute before splitting)."""
    out = X.copy()
    out["pulse_pressure"] = out["systolic_bp"] - out["diastolic_bp"]
    out["mean_arterial_pressure"] = out["diastolic_bp"] + out["pulse_pressure"] / 3.0
    out["cholesterol_ratio"] = out["ldl"] / (out["hdl"] + 1e-5)
    out["glucose_triglyceride_index"] = np.log(
        out["glucose"] * out["triglycerides"] + 1.0
    )
    return out


class ClinicalFeaturePipeline(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Raw clinical record -> 50-column model matrix.

    Stateful pieces (group-imputation modes and the scaler) are fitted on the
    training fold only. Categorical levels come from the frozen contract in
    ``settings`` rather than from the data, so a single-row request produces
    exactly the same columns as a training batch.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> ClinicalFeaturePipeline:
        base = add_clinical_interactions(X)
        self.imputation_modes_: dict[str, float] = {
            col: float(base[col].mode().iloc[0])
            for col in settings.IMPUTATION_CHECK_COLUMNS
        }
        unscaled = self._encode(base)
        continuous = {
            *settings.ORDINAL_MAPS,
            *settings.BASE_NUMERIC_COLUMNS,
            *settings.ENGINEERED_COLUMNS,
        }
        self.scaled_columns_: list[str] = [
            c for c in unscaled.columns if c in continuous
        ]
        self.scaler_ = StandardScaler().fit(unscaled[self.scaled_columns_])
        self.feature_names_out_: list[str] = list(unscaled.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("ClinicalFeaturePipeline is not fitted")
        out = self._encode(add_clinical_interactions(X))
        out[self.scaled_columns_] = self.scaler_.transform(out[self.scaled_columns_])
        return out[self.feature_names_out_].astype(np.float32)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def _encode(self, base: pd.DataFrame) -> pd.DataFrame:
        nominal = settings.NOMINAL_LEVELS
        passthrough = [c for c in settings.RAW_FEATURE_ORDER if c not in nominal]

        out = base[passthrough + settings.ENGINEERED_COLUMNS].copy()
        for col, mapping in settings.ORDINAL_MAPS.items():
            mapped = out[col].map(mapping)
            if mapped.isna().any():
                bad = sorted(out.loc[mapped.isna(), col].astype(str).unique())
                raise ValueError(f"Unknown levels for '{col}': {bad}")
            out[col] = mapped.astype(np.int64)

        for col, mode in self.imputation_modes_.items():
            out[f"{col}_is_imputed"] = np.isclose(
                base[col].to_numpy(dtype=float), mode, atol=IMPUTATION_ATOL
            ).astype(np.int8)

        for col, levels in nominal.items():
            values = base[col].astype(str)
            unknown = set(values.unique()) - set(levels)
            if unknown:
                raise ValueError(f"Unknown levels for '{col}': {sorted(unknown)}")
            for level in levels[1:]:
                out[f"{col}_{level}"] = (values == level).astype(np.int8).to_numpy()
        return out

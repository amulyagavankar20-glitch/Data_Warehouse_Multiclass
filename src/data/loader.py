"""Data loading, cleaning and target extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings


def load_raw_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load the raw warehouse CSV (defaults to ``settings.raw_data_path``)."""
    path = Path(filepath) if filepath is not None else settings.raw_data_path
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate records so they cannot straddle train/test folds.

    Target and leakage columns are kept here; they are separated from the
    predictors by :func:`split_features_targets`.
    """
    return df.drop_duplicates().reset_index(drop=True)


def parse_disease_flags(flags: pd.Series) -> np.ndarray:
    """Parse ``"DI,HT,HY"`` comma-delimited flags into an ``(n, 3)`` int array.

    Column order follows ``settings.DISEASE_FLAG_NAMES``:
    (Diabetes, Heart_Disease, Hypertension).
    """
    parsed = flags.astype(str).str.split(",", expand=True).astype(int)
    return parsed.to_numpy()


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the raw predictor frame from the 8-class sublabel target.

    Every leakage / target-clone column is quarantined; only the columns in
    ``settings.RAW_FEATURE_ORDER`` survive.
    """
    missing = [c for c in settings.RAW_FEATURE_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"Missing raw feature columns: {missing}")
    X = df[settings.RAW_FEATURE_ORDER].copy()
    y = df[settings.TARGET_COLUMN].astype(str).reset_index(drop=True)
    return X.reset_index(drop=True), y

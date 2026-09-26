"""Data splitting and leakage-safe preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import settings


@dataclass
class SplitData:
    """Container for train/test splits and preprocessing artifacts."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    feature_names: list[str]
    continuous_cols: list[str]
    ordinal_cols: list[str]
    nominal_cols: list[str]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Perform stratified train/test split.

    Drops leakage columns and the target, separates features from labels,
    and performs a stratified split to preserve class proportions.

    Args:
        df: Cleaned DataFrame.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        SplitData with train/test splits and encoders.
    """
    target = settings.TARGET_COLUMN

    leakage_cols = [c for c in settings.DROP_COLUMNS if c in df.columns]
    feature_cols = [
        c for c in df.columns
        if c not in leakage_cols and c not in [target, "label"]
    ]

    X = df[feature_cols].copy()
    y_raw = df[target].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_normal_mask = y_train != label_encoder.transform(["N"])[0]
    normal_label_idx = list(label_encoder.classes_).index("N") if "N" in label_encoder.classes_ else 0
    train_normal_mask = y_train != normal_label_idx

    continuous_cols = [c for c in settings.CONTINUOUS_COLUMNS if c in X_train.columns]
    ordinal_cols = [c for c in settings.ORDINAL_COLUMNS if c in X_train.columns]
    nominal_cols = [c for c in settings.NOMINAL_COLUMNS if c in X_train.columns]

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        feature_names=feature_cols,
        continuous_cols=continuous_cols,
        ordinal_cols=ordinal_cols,
        nominal_cols=nominal_cols,
    )


def get_normal_cohort(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Extract the normal (healthy) cohort from training data.

    Used for unsupervised anomaly detection training.

    Args:
        X_train: Training features.
        y_train: Training labels (encoded).
        label_encoder: Fitted label encoder.

    Returns:
        DataFrame containing only Normal cohort features.
    """
    normal_label_idx = list(label_encoder.classes_).index("N")
    normal_mask = y_train == normal_label_idx
    return X_train[normal_mask].copy()


def get_preprocessor_config(split: SplitData) -> dict[str, Any]:
    """Return preprocessing configuration for logging."""
    return {
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "n_features": len(split.feature_names),
        "continuous_cols": split.continuous_cols,
        "ordinal_cols": split.ordinal_cols,
        "nominal_cols": split.nominal_cols,
        "n_classes": len(split.label_encoder.classes_),
        "classes": list(split.label_encoder.classes_),
    }

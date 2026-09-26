"""Feature engineering pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.data.splitter import SplitData


def build_preprocessor(split: SplitData) -> ColumnTransformer:
    """Build a ColumnTransformer for feature encoding.

    - Ordinal encoding for ordinal categorical columns.
    - One-hot encoding for nominal categorical columns.
    - StandardScaler for continuous columns (fit on train only).

    Args:
        split: SplitData containing column lists.

    Returns:
        Configured ColumnTransformer.
    """
    transformers: list[tuple[str, Any, list[str]]] = []

    if split.ordinal_cols:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                split.ordinal_cols,
            )
        )

    if split.nominal_cols:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
                split.nominal_cols,
            )
        )

    if split.continuous_cols:
        transformers.append(
            ("continuous", StandardScaler(), split.continuous_cols)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_preprocessor(split: SplitData) -> ColumnTransformer:
    """Fit the preprocessing pipeline on training data only.

    Args:
        split: SplitData with training features.

    Returns:
        Fitted ColumnTransformer.
    """
    preprocessor = build_preprocessor(split)
    preprocessor.fit(split.X_train)
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Transform features using a fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Features to transform.
        feature_names: Optional expected feature names for alignment.

    Returns:
        Transformed DataFrame.
    """
    if feature_names is not None:
        X = X[feature_names].copy()

    transformed = preprocessor.transform(X)

    output_names = preprocessor.get_feature_names_out()
    cleaned_names = [name.split("__", 1)[-1] for name in output_names]

    return pd.DataFrame(
        transformed,
        columns=cleaned_names,
        index=X.index,
    )


def fit_transform_train(split: SplitData) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    """Fit preprocessor on train and transform both train and test.

    Args:
        split: SplitData object.

    Returns:
        Tuple of (fitted preprocessor, X_train_encoded, X_test_encoded).
    """
    preprocessor = fit_preprocessor(split)

    X_train_encoded = transform_features(preprocessor, split.X_train, split.feature_names)
    X_test_encoded = transform_features(preprocessor, split.X_test, split.feature_names)

    return preprocessor, X_train_encoded, X_test_encoded


def get_feature_columns_after_encoding(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """Get feature names after encoding."""
    names = preprocessor.get_feature_names_out()
    return [n.split("__", 1)[-1] for n in names]

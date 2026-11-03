"""Stratified train / validation / test partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import settings


@dataclass
class SplitData:
    """Raw (un-encoded) feature frames and encoded targets for each fold."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.label_encoder.classes_]


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> SplitData:
    """Split into train/val/test, stratified on the 8-class sublabel.

    Splitting happens before any statistic is fitted, so every downstream
    transformer sees only training rows.
    """
    val_size = settings.val_size if val_size is None else val_size
    test_size = settings.test_size if test_size is None else test_size
    random_state = settings.random_state if random_state is None else random_state

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    holdout = val_size + test_size
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y_enc, test_size=holdout, random_state=random_state, stratify=y_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold,
        y_hold,
        test_size=test_size / holdout,
        random_state=random_state,
        stratify=y_hold,
    )

    return SplitData(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        y_test=np.asarray(y_test),
        label_encoder=label_encoder,
    )


def normal_class_index(label_encoder: LabelEncoder) -> int:
    classes = list(label_encoder.classes_)
    if settings.NORMAL_CLASS not in classes:
        raise ValueError(f"Normal class '{settings.NORMAL_CLASS}' not in {classes}")
    return classes.index(settings.NORMAL_CLASS)

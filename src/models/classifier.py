"""Cost-sensitive LightGBM champion with per-class decision thresholds."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

from src.config import settings
from src.models.base import BaseModelWrapper


class LightGBMClassifier(BaseModelWrapper):
    """Multi-class GBDT trained with balanced sample weights.

    Class imbalance (~125:1) is handled by cost-sensitive weighting rather than
    synthetic oversampling, so no non-physiological biomarker vectors are
    fabricated. ``calibrate_thresholds`` then rescales per-class probabilities
    to maximise per-class F1 on the validation fold.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        super().__init__(
            name="lightgbm_classifier", params=params or settings.lgb_params
        )
        self.num_boost_round = num_boost_round or settings.lgb_num_boost_round
        self.early_stopping_rounds = (
            early_stopping_rounds or settings.lgb_early_stopping_rounds
        )
        self.thresholds: np.ndarray | None = None
        self.threshold_val_f1: dict[str, float] = {}
        self.class_weights: dict[int, float] = {}
        self.feature_names: list[str] = []

    @property
    def best_iteration(self) -> int:
        self._check_fitted()
        return int(self.model.best_iteration or self.model.current_iteration())

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> LightGBMClassifier:
        if y is None:
            raise ValueError("LightGBMClassifier requires labels")
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        self.class_weights = {
            int(c): float(w) for c, w in zip(classes, weights, strict=True)
        }
        sample_weight = np.array([self.class_weights[int(label)] for label in y])

        params = {**self.params, "num_class": int(classes.max()) + 1}
        train_set = lgb.Dataset(X, label=y, weight=sample_weight)
        valid_sets = [train_set]
        callbacks: list[Any] = []
        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=train_set))
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            )

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self.feature_names = list(X.columns)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        proba = self.model.predict(
            X[self.feature_names], num_iteration=self.best_iteration
        )
        return np.asarray(proba)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.decide(self.predict_proba(X))

    def decide(self, proba: np.ndarray) -> np.ndarray:
        """Arg-max over threshold-normalised probabilities."""
        if self.thresholds is None:
            return np.asarray(np.argmax(proba, axis=1))
        return np.asarray(np.argmax(proba / (self.thresholds + 1e-8), axis=1))

    def calibrate_thresholds(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> np.ndarray | None:
        """Pick, per class, the one-vs-rest cut-off that maximises F1 on val.

        Per-class optimal cut-offs do not always compose into a better
        multi-class decision, so the thresholds are kept only if they raise
        validation macro-F1 over plain arg-max; otherwise they are discarded.
        """
        clip_min = settings.threshold_clip_min if clip_min is None else clip_min
        clip_max = settings.threshold_clip_max if clip_max is None else clip_max
        self.thresholds = None
        proba = self.predict_proba(X_val)
        thresholds = np.full(proba.shape[1], 0.5)
        for k in range(proba.shape[1]):
            y_bin = (y_val == k).astype(int)
            if y_bin.sum() == 0:
                continue
            precision, recall, cuts = precision_recall_curve(y_bin, proba[:, k])
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            best = int(np.argmax(f1))
            if best < len(cuts):
                thresholds[k] = cuts[best]
        candidate = np.clip(thresholds, clip_min, clip_max)

        argmax_f1 = f1_score(y_val, np.argmax(proba, axis=1), average="macro")
        calibrated_f1 = f1_score(
            y_val, np.argmax(proba / (candidate + 1e-8), axis=1), average="macro"
        )
        self.threshold_val_f1 = {
            "argmax": float(argmax_f1),
            "calibrated": float(calibrated_f1),
        }
        if calibrated_f1 > argmax_f1:
            self.thresholds = candidate
        return self.thresholds

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False, ignore_index=True)

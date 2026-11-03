"""Tests for the classifier and anomaly detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import split_features_targets
from src.data.splitter import SplitData, split_data
from src.features.engineering import ClinicalFeaturePipeline
from src.models.anomaly import IsolationForestModel, OneClassSVMModel
from src.models.classifier import LightGBMClassifier


@pytest.fixture
def encoded(raw_df: pd.DataFrame) -> tuple[SplitData, pd.DataFrame, pd.DataFrame]:
    split = split_data(*split_features_targets(raw_df))
    pipe = ClinicalFeaturePipeline().fit(split.X_train)
    return split, pipe.transform(split.X_train), pipe.transform(split.X_val)


@pytest.fixture
def blobs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))


class TestLightGBMClassifier:
    def test_fit_predict_proba(self, encoded: tuple) -> None:
        split, X_train, X_val = encoded
        clf = LightGBMClassifier(num_boost_round=30).fit(
            X_train, split.y_train, X_val=X_val, y_val=split.y_val
        )
        proba = clf.predict_proba(X_val)
        assert proba.shape == (len(X_val), len(split.classes))
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
        assert clf.predict(X_val).shape == (len(X_val),)
        assert 1 <= clf.best_iteration <= 30

    def test_learns_signal(self, encoded: tuple) -> None:
        split, X_train, X_val = encoded
        clf = LightGBMClassifier(num_boost_round=60).fit(X_train, split.y_train)
        assert (clf.predict(X_val) == split.y_val).mean() > 0.5

    def test_balanced_class_weights(self, encoded: tuple) -> None:
        split, X_train, _ = encoded
        clf = LightGBMClassifier(num_boost_round=5).fit(X_train, split.y_train)
        counts = np.bincount(split.y_train)
        rare, common = int(np.argmin(counts)), int(np.argmax(counts))
        assert clf.class_weights[rare] > clf.class_weights[common]

    def test_threshold_guard(self, encoded: tuple) -> None:
        split, X_train, X_val = encoded
        clf = LightGBMClassifier(num_boost_round=30).fit(X_train, split.y_train)
        thresholds = clf.calibrate_thresholds(X_val, split.y_val)
        scores = clf.threshold_val_f1
        assert set(scores) == {"argmax", "calibrated"}
        if thresholds is None:
            assert scores["calibrated"] <= scores["argmax"]
        else:
            assert scores["calibrated"] > scores["argmax"]
            assert np.all((thresholds >= 0.05) & (thresholds <= 0.60))

    def test_decide_with_thresholds(self) -> None:
        clf = LightGBMClassifier()
        proba = np.array([[0.6, 0.4]])
        assert clf.decide(proba)[0] == 0
        clf.thresholds = np.array([0.9, 0.1])
        assert clf.decide(proba)[0] == 1

    def test_feature_importance(self, encoded: tuple) -> None:
        split, X_train, _ = encoded
        clf = LightGBMClassifier(num_boost_round=10).fit(X_train, split.y_train)
        fi = clf.get_feature_importance()
        assert list(fi.columns) == ["feature", "importance"]
        assert len(fi) == X_train.shape[1]

    def test_requires_labels(self, blobs: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            LightGBMClassifier().fit(blobs)

    def test_predict_before_fit_raises(self, blobs: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            LightGBMClassifier().predict_proba(blobs)


class TestIsolationForest:
    def test_fit_and_scores(self, blobs: pd.DataFrame) -> None:
        model = IsolationForestModel(n_estimators=50).fit(blobs)
        assert set(np.unique(model.predict(blobs))) <= {-1, 1}
        assert model.get_anomaly_score(blobs).shape == (300,)

    def test_outlier_scores_lower(self, blobs: pd.DataFrame) -> None:
        model = IsolationForestModel(n_estimators=100).fit(blobs)
        outlier = pd.DataFrame([[8.0, 8.0, 8.0, 8.0]], columns=blobs.columns)
        assert model.decision_function(outlier)[0] < np.median(
            model.decision_function(blobs)
        )

    def test_params(self) -> None:
        assert IsolationForestModel(n_estimators=7).get_params()["n_estimators"] == 7

    def test_not_fitted(self, blobs: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            IsolationForestModel().predict(blobs)


class TestOneClassSVM:
    def test_fit_and_predict(self, blobs: pd.DataFrame) -> None:
        model = OneClassSVMModel().fit(blobs)
        assert model.is_fitted
        assert set(np.unique(model.predict(blobs))) <= {-1, 1}
        assert model.get_anomaly_score(blobs).shape == (300,)

    def test_subsamples_large_cohort(self, blobs: pd.DataFrame) -> None:
        model = OneClassSVMModel(max_train_samples=50).fit(blobs)
        assert model.model.fit_status_ == 0
        assert len(model.model.support_) <= 50

    def test_not_fitted(self, blobs: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            OneClassSVMModel().decision_function(blobs)

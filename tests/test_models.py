"""Tests for model definitions."""

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly.autoencoder import DeepAutoencoderModel
from src.models.anomaly.isolation_forest import IsolationForestModel
from src.models.anomaly.one_class_svm import OneClassSVMModel
from src.models.classifier import HybridClassifier


class TestIsolationForest:
    """Tests for IsolationForestModel."""

    def test_fit_and_predict(self, small_features_df: pd.DataFrame) -> None:
        model = IsolationForestModel(n_estimators=10, contamination=0.1)
        model.fit(small_features_df)
        assert model.is_fitted
        anomaly_labels = model.predict(small_features_df)
        assert anomaly_labels.shape[0] == len(small_features_df)

    def test_decision_function(self, small_features_df: pd.DataFrame) -> None:
        model = IsolationForestModel(n_estimators=10, contamination=0.1)
        model.fit(small_features_df)
        scores = model.decision_function(small_features_df)
        assert scores.shape[0] == len(small_features_df)

    def test_get_anomaly_score(self, small_features_df: pd.DataFrame) -> None:
        model = IsolationForestModel(n_estimators=10, contamination=0.1)
        model.fit(small_features_df)
        scores = model.get_anomaly_score(small_features_df)
        assert scores.shape[0] == len(small_features_df)

    def test_predict_before_fit_raises(self, small_features_df: pd.DataFrame) -> None:
        model = IsolationForestModel(n_estimators=10)
        with pytest.raises(RuntimeError):
            model.predict(small_features_df)

    def test_params_stored(self) -> None:
        model = IsolationForestModel(n_estimators=50, contamination=0.15)
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["contamination"] == 0.15


class TestOneClassSVM:
    """Tests for OneClassSVMModel."""

    def test_fit_and_predict(self, small_features_df: pd.DataFrame) -> None:
        model = OneClassSVMModel(kernel="rbf", gamma="scale", nu=0.05)
        model.fit(small_features_df)
        assert model.is_fitted
        anomaly_labels = model.predict(small_features_df)
        assert anomaly_labels.shape[0] == len(small_features_df)

    def test_decision_function(self, small_features_df: pd.DataFrame) -> None:
        model = OneClassSVMModel()
        model.fit(small_features_df)
        scores = model.decision_function(small_features_df)
        assert scores.shape[0] == len(small_features_df)

    def test_predict_before_fit_raises(self, small_features_df: pd.DataFrame) -> None:
        model = OneClassSVMModel()
        with pytest.raises(RuntimeError):
            model.predict(small_features_df)


class TestDeepAutoencoder:
    """Tests for DeepAutoencoderModel."""

    def test_fit_returns_self(self, small_features_df: pd.DataFrame) -> None:
        model = DeepAutoencoderModel(
            input_dim=4, encoding_dim=8, max_epochs=5, batch_size=16,
        )
        result = model.fit(small_features_df)
        assert result is model
        assert model.is_fitted

    def test_predict_shape(self, small_features_df: pd.DataFrame) -> None:
        model = DeepAutoencoderModel(
            input_dim=4, encoding_dim=8, max_epochs=5, batch_size=16,
        )
        model.fit(small_features_df)
        reconstructed = model.predict(small_features_df)
        assert reconstructed.shape == small_features_df.values.shape

    def test_get_anomaly_score(self, small_features_df: pd.DataFrame) -> None:
        model = DeepAutoencoderModel(
            input_dim=4, encoding_dim=8, max_epochs=5, batch_size=16,
        )
        model.fit(small_features_df)
        scores = model.get_anomaly_score(small_features_df)
        assert scores.shape[0] == len(small_features_df)
        assert (scores >= 0).all()  # MSE is non-negative

    def test_predict_before_fit_raises(self, small_features_df: pd.DataFrame) -> None:
        model = DeepAutoencoderModel(input_dim=4)
        with pytest.raises(RuntimeError):
            model.predict(small_features_df)

    def test_autoencoder_detects_anomalies(
        self, small_features_df: pd.DataFrame
    ) -> None:
        model = DeepAutoencoderModel(
            input_dim=4, encoding_dim=8, max_epochs=10, batch_size=16,
        )
        model.fit(small_features_df)

        normal_scores = model.get_anomaly_score(small_features_df)
        anomaly_input = small_features_df.copy()
        anomaly_input.iloc[0] = [10.0, 10.0, 10.0, 10.0]  # extreme outlier
        anomaly_scores = model.get_anomaly_score(anomaly_input)

        assert anomaly_scores[0] > normal_scores[0]


class TestHybridClassifier:
    """Tests for HybridClassifier."""

    def test_fit_and_predict(self, small_features_df: pd.DataFrame) -> None:
        np.random.seed(42)
        y = np.random.choice([0, 1, 2], len(small_features_df))
        model = HybridClassifier(n_estimators=10)
        model.fit(small_features_df, y)
        assert model.is_fitted
        preds = model.predict(small_features_df)
        assert preds.shape[0] == len(small_features_df)

    def test_predict_proba(self, small_features_df: pd.DataFrame) -> None:
        np.random.seed(42)
        y = np.random.choice([0, 1, 2], len(small_features_df))
        model = HybridClassifier(n_estimators=10)
        model.fit(small_features_df, y)
        proba = model.predict_proba(small_features_df)
        assert proba.shape[0] == len(small_features_df)
        assert proba.shape[1] == 3  # 3 classes

    def test_feature_importance(self, small_features_df: pd.DataFrame) -> None:
        np.random.seed(42)
        y = np.random.choice([0, 1, 2], len(small_features_df))
        model = HybridClassifier(n_estimators=10)
        model.fit(small_features_df, y)
        fi = model.get_feature_importance()
        assert fi is not None
        assert len(fi) == 4  # 4 features

    def test_predict_before_fit_raises(self, small_features_df: pd.DataFrame) -> None:
        model = HybridClassifier(n_estimators=10)
        with pytest.raises(RuntimeError):
            model.predict(small_features_df)

    def test_params_stored(self) -> None:
        model = HybridClassifier(n_estimators=50)
        params = model.get_params()
        assert params["n_estimators"] == 50

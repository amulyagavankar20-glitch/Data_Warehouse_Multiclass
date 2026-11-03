"""Tests for evaluation metrics and plots."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.models.evaluator import (
    disease_marginals,
    evaluate_classifier,
    evaluate_disease_heads,
    plot_confusion_matrix,
    plot_feature_importance,
)

CLASSES = ["DI", "DI_HY", "HY", "N"]


@pytest.fixture
def outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 400)
    logits = rng.normal(size=(400, 4))
    logits[np.arange(400), y_true] += 2.0
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y_true, proba.argmax(axis=1), proba


class TestEvaluateClassifier:
    def test_metric_ranges(self, outputs: tuple) -> None:
        r = evaluate_classifier(*outputs, class_labels=CLASSES)
        for value in (r.accuracy, r.f1_macro, r.f1_weighted, r.precision_macro):
            assert 0 <= value <= 1
        assert r.roc_auc is not None and r.roc_auc > 0.7
        assert r.confusion_matrix.shape == (4, 4)

    def test_to_dict_uses_class_names(self, outputs: tuple) -> None:
        d = evaluate_classifier(*outputs, class_labels=CLASSES).to_dict(prefix="test_")
        assert "test_f1_DI_HY" in d
        assert "test_roc_auc_macro" in d
        assert all(isinstance(v, float) for v in d.values())

    def test_missing_predicted_class_keeps_matrix_shape(self, outputs: tuple) -> None:
        y_true, _, proba = outputs
        y_pred = np.zeros_like(y_true)
        r = evaluate_classifier(y_true, y_pred, proba, CLASSES)
        assert r.confusion_matrix.shape == (4, 4)

    def test_binary_auc(self) -> None:
        y = np.array([0, 0, 1, 1])
        proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
        assert evaluate_classifier(y, proba.argmax(1), proba).roc_auc == 1.0

    def test_no_proba(self) -> None:
        r = evaluate_classifier(np.array([0, 1, 1]), np.array([0, 1, 0]))
        assert r.roc_auc is None
        assert "roc_auc_macro" not in r.to_dict()

    def test_auc_undefined_returns_none(self) -> None:
        y = np.array([0, 0, 0])
        proba = np.array([[0.5, 0.3, 0.2]] * 3)
        assert evaluate_classifier(y, y, proba).roc_auc is None


class TestDiseaseHeads:
    def test_marginals(self) -> None:
        proba = np.array([[0.1, 0.2, 0.3, 0.4]])
        m = disease_marginals(proba, CLASSES)[0]
        assert m[0] == pytest.approx(0.3)
        assert m[1] == pytest.approx(0.0)
        assert m[2] == pytest.approx(0.5)

    def test_disease_auc(self, outputs: tuple) -> None:
        y_true, _, proba = outputs
        metrics = evaluate_disease_heads(y_true, proba, CLASSES)
        assert set(metrics) == {"auc_Diabetes", "auc_Hypertension"}
        assert all(0.5 < v <= 1 for v in metrics.values())


class TestPlots:
    def test_confusion_matrix_plot(self) -> None:
        fig = plot_confusion_matrix(np.array([[5, 1], [0, 0]]), ["a", "b"])
        assert isinstance(fig, Figure)

    def test_feature_importance_plot(self) -> None:
        fig = plot_feature_importance({"x": 3.0, "y": 1.0}, top_n=1)
        assert isinstance(fig, Figure)

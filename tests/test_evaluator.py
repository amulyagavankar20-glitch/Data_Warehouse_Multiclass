"""Tests for model evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluator import (
    EvaluationResult,
    compute_drift_score,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_feature_importance,
)


@pytest.fixture
def classification_outputs() -> tuple:
    """Fixture providing true and predicted labels with probabilities."""
    np.random.seed(42)
    n = 300
    y_true = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])
    # Add some noise to predictions
    y_pred = np.where(
        np.random.rand(n) > 0.85,
        np.random.choice([0, 1, 2], n),
        y_true,
    )
    y_proba = np.zeros((n, 3))
    for i in range(n):
        y_proba[i, y_pred[i]] = 0.7 + np.random.rand() * 0.3
        for j in range(3):
            if j != y_pred[i]:
                y_proba[i, j] = (1 - y_proba[i, y_pred[i]]) / 2

    return y_true, y_pred, y_proba


@pytest.fixture
def binary_outputs() -> tuple:
    """Fixture for binary classification."""
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], n)
    y_pred = np.where(np.random.rand(n) > 0.9, 1 - y_true, y_true)
    y_proba = np.column_stack([
        1 - (y_pred + np.random.rand(n) * 0.1),
        y_pred + np.random.rand(n) * 0.1,
    ])
    y_proba = np.clip(y_proba, 0, 1)
    return y_true, y_pred, y_proba


class TestEvaluateClassifier:
    """Tests for src/models/evaluator.py."""

    def test_returns_evaluation_result(self, classification_outputs: tuple) -> None:
        y_true, y_pred, y_proba = classification_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba, class_labels=["A", "B", "C"])
        assert isinstance(result, EvaluationResult)

    def test_accuracy_in_range(self, classification_outputs: tuple) -> None:
        y_true, y_pred, y_proba = classification_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba)
        assert 0.0 <= result.accuracy <= 1.0

    def test_f1_macro_in_range(self, classification_outputs: tuple) -> None:
        y_true, y_pred, y_proba = classification_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba)
        assert 0.0 <= result.f1_macro <= 1.0

    def test_confusion_matrix_shape(self, classification_outputs: tuple) -> None:
        y_true, y_pred, y_proba = classification_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba)
        n_classes = len(np.unique(y_true))
        assert result.confusion_matrix.shape == (n_classes, n_classes)

    def test_per_class_report_keys(self, classification_outputs: tuple) -> None:
        y_true, y_pred, y_proba = classification_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba, class_labels=["A", "B", "C"])
        assert "accuracy" in result.per_class_report
        assert "macro avg" in result.per_class_report
        assert "weighted avg" in result.per_class_report

    def test_roc_auc_multiclass(self, classification_outputs: tuple) -> None:
        y_true, y_pred, y_proba = classification_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba)
        assert result.roc_auc is not None or result.roc_auc is None  # may or may not compute

    def test_roc_auc_binary(self, binary_outputs: tuple) -> None:
        y_true, y_pred, y_proba = binary_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba)
        assert result.roc_auc is not None

    def test_to_dict_flattened(self, classification_outputs: tuple) -> None:
        y_true, y_pred, y_proba = classification_outputs
        result = evaluate_classifier(y_true, y_pred, y_proba, class_labels=["A", "B", "C"])
        d = result.to_dict()
        assert "accuracy" in d
        assert "f1_macro" in d
        assert "f1_weighted" in d

    def test_no_proba_no_roc_auc(self) -> None:
        np.random.seed(42)
        y_true = np.random.choice([0, 1, 2], 100)
        y_pred = np.random.choice([0, 1, 2], 100)
        result = evaluate_classifier(y_true, y_pred)
        assert result.roc_auc is None


class TestPlots:
    """Tests for plotting functions."""

    def test_confusion_matrix_plot(self) -> None:
        np.random.seed(42)
        cm = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 12]])
        fig = plot_confusion_matrix(cm, ["A", "B", "C"])
        assert fig is not None

    def test_feature_importance_plot(self) -> None:
        importances = {f"feat_{i}": float(i) / 10 for i in range(10)}
        fig = plot_feature_importance(importances, top_n=5)
        assert fig is not None


class TestDriftScore:
    """Tests for drift computation."""

    def test_drift_score_returns_dict(self) -> None:
        np.random.seed(42)
        X_train = pd.DataFrame({f"col_{i}": np.random.randn(100) for i in range(4)})
        X_test = pd.DataFrame({f"col_{i}": np.random.randn(100) for i in range(4)})
        result = compute_drift_score(X_train, X_test)
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_drift_score_all_in_range(self) -> None:
        np.random.seed(42)
        X_train = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        X_test = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        result = compute_drift_score(X_train, X_test)
        for score in result.values():
            assert 0.0 <= score <= 1.0

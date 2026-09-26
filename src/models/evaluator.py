"""Model evaluation and metrics computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """Container for model evaluation results."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    per_class_report: dict[str, Any]
    confusion_matrix: np.ndarray
    roc_auc: float | None = None
    class_labels: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] | None = None
    drift_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a flat dictionary for MLflow logging."""
        result: dict[str, Any] = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "roc_auc": self.roc_auc if self.roc_auc else 0.0,
        }

        for class_label, metrics in self.per_class_report.items():
            if isinstance(class_label, str) and isinstance(metrics, dict):
                result[f"precision_{class_label}"] = metrics.get("precision", 0)
                result[f"recall_{class_label}"] = metrics.get("recall", 0)
                result[f"f1_{class_label}"] = metrics.get("f1-score", 0)

        return result


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_labels: list[str] | None = None,
) -> EvaluationResult:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Prediction probabilities (for ROC-AUC).
        class_labels: List of class label names.

    Returns:
        EvaluationResult with all metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) > 2:
        try:
            roc_auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro",
                labels=np.unique(y_true),
            )
        except (ValueError, TypeError):
            roc_auc = None
    elif y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        except (ValueError, TypeError):
            roc_auc = None

    labels = class_labels or [str(c) for c in sorted(set(y_true) | set(y_pred))]

    return EvaluationResult(
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        precision_macro=float(precision),
        recall_macro=float(recall),
        per_class_report=report,
        confusion_matrix=cm,
        roc_auc=roc_auc,
        class_labels=labels,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list[str],
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Generate a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: dict[str, float],
    top_n: int = 20,
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Generate a feature importance bar plot."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    fig.tight_layout()
    return fig


def compute_drift_score(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, float]:
    """Compute simple distribution drift using Kolmogorov-Smirnov test.

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        Dictionary of column -> drift score (KS statistic).
    """
    from scipy.stats import ks_2samp

    if not hasattr(ks_2samp, "__call__"):
        return {}

    drift_scores: dict[str, float] = {}
    for col in X_train.columns:
        if col in X_test.columns:
            train_data = X_train[col].values
            test_data = X_test[col].values
            if len(train_data) > 0 and len(test_data) > 0:
                stat, _ = ks_2samp(train_data, test_data)
                drift_scores[col] = float(stat)

    return drift_scores

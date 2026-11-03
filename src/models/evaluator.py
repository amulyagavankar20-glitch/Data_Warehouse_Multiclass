"""Model evaluation and metrics computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import settings  # noqa: E402


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

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Flatten to ``{name: value}`` for MLflow / DVC metrics."""
        result: dict[str, float] = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
        }
        if self.roc_auc is not None:
            result["roc_auc_macro"] = self.roc_auc
        for label in self.class_labels:
            metrics = self.per_class_report.get(label)
            if isinstance(metrics, dict):
                result[f"precision_{label}"] = float(metrics["precision"])
                result[f"recall_{label}"] = float(metrics["recall"])
                result[f"f1_{label}"] = float(metrics["f1-score"])
        return {f"{prefix}{k}": float(v) for k, v in result.items()}


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_labels: list[str] | None = None,
) -> EvaluationResult:
    """Compute classification metrics; ROC-AUC is macro one-vs-rest."""
    n_classes = y_proba.shape[1] if y_proba is not None else None
    labels_idx = (
        list(range(n_classes))
        if n_classes is not None
        else sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    )
    labels = class_labels or [str(i) for i in labels_idx]

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    roc_auc: float | None = None
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=labels_idx,
                    )
                )
        except ValueError:
            roc_auc = None
        if roc_auc is not None and not np.isfinite(roc_auc):
            roc_auc = None

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(
            f1_score(
                y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
            )
        ),
        f1_weighted=float(
            f1_score(
                y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0
            )
        ),
        precision_macro=float(precision),
        recall_macro=float(recall),
        per_class_report=report,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels_idx),
        roc_auc=roc_auc,
        class_labels=labels,
    )


def disease_marginals(proba: np.ndarray, class_labels: list[str]) -> np.ndarray:
    """Collapse 8-class sublabel probabilities into per-disease probabilities.

    ``P(Diabetes) = sum of P(c)`` over every sublabel ``c`` containing ``DI``,
    and likewise for heart disease (``HT``) and hypertension (``HY``).
    Columns follow ``settings.DISEASE_FLAG_NAMES``.
    """
    tokens = [label.split("_") for label in class_labels]
    out = np.zeros((proba.shape[0], len(settings.DISEASE_FLAG_NAMES)))
    for j, disease in enumerate(settings.DISEASE_FLAG_NAMES):
        code = settings.SUBLABEL_TOKENS[disease]
        mask = np.array([code in t for t in tokens])
        out[:, j] = proba[:, mask].sum(axis=1)
    return out


def evaluate_disease_heads(
    y_true: np.ndarray, proba: np.ndarray, class_labels: list[str]
) -> dict[str, float]:
    """ROC-AUC per disease, derived from the multi-class probabilities."""
    marginals = disease_marginals(proba, class_labels)
    tokens = [class_labels[i].split("_") for i in y_true]
    metrics: dict[str, float] = {}
    for j, disease in enumerate(settings.DISEASE_FLAG_NAMES):
        code = settings.SUBLABEL_TOKENS[disease]
        y_bin = np.array([code in t for t in tokens], dtype=int)
        if 0 < y_bin.sum() < len(y_bin):
            metrics[f"auc_{disease}"] = float(roc_auc_score(y_bin, marginals[:, j]))
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list[str],
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Row-normalised confusion matrix (recall on the diagonal)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title="Normalised Confusion Matrix",
        ylabel="True sublabel",
        xlabel="Predicted sublabel",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
            )
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances: dict[str, float],
    top_n: int = 20,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Horizontal bar chart of the ``top_n`` most important features."""
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance (LightGBM gain)")
    fig.tight_layout()
    return fig

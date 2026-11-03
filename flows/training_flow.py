"""Prefect training flow for the disease detection / early-warning pipeline."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from sklearn.metrics import roc_auc_score

from src.config import settings
from src.data.loader import clean_data, load_raw_data, split_features_targets
from src.data.splitter import SplitData, normal_class_index, split_data
from src.data.validator import validate_data
from src.features.engineering import ClinicalFeaturePipeline
from src.models.anomaly.isolation_forest import IsolationForestModel
from src.models.anomaly.one_class_svm import OneClassSVMModel
from src.models.classifier import LightGBMClassifier
from src.models.evaluator import (
    evaluate_classifier,
    evaluate_disease_heads,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.monitoring.drift import build_reference, compute_drift
from src.serving.bundle import ModelBundle
from src.utils import mlflow_utils

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Raised when a trained model fails the minimum quality bar."""


@task(cache_policy=NO_CACHE)
def extract(data_path: Path | None) -> pd.DataFrame:
    df = load_raw_data(data_path)
    logger.info("Loaded raw data: %s", df.shape)
    return df


@task(cache_policy=NO_CACHE)
def validate(df: pd.DataFrame) -> pd.DataFrame:
    result = validate_data(df)
    for msg in result.warnings:
        logger.warning("Validation warning: %s", msg)
    if not result.passed:
        raise ValueError(f"Data validation failed: {result.errors}")
    return df


@task(cache_policy=NO_CACHE)
def prepare(df: pd.DataFrame) -> SplitData:
    cleaned = clean_data(df)
    logger.info("Dropped %d duplicate rows", len(df) - len(cleaned))
    X, y = split_features_targets(cleaned)
    return split_data(X, y)


@task(cache_policy=NO_CACHE)
def fit_features(
    split: SplitData,
) -> tuple[ClinicalFeaturePipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline = ClinicalFeaturePipeline().fit(split.X_train)
    return (
        pipeline,
        pipeline.transform(split.X_train),
        pipeline.transform(split.X_val),
        pipeline.transform(split.X_test),
    )


@task(cache_policy=NO_CACHE)
def train_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    num_boost_round: int | None,
) -> LightGBMClassifier:
    clf = LightGBMClassifier(num_boost_round=num_boost_round)
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    clf.calibrate_thresholds(X_val, y_val)
    logger.info("LightGBM best iteration: %d", clf.best_iteration)
    return clf


@task(cache_policy=NO_CACHE)
def train_anomaly_detectors(
    X_train: pd.DataFrame, split: SplitData
) -> tuple[IsolationForestModel, OneClassSVMModel]:
    healthy = X_train[split.y_train == normal_class_index(split.label_encoder)]
    logger.info("Healthy cohort for anomaly baselines: %d rows", len(healthy))
    iso = IsolationForestModel(
        n_estimators=settings.isolation_forest_n_estimators,
        contamination=settings.isolation_forest_contamination,
        random_state=settings.random_state,
    ).fit(healthy)
    svm = OneClassSVMModel(
        nu=settings.ocsvm_nu,
        max_train_samples=settings.ocsvm_max_train_samples,
        random_state=settings.random_state,
    ).fit(healthy)
    return iso, svm


@task(cache_policy=NO_CACHE)
def evaluate(
    clf: LightGBMClassifier,
    iso: IsolationForestModel,
    svm: OneClassSVMModel,
    X_test: pd.DataFrame,
    split: SplitData,
) -> tuple[dict[str, float], dict[str, Any], np.ndarray]:
    classes = split.classes
    proba = clf.predict_proba(X_test)

    calibrated = evaluate_classifier(split.y_test, clf.decide(proba), proba, classes)
    argmax = evaluate_classifier(split.y_test, np.argmax(proba, axis=1), proba, classes)
    metrics = {
        **calibrated.to_dict(prefix="test_"),
        "test_argmax_f1_macro": argmax.f1_macro,
        "test_argmax_accuracy": argmax.accuracy,
        **{
            f"test_{k}": v
            for k, v in evaluate_disease_heads(split.y_test, proba, classes).items()
        },
        "best_iteration": float(clf.best_iteration),
        "thresholds_applied": float(clf.thresholds is not None),
        **{f"val_{k}_f1_macro": v for k, v in clf.threshold_val_f1.items()},
    }

    iso_scores = iso.decision_function(X_test)
    svm_scores = svm.decision_function(X_test)
    abnormal = (split.y_test != normal_class_index(split.label_encoder)).astype(int)
    if 0 < abnormal.sum() < len(abnormal):
        metrics["test_anomaly_auc_isolation"] = float(
            roc_auc_score(abnormal, -iso_scores)
        )
        metrics["test_anomaly_auc_ocsvm"] = float(roc_auc_score(abnormal, -svm_scores))

    labels = split.label_encoder.inverse_transform(split.y_test)
    audit = (
        pd.DataFrame({"sublabel": labels, "iso": iso_scores, "svm": svm_scores})
        .groupby("sublabel")
        .agg(
            count=("iso", "size"),
            mean_iso_score=("iso", "mean"),
            mean_ocsvm_score=("svm", "mean"),
            iso_flag_rate=("iso", lambda s: float((s < 0).mean())),
            ocsvm_flag_rate=("svm", lambda s: float((s < 0).mean())),
        )
        .sort_values("mean_iso_score")
    )
    audit_dict = {str(k): v for k, v in audit.to_dict(orient="index").items()}
    return metrics, audit_dict, calibrated.confusion_matrix


@task(cache_policy=NO_CACHE)
def quality_gate(metrics: dict[str, float], min_auc: float) -> None:
    auc = metrics.get("test_roc_auc_macro", 0.0)
    if auc < min_auc:
        raise QualityGateError(f"test macro ROC-AUC {auc:.4f} < required {min_auc:.4f}")
    logger.info("Quality gate passed: test macro ROC-AUC %.4f >= %.4f", auc, min_auc)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=float) + "\n",
        encoding="utf-8",
        newline="\n",
    )


@flow(name="disease-anomaly-training")
def run_training(
    data_path: Path | None = None,
    model_dir: Path | None = None,
    reports_dir: Path | None = None,
    enable_mlflow: bool | None = None,
    min_test_macro_auc: float | None = None,
    num_boost_round: int | None = None,
) -> dict[str, float]:
    """extract -> validate -> split -> features -> train -> evaluate -> gate ->
    package -> track/register/promote. Returns the test metrics."""
    model_dir = model_dir or settings.model_dir
    reports_dir = reports_dir or settings.reports_dir
    enable_mlflow = settings.enable_mlflow if enable_mlflow is None else enable_mlflow
    min_auc = (
        settings.min_test_macro_auc
        if min_test_macro_auc is None
        else min_test_macro_auc
    )

    df = validate(extract(data_path))
    split = prepare(df)
    pipeline, X_train, X_val, X_test = fit_features(split)
    clf = train_classifier(X_train, split.y_train, X_val, split.y_val, num_boost_round)
    iso, svm = train_anomaly_detectors(X_train, split)
    metrics, audit, cm = evaluate(clf, iso, svm, X_test, split)

    drift_reference = build_reference(split.X_train)
    holdout_drift = compute_drift(drift_reference, split.X_test)
    metrics["test_max_psi"] = holdout_drift.max_psi

    quality_gate(metrics, min_auc)

    trained_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle = ModelBundle(
        feature_pipeline=pipeline,
        classifier=clf,
        iso_forest=iso,
        ocsvm=svm,
        classes=split.classes,
        drift_reference=drift_reference,
        metadata={
            "model_version": f"{trained_at}-{mlflow_utils.git_commit()}",
            "trained_at": trained_at,
            "n_train": len(split.X_train),
            "n_val": len(split.X_val),
            "n_test": len(split.X_test),
            "metrics": metrics,
        },
    )
    bundle_path = bundle.save(model_dir / "bundle.joblib")
    _write_json(model_dir / "serving_contract.json", bundle.serving_contract())
    _write_json(reports_dir / "metrics.json", metrics)
    _write_json(reports_dir / "anomaly_audit.json", audit)
    _write_json(reports_dir / "drift.json", holdout_drift.to_dict())
    importance = dict(
        clf.get_feature_importance().set_index("feature")["importance"].astype(float)
    )
    _write_json(reports_dir / "feature_importance.json", importance)
    fig_cm = plot_confusion_matrix(cm, split.classes)
    fig_cm.savefig(reports_dir / "confusion_matrix.png", dpi=120)
    fig_fi = plot_feature_importance(importance)

    if enable_mlflow:
        mlflow_utils.setup_mlflow()
        with mlflow.start_run(
            run_name=f"train-{trained_at}", tags=mlflow_utils.run_tags()
        ):
            mlflow_utils.log_params(
                {
                    **clf.params,
                    "num_boost_round": clf.num_boost_round,
                    "best_iteration": clf.best_iteration,
                    "n_features": len(pipeline.feature_names_out_),
                    "n_train": len(split.X_train),
                    "model_version": bundle.version,
                }
            )
            mlflow_utils.log_metrics(metrics)
            mlflow.log_dict(audit, "reports/anomaly_audit.json")
            mlflow.log_dict(holdout_drift.to_dict(), "reports/drift.json")
            mlflow.log_dict(bundle.serving_contract(), "serving_contract.json")
            mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
            mlflow.log_figure(fig_fi, "plots/feature_importance.png")
            example = split.X_test.head(2).astype(
                dict.fromkeys(settings.BASE_NUMERIC_COLUMNS, float)
            )
            version = mlflow_utils.log_model_bundle(bundle_path, input_example=example)
            if version is not None:
                promoted = mlflow_utils.promote_if_better(
                    version, metrics["test_roc_auc_macro"]
                )
                mlflow.set_tag("promoted_to_champion", str(promoted))
    plt.close("all")

    logger.info(
        "Training complete: macro AUC %.4f, macro F1 %.4f",
        metrics["test_roc_auc_macro"],
        metrics["test_f1_macro"],
    )
    return metrics


if __name__ == "__main__":
    run_training()

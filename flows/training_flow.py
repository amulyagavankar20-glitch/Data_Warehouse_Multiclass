"""Prefect training flow for the disease anomaly pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from prefect import flow, task

from src.config import settings
from src.data.loader import clean_data, load_raw_data
from src.data.splitter import get_normal_cohort, split_data
from src.features.engineering import fit_transform_train
from src.models.anomaly.autoencoder import DeepAutoencoderModel
from src.models.anomaly.isolation_forest import IsolationForestModel
from src.models.anomaly.one_class_svm import OneClassSVMModel
from src.models.classifier import HybridClassifier
from src.models.evaluator import evaluate_classifier, plot_confusion_matrix
from src.utils.mlflow_utils import (
    log_artifacts_dict,
    log_metrics,
    log_plot,
    setup_mlflow,
    start_run,
    transition_model_to_stage,
)

logger = logging.getLogger(__name__)


@task
def extract_data() -> pd.DataFrame:
    """Extract raw data from CSV."""
    df = load_raw_data()
    logger.info("Loaded raw data: %s", df.shape)
    return df


@task
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data quality; fail if critical checks fail."""
    from src.data.validator import validate_data as validate

    result = validate(df)

    for error in result.errors:
        logger.error("Validation error: %s", error)
    for warning in result.warnings:
        logger.warning("Validation warning: %s", warning)

    if not result.passed:
        raise ValueError(f"Data validation failed: {result.errors}")

    logger.info("Data validation passed with %d warnings", len(result.warnings))
    return df


@task
def preprocess_data(df: pd.DataFrame):
    """Clean data and perform leakage-safe stratified split."""
    cleaned = clean_data(df)
    logger.info("Cleaned data shape: %s", cleaned.shape)

    split = split_data(
        cleaned,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    preprocessor, X_train_enc, X_test_enc = fit_transform_train(split)

    return split, preprocessor, X_train_enc, X_test_enc


@task
def train_anomaly_models(X_train_enc, split):
    """Train unsupervised anomaly detection models on normal cohort."""
    normal_cohort = get_normal_cohort(
        X_train_enc,
        split.y_train,
        split.label_encoder,
    )
    logger.info("Normal cohort shape: %s", normal_cohort.shape)

    iso_model = IsolationForestModel(
        n_estimators=settings.isolation_forest_n_estimators,
        contamination=settings.isolation_forest_contamination,
        random_state=settings.random_state,
    )
    iso_model.fit(normal_cohort)

    ocs_model = OneClassSVMModel(
        kernel="rbf",
        gamma="scale",
        nu=settings.isolation_forest_contamination,
    )
    ocs_model.fit(normal_cohort)

    ae_model = DeepAutoencoderModel(
        input_dim=X_train_enc.shape[1],
        encoding_dim=settings.autoencoder_encoding_dim,
        dropout=settings.autoencoder_dropout,
        patience=settings.autoencoder_patience,
        max_epochs=50,
        batch_size=512,
    )
    ae_model.fit(normal_cohort)

    return iso_model, ocs_model, ae_model


@task
def compute_meta_features(iso_model, ocs_model, ae_model, X_train_enc, X_test_enc):
    """Compute anomaly meta-features for classifier training."""
    iso_scores_train = iso_model.get_anomaly_score(X_train_enc)
    iso_scores_test = iso_model.get_anomaly_score(X_test_enc)

    ae_mse_train = ae_model.get_anomaly_score(X_train_enc)
    ae_mse_test = ae_model.get_anomaly_score(X_test_enc)

    X_train_meta = X_train_enc.copy()
    X_train_meta["iso_score"] = iso_scores_train
    X_train_meta["ae_mse"] = ae_mse_train

    X_test_meta = X_test_enc.copy()
    X_test_meta["iso_score"] = iso_scores_test
    X_test_meta["ae_mse"] = ae_mse_test

    return X_train_meta, X_test_meta


@task
def train_classifier(X_train_meta, y_train):
    """Train the hybrid RandomForest classifier."""
    classifier = HybridClassifier(
        n_estimators=settings.rf_n_estimators,
        class_weight="balanced",
        random_state=settings.random_state,
    )
    classifier.fit(X_train_meta, y_train)
    return classifier


@task
def evaluate_and_log(
    classifier,
    X_test_meta,
    y_test,
    split,
    iso_model,
    ocs_model,
    ae_model,
):
    """Evaluate the classifier and log results to MLflow."""
    y_pred = classifier.predict(X_test_meta)
    y_proba = classifier.predict_proba(X_test_meta)

    result = evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=list(split.label_encoder.classes_),
    )

    metrics = result.to_dict()
    log_metrics(metrics)

    fig_cm = plot_confusion_matrix(
        result.confusion_matrix,
        result.class_labels,
    )
    log_plot(fig_cm, "confusion_matrix.png")

    fi_df = classifier.get_feature_importance()
    if fi_df is not None:
        fi_dict = dict(zip(fi_df["feature"], fi_df["importance"]))
        log_artifacts_dict(fi_dict, "feature_importance.json")

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param(
        "n_features",
        len(X_test_meta.columns) - 2,  # exclude meta-features
    )

    return result


@flow()
def run_training():
    """Full MLOps training pipeline: extract → validate → preprocess →
    train anomaly models → compute meta-features → train classifier →
    evaluate → log to MLflow → register model.
    """
    setup_mlflow()

    with start_run(run_name="training-run"):
        df = extract_data()
        df = validate_data(df)
        split, preprocessor, X_train_enc, X_test_enc = preprocess_data(df)

        iso_model, ocs_model, ae_model = train_anomaly_models(X_train_enc, split)

        X_train_meta, X_test_meta = compute_meta_features(
            iso_model, ocs_model, ae_model, X_train_enc, X_test_enc,
        )

        classifier = train_classifier(X_train_meta, split.y_train)

        result = evaluate_and_log(
            classifier,
            X_test_meta,
            split.y_test,
            split,
            iso_model,
            ocs_model,
            ae_model,
        )

        model_bundle = {
            "model": classifier,
            "classes": list(split.label_encoder.classes_),
            "preprocessor": preprocessor,
            "label_encoder": split.label_encoder,
            "feature_names": split.feature_names,
            "iso_forest": iso_model,
            "ocsvm": ocs_model,
            "autoencoder": ae_model,
        }

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(model_bundle, f.name)
            bundle_path = f.name

        mlflow.log_artifact(bundle_path, "model_bundle")
        os.unlink(bundle_path)

        # Register the sklearn classifier to the model registry
        try:
            mlflow.sklearn.log_model(
                sk_model=classifier.model,
                artifact_path="classifier",
                registered_model_name=settings.mlflow_model_name,
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        except Exception:
            mlflow.sklearn.log_model(
                sk_model=classifier.model,
                artifact_path="classifier",
                registered_model_name=settings.mlflow_model_name,
            )

        # Save model bundle for API serving
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        joblib.dump(model_bundle, models_dir / "model.joblib")

        transition_model_to_stage(settings.mlflow_model_name, "Staging")

        logger.info("Training pipeline complete. Accuracy: %.4f, F1-macro: %.4f",
                     result.accuracy, result.f1_macro)

        return {
            "accuracy": result.accuracy,
            "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
            "roc_auc": result.roc_auc,
        }


if __name__ == "__main__":
    run_training()

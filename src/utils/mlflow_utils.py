"""MLflow logging and model management utilities."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str | None = None) -> str:
    """Configure MLflow tracking.

    Args:
        tracking_uri: Optional tracking URI. Defaults to settings.

    Returns:
        The tracking URI.
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return uri


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))


def log_metrics(metrics: dict[str, float]) -> None:
    """Log metrics to MLflow."""
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            mlflow.log_metric(key, float(value))


def log_artifacts_dict(data: dict[str, Any], filename: str) -> None:
    """Log a dictionary as a JSON artifact."""
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2, default=str)
        temp_path = f.name

    mlflow.log_artifact(temp_path, filename)
    os.unlink(temp_path)


def log_plot(fig: Any, filename: str) -> None:
    """Log a matplotlib figure as an artifact."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches="tight")
        temp_path = f.name
        import matplotlib
        matplotlib.pyplot.close(fig)

    mlflow.log_artifact(temp_path, "plots")
    os.unlink(temp_path)


def save_model(model: Any, model_name: str | None = None) -> str:
    """Save a model to MLflow model registry.

    Args:
        model: Model object to save.
        model_name: Registered model name.

    Returns:
        Model URI.
    """
    name = model_name or settings.mlflow_model_name
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=name,
    )
    model_uri = f"models:/{name}/None"
    return model_uri


def load_production_model(model_name: str | None = None) -> Any:
    """Load a model from MLflow registry in Production stage.

    Args:
        model_name: Registered model name.

    Returns:
        Loaded model.
    """
    name = model_name or settings.mlflow_model_name
    model_uri = f"models:/{name}/{settings.mlflow_model_stage}"
    return mlflow.sklearn.load_model(model_uri)


def register_model_from_run(
    model: Any,
    model_name: str | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Register a model to the MLflow model registry.

    Args:
        model: Model to register.
        model_name: Registered model name.
        metrics: Optional metrics for tagging.

    Returns:
        Run ID.
    """
    name = model_name or settings.mlflow_model_name

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=name,
    )

    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
    return run_id


def transition_model_to_stage(
    model_name: str,
    stage: str,
    version: int | None = None,
) -> None:
    """Transition a model version to a new stage.

    Args:
        model_name: Registered model name.
        stage: Target stage (Staging, Production, None).
        version: Optional model version.
    """
    client = mlflow.MlflowClient()
    if version is None:
        versions = client.search_model_versions(f"name = '{model_name}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            version = int(latest.version)

    if version is not None:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        logger.info("Transitioned model %s v%d to %s", model_name, version, stage)


def start_run(run_name: str | None = None) -> Any:
    """Start an MLflow run with standard tags."""
    tags = {
        "dataset_version": "1.0",
        "pipeline_version": "1.0.0",
        "author": os.environ.get("USER", "unknown"),
        "source": "disease-anomaly-pipeline",
    }
    if run_name:
        tags["mlflow.runName"] = run_name

    return mlflow.start_run(tags=tags)

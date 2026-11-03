"""MLflow tracking, registration and alias-based promotion."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

PROMOTION_METRIC = "test_roc_auc_macro"


def setup_mlflow(tracking_uri: str | None = None) -> str:
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return uri


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


def run_tags() -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "environment": settings.environment,
        "source": "disease-anomaly-pipeline",
    }


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params(
        {
            k: v if isinstance(v, str | int | float | bool) else str(v)
            for k, v in params.items()
        }
    )


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(
        {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float | np.integer | np.floating)
        }
    )


def log_model_bundle(
    bundle_path: Path, input_example: Any = None, register: bool = True
) -> int | None:
    """Log the bundle as a pyfunc model and (optionally) register a version."""
    from src.serving.pyfunc import DiseaseModel

    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=DiseaseModel(),
        artifacts={"bundle": str(bundle_path)},
        code_paths=[str(PROJECT_ROOT / "src")],
        input_example=input_example,
        registered_model_name=settings.mlflow_model_name if register else None,
    )
    version = getattr(info, "registered_model_version", None)
    return int(version) if version is not None else None


def _champion_metric(client: MlflowClient) -> float | None:
    try:
        mv = client.get_model_version_by_alias(
            settings.mlflow_model_name, settings.mlflow_champion_alias
        )
    except MlflowException:
        return None
    if mv.run_id is None:
        return None
    value = client.get_run(mv.run_id).data.metrics.get(PROMOTION_METRIC)
    return float(value) if value is not None else None


def promote_if_better(version: int, candidate_metric: float) -> bool:
    """Tag ``version`` as candidate; move the champion alias only if it wins."""
    client = MlflowClient()
    name = settings.mlflow_model_name
    client.set_registered_model_alias(
        name, settings.mlflow_candidate_alias, str(version)
    )

    current = _champion_metric(client)
    if current is not None and candidate_metric <= current:
        logger.info(
            "Keeping champion (%s=%.4f >= candidate %.4f)",
            PROMOTION_METRIC,
            current,
            candidate_metric,
        )
        return False

    client.set_registered_model_alias(
        name, settings.mlflow_champion_alias, str(version)
    )
    logger.info("Promoted %s v%d to @%s", name, version, settings.mlflow_champion_alias)
    return True

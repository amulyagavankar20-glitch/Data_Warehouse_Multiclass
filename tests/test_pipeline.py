"""End-to-end tests of the Prefect training flow, MLflow registry and bundle."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from prefect.testing.utilities import prefect_test_harness

from flows.training_flow import QualityGateError, run_training
from src.config import settings
from src.data.loader import load_raw_data
from src.serving.bundle import ModelBundle


@pytest.fixture(scope="module", autouse=True)
def prefect_harness() -> Iterator[None]:
    with prefect_test_harness():
        yield


def _train(raw_csv: Path, tmp_path: Path, **kwargs: object) -> dict[str, float]:
    return run_training(
        data_path=raw_csv,
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        num_boost_round=30,
        **kwargs,  # type: ignore[arg-type]
    )


def test_flow_writes_artifacts(raw_csv: Path, tmp_path: Path) -> None:
    metrics = _train(raw_csv, tmp_path, enable_mlflow=False, min_test_macro_auc=0.0)
    assert 0.5 < metrics["test_roc_auc_macro"] <= 1.0
    for name in (
        "metrics.json",
        "drift.json",
        "anomaly_audit.json",
        "confusion_matrix.png",
    ):
        assert (tmp_path / "reports" / name).exists()

    contract = json.loads((tmp_path / "models" / "serving_contract.json").read_text())
    assert contract["input_feature_count"] == 50

    bundle = ModelBundle.load(tmp_path / "models" / "bundle.joblib")
    preds = bundle.predict(load_raw_data(raw_csv).head(3))
    assert len(preds) == 3
    assert bundle.drift_reference
    assert (
        bundle.metadata["metrics"]["test_roc_auc_macro"]
        == metrics["test_roc_auc_macro"]
    )


def test_quality_gate_blocks_weak_model(raw_csv: Path, tmp_path: Path) -> None:
    with pytest.raises(QualityGateError):
        _train(raw_csv, tmp_path, enable_mlflow=False, min_test_macro_auc=0.999)
    assert not (tmp_path / "models" / "bundle.joblib").exists()


def test_invalid_data_fails_fast(raw_csv: Path, tmp_path: Path) -> None:
    df = load_raw_data(raw_csv)
    df.loc[0, "systolic_bp"] = -5
    df.to_csv(raw_csv, index=False)
    with pytest.raises(ValueError, match="validation failed"):
        _train(raw_csv, tmp_path, enable_mlflow=False, min_test_macro_auc=0.0)


def test_mlflow_registration_and_promotion(
    raw_csv: Path, tmp_path: Path, mlflow_tmp: str
) -> None:
    _train(raw_csv, tmp_path, enable_mlflow=True, min_test_macro_auc=0.0)
    client = MlflowClient(tracking_uri=mlflow_tmp)
    name = settings.mlflow_model_name
    champion = client.get_model_version_by_alias(name, settings.mlflow_champion_alias)
    assert int(champion.version) == 1
    run = client.get_run(champion.run_id)
    assert "test_roc_auc_macro" in run.data.metrics
    assert "git_commit" in run.data.tags

    _train(raw_csv, tmp_path, enable_mlflow=True, min_test_macro_auc=0.0)
    candidate = client.get_model_version_by_alias(name, settings.mlflow_candidate_alias)
    champion = client.get_model_version_by_alias(name, settings.mlflow_champion_alias)
    assert int(candidate.version) == 2
    assert int(champion.version) == 1

    bundle = ModelBundle.from_mlflow(f"models:/{name}@{settings.mlflow_champion_alias}")
    assert bundle.classes

    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model(f"models:/{name}@{settings.mlflow_champion_alias}")
    df = load_raw_data(raw_csv).head(4)[settings.RAW_FEATURE_ORDER]
    df = df.astype(dict.fromkeys(settings.BASE_NUMERIC_COLUMNS, float))
    out = model.predict(df)
    assert list(out.columns[:2]) == ["predicted_sublabel", "confidence"]
    assert len(out) == 4


def test_bundle_load_rejects_wrong_type(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "x.joblib"
    joblib.dump({"not": "a bundle"}, path)
    with pytest.raises(TypeError):
        ModelBundle.load(path)

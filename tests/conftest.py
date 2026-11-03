"""Shared test fixtures: schema-faithful synthetic warehouse data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import settings

SUBLABELS = ["N", "HY", "DI_HY", "DI", "HT", "DI_HT", "HT_HY", "DI_HT_HY"]
SUBLABEL_P = [0.30, 0.20, 0.18, 0.12, 0.06, 0.05, 0.05, 0.04]


def make_raw_frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic records with the real columns, levels and a learnable signal."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(SUBLABELS, size=n, p=SUBLABEL_P)
    di = np.array(["DI" in s for s in sub])
    ht = np.array(["HT" in s for s in sub])
    hy = np.array(["HY" in s for s in sub])

    def pick(levels: list[str]) -> np.ndarray:
        return rng.choice(levels, size=n)

    systolic = rng.normal(125, 10, n) + 20 * hy
    diastolic = rng.normal(80, 6, n) + 10 * hy
    df = pd.DataFrame(
        {
            "composite_key": [f"k{i % 50}" for i in range(n)],
            "age_level": pick(list(settings.ORDINAL_MAPS["age_level"])),
            "gender": pick(settings.NOMINAL_LEVELS["gender"]),
            "bmi_level": pick(list(settings.ORDINAL_MAPS["bmi_level"])),
            "smoking": pick(settings.NOMINAL_LEVELS["smoking"]),
            "diabetes": np.where(di, "Yes", "No"),
            "age": rng.integers(2, 90, n),
            "age_normalized": rng.uniform(0, 1, n),
            "bmi": rng.normal(27, 5, n).clip(12, 80),
            "hypertension": hy.astype(float),
            "heart_disease": ht.astype(float),
            "HbA1c_level": (rng.normal(5.4, 0.4, n) + 1.6 * di).clip(3.5, 9),
            "glucose": np.round(rng.normal(110, 15, n) + 60 * di).clip(70, 300),
            "cholesterol": np.full(n, 214.7),
            "sleep_hours": rng.uniform(4, 10, n),
            "triglycerides": rng.uniform(50, 400, n),
            "physical_activity": pick(settings.NOMINAL_LEVELS["physical_activity"]),
            "family_history": pick(settings.NOMINAL_LEVELS["family_history"]),
            "stress_level": pick(list(settings.ORDINAL_MAPS["stress_level"])),
            "low_hdl_cholesterol": pick(settings.NOMINAL_LEVELS["low_hdl_cholesterol"]),
            "high_ldl_cholesterol": pick(
                settings.NOMINAL_LEVELS["high_ldl_cholesterol"]
            ),
            "blood_pressure": (rng.normal(135, 8, n) + 25 * hy).clip(120, 180),
            "high_blood_pressure": np.where(hy, "Yes", pick(["No", "Yes"])),
            "sugar_consumption": pick(settings.NOMINAL_LEVELS["sugar_consumption"]),
            "crp_level": (rng.uniform(0, 8, n) + 6 * ht).clip(0, 15),
            "homocysteine_level": (rng.uniform(5, 14, n) + 5 * ht).clip(5, 20),
            "systolic_bp": systolic.clip(90, 179),
            "diastolic_bp": diastolic.clip(60, 119),
            "alcohol_intake": rng.uniform(0, 30, n),
            "salt_intake": rng.uniform(2, 15, n),
            "heart_rate": (rng.normal(72, 6, n) + 12 * ht).clip(50, 99),
            "hdl": rng.uniform(30, 99, n),
            "ldl": rng.uniform(70, 189, n),
            "education_level": pick(list(settings.ORDINAL_MAPS["education_level"])),
            "employment_status": pick(settings.NOMINAL_LEVELS["employment_status"]),
            "source_dataset": pick(["diabetes", "hypertension", "heart"]),
            "disease_flags": [
                f"{int(a)},{int(b)},{int(c)}"
                for a, b, c in zip(di, ht, hy, strict=True)
            ],
            "sublabel": sub,
            "label": np.where(sub == "N", "Normal", "Abnormal"),
        }
    )
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def raw_csv(tmp_path: Path, raw_df: pd.DataFrame) -> Path:
    path = tmp_path / "warehouse.csv"
    raw_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a small real bundle once per session (no MLflow)."""
    from prefect.testing.utilities import prefect_test_harness

    from flows.training_flow import run_training

    root = tmp_path_factory.mktemp("trained")
    csv = root / "warehouse.csv"
    make_raw_frame(n=2000, seed=7).to_csv(csv, index=False)
    with prefect_test_harness():
        run_training(
            data_path=csv,
            model_dir=root / "models",
            reports_dir=root / "reports",
            enable_mlflow=False,
            min_test_macro_auc=0.0,
            num_boost_round=40,
        )
    return root / "models" / "bundle.joblib"


@pytest.fixture
def mlflow_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    import mlflow

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setattr(settings, "mlflow_tracking_uri", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    if mlflow.active_run():
        mlflow.end_run()

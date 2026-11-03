"""Tests for PSI drift monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import split_features_targets
from src.monitoring.drift import build_reference, compute_drift, monitored_columns, psi
from tests.conftest import make_raw_frame


@pytest.fixture
def reference(raw_df: pd.DataFrame) -> dict:
    X, _ = split_features_targets(raw_df)
    return build_reference(X)


def test_psi_identical_is_zero() -> None:
    p = np.array([0.25, 0.25, 0.5])
    assert psi(p, p) == pytest.approx(0.0)


def test_reference_covers_continuous_features(reference: dict) -> None:
    assert "pulse_pressure" in reference
    assert "cholesterol" not in reference
    assert set(reference) <= set(monitored_columns())
    for ref in reference.values():
        assert sum(ref["proportions"]) == pytest.approx(1.0)


def test_same_distribution_is_stable(reference: dict) -> None:
    X_new, _ = split_features_targets(make_raw_frame(n=1500, seed=99))
    report = compute_drift(reference, X_new)
    assert report.max_psi < 0.1
    assert report.drifted_features == []


def test_shift_is_detected(reference: dict) -> None:
    X_new, _ = split_features_targets(make_raw_frame(n=1500, seed=99))
    X_new["glucose"] = X_new["glucose"] + 80
    report = compute_drift(reference, X_new, threshold=0.25)
    assert "glucose" in report.drifted_features
    payload = report.to_dict()
    assert payload["max_psi"] >= 0.25
    assert payload["threshold"] == 0.25


def test_cli_exit_codes(
    trained_bundle_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.monitoring.__main__ import main

    batch = make_raw_frame(n=600, seed=5)
    stable_csv = tmp_path / "stable.csv"
    batch.to_csv(stable_csv, index=False)
    out = tmp_path / "drift.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drift",
            str(stable_csv),
            "--bundle",
            str(trained_bundle_path),
            "--output",
            str(out),
        ],
    )
    assert main() == 0
    assert "psi" in json.loads(out.read_text())

    batch["glucose"] = batch["glucose"] + 120
    shifted_csv = tmp_path / "shifted.csv"
    batch.to_csv(shifted_csv, index=False)
    monkeypatch.setattr(
        sys, "argv", ["drift", str(shifted_csv), "--bundle", str(trained_bundle_path)]
    )
    assert main() == 2

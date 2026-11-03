"""Tests for the clinical feature pipeline (training/serving parity)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import split_features_targets
from src.data.splitter import split_data
from src.features.engineering import ClinicalFeaturePipeline, add_clinical_interactions

EXPECTED_FEATURES = [
    "age_level", "bmi_level", "age", "bmi", "HbA1c_level", "glucose", "cholesterol",
    "sleep_hours", "triglycerides", "stress_level", "blood_pressure", "crp_level",
    "homocysteine_level", "systolic_bp", "diastolic_bp", "alcohol_intake",
    "salt_intake", "heart_rate", "hdl", "ldl", "education_level", "pulse_pressure",
    "mean_arterial_pressure", "cholesterol_ratio", "glucose_triglyceride_index",
    "age_is_imputed", "glucose_is_imputed", "HbA1c_level_is_imputed",
    "heart_rate_is_imputed", "diastolic_bp_is_imputed", "hdl_is_imputed",
    "systolic_bp_is_imputed", "ldl_is_imputed", "salt_intake_is_imputed",
    "cholesterol_is_imputed", "alcohol_intake_is_imputed",
    "triglycerides_is_imputed", "gender_Male", "smoking_Former", "smoking_Never",
    "physical_activity_Low", "physical_activity_Moderate", "family_history_Yes",
    "low_hdl_cholesterol_Yes", "high_ldl_cholesterol_Yes", "high_blood_pressure_Yes",
    "sugar_consumption_Low", "sugar_consumption_Medium", "employment_status_Retired",
    "employment_status_Unemployed",
]  # fmt: skip


@pytest.fixture
def split_and_pipeline(
    raw_df: pd.DataFrame,
) -> tuple[ClinicalFeaturePipeline, pd.DataFrame, pd.DataFrame]:
    split = split_data(*split_features_targets(raw_df))
    return ClinicalFeaturePipeline().fit(split.X_train), split.X_train, split.X_test


def test_interactions() -> None:
    row = pd.DataFrame(
        {"systolic_bp": [120.0], "diastolic_bp": [80.0], "ldl": [100.0],
         "hdl": [50.0], "glucose": [100.0], "triglycerides": [150.0]}
    )  # fmt: skip
    out = add_clinical_interactions(row).iloc[0]
    assert out["pulse_pressure"] == 40
    assert out["mean_arterial_pressure"] == pytest.approx(80 + 40 / 3)
    assert out["cholesterol_ratio"] == pytest.approx(2.0, rel=1e-4)
    assert out["glucose_triglyceride_index"] == pytest.approx(np.log(15001))


def test_contract_matches_notebook_champion(split_and_pipeline: tuple) -> None:
    pipeline, X_train, _ = split_and_pipeline
    out = pipeline.transform(X_train)
    assert list(out.columns) == EXPECTED_FEATURES
    assert out.shape[1] == 50
    assert out.dtypes.eq(np.float32).all()


def test_one_hot_columns_are_not_scaled(split_and_pipeline: tuple) -> None:
    pipeline, X_train, _ = split_and_pipeline
    assert "physical_activity_Moderate" not in pipeline.scaled_columns_
    out = pipeline.transform(X_train)
    assert set(np.unique(out["physical_activity_Moderate"])) <= {0.0, 1.0}
    assert len(pipeline.scaled_columns_) == 25


def test_scaler_fit_on_train_only(split_and_pipeline: tuple) -> None:
    pipeline, X_train, X_test = split_and_pipeline
    train_out = pipeline.transform(X_train)
    assert train_out["glucose"].mean() == pytest.approx(0, abs=1e-5)
    assert pipeline.transform(X_test)["glucose"].mean() != pytest.approx(0, abs=1e-5)


def test_imputation_modes_from_train(split_and_pipeline: tuple) -> None:
    pipeline, X_train, _ = split_and_pipeline
    assert pipeline.imputation_modes_["cholesterol"] == pytest.approx(214.7)
    assert pipeline.transform(X_train)["cholesterol_is_imputed"].eq(1).all()


def test_single_row_equals_batch_row(split_and_pipeline: tuple) -> None:
    pipeline, _, X_test = split_and_pipeline
    batch = pipeline.transform(X_test)
    single = pipeline.transform(X_test.iloc[[3]])
    np.testing.assert_allclose(single.to_numpy()[0], batch.to_numpy()[3])


def test_unknown_nominal_level_raises(split_and_pipeline: tuple) -> None:
    pipeline, _, X_test = split_and_pipeline
    bad = X_test.head(1).copy()
    bad["smoking"] = "Vape"
    with pytest.raises(ValueError, match="smoking"):
        pipeline.transform(bad)


def test_unknown_ordinal_level_raises(split_and_pipeline: tuple) -> None:
    pipeline, _, X_test = split_and_pipeline
    bad = X_test.head(1).copy()
    bad["age_level"] = "Ancient"
    with pytest.raises(ValueError, match="age_level"):
        pipeline.transform(bad)


def test_transform_before_fit_raises(raw_df: pd.DataFrame) -> None:
    X, _ = split_features_targets(raw_df)
    with pytest.raises(RuntimeError):
        ClinicalFeaturePipeline().transform(X)


def test_feature_names_out(split_and_pipeline: tuple) -> None:
    pipeline, _, _ = split_and_pipeline
    assert list(pipeline.get_feature_names_out()) == EXPECTED_FEATURES

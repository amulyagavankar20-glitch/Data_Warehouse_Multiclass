"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Small synthetic dataset matching the real data schema."""
    np.random.seed(42)
    n = 1000

    data = {
        "composite_key": [f"key_{i}" for i in range(n)],
        "age_level": np.random.choice(["Young", "Middle", "Senior"], n),
        "gender": np.random.choice(["Male", "Female"], n),
        "bmi_level": np.random.choice(["Underweight", "Normal", "Overweight", "Obese"], n),
        "smoking": np.random.choice(["Never", "Former", "Current"], n),
        "diabetes": np.random.choice(["Yes", "No"], n),
        "age": np.random.randint(18, 100, n),
        "age_normalized": np.random.uniform(0, 1, n),
        "bmi": np.random.uniform(15, 45, n),
        "hypertension": np.random.choice([0, 1], n),
        "heart_disease": np.random.choice([0, 1], n),
        "HbA1c_level": np.random.uniform(4, 12, n),
        "glucose": np.random.randint(50, 300, n),
        "cholesterol": np.random.uniform(100, 400, n),
        "sleep_hours": np.random.uniform(3, 12, n),
        "triglycerides": np.random.uniform(50, 500, n),
        "physical_activity": np.random.choice(["Low", "Medium", "High"], n),
        "family_history": np.random.choice(["Yes", "No"], n),
        "stress_level": np.random.choice(["Low", "Medium", "High"], n),
        "low_hdl_cholesterol": np.random.choice(["Yes", "No"], n),
        "high_ldl_cholesterol": np.random.choice(["Yes", "No"], n),
        "blood_pressure": np.random.uniform(80, 200, n),
        "high_blood_pressure": np.random.choice(["Yes", "No"], n),
        "sugar_consumption": np.random.choice(["Low", "Medium", "High"], n),
        "crp_level": np.random.uniform(0, 50, n),
        "homocysteine_level": np.random.uniform(5, 40, n),
        "systolic_bp": np.random.uniform(90, 200, n),
        "diastolic_bp": np.random.uniform(50, 120, n),
        "alcohol_intake": np.random.uniform(0, 10, n),
        "salt_intake": np.random.uniform(0, 10, n),
        "heart_rate": np.random.uniform(50, 120, n),
        "hdl": np.random.uniform(20, 80, n),
        "ldl": np.random.uniform(30, 200, n),
        "education_level": np.random.choice(["HighSchool", "Bachelor", "Master", "PhD"], n),
        "employment_status": np.random.choice(["Employed", "Unemployed", "Retired"], n),
        "source_dataset": np.random.choice(["diabetes", "hypertension", "heart_disease"], n),
        "disease_flags": np.random.choice(["0,0,0", "1,0,0", "0,1,0", "0,0,1", "1,1,0", "1,0,1", "0,1,1", "1,1,1"], n),
        "sublabel": np.random.choice(["N", "DI", "HY", "HT", "DI_HY", "DI_HT", "HT_HY", "DI_HT_HY"], n, p=[0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.08, 0.02]),
        "label": np.where(np.random.rand(n) < 0.5, "Normal", "Abnormal"),
    }

    return pd.DataFrame(data)


@pytest.fixture
def cleaned_synthetic_data(synthetic_data: pd.DataFrame) -> pd.DataFrame:
    """Cleaned synthetic data (after clean_data)."""
    from src.data.loader import clean_data
    return clean_data(synthetic_data)


@pytest.fixture
def small_features_df() -> pd.DataFrame:
    """Small DataFrame for model tests."""
    np.random.seed(42)
    n = 100
    data = {
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
        "feat_4": np.random.randn(n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_split(small_features_df: pd.DataFrame) -> object:
    """Mock SplitData object for testing."""
    from src.data.splitter import SplitData

    n = len(small_features_df)
    y = np.random.choice([0, 1, 2], n)

    le = LabelEncoder()
    y_encoded = le.fit_transform(np.random.choice(["N", "DI", "HY"], n))

    n_train = int(n * 0.8)
    return SplitData(
        X_train=small_features_df.iloc[:n_train].reset_index(drop=True),
        X_test=small_features_df.iloc[n_train:].reset_index(drop=True),
        y_train=y_encoded[:n_train],
        y_test=y_encoded[n_train:],
        label_encoder=le,
        feature_names=list(small_features_df.columns),
        continuous_cols=["feat_1", "feat_2", "feat_3", "feat_4"],
        ordinal_cols=[],
        nominal_cols=[],
    )


@pytest.fixture
def tmp_mlflow_run(tmp_path):
    """Set up a temporary MLflow run for testing."""
    import mlflow

    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test-experiment")
    mlflow.start_run()
    yield
    mlflow.end_run()

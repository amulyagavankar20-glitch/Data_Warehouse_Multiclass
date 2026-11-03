"""Central configuration: runtime settings plus the frozen feature contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings, overridable via environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    raw_data_path: Path = PROJECT_ROOT / "data" / "Data Warehouse Multiclass.csv"
    model_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    mlflow_tracking_uri: str = (PROJECT_ROOT / "mlruns").as_uri()
    mlflow_experiment_name: str = "disease-anomaly"
    mlflow_model_name: str = "disease-anomaly-classifier"
    mlflow_champion_alias: str = "champion"
    mlflow_candidate_alias: str = "candidate"
    enable_mlflow: bool = True

    model_uri: str | None = None

    random_state: int = 42
    val_size: float = 0.15
    test_size: float = 0.15

    lgb_num_boost_round: int = 600
    lgb_early_stopping_rounds: int = 30
    threshold_clip_min: float = 0.05
    threshold_clip_max: float = 0.60

    isolation_forest_n_estimators: int = 200
    isolation_forest_contamination: float = 0.05
    ocsvm_nu: float = 0.05
    ocsvm_max_train_samples: int = 10_000

    min_test_macro_auc: float = 0.90
    psi_alert_threshold: float = 0.25

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"

    @property
    def bundle_path(self) -> Path:
        return self.model_dir / "bundle.joblib"

    @property
    def lgb_params(self) -> dict[str, Any]:
        return {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 7,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.5,
            "lambda_l2": 3.0,
            "min_child_samples": 20,
            "verbose": -1,
            "seed": self.random_state,
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": -1,
        }

    TARGET_COLUMN: ClassVar[str] = "sublabel"
    NORMAL_CLASS: ClassVar[str] = "N"

    LEAKAGE_COLUMNS: ClassVar[list[str]] = [
        "label",
        "sublabel",
        "disease_flags",
        "diabetes",
        "hypertension",
        "heart_disease",
        "composite_key",
        "age_normalized",
        "source_dataset",
    ]

    DISEASE_FLAG_NAMES: ClassVar[list[str]] = [
        "Diabetes",
        "Heart_Disease",
        "Hypertension",
    ]
    SUBLABEL_TOKENS: ClassVar[dict[str, str]] = {
        "Diabetes": "DI",
        "Heart_Disease": "HT",
        "Hypertension": "HY",
    }

    ORDINAL_MAPS: ClassVar[dict[str, dict[str, int]]] = {
        "age_level": {"Young": 0, "Middle": 1, "Mature": 2, "Senior": 3},
        "bmi_level": {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3},
        "stress_level": {"Low": 0, "Medium": 1, "High": 2},
        "education_level": {"Primary": 0, "Secondary": 1, "Tertiary": 2},
    }

    NOMINAL_LEVELS: ClassVar[dict[str, list[str]]] = {
        "gender": ["Female", "Male"],
        "smoking": ["Current", "Former", "Never"],
        "physical_activity": ["High", "Low", "Moderate"],
        "family_history": ["No", "Yes"],
        "low_hdl_cholesterol": ["No", "Yes"],
        "high_ldl_cholesterol": ["No", "Yes"],
        "high_blood_pressure": ["No", "Yes"],
        "sugar_consumption": ["High", "Low", "Medium"],
        "employment_status": ["Employed", "Retired", "Unemployed"],
    }

    BASE_NUMERIC_COLUMNS: ClassVar[list[str]] = [
        "age",
        "bmi",
        "HbA1c_level",
        "glucose",
        "cholesterol",
        "sleep_hours",
        "triglycerides",
        "blood_pressure",
        "crp_level",
        "homocysteine_level",
        "systolic_bp",
        "diastolic_bp",
        "alcohol_intake",
        "salt_intake",
        "heart_rate",
        "hdl",
        "ldl",
    ]

    ENGINEERED_COLUMNS: ClassVar[list[str]] = [
        "pulse_pressure",
        "mean_arterial_pressure",
        "cholesterol_ratio",
        "glucose_triglyceride_index",
    ]

    IMPUTATION_CHECK_COLUMNS: ClassVar[list[str]] = [
        "age",
        "glucose",
        "HbA1c_level",
        "heart_rate",
        "diastolic_bp",
        "hdl",
        "systolic_bp",
        "ldl",
        "salt_intake",
        "cholesterol",
        "alcohol_intake",
        "triglycerides",
    ]

    RAW_FEATURE_ORDER: ClassVar[list[str]] = [
        "age_level",
        "gender",
        "bmi_level",
        "smoking",
        "age",
        "bmi",
        "HbA1c_level",
        "glucose",
        "cholesterol",
        "sleep_hours",
        "triglycerides",
        "physical_activity",
        "family_history",
        "stress_level",
        "low_hdl_cholesterol",
        "high_ldl_cholesterol",
        "blood_pressure",
        "high_blood_pressure",
        "sugar_consumption",
        "crp_level",
        "homocysteine_level",
        "systolic_bp",
        "diastolic_bp",
        "alcohol_intake",
        "salt_intake",
        "heart_rate",
        "hdl",
        "ldl",
        "education_level",
        "employment_status",
    ]


settings = Settings()

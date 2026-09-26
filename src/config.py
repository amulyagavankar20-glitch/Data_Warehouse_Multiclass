"""Configuration management using Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_path: Path = Path("data/raw/Data Warehouse Multiclass.csv")
    processed_data_dir: Path = Path("data/processed")

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_backend_store_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_root: str = "./mlruns"
    mlflow_experiment_name: str = "disease-anomaly"
    mlflow_model_name: str = "disease-anomaly-classifier"
    mlflow_model_stage: str = "Production"

    # Model parameters
    random_state: int = 42
    test_size: float = 0.2
    isolation_forest_contamination: float = 0.05
    isolation_forest_n_estimators: int = 200
    rf_n_estimators: int = 200
    autoencoder_encoding_dim: int = 16
    autoencoder_dropout: float = 0.2
    autoencoder_patience: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Environment
    environment: str = "development"

    # Column configuration
    DROP_COLUMNS: ClassVar[list[str]] = [
        "composite_key",
        "source_dataset",
        "disease_flags",
        "label",
        "sublabel",
        "hypertension",
        "heart_disease",
        "diabetes",
    ]

    ORDINAL_COLUMNS: ClassVar[list[str]] = [
        "age_level",
        "bmi_level",
        "education_level",
        "stress_level",
    ]

    NOMINAL_COLUMNS: ClassVar[list[str]] = [
        "gender",
        "smoking",
        "physical_activity",
        "family_history",
        "low_hdl_cholesterol",
        "high_ldl_cholesterol",
        "high_blood_pressure",
        "sugar_consumption",
        "employment_status",
    ]

    CONTINUOUS_COLUMNS: ClassVar[list[str]] = [
        "age",
        "age_normalized",
        "bmi",
        "HbA1c_level",
        "glucose",
        "cholesterol",
        "sleep_hours",
        "triglycerides",
        "crp_level",
        "homocysteine_level",
        "systolic_bp",
        "diastolic_bp",
        "blood_pressure",
        "alcohol_intake",
        "salt_intake",
        "heart_rate",
        "hdl",
        "ldl",
    ]

    TARGET_COLUMN: ClassVar[str] = "sublabel"

    MODELCONFIG_PATH: ClassVar[Path] = Path("src/models/model_config.json")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

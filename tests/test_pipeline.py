"""End-to-end integration tests for the training pipeline."""

from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_data(tmp_path):
    """Create temporary synthetic data file for pipeline testing."""
    np.random.seed(42)
    n = 500
    data = {
        "composite_key": [f"key_{i}" for i in range(n)],
        "age_level": np.random.choice(["Young", "Middle", "Senior"], n),
        "gender": np.random.choice(["Male", "Female"], n),
        "bmi_level": np.random.choice(["Normal", "Overweight", "Obese"], n),
        "smoking": np.random.choice(["Never", "Current"], n),
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
        "stress_level": np.random.choice(["Low", "High"], n),
        "low_hdl_cholesterol": np.random.choice(["Yes", "No"], n),
        "high_ldl_cholesterol": np.random.choice(["Yes", "No"], n),
        "blood_pressure": np.random.uniform(80, 200, n),
        "high_blood_pressure": np.random.choice(["Yes", "No"], n),
        "sugar_consumption": np.random.choice(["Low", "High"], n),
        "crp_level": np.random.uniform(0, 50, n),
        "homocysteine_level": np.random.uniform(5, 40, n),
        "systolic_bp": np.random.uniform(90, 200, n),
        "diastolic_bp": np.random.uniform(50, 120, n),
        "alcohol_intake": np.random.uniform(0, 10, n),
        "salt_intake": np.random.uniform(0, 10, n),
        "heart_rate": np.random.uniform(50, 120, n),
        "hdl": np.random.uniform(20, 80, n),
        "ldl": np.random.uniform(30, 200, n),
        "education_level": np.random.choice(["HighSchool", "Bachelor"], n),
        "employment_status": np.random.choice(["Employed", "Retired"], n),
        "source_dataset": np.random.choice(["diabetes", "hypertension"], n),
        "disease_flags": np.random.choice(["0,0,0", "1,0,0", "0,1,0", "1,1,0"], n),
        "sublabel": np.random.choice(
            ["N", "DI", "HY", "HT", "DI_HY", "DI_HT", "HT_HY", "DI_HT_HY"],
            n, p=[0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.08, 0.02],
        ),
        "label": np.where(np.random.rand(n) < 0.5, "Normal", "Abnormal"),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestPipelineComponents:
    """Integration tests for pipeline components working together."""

    def test_data_loading_and_cleaning(self, temp_data):
        from src.data.loader import clean_data, load_raw_data
        df = load_raw_data(str(temp_data))
        assert len(df) == 500
        cleaned = clean_data(df)
        assert "disease_flags" not in cleaned.columns
        assert "hypertension" not in cleaned.columns
        assert "diabetes" not in cleaned.columns

    def test_validation_passes(self, temp_data):
        from src.data.loader import load_raw_data
        from src.data.validator import validate_data
        df = load_raw_data(str(temp_data))
        result = validate_data(df)
        assert result.passed

    def test_split_and_encode(self, temp_data):
        from src.data.loader import clean_data, load_raw_data
        from src.data.splitter import split_data
        from src.features.engineering import fit_transform_train
        df = load_raw_data(str(temp_data))
        cleaned = clean_data(df)
        split = split_data(cleaned, test_size=0.2, random_state=42)
        preprocessor, X_train_enc, X_test_enc = fit_transform_train(split)
        assert X_train_enc.shape[0] > 0
        assert X_test_enc.shape[0] > 0

    def test_anomaly_models_fit(self, temp_data):
        from src.data.loader import clean_data, load_raw_data
        from src.data.splitter import get_normal_cohort, split_data
        from src.features.engineering import fit_transform_train
        from src.models.anomaly.autoencoder import DeepAutoencoderModel
        from src.models.anomaly.isolation_forest import IsolationForestModel
        from src.models.anomaly.one_class_svm import OneClassSVMModel
        df = load_raw_data(str(temp_data))
        cleaned = clean_data(df)
        split = split_data(cleaned, test_size=0.2, random_state=42)
        _, X_train_enc, _ = fit_transform_train(split)
        normal = get_normal_cohort(X_train_enc, split.y_train, split.label_encoder)

        iso = IsolationForestModel(n_estimators=10, random_state=42)
        iso.fit(normal)
        assert iso.is_fitted

        svm = OneClassSVMModel()
        svm.fit(normal)
        assert svm.is_fitted

        ae = DeepAutoencoderModel(
            input_dim=X_train_enc.shape[1], encoding_dim=8,
            max_epochs=5, batch_size=32, patience=3,
        )
        ae.fit(normal)
        assert ae.is_fitted

    def test_classifier_fits_and_predicts(self, temp_data):
        from src.data.loader import clean_data, load_raw_data
        from src.data.splitter import split_data
        from src.features.engineering import fit_transform_train
        from src.models.classifier import HybridClassifier
        df = load_raw_data(str(temp_data))
        cleaned = clean_data(df)
        split = split_data(cleaned, test_size=0.2, random_state=42)
        _, X_train_enc, _ = fit_transform_train(split)

        X_train_meta = X_train_enc.copy()
        X_train_meta["iso_score"] = np.random.randn(len(X_train_enc))
        X_train_meta["ae_mse"] = np.random.rand(len(X_train_enc))

        clf = HybridClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train_meta, split.y_train)
        preds = clf.predict(X_train_meta)
        assert len(preds) == len(X_train_meta)


class TestPrefectFlow:
    """Tests for the full Prefect training flow."""

    def test_full_flow_with_mlflow(self, temp_data, tmp_path):
        """Run the full Prefect flow with temporary data and MLflow tracking."""
        from flows.training_flow import run_training
        from src.config import settings

        mlflow_db = tmp_path / "mlflow.db"
        artifact_root = str(tmp_path / "mlruns")
        mlflow_uri = f"sqlite:///{mlflow_db}"

        # Patch settings for test execution
        with patch.object(settings, "raw_data_path", temp_data), \
             patch.object(settings, "mlflow_tracking_uri", mlflow_uri), \
             patch.object(settings, "mlflow_backend_store_uri", mlflow_uri), \
             patch.object(settings, "mlflow_artifact_root", artifact_root), \
             patch.object(settings, "mlflow_experiment_name", "test-flow-exp"), \
             patch.object(settings, "mlflow_model_name", "test-flow-model"), \
             patch.object(settings, "mlflow_model_stage", "Staging"), \
             patch.object(settings, "isolation_forest_n_estimators", 10), \
             patch.object(settings, "autoencoder_patience", 3), \
             patch.object(settings, "autoencoder_encoding_dim", 8), \
             patch.object(settings, "rf_n_estimators", 10), \
             patch.object(settings, "test_size", 0.2):

            result = run_training()

        assert "accuracy" in result
        assert "f1_macro" in result
        assert 0.0 <= result["accuracy"] <= 1.0

        # Verify MLflow run was created
        client = mlflow.tracking.MlflowClient(mlflow_uri)
        experiments = client.search_experiments()
        exp_names = [e.name for e in experiments]
        assert "test-flow-exp" in exp_names

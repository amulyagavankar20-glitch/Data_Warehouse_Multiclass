"""Tests for FastAPI endpoints."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.api.main import app


@pytest.fixture
def api_client(tmp_path):
    """Create a test client with a mock model bundle."""
    np.random.seed(42)
    n = 200
    feature_cols = [
        "age", "age_normalized", "bmi", "HbA1c_level", "glucose",
        "cholesterol", "sleep_hours", "triglycerides",
        "crp_level", "homocysteine_level", "systolic_bp",
        "diastolic_bp", "blood_pressure", "alcohol_intake",
        "salt_intake", "heart_rate", "hdl", "ldl",
        "age_level", "gender", "bmi_level", "smoking",
        "physical_activity", "family_history", "stress_level",
        "low_hdl_cholesterol", "high_ldl_cholesterol",
        "high_blood_pressure", "sugar_consumption",
        "education_level", "employment_status",
    ]
    numeric_cols = [c for c in feature_cols if c not in [
        "age_level", "gender", "bmi_level", "smoking",
        "physical_activity", "family_history", "stress_level",
        "low_hdl_cholesterol", "high_ldl_cholesterol",
        "high_blood_pressure", "sugar_consumption",
        "education_level", "employment_status",
    ]]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    X = pd.DataFrame({
        col: np.random.randn(n) if col in numeric_cols
        else np.random.choice(["a", "b", "c"], n)
        for col in feature_cols
    })
    y = np.random.choice(["N", "DI", "HY"], n)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    X_encoded = preprocessor.fit_transform(X)
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_cols)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_encoded_df, y_enc)

    from src.models.anomaly.autoencoder import DeepAutoencoderModel
    from src.models.anomaly.isolation_forest import IsolationForestModel
    from src.models.anomaly.one_class_svm import OneClassSVMModel

    iso_model = IsolationForestModel(n_estimators=5, random_state=42)
    iso_model.fit(X_encoded_df)

    ocs_model = OneClassSVMModel()
    ocs_model.fit(X_encoded_df)

    ae_model = DeepAutoencoderModel(
        input_dim=X_encoded_df.shape[1], encoding_dim=8,
        max_epochs=2, batch_size=32,
    )
    ae_model.fit(X_encoded_df)

    import src.api.main as main_module
    main_module.MODEL_LOADED = True
    main_module.MODEL_OBJECT = {
        "model": model,
        "classes": list(le.classes_),
        "preprocessor": preprocessor,
        "label_encoder": le,
        "feature_names": feature_cols,
        "iso_forest": iso_model,
        "ocsvm": ocs_model,
        "autoencoder": ae_model,
    }

    client = TestClient(app)
    yield client

    main_module.MODEL_LOADED = False
    main_module.MODEL_OBJECT = {}


def _valid_payload():
    """Return a valid patient features payload."""
    return {
        "age": 45, "age_normalized": 0.45, "bmi": 25.5,
        "HbA1c_level": 6.0, "glucose": 120, "cholesterol": 200,
        "sleep_hours": 7.5, "triglycerides": 150,
        "crp_level": 3.0, "homocysteine_level": 12,
        "systolic_bp": 120, "diastolic_bp": 80,
        "blood_pressure": 120, "alcohol_intake": 2,
        "salt_intake": 5, "heart_rate": 70, "hdl": 50, "ldl": 100,
        "age_level": "Middle", "gender": "Female",
        "bmi_level": "Normal", "smoking": "Never",
        "physical_activity": "Medium", "family_history": "No",
        "stress_level": "Medium", "low_hdl_cholesterol": "No",
        "high_ldl_cholesterol": "No", "high_blood_pressure": "No",
        "sugar_consumption": "Medium", "education_level": "Bachelor",
        "employment_status": "Employed",
    }


class TestHealthEndpoint:
    def test_health_check(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestMetricsEndpoint:
    def test_metrics(self, api_client: TestClient) -> None:
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert b"http_requests_total" in response.content


class TestPredictEndpoint:
    def test_predict_success(self, api_client: TestClient) -> None:
        response = api_client.post("/predict", json=_valid_payload())
        assert response.status_code == 200
        data = response.json()
        assert "predicted_sublabel" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_validation_error(self, api_client: TestClient) -> None:
        payload = {"age": "invalid", "bmi": 25}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_field(self, api_client: TestClient) -> None:
        payload = {"age": 45}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422


class TestAnomalyScoreEndpoint:
    def test_anomaly_score_success(self, api_client: TestClient) -> None:
        response = api_client.post("/anomaly-score", json=_valid_payload())
        assert response.status_code == 200
        data = response.json()
        assert "iso_score" in data
        assert "ae_mse" in data
        assert "is_anomaly_isolation" in data
        assert "is_anomaly_svm" in data

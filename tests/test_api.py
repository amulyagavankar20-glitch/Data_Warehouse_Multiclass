"""Tests for the FastAPI serving layer against a really-trained bundle."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.api.schemas import PatientFeatures
from src.serving.bundle import ModelBundle

PAYLOAD: dict[str, Any] = PatientFeatures.model_config["json_schema_extra"]["example"]  # type: ignore[index]


@pytest.fixture
def client(
    trained_bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        api, "load_bundle", lambda: ModelBundle.load(trained_bundle_path)
    )
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def offline_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(api, "load_bundle", lambda: None)
    with TestClient(api.app) as c:
        yield c


class TestOps:
    def test_health(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["model_version"]

    def test_ready(self, client: TestClient) -> None:
        assert client.get("/ready").status_code == 200

    def test_metrics_prometheus_format(self, client: TestClient) -> None:
        client.post("/predict", json=PAYLOAD)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert 'endpoint="/predict"' in resp.text
        assert "model_predictions_total" in resp.text

    def test_contract(self, client: TestClient) -> None:
        body = client.get("/model/contract").json()
        assert body["input_feature_count"] == 50
        assert len(body["target_classes"]) == 8


class TestOffline:
    def test_health_degraded(self, offline_client: TestClient) -> None:
        assert offline_client.get("/health").json()["model_loaded"] is False

    def test_not_ready(self, offline_client: TestClient) -> None:
        assert offline_client.get("/ready").status_code == 503

    def test_predict_503(self, offline_client: TestClient) -> None:
        assert offline_client.post("/predict", json=PAYLOAD).status_code == 503


class TestPredict:
    def test_predict(self, client: TestClient) -> None:
        resp = client.post("/predict", json=PAYLOAD)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["predicted_sublabel"] in body["probabilities"]
        assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
        assert set(body["disease_probabilities"]) == {
            "Diabetes",
            "Heart_Disease",
            "Hypertension",
        }

    def test_paediatric_patient_accepted(self, client: TestClient) -> None:
        assert client.post("/predict", json={**PAYLOAD, "age": 6}).status_code == 200

    def test_batch(self, client: TestClient) -> None:
        resp = client.post("/predict/batch", json={"records": [PAYLOAD, PAYLOAD]})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 2

    def test_unknown_category_rejected(self, client: TestClient) -> None:
        resp = client.post("/predict", json={**PAYLOAD, "physical_activity": "Medium"})
        assert resp.status_code == 422

    def test_missing_field(self, client: TestClient) -> None:
        payload = {k: v for k, v in PAYLOAD.items() if k != "glucose"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_extra_leakage_field_rejected(self, client: TestClient) -> None:
        assert (
            client.post("/predict", json={**PAYLOAD, "diabetes": "Yes"}).status_code
            == 422
        )

    def test_out_of_range(self, client: TestClient) -> None:
        assert (
            client.post("/predict", json={**PAYLOAD, "glucose": 5000}).status_code
            == 422
        )

    def test_pipeline_value_error_maps_to_422(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_: Any) -> None:
            raise ValueError("bad input")

        assert api.STATE.bundle is not None
        monkeypatch.setattr(api.STATE.bundle, "predict", boom)
        monkeypatch.setattr(api.STATE.bundle, "score_anomaly", boom)
        assert client.post("/predict", json=PAYLOAD).status_code == 422
        assert client.post("/anomaly-score", json=PAYLOAD).status_code == 422


class TestAnomaly:
    def test_anomaly_score(self, client: TestClient) -> None:
        body = client.post("/anomaly-score", json=PAYLOAD).json()
        assert {
            "iso_score",
            "ocsvm_score",
            "is_anomaly_isolation",
            "is_anomaly_svm",
        } <= set(body)

    def test_extreme_patient_flagged(self, client: TestClient) -> None:
        extreme = {
            **PAYLOAD,
            "glucose": 650,
            "HbA1c_level": 19,
            "systolic_bp": 290,
            "crp_level": 280,
            "bmi": 100,
        }
        body = client.post("/anomaly-score", json=extreme).json()
        assert body["is_anomaly_isolation"] or body["is_anomaly_svm"]

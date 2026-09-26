"""FastAPI application for disease anomaly detection serving."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
import prometheus_client
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram

from src.api.schemas import (
    AnomalyScoreResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.config import settings

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)
MODEL_LOADED = False
MODEL_OBJECT: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED, MODEL_OBJECT
    logger.info("Starting API service...")
    try:
        MODEL_OBJECT = joblib.load("models/model.joblib")
        MODEL_LOADED = True
        logger.info("Model loaded successfully from %s", "models/model.joblib")
    except FileNotFoundError:
        logger.warning("No model found at models/model.joblib. API will serve in offline mode.")
        MODEL_LOADED = False
        MODEL_OBJECT = {}
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        MODEL_LOADED = False
        MODEL_OBJECT = {}
    yield
    logger.info("Shutting down API service...")


app = FastAPI(
    title="Disease Anomaly Early-Warning System",
    description="MLOps API for multi-class disease detection and anomaly scoring",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    method = request.method
    path = request.url.path

    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(method=method, endpoint=path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)

    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_LOADED,
        model_name=settings.mlflow_model_name,
    )


@app.get("/metrics")
async def metrics():
    return JSONResponse(
        content=prometheus_client.generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4",
    )


def _features_to_dataframe(features: PatientFeatures) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame for model input."""
    data = features.model_dump()
    return pd.DataFrame([data])


def _preprocess_features(X_raw: pd.DataFrame) -> pd.DataFrame | np.ndarray:
    """Apply the loaded preprocessor to raw features.

    Falls back to raw DataFrame if no preprocessor is available.
    """
    preprocessor = MODEL_OBJECT.get("preprocessor")
    feature_names = MODEL_OBJECT.get("feature_names")

    if feature_names:
        available = [c for c in feature_names if c in X_raw.columns]
        X_raw = X_raw[available]

    if preprocessor is not None:
        try:
            return preprocessor.transform(X_raw)
        except Exception:
            pass

    return X_raw


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PatientFeatures):
    if not MODEL_LOADED or "model" not in MODEL_OBJECT:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X_raw = _features_to_dataframe(features)
        X_encoded = _preprocess_features(X_raw)

        model = MODEL_OBJECT["model"]
        prediction = model.predict(X_encoded)[0]
        probabilities = model.predict_proba(X_encoded)[0]
        class_labels = MODEL_OBJECT["classes"]

        pred_label = class_labels[prediction]
        pred_proba = float(probabilities.max())

        proba_dict = {
            cls: float(probabilities[i])
            for i, cls in enumerate(class_labels)
        }

        return PredictionResponse(
            predicted_sublabel=pred_label,
            confidence=pred_proba,
            probabilities=proba_dict,
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anomaly-score", response_model=AnomalyScoreResponse)
async def anomaly_score(features: PatientFeatures):
    if not MODEL_LOADED or "iso_forest" not in MODEL_OBJECT:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X_raw = _features_to_dataframe(features)
        X_encoded = _preprocess_features(X_raw)
        X_values = X_encoded.values if isinstance(X_encoded, pd.DataFrame) else X_encoded

        iso_model: Any = MODEL_OBJECT["iso_forest"]
        ocs_model: Any = MODEL_OBJECT["ocsvm"]
        ae_model: Any = MODEL_OBJECT["autoencoder"]

        iso_score = float(iso_model.decision_function(X_values)[0])
        ae_mse = float(ae_model.get_anomaly_score(X_values)[0])

        is_anomaly_if = bool(iso_model.predict(X_values)[0] == -1)
        is_anomaly_svm = bool(ocs_model.predict(X_values)[0] == -1)

        return AnomalyScoreResponse(
            iso_score=iso_score,
            ae_mse=ae_mse,
            is_anomaly_isolation=is_anomaly_if,
            is_anomaly_svm=is_anomaly_svm,
        )
    except Exception as e:
        logger.error("Anomaly score error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

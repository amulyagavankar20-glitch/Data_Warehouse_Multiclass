"""FastAPI serving layer for sublabel prediction and early-warning scoring."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.routing import Match

from src.api.schemas import (
    AnomalyScoreResponse,
    BatchPredictionResponse,
    BatchRequest,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.config import settings
from src.serving.bundle import ModelBundle

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)
PREDICTIONS = Counter(
    "model_predictions_total", "Predicted sublabels served", ["sublabel"]
)
ANOMALY_FLAGS = Counter(
    "model_anomaly_flags_total", "Records flagged as anomalous", ["detector"]
)
MODEL_INFO = Gauge("model_info", "Loaded model version", ["version"])


@dataclass
class ServiceState:
    bundle: ModelBundle | None = None


STATE = ServiceState()


def load_bundle() -> ModelBundle | None:
    """Prefer an MLflow registry URI; fall back to the local bundle file."""
    try:
        if settings.model_uri:
            return ModelBundle.from_mlflow(settings.model_uri)
        if settings.bundle_path.exists():
            return ModelBundle.load(settings.bundle_path)
        logger.warning("No model at %s; serving in degraded mode", settings.bundle_path)
    except Exception:
        logger.exception("Failed to load model bundle")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    STATE.bundle = load_bundle()
    if STATE.bundle is not None:
        MODEL_INFO.labels(version=STATE.bundle.version).set(1)
        logger.info("Loaded model %s", STATE.bundle.version)
    yield
    STATE.bundle = None


app = FastAPI(
    title="Disease Anomaly Early-Warning System",
    description=(
        "Multi-class comorbidity prediction and healthy-baseline anomaly scoring"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _route_template(request: Request) -> str:
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return str(getattr(route, "path", request.url.path))
    return "unmatched"


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = _route_template(request)
    REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(
        time.perf_counter() - start
    )
    return response


def _require_bundle() -> ModelBundle:
    if STATE.bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return STATE.bundle


def _to_frame(records: list[PatientFeatures]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


def _predict(records: list[PatientFeatures]) -> list[PredictionResponse]:
    bundle = _require_bundle()
    try:
        results = bundle.predict(_to_frame(records))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for r in results:
        PREDICTIONS.labels(r["predicted_sublabel"]).inc()
    return [PredictionResponse(**r, model_version=bundle.version) for r in results]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up (a model may not be loaded)."""
    bundle = STATE.bundle
    return HealthResponse(
        status="healthy",
        model_loaded=bundle is not None,
        model_name=settings.mlflow_model_name,
        model_version=bundle.version if bundle else None,
    )


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Readiness: 503 until a model is loaded, so traffic is held back."""
    bundle = _require_bundle()
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_name=settings.mlflow_model_name,
        model_version=bundle.version,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/contract")
def contract() -> dict[str, object]:
    return _require_bundle().serving_contract()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    return _predict([features])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchRequest) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=_predict(batch.records))


@app.post("/anomaly-score", response_model=AnomalyScoreResponse)
def anomaly_score(features: PatientFeatures) -> AnomalyScoreResponse:
    bundle = _require_bundle()
    try:
        result = bundle.score_anomaly(_to_frame([features]))[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result["is_anomaly_isolation"]:
        ANOMALY_FLAGS.labels("isolation_forest").inc()
    if result["is_anomaly_svm"]:
        ANOMALY_FLAGS.labels("one_class_svm").inc()
    return AnomalyScoreResponse(**result, model_version=bundle.version)

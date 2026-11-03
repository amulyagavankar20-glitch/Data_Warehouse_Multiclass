# API Specification

**Service:** Disease Anomaly Early-Warning System · **Version:** 1.0.0 ·
**Base URL (local):** `http://localhost:8000`

> [!TIP]
> The running service publishes an interactive OpenAPI UI at `/docs` and the
> machine-readable schema at `/openapi.json`. This document is the
> human-readable reference.

## Endpoints

| Method | Path | Purpose | Success | Needs model |
| :-- | :-- | :-- | :-: | :-: |
| `GET` | [`/health`](#get-health) | Liveness probe | `200` | No |
| `GET` | [`/ready`](#get-ready) | Readiness probe | `200` | Yes |
| `GET` | [`/metrics`](#get-metrics) | Prometheus exposition | `200` | No |
| `GET` | [`/model/contract`](#get-modelcontract) | Contract of the loaded model | `200` | Yes |
| `POST` | [`/predict`](#post-predict) | Predict one patient | `200` | Yes |
| `POST` | [`/predict/batch`](#post-predictbatch) | Predict up to 1,000 patients | `200` | Yes |
| `POST` | [`/anomaly-score`](#post-anomaly-score) | Early-warning deviation score | `200` | Yes |

## Conventions

- Request and response bodies are JSON (`Content-Type: application/json`).
- Probabilities are floats in `[0, 1]`. Per-class probabilities sum to 1.
- Every inference response carries `model_version`
  (`<UTC timestamp>-<git sha>`), so predictions are traceable to an MLflow run.
- The API is stateless, so requests can go to any replica.

> [!WARNING]
> The service has no built-in authentication. Deploy it behind an API gateway
> or service mesh that enforces authN/Z and TLS before exposing it outside a
> trusted network. Requests contain health data.

---

## `GET /health`

Liveness. Returns `200` whenever the process is up, including when no model is
loaded.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "disease-anomaly-classifier",
  "model_version": "20261103T140735Z-9705890"
}
```

## `GET /ready`

Readiness. Returns the same body as `/health` with `"status": "ready"`, or
`503` if no model is loaded. Use it as the load-balancer and Kubernetes
readiness check.

## `GET /metrics`

Prometheus text format (`text/plain; version=0.0.4`).

| Metric | Type | Labels | Meaning |
| :-- | :-- | :-- | :-- |
| `http_requests_total` | counter | `method`, `endpoint`, `status` | Requests by route template |
| `http_request_duration_seconds` | histogram | `method`, `endpoint` | Request latency |
| `model_predictions_total` | counter | `sublabel` | Predictions served per class |
| `model_anomaly_flags_total` | counter | `detector` | Records flagged anomalous |
| `model_info` | gauge | `version` | `1` for the loaded model version |

> [!NOTE]
> `endpoint` is the **route template** (for example `/predict`), not the raw
> URL, which keeps label cardinality bounded.

## `GET /model/contract`

Returns the serving contract of the loaded model: raw input columns, the 50
engineered features, scaled columns, target classes, decision thresholds
(`null` when arg-max is used) and the test metrics recorded at training time.

---

## Request schema: `PatientFeatures`

One raw clinical record. **Unknown fields are rejected**, including the
quarantined leakage columns such as `diabetes` and `disease_flags`.

### Categorical fields

| Field | Allowed values |
| :-- | :-- |
| `age_level` | `Young`, `Middle`, `Mature`, `Senior` |
| `gender` | `Female`, `Male` |
| `bmi_level` | `Underweight`, `Normal`, `Overweight`, `Obese` |
| `smoking` | `Current`, `Former`, `Never` |
| `physical_activity` | `High`, `Low`, `Moderate` |
| `family_history` | `Yes`, `No` |
| `stress_level` | `Low`, `Medium`, `High` |
| `low_hdl_cholesterol` | `Yes`, `No` |
| `high_ldl_cholesterol` | `Yes`, `No` |
| `high_blood_pressure` | `Yes`, `No` |
| `sugar_consumption` | `High`, `Low`, `Medium` |
| `education_level` | `Primary`, `Secondary`, `Tertiary` |
| `employment_status` | `Employed`, `Retired`, `Unemployed` |

### Numeric fields

Bounds are physiological limits, not the training range, so paediatric and
extreme-but-real patients are accepted.

| Field | Unit / meaning | Min | Max |
| :-- | :-- | --: | --: |
| `age` | years | 0 | 120 |
| `bmi` | kg/m² | 8 | 110 |
| `HbA1c_level` | % | 2 | 20 |
| `glucose` | mg/dL | 20 | 700 |
| `cholesterol` | total, mg/dL | 50 | 700 |
| `sleep_hours` | hours/day | 0 | 24 |
| `triglycerides` | mg/dL | 10 | 3000 |
| `blood_pressure` | mmHg | 40 | 300 |
| `crp_level` | mg/L | 0 | 300 |
| `homocysteine_level` | µmol/L | 0 | 150 |
| `systolic_bp` | mmHg | 50 | 300 |
| `diastolic_bp` | mmHg | 20 | 200 |
| `alcohol_intake` | units | 0 | 100 |
| `salt_intake` | g/day | 0 | 60 |
| `heart_rate` | bpm | 20 | 250 |
| `hdl` | mg/dL | 5 | 200 |
| `ldl` | mg/dL | 10 | 400 |

All 30 fields are required. A complete example is in
[`examples/patient.json`](examples/patient.json).

---

## `POST /predict`

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d @docs/examples/patient.json
```

**Response `200`: `PredictionResponse`**

```json
{
  "predicted_sublabel": "DI",
  "confidence": 0.5368,
  "probabilities": {
    "DI": 0.5368, "DI_HT": 0.0001, "DI_HT_HY": 0.0032, "DI_HY": 0.3490,
    "HT": 0.0004, "HT_HY": 0.0001, "HY": 0.0388, "N": 0.0717
  },
  "disease_probabilities": {
    "Diabetes": 0.8891,
    "Heart_Disease": 0.0038,
    "Hypertension": 0.3910
  },
  "model_version": "20261103T140735Z-9705890"
}
```

| Field | Type | Description |
| :-- | :-- | :-- |
| `predicted_sublabel` | string | Decided class (arg-max, or threshold-normalised when calibration is active) |
| `confidence` | float | Probability of the predicted class |
| `probabilities` | object | Probability of each of the 8 sublabels |
| `disease_probabilities` | object | Marginal probability of each condition: the sum over every sublabel containing it |
| `model_version` | string | Version of the bundle that served the request |

> [!TIP]
> In the example, `DI` wins with 0.54, but `DI_HY` holds 0.35 and
> `P(Hypertension)` is 0.39. When `confidence` is low, show
> `disease_probabilities` to clinicians rather than only the top sublabel.

## `POST /predict/batch`

```json
{ "records": [ { "...PatientFeatures..." }, { "...PatientFeatures..." } ] }
```

Between 1 and 1,000 records. The response is `{ "predictions": [PredictionResponse, …] }`,
in request order. Validation is all-or-nothing: one invalid record fails the
whole batch with `422`.

## `POST /anomaly-score`

Same request body as `/predict`.

```json
{
  "iso_score": 0.0912,
  "ocsvm_score": 19.1088,
  "is_anomaly_isolation": false,
  "is_anomaly_svm": false,
  "model_version": "20261103T140735Z-9705890"
}
```

| Field | Interpretation |
| :-- | :-- |
| `iso_score` | IsolationForest decision score; **lower is more anomalous**; `< 0` flags an anomaly |
| `ocsvm_score` | One-Class SVM signed distance; **negative is anomalous** |
| `is_anomaly_isolation` / `is_anomaly_svm` | Per-detector flag |

Both detectors were fitted only on healthy (`N`) patients, so the score
measures distance from a healthy profile and not the presence of a specific
disease.

---

## Errors

| Status | When | Body |
| :-: | :-- | :-- |
| `422` | Missing or unknown field, wrong type, out-of-range value, unknown category, batch size out of range | FastAPI validation detail (`{"detail": [...]}`) |
| `422` | Record passes the schema but the feature pipeline rejects it | `{"detail": "<reason>"}` |
| `503` | No model loaded | `{"detail": "Model not loaded"}` |
| `500` | Unexpected server error | Logged with a stack trace; alerted via `DiseaseApiHighErrorRate` |

## Versioning & compatibility

- The **request schema** is tied to `settings.RAW_FEATURE_ORDER` and the frozen
  categorical levels. Adding a category or column is a breaking change: it
  needs a retrain and a new API version.
- **Model versions** change without API changes. Clients should log
  `model_version` with every stored prediction.

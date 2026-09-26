# Disease Anomaly Early-Warning System

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Docker Compose (local-first)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌───────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ PostgreSQL │──│ MLflow   │──│ FastAPI │──│Prom-    │──│ Grafana │ │
│  │ (Backend)  │  │ (Tracking│  │ (Serving│  │ etheus  │  │ (Dashboards)│ │
│  │            │  │ +Registry)│  │ API)   │  │         │  │          │ │
│  └───────────┘  └──────────┘  └─────────┘  └─────────┘  └─────────┘ │
│         │              │          │                                 │
│         └─── DVC ──────┴── Prefect (Orchestration)                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

   Data: DVC-tracked CSV (280K rows)
   Training: Prefect flow → MLflow logging → Model registry
   Serving: FastAPI loads model from MLflow Production stage
   Monitoring: Prometheus + Grafana for API metrics, Evidently AI for drift
```

## Overview

This project implements a full MLOps stack for multi-class disease detection and anomaly detection:

| Component | Technology |
|---|---|
| Experiment tracking | MLflow (PostgreSQL backend, local artifact store) |
| Orchestration | Prefect 3 |
| Model serving | FastAPI + Uvicorn |
| Data versioning | DVC |
| Dependency management | Poetry |
| Code quality | ruff, black, mypy, pre-commit |
| Testing | pytest with 80%+ coverage gate |
| Monitoring | Prometheus + Grafana |
| Drift detection | Evidently AI |

## Key Design Decision: Target Leakage Fix

The original notebook achieved ~1.00 precision/recall/F1 — explained by `disease_flags` directly encoding the sublabel. This refactor **drops all leakage columns**:

- `disease_flags` — comma-delimited string encoding disease state
- `hypertension` — binary indicator (0/1)
- `heart_disease` — binary indicator (0/1)
- `diabetes` — Yes/No indicator

**Expected realistic accuracy: 85-95%** (not 1.00). Honest metrics on a real problem are more valuable.

## Quick Start

```bash
# Install dependencies
make install

# Start MLflow UI (local)
make mlflow-ui

# Run the training pipeline
make train

# Start the API server
make serve

# Run tests
make test

# Lint
make lint
```

### Docker Compose (All Services)

```bash
docker compose up -d
```

Services:
- **PostgreSQL** (port 5432): MLflow backend store
- **MLflow** (port 5000): Tracking + model registry UI
- **API** (port 8000): FastAPI serving
- **Prefect** (port 4200): Pipeline orchestration
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3000): Dashboards (admin/admin123)

## API Endpoints

### `GET /health`
Check service health and model loading status.

### `POST /predict`
Predict disease sublabel from patient biomarkers.

```json
{
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
  "employment_status": "Employed"
}
```

### `POST /anomaly-score`
Score patient biomarkers for anomaly using isolation forest, one-class SVM, and autoencoder.

### `GET /metrics`
Prometheus-format metrics (request count, latency, error rate).

## Project Structure

```
┌── .github/workflows/     CI/CD pipelines
├─  .grafana/              Grafana dashboards + provisioning
├─  data/                  Data (DVC-tracked)
├─  src/                   Source package
│   ├─  config.py          Pydantic-settings configuration
│   ├─  data/              Data loading, validation, splitting
│   ├─  features/          Feature engineering (encoding, scaling)
│   ├─  models/            Anomaly detection + classifier
│   ├─  api/               FastAPI application + schemas
│   └─  utils/             MLflow logging utilities
├─  flows/                 Prefect training flow
├─  tests/                 pytest test suite
├─  Dockerfile             Multi-stage API image
├─  docker-compose.yml     All services
├─  dvc.yaml               DVC pipeline stages
├─  pyproject.toml          Dependencies + tool config
├─  run_pipeline.py         CLI entrypoint
└─  Makefile               Convenience targets
```

## Data Pipeline

1. **Extract**: Load CSV via DVC
2. **Validate**: Schema, nulls, physiological ranges, class distribution
3. **Preprocess**: Drop leakage columns → ordinal encode → one-hot encode → StandardScaler (fit on train only) → stratified 80/20 split
4. **Train Anomaly**: Isolation Forest + One-Class SVM + Deep Autoencoder on Normal cohort
5. **Compute Meta-features**: iso_score, ae_mse from anomaly models
6. **Train Classifier**: RandomForest on biomarkers + meta-features
7. **Evaluate**: Classification report, confusion matrix, ROC-AUC
8. **Log to MLflow**: Parameters, metrics, artifacts, model registry
9. **Register**: Transition model to Staging

## Environment Requirements

- Python 3.10+
- Docker + Docker Compose v2
- DVC (`pip install dvc`)

# AGENTS.md — Code Agent Context

## Project Overview

MLOps project transforming a flat Jupyter notebook into a production-grade MLOps pipeline for multi-class disease detection and anomaly scoring.

## Architecture Stack

- **Orchestration**: Prefect 3 flows in `flows/`
- **Experiment tracking**: MLflow (SQLite backend locally, PostgreSQL in Docker)
- **Model serving**: FastAPI on port 8000
- **Data versioning**: DVC (local remote at `/tmp/dvc-cache`)
- **Monitoring**: Prometheus + Grafana (API metrics), Evidently AI (drift)
- **CI/CD**: GitHub Actions (lint, test with coverage gate, Docker build)

## Code Conventions

- All source code in `src/` package, importable as `from src.module import ...`
- Type hints on all functions (mypy strict mode)
- Pydantic settings in `src/config.py` — do not hardcode paths or params
- No comments in code unless explicitly requested
- Tests in `tests/`, use pytest fixtures from `conftest.py`
- Coverage gate: 80% minimum (enforced in CI)

## Running Tasks

```bash
make lint    # ruff + black --check + mypy
make test    # pytest with coverage
make train   # run Prefect flow (requires MLflow running)
make serve   # FastAPI on :8000
```

## Key Files

- `flows/training_flow.py`: Full Prefect flow (extract → validate → preprocess → train → evaluate → log → register)
- `src/config.py`: Central configuration (Settings singleton)
- `src/data/`: Data loading, validation, splitting (leakage-safe)
- `src/models/`: Anomaly detection (isolation forest, one-class SVM, autoencoder) + hybrid classifier
- `src/api/main.py`: FastAPI app with Prometheus middleware
- `src/utils/mlflow_utils.py`: MLflow logging and registry helpers

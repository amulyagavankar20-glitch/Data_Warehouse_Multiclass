# AGENTS.md — Code Agent Context

## Project Overview

Production MLOps pipeline for 8-class comorbidity sublabel prediction (`N`, `DI`,
`HT`, `HY` and their combinations) plus healthy-baseline anomaly scoring as an
early-warning signal. The research notebooks (`Eda.ipynb`,
`Multi_Class_Disease_Detection_Anomaly_Detection.ipynb`) are the source of the
modelling decisions; `src/` is the production implementation of them.

## Non-negotiable modelling invariants

- **Leakage quarantine**: `diabetes`, `hypertension`, `heart_disease`,
  `disease_flags`, `label`, `sublabel`, `composite_key`, `age_normalized` and
  `source_dataset` never enter the feature matrix (`settings.LEAKAGE_COLUMNS`).
  Including them yields a spurious ~1.00 F1.
- Exact duplicates are dropped **before** splitting.
- Every fitted statistic (imputation-flag modes, scaler) comes from the train
  fold only — see `ClinicalFeaturePipeline`.
- Categorical levels are frozen in `src/config.py`; unknown levels must raise,
  never be silently encoded.
- `disease_flags` order is `(Diabetes, Heart_Disease, Hypertension)` =
  `(DI, HT, HY)`.

## Architecture Stack

- **Orchestration**: Prefect 3 flow in `flows/training_flow.py`
- **Experiment tracking / registry**: MLflow; models promoted via the
  `@champion` alias only when they beat the current champion's test macro AUC
- **Model serving**: FastAPI (`src/api`), loads `models/bundle.joblib` or
  `MODEL_URI` (e.g. `models:/disease-anomaly-classifier@champion`)
- **Data / pipeline versioning**: DVC (`dvc.yaml`); raw CSV is git-tracked
- **Monitoring**: Prometheus + Grafana (API + prediction mix), PSI drift CLI
  (`python -m src.monitoring batch.csv`)
- **CI/CD**: GitHub Actions — lint, test (80% coverage gate), DVC/data contract
  check, Docker build + Trivy + smoke test, GHCR release, scheduled retraining

## Code Conventions

- All source in `src/`, imported as `from src.module import ...`
- Type hints on all functions (mypy, `disallow_untyped_defs`)
- Settings in `src/config.py` — do not hardcode paths or params
- Docstrings over inline comments
- Tests in `tests/`; `tests/conftest.py::make_raw_frame` produces
  schema-faithful synthetic data — use it rather than ad-hoc frames

## Running Tasks

```bash
make lint    # ruff + black --check + mypy
make test    # pytest with coverage gate
make train   # Prefect training flow (local MLflow store in ./mlruns)
make serve   # FastAPI on :8000
```

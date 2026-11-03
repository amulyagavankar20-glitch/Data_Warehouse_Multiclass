# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE` (MIT; the dataset is excluded), `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `CONTRIBUTING.md`, `CITATION.cff`, `SUPPORT.md` and `.editorconfig`.
- Issue forms, a pull request template and `CODEOWNERS`.
- `.gitattributes` enforcing LF line endings.

### Fixed

- On Windows checkouts (`core.autocrlf=true`) DVC saw every file as changed,
  because CRLF bytes changed the content hashes. Line endings are now
  normalised to LF.
- Pipeline JSON reports now end with a newline and are always written with LF
  (`Path.write_text` produced CRLF on Windows), so they no longer conflict
  with the `end-of-file-fixer` hook.
- Training was not bit-reproducible: metrics varied around the 7th
  significant digit between runs. LightGBM now runs with `deterministic=True`
  and `force_row_wise=True`; consecutive runs produce byte-identical reports
  (macro F1 0.4256 → 0.4257).

## [1.0.0] - 2026-11-03

### Added

- Production pipeline: Prefect training flow with data-contract validation, a
  stratified 70/15/15 split and a quality gate (test macro AUC ≥ 0.90).
- `ClinicalFeaturePipeline`, shared by training and serving, which reproduces
  the research notebook's 50-feature contract.
- Cost-sensitive LightGBM champion with a guarded per-class threshold
  calibration step.
- IsolationForest and One-Class SVM early-warning detectors fitted on the
  healthy cohort.
- MLflow pyfunc registration with `@candidate` / `@champion` alias promotion.
- FastAPI service: `/predict`, `/predict/batch`, `/anomaly-score`,
  `/model/contract`, `/health`, `/ready` and `/metrics`.
- PSI drift reference in the model bundle, and the `python -m src.monitoring`
  CLI.
- DVC pipeline; Docker images; Compose stack with Postgres, MLflow, Prefect,
  Prometheus (with alert rules) and Grafana.
- CI (lint, typecheck, tests with an 80% coverage gate, DVC and data checks,
  Docker build, Trivy scan, smoke test), GHCR release and weekly retraining
  workflows.
- Documentation: architecture, API specification, data contract, model card,
  operations runbook and development guide.

### Fixed

- `/predict` failed with a real model bundle (the classifier expected
  meta-features that serving did not compute).
- The input schema rejected real patients (age below 18, BMI above 60).
- `/metrics` returned a JSON string instead of Prometheus exposition format.
- `disease_flags` columns were mislabelled; the order is `(DI, HT, HY)`.
- Imputation-flag modes were computed on the full dataset; they now use the
  training fold only.
- The `physical_activity_Moderate` one-hot column was standard-scaled by
  mistake.

### Removed

- Target-leaking features (`diabetes`, `hypertension`, `heart_disease`,
  `disease_flags`, `age_normalized`, among others). The previously reported
  1.00 F1 was an artifact of this leakage.
- The PyTorch autoencoder, notebook-era `production_artifacts/`, and the Poetry
  configuration.

[Unreleased]: https://github.com/amulyagavankar20-glitch/Data_Warehouse_Multiclass/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/amulyagavankar20-glitch/Data_Warehouse_Multiclass/releases/tag/v1.0.0

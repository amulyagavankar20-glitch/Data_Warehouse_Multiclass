# Architecture

This document describes how the system is put together, how a model moves from
raw data to production, and why the main design decisions were made.

- [System context](#system-context)
- [Components](#components)
- [Training pipeline](#training-pipeline)
- [Feature pipeline](#feature-pipeline)
- [Model lifecycle](#model-lifecycle)
- [Serving path](#serving-path)
- [Observability](#observability)
- [Deployment topology](#deployment-topology)
- [Design decisions](#design-decisions)

---

## System context

```mermaid
flowchart TB
    subgraph Sources
        WH[("Clinical data warehouse<br/>CSV export")]
    end

    subgraph Training["Training plane"]
        PF["Prefect flow"]
        DVC["DVC pipeline"]
        GHA["GitHub Actions<br/>retrain.yml"]
    end

    subgraph Registry["Model registry"]
        ML[("MLflow<br/>Postgres + artifact store")]
    end

    subgraph Serving["Serving plane"]
        API["FastAPI service"]
    end

    subgraph Observability
        PROM["Prometheus"]
        GRAF["Grafana"]
        DRIFT["PSI drift CLI"]
    end

    CLIENT(["Client applications"])

    WH --> PF
    DVC --> PF
    GHA --> PF
    PF -->|"log, register, promote"| ML
    ML -->|"models:/…@champion"| API
    CLIENT -->|"HTTPS / JSON"| API
    API -->|"/metrics"| PROM --> GRAF
    WH -.->|"new batches"| DRIFT
    ML -.->|"drift reference"| DRIFT
```

## Components

| Component | Module | Responsibility |
| :-- | :-- | :-- |
| Settings | `src/config.py` | Runtime settings (env-overridable) and the frozen feature contract |
| Loader | `src/data/loader.py` | Read the CSV, drop duplicates, separate predictors from targets |
| Validator | `src/data/validator.py` | Schema, null, range, category and class-presence checks |
| Splitter | `src/data/splitter.py` | Stratified 70 / 15 / 15 train / validation / test split |
| Feature pipeline | `src/features/engineering.py` | Raw record → 50-column model matrix; fitted on train only |
| Classifier | `src/models/classifier.py` | Cost-sensitive LightGBM, early stopping, guarded threshold calibration |
| Anomaly detectors | `src/models/anomaly/` | IsolationForest and One-Class SVM fitted on the healthy cohort |
| Evaluator | `src/models/evaluator.py` | Multi-class metrics, per-disease AUC, plots |
| Drift monitor | `src/monitoring/` | PSI reference from train deciles; CLI for new batches |
| Model bundle | `src/serving/bundle.py` | One versioned artifact holding everything inference needs |
| pyfunc wrapper | `src/serving/pyfunc.py` | Makes the registered model loadable with `mlflow.pyfunc` |
| Registry utils | `src/utils/mlflow_utils.py` | Tracking, registration, champion/challenger promotion |
| Training flow | `flows/training_flow.py` | Prefect flow that runs every stage end to end |
| API | `src/api/` | HTTP interface, input validation, Prometheus instrumentation |

## Training pipeline

```mermaid
flowchart TD
    A["extract<br/>load CSV"] --> B{"validate<br/>data contract"}
    B -->|errors| X1["fail fast<br/>ValueError"]
    B -->|ok| C["prepare<br/>dedupe → stratified split"]
    C --> D["fit_features<br/>on train fold"]
    D --> E["train_classifier<br/>LightGBM + early stopping on val"]
    D --> F["train_anomaly_detectors<br/>healthy N cohort"]
    E --> E2["calibrate_thresholds<br/>keep only if val F1 improves"]
    E2 --> G["evaluate on test"]
    F --> G
    C --> R["build PSI reference<br/>train deciles"]
    G --> H{"quality_gate<br/>macro AUC ≥ 0.90"}
    R --> H
    H -->|fail| X2["QualityGateError<br/>nothing written"]
    H -->|pass| I["package ModelBundle<br/>+ reports"]
    I --> J["MLflow: log params, metrics,<br/>plots, pyfunc model"]
    J --> K["register version<br/>→ promotion"]
```

Each box is a Prefect task. The flow returns the test metrics and writes:

| Output | Tracked by | Purpose |
| :-- | :-- | :-- |
| `models/bundle.joblib` | DVC cache | Inference artifact loaded by the API |
| `models/serving_contract.json` | git | Feature list, classes, thresholds, metrics (reviewable in PRs) |
| `reports/metrics.json` | git (DVC metric) | Test metrics; `dvc metrics diff` compares runs |
| `reports/drift.json` | git (DVC metric) | PSI of the test fold against the train reference |
| `reports/anomaly_audit.json` | git | Anomaly scores and flag rates per sublabel |
| `reports/feature_importance.json` | git | LightGBM gain per feature |
| `reports/confusion_matrix.png` | git (DVC plot) | Row-normalised confusion matrix |

> [!NOTE]
> The quality gate runs **before** anything is written. A model that fails it
> never reaches `models/` or the registry.

## Feature pipeline

`ClinicalFeaturePipeline` is a scikit-learn transformer. Training and serving run
the same instance, which removes training/serving skew by construction.

```mermaid
flowchart LR
    R["30 raw columns"] --> I["Clinical interactions<br/>pulse pressure · MAP ·<br/>LDL/HDL · log(glucose × TG)"]
    I --> O["Ordinal encode<br/>age · BMI · stress · education"]
    I --> M["12 imputation flags<br/>value == train mode"]
    I --> N["One-hot encode<br/>9 nominal columns, frozen levels"]
    O & M & N --> S["StandardScaler on 25<br/>continuous/ordinal columns"]
    S --> F["50 float32 features"]
```

| Stage | Stateful? | Fitted on |
| :-- | :-: | :-- |
| Clinical interactions | No | — |
| Ordinal / one-hot encoding | No | Levels frozen in `settings` |
| Imputation flags | Yes | Train-fold mode of each column |
| Scaling | Yes | Train fold |

See the [data contract](data-contract.md) for the full feature list.

## Model lifecycle

```mermaid
stateDiagram-v2
    [*] --> Trained: run_training()
    Trained --> Rejected: macro AUC < 0.90
    Rejected --> [*]
    Trained --> Candidate: registered (@candidate)
    Candidate --> Champion: beats champion test macro AUC, or no champion yet
    Candidate --> Archived: does not beat champion
    Champion --> Archived: superseded by a better candidate
    Champion --> Serving: API loads the champion alias
```

Promotion compares on `test_roc_auc_macro`, which is threshold-independent and
stable under class imbalance. The comparison is strict (`>`), so a retrain that
merely ties does not churn production.

## Serving path

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as FastAPI
    participant V as Pydantic schema
    participant B as ModelBundle
    participant P as Prometheus registry

    C->>A: POST /predict (raw record)
    A->>V: validate types, bounds, enums
    alt invalid
        V-->>C: 422 Unprocessable Entity
    end
    A->>B: predict(DataFrame)
    B->>B: feature pipeline → LightGBM → decide()
    B->>B: collapse to disease marginals
    B-->>A: sublabel, probabilities
    A->>P: model_predictions_total{sublabel}++
    A-->>C: 200 PredictionResponse
    Note over A,P: middleware records http_requests_total<br/>and latency by route template
```

At startup the API loads the bundle from `MODEL_URI` (an MLflow registry URI) if
it is set, and from `models/bundle.joblib` otherwise. If neither is available,
the process still starts: `/health` returns 200, while `/ready` and the
inference endpoints return 503. An orchestrator therefore keeps the pod out of
rotation instead of restarting it in a loop.

## Observability

| Signal | Source | Where to look |
| :-- | :-- | :-- |
| Request rate, errors, latency | `http_requests_total`, `http_request_duration_seconds` | Grafana · *Disease Anomaly API* |
| Prediction mix (prediction drift) | `model_predictions_total{sublabel}` | Grafana panel *Predicted sublabel mix* |
| Early-warning volume | `model_anomaly_flags_total{detector}` | Grafana panel *Early-warning anomaly flags* |
| Loaded model version | `model_info{version}` | Grafana stat · `/health` |
| Input data drift | PSI per feature | `python -m src.monitoring batch.csv` |
| Training history | MLflow runs | MLflow UI |

Alert rules live in `prometheus_alerts.yml`. Their response procedures are in
the [operations runbook](operations.md#alerts).

## Deployment topology

```mermaid
flowchart LR
    subgraph compose["docker compose"]
        PG[("postgres:16")]
        MLF["mlflow server<br/>:5000"]
        PFS["prefect server<br/>:4200"]
        TR["trainer<br/>(profile: train)"]
        API["api<br/>:8000 · 2 workers"]
        PR["prometheus<br/>:9090"]
        GR["grafana<br/>:3000"]
    end
    PG --- MLF
    TR -->|"tracking + artifacts"| MLF
    TR -->|"flow runs"| PFS
    API -->|"MODEL_URI"| MLF
    PR -->|"scrape"| API
    PR -->|"scrape"| MLF
    GR --> PR
```

| Service | Image | Persistent volume |
| :-- | :-- | :-- |
| `postgres` | `postgres:16-alpine` | `postgres_data` |
| `mlflow` | `Dockerfile.mlflow` (MLflow 2.22.5) | `mlflow_artifacts` (proxied, no S3 needed) |
| `prefect` | `prefecthq/prefect:3.8.8-python3.11` | `prefect_data` |
| `trainer` / `api` | `Dockerfile` (non-root, uv-locked) | bind mounts `./models`, `./reports` |
| `prometheus` | `prom/prometheus:v2.54.1` | `prometheus_data` |
| `grafana` | `grafana/grafana:11.2.2` | `grafana_data` |

## Design decisions

| Decision | Alternatives considered | Rationale |
| :-- | :-- | :-- |
| LightGBM as the classifier | Logistic regression (val AUC 0.826), random forest (0.940), multi-task NN (0.952 AUC but 0.353 F1) | Best macro F1 with near-best AUC and low latency; the notebook benchmark covers all four |
| Balanced sample weights | SMOTE | Synthetic biomarker vectors can be physiologically impossible |
| Threshold calibration guarded by val F1 | Always calibrate | Per-class one-vs-rest thresholds do not compose here (val F1 0.353 vs 0.419 for arg-max) |
| IsolationForest + OCSVM | Adding a PyTorch autoencoder | Avoids ~700 MB of torch in the image; OCSVM capped at 10k training rows because its cost is O(n²) |
| One bundle artifact | Separate scaler / encoder / model files | Atomic versioning; serving cannot load mismatched pieces |
| Registry aliases | MLflow stages | Stages are deprecated since MLflow 2.9 |
| Raw CSV in git, model in DVC | Everything in DVC | No shared DVC remote exists yet; clones and CI work with no extra setup |
| Strict category enums in the API | Map unknown categories to 0 | Silent mis-encoding is worse than a 422 |

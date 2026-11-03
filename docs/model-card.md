# Model Card: `disease-anomaly-classifier`

| | |
| :-- | :-- |
| **Model type** | LightGBM multi-class GBDT (8 classes) + IsolationForest and One-Class SVM anomaly detectors |
| **Version** | `20261103T140735Z-9705890` (see `/health` for the deployed version) |
| **Training data** | 195,090 patients (70% of the 278,701 deduplicated records) |
| **Evaluation data** | 41,806-patient held-out test fold, stratified by sublabel |
| **Owner** | Data Warehouse Multiclass project |
| **Registry** | MLflow `disease-anomaly-classifier`, alias `@champion` |

> [!CAUTION]
> This is a **decision-support research model**. It is not a medical device
> and has not been clinically validated. It must not be used to diagnose,
> rule out, or treat any condition without review by a qualified clinician.

## Intended use

| | |
| :-- | :-- |
| ✅ Intended | Population-level risk stratification; prioritising patients for follow-up screening; surfacing sub-clinical deviation from a healthy profile |
| ⚠️ Use with care | Individual-level decisions (always alongside clinical judgement) |
| ❌ Out of scope | Autonomous diagnosis; populations or measurement protocols unlike the source warehouse; patients with heart-disease-only presentations where high recall is required |

## Training procedure

| Setting | Value |
| :-- | :-- |
| Split | Stratified 70 / 15 / 15 (train / validation / test), seed 42 |
| Imbalance handling | Balanced sample weights (0.33 for `N` to 41.2 for `DI_HT_HY`); no oversampling |
| Hyperparameters | `num_leaves=63`, `max_depth=7`, `learning_rate=0.05`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `lambda_l1=1.5`, `lambda_l2=3.0` |
| Boosting | Up to 600 rounds, early stopping (30) on validation multi-logloss |
| Decision rule | Arg-max; per-class threshold calibration was evaluated and rejected (val macro F1 0.353 vs 0.419) |
| Anomaly detectors | Fitted on healthy (`N`) training patients only; IsolationForest 200 trees, contamination 0.05; OCSVM RBF, ν = 0.05, 10k-row subsample |

## Performance

### Overall (test fold)

| Metric | Value |
| :-- | --: |
| Macro one-vs-rest ROC-AUC | 0.944 |
| Macro F1 | 0.426 |
| Weighted F1 | 0.715 |
| Accuracy | 0.708 |

### Per sublabel

| Sublabel | Support | Precision | Recall | F1 |
| :-- | --: | --: | --: | --: |
| `N` | 15,698 | 0.938 | 0.647 | 0.766 |
| `HY` | 10,468 | 0.714 | 0.914 | 0.802 |
| `DI_HY` | 9,971 | 0.724 | 0.816 | 0.767 |
| `DI` | 4,786 | 0.439 | 0.293 | 0.351 |
| `HT` | 388 | 0.103 | 0.536 | 0.173 |
| `DI_HT` | 209 | 0.165 | 0.340 | 0.222 |
| `HT_HY` | 160 | 0.090 | 0.269 | 0.135 |
| `DI_HT_HY` | 126 | 0.145 | 0.270 | 0.188 |

### Per condition (derived from the sublabel probabilities)

| Condition | ROC-AUC |
| :-- | --: |
| Diabetes | 1.000 |
| Heart disease | 0.926 |
| Hypertension | 0.805 |

### Early-warning detectors (abnormal vs `N`)

| Detector | ROC-AUC |
| :-- | --: |
| IsolationForest | 0.813 |
| One-Class SVM | 0.824 |

### Top features (LightGBM gain share)

| Feature | Share |
| :-- | --: |
| `HbA1c_level` | 31.3% |
| `blood_pressure` | 6.9% |
| `bmi` | 5.9% |
| `age` | 5.8% |
| `high_blood_pressure_Yes` | 4.8% |
| `glucose` | 4.5% |
| `ldl` | 4.4% |
| `homocysteine_level` | 3.7% |
| `glucose_triglyceride_index` | 3.5% |
| `systolic_bp` | 3.4% |

## Limitations

> [!WARNING]
> **Heart disease is the weak spot.** The four heart-disease sublabels make up
> under 2% of rows and reach F1 of only 0.13–0.22. The anomaly detectors also
> flag `HT` patients least often (IsolationForest flags 0.8% of them). Do not
> rely on this model to rule out heart disease.

- **Diabetes is close to definitional.** The diabetes AUC of 1.000 reflects that
  diabetes in the source data is determined by HbA1c and glucose. The model
  restates the labelling rule, which is not the same as early detection.
- **Precision/recall trade-off on `N`.** Precision for healthy patients is high
  (0.94) but recall is 0.65: about a third of healthy patients are assigned a
  disease sublabel. This is the intended direction for screening, but it
  creates follow-up load.
- **Source bias.** The three source studies have different measurement and
  imputation regimes (adversarial validation AUC 0.96). Performance on data
  from a new source is unknown; run the drift check first.
- **Imputed values.** Many biomarkers were group-mean imputed in the
  warehouse. The model partly learns from imputation patterns (the
  `*_is_imputed` flags), which will not transfer to sites that impute
  differently.
- **Cross-sectional data.** There is no temporal validation. "Early warning"
  means deviation at a single point in time, not a forecast.

## Ethical considerations

| Topic | Status |
| :-- | :-- |
| Sensitive attributes | `gender` and `age` are model inputs. No subgroup fairness audit has been run yet. **Recommended before any clinical pilot.** |
| Privacy | The training data is a warehouse export. The API is stateless and does not log request bodies, but deployments must add authentication and TLS ([API spec](api-spec.md#conventions)). |
| Human oversight | Outputs are probabilities with a model version and are meant for review by a clinician. |

## Monitoring & maintenance

- Retrains weekly or on demand (`.github/workflows/retrain.yml`). A new model is
  promoted only if it beats the current champion's macro AUC, and it must clear
  the 0.90 quality gate.
- Input drift is measured with PSI against the training reference. Test-fold
  PSI at training time was 0.0007 (stable).
- Responses are documented in the [operations runbook](operations.md).

## Reproducibility

```bash
uv sync --frozen && uv run dvc repro
```

The pipeline is deterministic with fixed seeds: repeated runs reproduce
`reports/metrics.json` exactly.

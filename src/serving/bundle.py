"""Versioned inference bundle shared by the API, batch scoring and MLflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.features.engineering import ClinicalFeaturePipeline
from src.models.anomaly.isolation_forest import IsolationForestModel
from src.models.anomaly.one_class_svm import OneClassSVMModel
from src.models.classifier import LightGBMClassifier
from src.models.evaluator import disease_marginals

BUNDLE_FILENAME = "bundle.joblib"


@dataclass
class ModelBundle:
    feature_pipeline: ClinicalFeaturePipeline
    classifier: LightGBMClassifier
    iso_forest: IsolationForestModel
    ocsvm: OneClassSVMModel
    classes: list[str]
    drift_reference: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unversioned"))

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        return self.feature_pipeline.transform(X_raw[settings.RAW_FEATURE_ORDER])

    def predict(self, X_raw: pd.DataFrame) -> list[dict[str, Any]]:
        """Sublabel prediction, per-class and per-disease probabilities."""
        X = self.transform(X_raw)
        proba = self.classifier.predict_proba(X)
        pred = self.classifier.decide(proba)
        diseases = disease_marginals(proba, self.classes)
        results = []
        for i in range(len(X)):
            k = int(pred[i])
            results.append(
                {
                    "predicted_sublabel": self.classes[k],
                    "confidence": float(proba[i, k]),
                    "probabilities": {
                        c: float(proba[i, j]) for j, c in enumerate(self.classes)
                    },
                    "disease_probabilities": {
                        d: float(diseases[i, j])
                        for j, d in enumerate(settings.DISEASE_FLAG_NAMES)
                    },
                }
            )
        return results

    def score_anomaly(self, X_raw: pd.DataFrame) -> list[dict[str, Any]]:
        """Early-warning deviation from the healthy-cohort profile."""
        X = self.transform(X_raw).to_numpy()
        iso = self.iso_forest.decision_function(X)
        svm = self.ocsvm.decision_function(X)
        return [
            {
                "iso_score": float(iso[i]),
                "ocsvm_score": float(svm[i]),
                "is_anomaly_isolation": bool(iso[i] < 0),
                "is_anomaly_svm": bool(svm[i] < 0),
            }
            for i in range(len(X))
        ]

    def serving_contract(self) -> dict[str, Any]:
        thresholds = self.classifier.thresholds
        return {
            "model_version": self.version,
            "raw_input_features": list(settings.RAW_FEATURE_ORDER),
            "input_feature_count": len(self.feature_pipeline.feature_names_out_),
            "input_features": list(self.feature_pipeline.feature_names_out_),
            "numeric_scaled_columns": list(self.feature_pipeline.scaled_columns_),
            "target_classes": list(self.classes),
            "decision_thresholds": (
                dict(zip(self.classes, np.asarray(thresholds).tolist(), strict=True))
                if thresholds is not None
                else None
            ),
            "multilabel_heads": list(settings.DISEASE_FLAG_NAMES),
            "metrics": self.metadata.get("metrics", {}),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj

    @classmethod
    def from_mlflow(cls, model_uri: str) -> ModelBundle:
        """Load from a registry URI such as ``models:/<name>@champion``."""
        import mlflow

        local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
        matches = sorted(local_dir.rglob(BUNDLE_FILENAME))
        if not matches:
            raise FileNotFoundError(f"No {BUNDLE_FILENAME} under {model_uri}")
        return cls.load(matches[0])

"""MLflow ``pyfunc`` flavour so the registered model is directly servable."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mlflow.pyfunc import PythonModel

from src.serving.bundle import ModelBundle


class DiseaseModel(PythonModel):  # type: ignore[misc]
    """Raw patient records in, sublabel + disease probabilities out."""

    def load_context(self, context: Any) -> None:
        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: Any = None
    ) -> pd.DataFrame:
        preds = self.bundle.predict(model_input)
        out = pd.DataFrame(
            {
                "predicted_sublabel": [p["predicted_sublabel"] for p in preds],
                "confidence": [p["confidence"] for p in preds],
            }
        )
        for cls in self.bundle.classes:
            out[f"proba_{cls}"] = [p["probabilities"][cls] for p in preds]
        return out

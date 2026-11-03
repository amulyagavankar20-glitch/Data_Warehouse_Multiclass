#!/usr/bin/env python
"""CLI entrypoint for the disease anomaly training pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from flows.training_flow import QualityGateError, run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the disease anomaly training pipeline"
    )
    parser.add_argument("--data", type=Path, default=None, help="Raw CSV path")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Skip MLflow tracking/registry"
    )
    parser.add_argument(
        "--min-auc", type=float, default=None, help="Override the quality gate"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        metrics = run_training(
            data_path=args.data,
            enable_mlflow=False if args.no_mlflow else None,
            min_test_macro_auc=args.min_auc,
        )
    except QualityGateError as exc:
        print(f"\nQuality gate failed: {exc}", file=sys.stderr)
        sys.exit(3)

    print("\nPipeline completed successfully:")
    for key in ("test_roc_auc_macro", "test_f1_macro", "test_accuracy"):
        print(f"  {key:<22} {metrics[key]:.4f}")


if __name__ == "__main__":
    main()

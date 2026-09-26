#!/usr/bin/env python
"""CLI entrypoint for the disease anomaly training pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from flows.training_flow import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the disease anomaly training pipeline"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Run without MLflow tracking",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri("")
        mlflow.end_run()

    try:
        result = run_training()
        print(f"\nPipeline completed successfully:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  F1 (macro): {result['f1_macro']:.4f}")
        print(f"  F1 (weighted): {result['f1_weighted']:.4f}")
        if result.get("roc_auc"):
            print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

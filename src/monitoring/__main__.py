"""CLI: ``python -m src.monitoring <batch.csv>`` -> PSI drift report.

Exits with status 2 when any feature breaches the PSI alert threshold, so the
command can gate a scheduled retraining job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import settings
from src.data.loader import load_raw_data
from src.monitoring.drift import compute_drift
from src.serving.bundle import ModelBundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="CSV of new raw patient records")
    parser.add_argument("--bundle", type=Path, default=settings.bundle_path)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bundle = ModelBundle.load(args.bundle)
    report = compute_drift(
        bundle.drift_reference, load_raw_data(args.batch), args.threshold
    )
    payload = json.dumps(report.to_dict(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 2 if report.drifted_features else 0


if __name__ == "__main__":
    sys.exit(main())

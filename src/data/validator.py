"""Contract-driven data validation for training batches."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import settings


@dataclass
class ValidationResult:
    """Outcome of a validation run; any error fails the pipeline."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: ValidationResult) -> None:
        for msg in other.errors:
            self.add_error(msg)
        self.warnings.extend(other.warnings)


EXPECTED_COLUMNS: list[str] = [
    *settings.RAW_FEATURE_ORDER,
    settings.TARGET_COLUMN,
]

PHYSIOLOGICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "bmi": (8, 110),
    "HbA1c_level": (2, 20),
    "glucose": (20, 700),
    "cholesterol": (50, 700),
    "sleep_hours": (0, 24),
    "triglycerides": (10, 3000),
    "blood_pressure": (40, 300),
    "crp_level": (0, 300),
    "homocysteine_level": (0, 150),
    "systolic_bp": (50, 300),
    "diastolic_bp": (20, 200),
    "alcohol_intake": (0, 100),
    "salt_intake": (0, 60),
    "heart_rate": (20, 250),
    "hdl": (5, 200),
    "ldl": (10, 400),
}

VALID_SUBLABELS: set[str] = {
    "N",
    "DI",
    "HT",
    "HY",
    "DI_HT",
    "DI_HY",
    "HT_HY",
    "DI_HT_HY",
}

MIN_CLASS_FRACTION = 0.001


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        result.add_error(f"Missing columns: {missing}")
    return result


def validate_nulls(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()
    for col in EXPECTED_COLUMNS:
        if col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null:
                result.add_error(f"Column '{col}' has {n_null} null values")
    return result


def validate_ranges(df: pd.DataFrame) -> ValidationResult:
    """Values outside physiologically possible bounds are hard errors."""
    result = ValidationResult()
    for col, (lo, hi) in PHYSIOLOGICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(((values < lo) | (values > hi) | values.isna()).sum())
        if n_bad:
            result.add_error(f"Column '{col}' has {n_bad} values outside [{lo}, {hi}]")
    return result


def validate_categories(df: pd.DataFrame) -> ValidationResult:
    """Unknown categorical levels would silently corrupt the encoding."""
    result = ValidationResult()
    allowed: dict[str, set[str]] = {
        **{c: set(m) for c, m in settings.ORDINAL_MAPS.items()},
        **{c: set(levels) for c, levels in settings.NOMINAL_LEVELS.items()},
        settings.TARGET_COLUMN: VALID_SUBLABELS,
    }
    for col, levels in allowed.items():
        if col not in df.columns:
            continue
        unknown = set(df[col].dropna().astype(str).unique()) - levels
        if unknown:
            result.add_error(f"Column '{col}' has unknown levels: {sorted(unknown)}")
    return result


def validate_class_distribution(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()
    target = settings.TARGET_COLUMN
    if target not in df.columns:
        result.add_error(f"Target column '{target}' missing")
        return result
    fractions = df[target].value_counts(normalize=True)
    present = {str(k) for k in fractions.index}
    for absent in sorted(VALID_SUBLABELS - present):
        result.add_error(f"Class '{absent}' is absent from the batch")
    for label, frac in fractions.items():
        if frac < MIN_CLASS_FRACTION:
            result.add_warning(f"Class '{label}' is only {frac:.3%} of the batch")
    return result


def validate_data(df: pd.DataFrame) -> ValidationResult:
    """Run every check and return the combined result."""
    result = ValidationResult()
    if df.empty:
        result.add_error("Dataset is empty")
        return result
    for check in (
        validate_schema,
        validate_nulls,
        validate_ranges,
        validate_categories,
        validate_class_distribution,
    ):
        result.merge(check(df))
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        result.add_warning(f"{n_dupes} exact duplicate rows will be dropped")
    return result

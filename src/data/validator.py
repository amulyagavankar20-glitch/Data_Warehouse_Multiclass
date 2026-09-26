"""Data validation module with quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation checks."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


EXPECTED_COLUMNS: list[str] = [
    "age", "gender", "bmi", "HbA1c_level", "glucose", "cholesterol",
    "sleep_hours", "triglycerides", "systolic_bp", "diastolic_bp",
    "heart_rate", "hdl", "ldl", "crp_level", "homocysteine_level",
    "sublabel", "label",
]

CONTINUOUS_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "bmi": (10, 60),
    "HbA1c_level": (3, 15),
    "glucose": (20, 600),
    "cholesterol": (50, 600),
    "sleep_hours": (0, 24),
    "triglycerides": (0, 2000),
    "systolic_bp": (50, 250),
    "diastolic_bp": (30, 150),
    "heart_rate": (30, 220),
    "hdl": (10, 100),
    "ldl": (10, 300),
    "crp_level": (0, 200),
    "homocysteine_level": (0, 100),
    "blood_pressure": (30, 250),
    "alcohol_intake": (0, 20),
    "salt_intake": (0, 20),
}


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    """Check that all expected columns are present."""
    result = ValidationResult()
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        result.add_error(f"Missing columns: {sorted(missing)}")
    return result


def validate_nulls(df: pd.DataFrame) -> ValidationResult:
    """Check for nulls in critical biomarker columns."""
    result = ValidationResult()
    critical_cols = ["age", "bmi", "HbA1c_level", "glucose", "cholesterol", "sublabel"]
    for col in critical_cols:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            result.add_error(f"Column '{col}' has {null_count} null values")
    return result


def validate_ranges(df: pd.DataFrame) -> ValidationResult:
    """Check that continuous columns are within expected physiological ranges."""
    result = ValidationResult()
    for col, (min_val, max_val) in CONTINUOUS_RANGES.items():
        if col in df.columns:
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            count = out_of_range.sum()
            if count > 0:
                result.add_warning(
                    f"Column '{col}' has {count} values outside range [{min_val}, {max_val}]"
                )
    return result


def validate_class_distribution(df: pd.DataFrame) -> ValidationResult:
    """Check that no class has less than 0.5% representation."""
    result = ValidationResult()
    if "sublabel" not in df.columns:
        result.add_error("sublabel column missing")
        return result

    total = len(df)
    counts = df["sublabel"].value_counts()
    for label, count in counts.items():
        pct = count / total * 100
        if pct < 0.5:
            result.add_warning(
                f"Class '{label}' has only {pct:.2f}% of data (below 0.5% threshold)"
            )
    return result


def validate_data(df: pd.DataFrame) -> ValidationResult:
    """Run all data quality checks.

    Args:
        df: DataFrame to validate.

    Returns:
        Combined ValidationResult from all checks.
    """
    result = ValidationResult()

    for check in [validate_schema, validate_nulls, validate_ranges, validate_class_distribution]:
        check_result = check(df)
        result.errors.extend(check_result.errors)
        result.warnings.extend(check_result.warnings)

    result.passed = len(result.errors) == 0
    return result

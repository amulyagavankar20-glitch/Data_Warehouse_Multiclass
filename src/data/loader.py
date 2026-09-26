"""Data loading and cleaning module."""

from __future__ import annotations

import pandas as pd

from src.config import settings


def load_raw_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset.

    Args:
        filepath: Optional path to the CSV file. Defaults to settings.raw_data_path.

    Returns:
        DataFrame with raw disease data.
    """
    path = filepath or settings.raw_data_path
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset.

    - Drops full-row duplicates.
    - Parses disease_flags into individual binary columns (for EDA/documentation).
    - Drops leakage columns (disease_flags, hypertension, heart_disease, diabetes).

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame with parsed disease flag columns and leakage columns removed.
    """
    df = df.drop_duplicates().copy()

    df = parse_disease_flags(df)

    leakage_cols = ["hypertension", "heart_disease", "diabetes", "disease_flags"]
    existing_leakage = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=existing_leakage)

    return df


def parse_disease_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the disease_flags column into individual binary columns.

    The disease_flags column contains a comma-delimited string like '1,0,1'
    representing (diabetes, hypertension, heart_disease) indicators.

    Args:
        df: DataFrame with disease_flags column.

    Returns:
        DataFrame with parsed binary columns: has_diabetes_flag, has_hypertension_flag,
        has_heart_disease_flag.
    """
    if "disease_flags" not in df.columns:
        return df

    parsed = df["disease_flags"].str.split(",", expand=True)
    parsed.columns = ["has_diabetes_flag", "has_hypertension_flag", "has_heart_disease_flag"]
    parsed = parsed.astype(int)

    df = pd.concat([df, parsed], axis=1)
    return df


def get_feature_names() -> list[str]:
    """Return the list of feature column names used by the pipeline."""
    return settings.ORDINAL_COLUMNS + settings.NOMINAL_COLUMNS + settings.CONTINUOUS_COLUMNS

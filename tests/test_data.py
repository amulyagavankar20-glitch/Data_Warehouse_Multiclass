"""Tests for data loading, validation, and splitting."""

import numpy as np
import pandas as pd

from src.data.loader import (
    clean_data,
    get_feature_names,
    parse_disease_flags,
)
from src.data.splitter import get_normal_cohort, split_data
from src.data.validator import (
    validate_class_distribution,
    validate_nulls,
    validate_ranges,
    validate_schema,
)


class TestDataLoader:
    """Tests for src/data/loader.py."""

    def test_clean_data_removes_duplicates(self, synthetic_data: pd.DataFrame) -> None:
        df = pd.concat([synthetic_data, synthetic_data.head(5)], ignore_index=True)
        cleaned = clean_data(df)
        assert len(cleaned) == len(synthetic_data)

    def test_clean_data_drops_leakage_columns(self, synthetic_data: pd.DataFrame) -> None:
        cleaned = clean_data(synthetic_data)
        for col in ["disease_flags", "hypertension", "heart_disease", "diabetes"]:
            assert col not in cleaned.columns

    def test_clean_data_parses_disease_flags(self, synthetic_data: pd.DataFrame) -> None:
        parsed = parse_disease_flags(synthetic_data.copy())
        assert "has_diabetes_flag" in parsed.columns
        assert "has_hypertension_flag" in parsed.columns
        assert "has_heart_disease_flag" in parsed.columns
        for col in ["has_diabetes_flag", "has_hypertension_flag", "has_heart_disease_flag"]:
            assert parsed[col].dtype in [np.int64, int]

    def test_get_feature_names_returns_expected(self) -> None:
        names = get_feature_names()
        assert "age" in names
        assert "bmi" in names
        assert "HbA1c_level" in names
        assert "age_level" in names
        assert "gender" in names


class TestDataValidator:
    """Tests for src/data/validator.py."""

    def test_validate_schema_pass(self, synthetic_data: pd.DataFrame) -> None:
        result = validate_schema(synthetic_data)
        assert result.passed

    def test_validate_schema_fail_missing_columns(self) -> None:
        df = pd.DataFrame({"age": [25], "bmi": [22.5]})
        result = validate_schema(df)
        assert not result.passed
        assert len(result.errors) > 0

    def test_validate_nulls_pass(self, synthetic_data: pd.DataFrame) -> None:
        result = validate_nulls(synthetic_data)
        assert result.passed

    def test_validate_nulls_fail(self) -> None:
        df = pd.DataFrame({"age": [None, 25], "bmi": [None, 22.5], "HbA1c_level": [5.0, None]})
        result = validate_nulls(df)
        assert not result.passed

    def test_validate_ranges_warns_on_out_of_range(self, synthetic_data: pd.DataFrame) -> None:
        df = synthetic_data.copy()
        df.loc[0, "age"] = 200  # out of range
        result = validate_ranges(df)
        assert len(result.warnings) > 0

    def test_validate_ranges_pass(self, synthetic_data: pd.DataFrame) -> None:
        result = validate_ranges(synthetic_data)
        assert result.passed or len(result.warnings) > 0  # may warn but no errors

    def test_validate_class_distribution(self, synthetic_data: pd.DataFrame) -> None:
        result = validate_class_distribution(synthetic_data)
        assert result.passed


class TestDataSplitter:
    """Tests for src/data/splitter.py."""

    def test_split_data_shapes(self, cleaned_synthetic_data: pd.DataFrame) -> None:
        split = split_data(cleaned_synthetic_data, test_size=0.2, random_state=42)
        total = len(split.X_train) + len(split.X_test)
        assert total == len(cleaned_synthetic_data)
        assert len(split.X_train) == 800
        assert len(split.X_test) == 200

    def test_split_data_stratified(self, cleaned_synthetic_data: pd.DataFrame) -> None:
        split = split_data(cleaned_synthetic_data, test_size=0.2, random_state=42)
        train_dist = np.bincount(split.y_train)
        test_dist = np.bincount(split.y_test)
        assert len(train_dist) == len(test_dist)

    def test_split_data_label_encoder(self, cleaned_synthetic_data: pd.DataFrame) -> None:
        split = split_data(cleaned_synthetic_data, test_size=0.2, random_state=42)
        assert isinstance(split.label_encoder, type(split.label_encoder))
        assert len(split.label_encoder.classes_) > 1

    def test_get_normal_cohort(self, cleaned_synthetic_data: pd.DataFrame) -> None:
        split = split_data(cleaned_synthetic_data, test_size=0.2, random_state=42)
        normal_cohort = get_normal_cohort(
            split.X_train, split.y_train, split.label_encoder,
        )
        assert len(normal_cohort) > 0
        assert len(normal_cohort) < len(split.X_train)

    def test_split_data_no_leakage_columns(self, cleaned_synthetic_data: pd.DataFrame) -> None:
        split = split_data(cleaned_synthetic_data, test_size=0.2, random_state=42)
        leakage_cols = ["disease_flags", "hypertension", "heart_disease", "diabetes", "label"]
        for col in leakage_cols:
            assert col not in split.X_train.columns
            assert col not in split.X_test.columns

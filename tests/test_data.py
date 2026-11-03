"""Tests for loading, validation and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.data.loader import (
    clean_data,
    load_raw_data,
    parse_disease_flags,
    split_features_targets,
)
from src.data.splitter import normal_class_index, split_data
from src.data.validator import validate_data


class TestLoader:
    def test_load_raw_data(self, raw_csv: Path) -> None:
        assert len(load_raw_data(raw_csv)) == 1500

    def test_clean_data_removes_duplicates(self, raw_df: pd.DataFrame) -> None:
        doubled = pd.concat([raw_df, raw_df.head(10)])
        assert len(clean_data(doubled)) == len(raw_df)

    def test_disease_flag_order_matches_sublabels(self, raw_df: pd.DataFrame) -> None:
        flags = parse_disease_flags(raw_df["disease_flags"])
        for (di, ht, hy), sub in zip(flags, raw_df["sublabel"], strict=True):
            tokens = sub.split("_")
            assert bool(di) == ("DI" in tokens)
            assert bool(ht) == ("HT" in tokens)
            assert bool(hy) == ("HY" in tokens)

    def test_split_features_targets_quarantines_leakage(
        self, raw_df: pd.DataFrame
    ) -> None:
        X, y = split_features_targets(raw_df)
        assert list(X.columns) == settings.RAW_FEATURE_ORDER
        assert not set(settings.LEAKAGE_COLUMNS) & set(X.columns)
        assert len(y) == len(X)

    def test_split_features_targets_missing_column(self, raw_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            split_features_targets(raw_df.drop(columns=["glucose"]))


class TestValidator:
    def test_valid_data_passes(self, raw_df: pd.DataFrame) -> None:
        result = validate_data(raw_df)
        assert result.passed, result.errors

    def test_empty(self) -> None:
        assert not validate_data(pd.DataFrame()).passed

    def test_missing_columns(self, raw_df: pd.DataFrame) -> None:
        result = validate_data(raw_df.drop(columns=["hdl"]))
        assert not result.passed
        assert any("Missing columns" in e for e in result.errors)

    def test_nulls(self, raw_df: pd.DataFrame) -> None:
        raw_df.loc[0, "glucose"] = np.nan
        assert any("null" in e for e in validate_data(raw_df).errors)

    def test_out_of_range(self, raw_df: pd.DataFrame) -> None:
        raw_df.loc[0, "systolic_bp"] = 900
        assert any("systolic_bp" in e for e in validate_data(raw_df).errors)

    def test_paediatric_ages_are_valid(self, raw_df: pd.DataFrame) -> None:
        raw_df.loc[:5, "age"] = 3
        assert validate_data(raw_df).passed

    def test_unknown_category(self, raw_df: pd.DataFrame) -> None:
        raw_df.loc[0, "smoking"] = "Sometimes"
        assert any("smoking" in e for e in validate_data(raw_df).errors)

    def test_absent_class(self, raw_df: pd.DataFrame) -> None:
        result = validate_data(raw_df[raw_df["sublabel"] != "HT"])
        assert any("'HT' is absent" in e for e in result.errors)

    def test_duplicates_warn(self, raw_df: pd.DataFrame) -> None:
        result = validate_data(pd.concat([raw_df, raw_df.head(3)]))
        assert result.passed
        assert any("duplicate" in w for w in result.warnings)


class TestSplitter:
    def test_fold_sizes(self, raw_df: pd.DataFrame) -> None:
        split = split_data(*split_features_targets(raw_df))
        n = len(raw_df)
        assert len(split.X_train) + len(split.X_val) + len(split.X_test) == n
        assert abs(len(split.X_test) / n - settings.test_size) < 0.01
        assert abs(len(split.X_val) / n - settings.val_size) < 0.01

    def test_stratified_all_classes_everywhere(self, raw_df: pd.DataFrame) -> None:
        split = split_data(*split_features_targets(raw_df))
        n_classes = len(split.classes)
        for y in (split.y_train, split.y_val, split.y_test):
            assert len(np.unique(y)) == n_classes

    def test_normal_class_index(self, raw_df: pd.DataFrame) -> None:
        split = split_data(*split_features_targets(raw_df))
        assert split.classes[normal_class_index(split.label_encoder)] == "N"

    def test_normal_class_missing(self, raw_df: pd.DataFrame) -> None:
        X, y = split_features_targets(raw_df)
        keep = y != "N"
        split = split_data(X[keep.values], y[keep], val_size=0.2, test_size=0.2)
        with pytest.raises(ValueError):
            normal_class_index(split.label_encoder)

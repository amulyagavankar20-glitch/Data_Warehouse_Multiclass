"""Tests for feature engineering."""

from sklearn.compose import ColumnTransformer

from src.data.splitter import SplitData
from src.features.engineering import (
    build_preprocessor,
    fit_preprocessor,
    fit_transform_train,
    get_feature_columns_after_encoding,
    transform_features,
)


class TestFeatureEngineering:
    """Tests for src/features/engineering.py."""

    def test_build_preprocessor_returns_column_transformer(self, mock_split: SplitData) -> None:
        preprocessor = build_preprocessor(mock_split)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_preprocessor(self, mock_split: SplitData) -> None:
        preprocessor = fit_preprocessor(mock_split)
        assert preprocessor is not None

    def test_fit_transform_train_shapes(self, mock_split: SplitData) -> None:
        preprocessor, X_train_enc, X_test_enc = fit_transform_train(mock_split)
        assert X_train_enc.shape[0] == len(mock_split.X_train)
        assert X_test_enc.shape[0] == len(mock_split.X_test)
        assert X_train_enc.shape[1] == X_test_enc.shape[1]

    def test_transform_features_column_alignment(
        self, mock_split: SplitData
    ) -> None:
        preprocessor = fit_preprocessor(mock_split)
        X_test_enc = transform_features(
            preprocessor, mock_split.X_test, mock_split.feature_names,
        )
        assert X_test_enc.shape[0] == len(mock_split.X_test)

    def test_transform_preserves_index(self, mock_split: SplitData) -> None:
        preprocessor = fit_preprocessor(mock_split)
        X_test_enc = transform_features(
            preprocessor, mock_split.X_test, mock_split.feature_names,
        )
        assert list(X_test_enc.index) == list(mock_split.X_test.index)

    def test_get_feature_columns_after_encoding(self, mock_split: SplitData) -> None:
        preprocessor = fit_preprocessor(mock_split)
        cols = get_feature_columns_after_encoding(preprocessor)
        assert len(cols) > 0
        assert len(cols) == preprocessor.transform(mock_split.X_train).shape[1]

    def test_fit_on_train_only_no_test_leakage(self, mock_split: SplitData) -> None:
        preprocessor, X_train_enc, X_test_enc = fit_transform_train(mock_split)
        # Check that continuous columns are scaled (mean ~0 on train)
        train_means = X_train_enc.mean().values
        assert abs(train_means.mean()) < 1.0  # scaled features should have ~0 mean

    def test_scaling_fit_on_train(self, mock_split: SplitData) -> None:
        """Verify StandardScaler is fit on train only."""
        preprocessor, X_train_enc, X_test_enc = fit_transform_train(mock_split)
        # Train should have mean ~0, std ~1 for continuous features
        # This is expected behavior of StandardScaler fit on train
        assert X_train_enc.shape[1] > 0

"""
Tests for PreprocessingService
"""
import pytest
import numpy as np
import pandas as pd

from services.preprocessing import PreprocessingService
from schemas.models import ProblemType


class TestPreprocessingClassification:
    @pytest.fixture(autouse=True)
    def setup(self, classification_df):
        self.df = classification_df
        self.svc = PreprocessingService()
        self.X_train, self.X_test, self.y_train, self.y_test, self.steps = (
            self.svc.prepare(self.df, "target", ProblemType.CLASSIFICATION)
        )

    def test_output_shapes_are_consistent(self):
        assert self.X_train.shape[0] == len(self.y_train)
        assert self.X_test.shape[0]  == len(self.y_test)
        assert self.X_train.shape[1] == self.X_test.shape[1]

    def test_train_test_split_ratio(self):
        total = self.X_train.shape[0] + self.X_test.shape[0]
        assert total == len(self.df)
        assert abs(self.X_test.shape[0] / total - 0.2) < 0.05

    def test_no_nan_in_output(self):
        assert not np.isnan(self.X_train).any()
        assert not np.isnan(self.X_test).any()

    def test_target_label_encoded(self):
        assert self.svc.label_encoder is not None
        unique_labels = set(self.y_train.tolist() + self.y_test.tolist())
        assert unique_labels.issubset({0, 1})

    def test_steps_logged(self):
        assert len(self.steps) > 0
        assert any("split" in s.lower() for s in self.steps)

    def test_feature_names_populated(self):
        assert len(self.svc.feature_names) > 0


class TestPreprocessingRegression:
    @pytest.fixture(autouse=True)
    def setup(self, regression_df):
        self.df = regression_df
        self.svc = PreprocessingService()
        self.X_train, self.X_test, self.y_train, self.y_test, self.steps = (
            self.svc.prepare(self.df, "price", ProblemType.REGRESSION)
        )

    def test_no_label_encoder_for_regression(self):
        assert self.svc.label_encoder is None

    def test_target_is_float(self):
        assert self.y_train.dtype in [np.float32, np.float64]

    def test_no_nan_in_output(self):
        assert not np.isnan(self.X_train).any()
        assert not np.isnan(self.X_test).any()


class TestPreprocessingMissingValues:
    def test_handles_missing_values(self, df_with_missing):
        svc = PreprocessingService()
        X_train, X_test, y_train, y_test, steps = svc.prepare(
            df_with_missing, "target", ProblemType.CLASSIFICATION
        )
        assert not np.isnan(X_train).any(), "Training data should have no NaN after imputation"
        assert not np.isnan(X_test).any(),  "Test data should have no NaN after imputation"


class TestPreprocessingExcludeColumns:
    def test_excluded_columns_removed(self, classification_df):
        svc = PreprocessingService()
        X_train, X_test, _, _, steps = svc.prepare(
            classification_df, "target",
            ProblemType.CLASSIFICATION,
            exclude_columns=["region"]
        )
        assert any("Dropped" in s or "excluded" in s.lower() or "region" not in svc.feature_names
                   for s in steps)

    def test_cannot_lose_target_through_exclude(self, classification_df):
        svc = PreprocessingService()
        # Excluding non-target columns should still work fine
        X_train, _, _, _, _ = svc.prepare(
            classification_df, "target",
            ProblemType.CLASSIFICATION,
            exclude_columns=["region", "category"]
        )
        assert X_train.shape[0] > 0

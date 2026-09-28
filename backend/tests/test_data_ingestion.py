"""
Tests for DataIngestionService
"""
import io
import os
import pytest
import pandas as pd
import numpy as np

from services.data_ingestion import DataIngestionService
from schemas.models import ProblemType, ColumnType


@pytest.fixture
def service(tmp_path, monkeypatch):
    svc = DataIngestionService()
    monkeypatch.setattr(svc, "upload_dir", tmp_path)
    svc.upload_dir = tmp_path
    return svc


class TestMetadataExtraction:
    def test_basic_metadata(self, service, classification_df, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        classification_df.to_csv(csv_path, index=False)

        meta = service.extract_metadata(
            classification_df, "test-id", "test.csv", csv_path
        )

        assert meta.dataset_id == "test-id"
        assert meta.rows == len(classification_df)
        assert meta.columns == len(classification_df.columns)
        assert len(meta.column_info) == len(classification_df.columns)

    def test_numeric_column_detection(self, service, classification_df, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        classification_df.to_csv(csv_path, index=False)
        meta = service.extract_metadata(classification_df, "id", "f.csv", csv_path)

        num_cols = [c.name for c in meta.column_info if c.column_type == ColumnType.NUMERIC]
        assert "age" in num_cols
        assert "income" in num_cols
        assert "score" in num_cols

    def test_categorical_column_detection(self, service, classification_df, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        classification_df.to_csv(csv_path, index=False)
        meta = service.extract_metadata(classification_df, "id", "f.csv", csv_path)

        cat_cols = [c.name for c in meta.column_info if c.column_type == ColumnType.CATEGORICAL]
        assert "category" in cat_cols
        assert "region" in cat_cols

    def test_missing_value_reporting(self, service, df_with_missing, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        df_with_missing.to_csv(csv_path, index=False)
        meta = service.extract_metadata(df_with_missing, "id", "f.csv", csv_path)

        assert meta.has_missing is True
        age_info = next(c for c in meta.column_info if c.name == "age")
        assert age_info.missing_count == 10
        assert age_info.missing_pct == pytest.approx(5.0, abs=0.1)

    def test_duplicate_detection(self, service, tmp_path):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)
        meta = service.extract_metadata(df, "id", "f.csv", csv_path)
        assert meta.duplicate_rows == 1

    def test_no_duplicates(self, service, classification_df, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        classification_df.to_csv(csv_path, index=False)
        meta = service.extract_metadata(classification_df, "id", "f.csv", csv_path)
        assert meta.duplicate_rows == 0


class TestProblemTypeDetection:
    def test_detects_classification_binary(self, service):
        s = pd.Series([0, 1, 0, 1, 0])
        assert service._detect_problem_type(s) == ProblemType.CLASSIFICATION

    def test_detects_classification_categorical_string(self, service):
        s = pd.Series(["cat", "dog", "cat", "bird"])
        assert service._detect_problem_type(s) == ProblemType.CLASSIFICATION

    def test_detects_regression_continuous(self, service):
        np.random.seed(0)
        s = pd.Series(np.random.uniform(0, 10_000, 500))
        assert service._detect_problem_type(s) == ProblemType.REGRESSION

    def test_detects_classification_low_cardinality_numeric(self, service):
        # Only 5 unique values out of 100 → classification
        s = pd.Series([1, 2, 3, 4, 5] * 20)
        assert service._detect_problem_type(s) == ProblemType.CLASSIFICATION


class TestTargetDetection:
    def test_keyword_target(self, service):
        df = pd.DataFrame({"age": [1], "income": [2], "target": [0]})
        assert service._detect_target_column(df) == "target"

    def test_keyword_label(self, service):
        df = pd.DataFrame({"x": [1], "y": [2], "label": [0]})
        assert service._detect_target_column(df) == "label"

    def test_fallback_last_column(self, service):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [0]})
        assert service._detect_target_column(df) == "c"


class TestValidation:
    def test_valid_dataset_passes(self, service, classification_df):
        valid, errors = service.validate_dataset(classification_df, "target")
        assert valid is True
        assert errors == []

    def test_too_few_rows(self, service):
        df = pd.DataFrame({"x": range(5), "y": range(5)})
        valid, errors = service.validate_dataset(df, "y")
        assert valid is False
        assert any("too small" in e for e in errors)

    def test_missing_target_column(self, service, classification_df):
        valid, errors = service.validate_dataset(classification_df, "nonexistent")
        assert valid is False
        assert any("not found" in e for e in errors)

    def test_too_few_columns(self, service):
        df = pd.DataFrame({"only": range(50)})
        valid, errors = service.validate_dataset(df, "only")
        assert valid is False
        assert any("at least 2" in e for e in errors)

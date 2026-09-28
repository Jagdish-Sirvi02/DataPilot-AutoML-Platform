"""
DataPilot – Data Ingestion & Validation Service.
"""
from __future__ import annotations
import io
import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from backend.schemas.models import (
    DatasetMetadata, ColumnInfo, ColumnType, ProblemType,
)
from backend.utils.config import settings

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Save, load, validate and summarise CSV datasets."""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # ── Upload ─────────────────────────────────────────────────────────────────

    def save_upload(self, filename: str, content: bytes) -> Tuple[str, str]:
        """Persist raw bytes; return (dataset_id, absolute_filepath)."""
        dataset_id = str(uuid.uuid4())
        stem = Path(filename).stem[:60]          # guard against long names
        safe_name = f"{dataset_id}_{stem}.csv"
        filepath = self.upload_dir / safe_name

        with open(filepath, "wb") as fh:
            fh.write(content)

        logger.info(f"Dataset saved: {filepath}  ({len(content)/1024:.1f} KB)")
        return dataset_id, str(filepath)

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, str]:
        files = sorted(self.upload_dir.glob(f"{dataset_id}_*.csv"))
        if not files:
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found in upload store")
        filepath = files[0]
        df = pd.read_csv(filepath, low_memory=False)
        return df, str(filepath)

    def load_from_bytes(self, content: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(content), low_memory=False)

    # ── Metadata ───────────────────────────────────────────────────────────────

    def extract_metadata(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        filename: str,
        filepath: str,
    ) -> DatasetMetadata:
        column_info: List[ColumnInfo] = []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []

        for col in df.columns:
            dtype_str = str(df[col].dtype)
            missing = int(df[col].isna().sum())
            unique = int(df[col].nunique(dropna=True))
            samples = [str(v) for v in df[col].dropna().head(5).tolist()]

            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = ColumnType.NUMERIC
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_type = ColumnType.DATETIME
            elif pd.api.types.is_bool_dtype(df[col]):
                col_type = ColumnType.BOOLEAN
                categorical_cols.append(col)
            elif unique / max(len(df), 1) > 0.9 and unique > 100:
                col_type = ColumnType.TEXT   # high-cardinality string → treat as text
            else:
                col_type = ColumnType.CATEGORICAL
                categorical_cols.append(col)

            column_info.append(ColumnInfo(
                name=col,
                dtype=dtype_str,
                column_type=col_type,
                missing_count=missing,
                missing_pct=round(missing / max(len(df), 1) * 100, 2),
                unique_count=unique,
                sample_values=samples,
            ))

        target_col = self._detect_target(df)
        problem_type = self._detect_problem_type(df[target_col]) if target_col else None
        size_kb = round(os.path.getsize(filepath) / 1024, 2) if os.path.exists(filepath) else 0.0

        return DatasetMetadata(
            dataset_id=dataset_id,
            filename=filename,
            rows=len(df),
            columns=len(df.columns),
            file_size_kb=size_kb,
            target_column=target_col,
            problem_type=problem_type,
            column_info=column_info,
            duplicate_rows=int(df.duplicated().sum()),
            has_missing=bool(df.isnull().any().any()),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate_dataset(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        if len(df) < 30:
            errors.append(f"Too few rows ({len(df)}); minimum is 30.")

        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found.")
            return False, errors  # No point checking further

        if len(df.columns) < 2:
            errors.append("Need at least 2 columns (1 feature + target).")

        target_missing_pct = df[target_column].isna().sum() / len(df) * 100
        if target_missing_pct > 20:
            errors.append(
                f"Target column has {target_missing_pct:.1f}% missing values (max 20%)."
            )

        feature_cols = [c for c in df.columns if c != target_column]
        if not feature_cols:
            errors.append("No feature columns found after excluding target.")

        return len(errors) == 0, errors

    # ── Private helpers ────────────────────────────────────────────────────────

    def _detect_target(self, df: pd.DataFrame) -> Optional[str]:
        keywords = [
            "target", "label", "class", "output", "result", "outcome",
            "y", "churn", "survived", "diagnosis", "price", "salary",
            "income", "revenue", "sales", "score", "default", "fraud",
        ]
        lower_map = {c.lower(): c for c in df.columns}
        for kw in keywords:
            if kw in lower_map:
                return lower_map[kw]
        return df.columns[-1]   # fallback: last column

    def _detect_problem_type(self, series: pd.Series) -> ProblemType:
        if not pd.api.types.is_numeric_dtype(series):
            return ProblemType.CLASSIFICATION
        nunique = series.nunique()
        ratio = nunique / max(len(series), 1)
        if nunique <= 20 or ratio < 0.05:
            return ProblemType.CLASSIFICATION
        return ProblemType.REGRESSION

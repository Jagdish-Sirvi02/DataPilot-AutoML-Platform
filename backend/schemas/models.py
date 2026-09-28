"""
DataPilot – All Pydantic request/response schemas.
No auth dependencies — open-access platform.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION     = "regression"


class ColumnType(str, Enum):
    NUMERIC     = "numeric"
    CATEGORICAL = "categorical"
    DATETIME    = "datetime"
    BOOLEAN     = "boolean"
    TEXT        = "text"


# ── Dataset ────────────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name:          str
    dtype:         str
    column_type:   ColumnType
    missing_count: int
    missing_pct:   float
    unique_count:  int
    sample_values: List[str]


class DatasetMetadata(BaseModel):
    dataset_id:           str
    filename:             str
    rows:                 int
    columns:              int
    file_size_kb:         float
    target_column:        Optional[str]          = None
    problem_type:         Optional[ProblemType]  = None
    column_info:          List[ColumnInfo]
    duplicate_rows:       int
    has_missing:          bool
    numeric_columns:      List[str]
    categorical_columns:  List[str]


# ── AutoML ─────────────────────────────────────────────────────────────────────

class AutoMLRequest(BaseModel):
    dataset_id:      str
    target_column:   str
    problem_type:    Optional[ProblemType] = None
    exclude_columns: Optional[List[str]]   = []

    @field_validator("target_column")
    @classmethod
    def target_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target_column cannot be empty")
        return v.strip()


class ModelMetrics(BaseModel):
    model_name: str
    # Classification
    accuracy:  Optional[float] = None
    precision: Optional[float] = None
    recall:    Optional[float] = None
    f1_score:  Optional[float] = None
    roc_auc:   Optional[float] = None
    # Regression
    rmse: Optional[float] = None
    mae:  Optional[float] = None
    r2:   Optional[float] = None
    # Universal
    cv_score_mean:    float
    cv_score_std:     float
    training_time_sec: float


class AutoMLResult(BaseModel):
    job_id:              str
    dataset_id:          str
    problem_type:        ProblemType
    target_column:       str
    best_model:          str
    best_score:          float
    models:              List[ModelMetrics]
    feature_importance:  Optional[Dict[str, float]] = None
    preprocessing_steps: List[str]
    workflow_path:       Optional[str]        = None
    visualizations:      Dict[str, str]       = {}


# ── Job ────────────────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id:           str
    status:           str
    progress:         int
    current_step:     str
    steps_completed:  List[str]
    error:            Optional[str]              = None
    results:          Optional[Dict[str, Any]]   = None
    created_at:       Optional[str]              = None
    updated_at:       Optional[str]              = None

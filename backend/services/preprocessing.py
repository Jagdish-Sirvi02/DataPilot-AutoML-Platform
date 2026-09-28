"""
DataPilot – Preprocessing & Feature Engineering Service.
Builds a reproducible sklearn ColumnTransformer pipeline.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler,
)

from backend.schemas.models import ProblemType
from backend.utils.config import settings

logger = logging.getLogger(__name__)


class PreprocessingService:
    """
    Automated preprocessing:
    - Drop irrelevant / datetime columns
    - Remove duplicates & rows with missing target
    - Impute + scale numeric features
    - Impute + one-hot encode categorical features
    - Label-encode classification targets
    - Stratified train/test split
    """

    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names_in_: List[str] = []
        self.steps_applied: List[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def prepare(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: ProblemType,
        exclude_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Full pipeline.
        Returns: X_train, X_test, y_train, y_test, list_of_steps_applied
        """
        self.steps_applied = []
        df = df.copy()
        exclude_columns = list(exclude_columns or [])

        # 1 ── Drop excluded + datetime columns
        drop_set = set(exclude_columns)
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        drop_set.update(dt_cols)
        drop_set.discard(target_column)
        if drop_set:
            df.drop(columns=list(drop_set), inplace=True, errors="ignore")
            self.steps_applied.append(f"Dropped {len(drop_set)} column(s): {', '.join(sorted(drop_set))}")

        # 2 ── Remove duplicate rows
        n_before = len(df)
        df.drop_duplicates(inplace=True)
        n_removed = n_before - len(df)
        if n_removed:
            self.steps_applied.append(f"Removed {n_removed} duplicate rows")

        # 3 ── Drop rows with missing target
        df.dropna(subset=[target_column], inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.steps_applied.append("Dropped rows with missing target values")

        # 4 ── Separate X / y
        X = df.drop(columns=[target_column])
        y = df[target_column].copy()
        self.feature_names_in_ = X.columns.tolist()

        # 5 ── Encode target
        if problem_type == ProblemType.CLASSIFICATION:
            self.label_encoder = LabelEncoder()
            y = pd.Series(
                self.label_encoder.fit_transform(y.astype(str)),
                name=target_column,
            )
            self.steps_applied.append(
                f"Label-encoded target '{target_column}' "
                f"({len(self.label_encoder.classes_)} classes)"
            )
        else:
            y = y.astype(float)

        # 6 ── Column type detection
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        raw_cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Drop extremely high-cardinality categoricals (>50 unique) – too noisy
        high_card = [c for c in raw_cat_cols if X[c].nunique() > 50]
        if high_card:
            X.drop(columns=high_card, inplace=True, errors="ignore")
            raw_cat_cols = [c for c in raw_cat_cols if c not in high_card]
            self.steps_applied.append(
                f"Dropped {len(high_card)} high-cardinality column(s): {', '.join(high_card)}"
            )

        categorical_cols = raw_cat_cols

        # 7 ── Build sklearn transformers
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("impute",  SimpleImputer(strategy="most_frequent")),
            ("encode",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipe, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_pipe, categorical_cols))

        self.preprocessor = ColumnTransformer(transformers, remainder="drop")
        self.steps_applied += [
            f"Median imputation on {len(numeric_cols)} numeric column(s)",
            "StandardScaler applied to numeric features",
            f"One-Hot encoding on {len(categorical_cols)} categorical column(s)",
        ]

        # 8 ── Stratified split
        stratify = None
        if problem_type == ProblemType.CLASSIFICATION:
            counts = y.value_counts()
            if counts.min() >= 2:   # need at least 2 samples per class for stratify
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
            stratify=stratify,
        )
        self.steps_applied.append(
            f"Train/test split {int((1-settings.TEST_SIZE)*100)}%/"
            f"{int(settings.TEST_SIZE*100)}%"
            + (" (stratified)" if stratify is not None else "")
        )

        # 9 ── Fit + transform
        X_train_out = self.preprocessor.fit_transform(X_train)
        X_test_out  = self.preprocessor.transform(X_test)

        logger.info(
            f"Preprocessing done. Train={X_train_out.shape}, Test={X_test_out.shape}"
        )
        return (
            X_train_out, X_test_out,
            y_train.values, y_test.values,
            self.steps_applied,
        )

    # ── Feature names ──────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> List[str]:
        """Return output feature names after ColumnTransformer."""
        if self.preprocessor is None:
            return []
        names: List[str] = []
        for name, transformer, cols in self.preprocessor.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                try:
                    enc: OneHotEncoder = transformer.named_steps["encode"]
                    names.extend(enc.get_feature_names_out(cols).tolist())
                except Exception:
                    names.extend(cols)
        return names

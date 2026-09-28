"""
DataPilot Test Fixtures
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Make backend importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Sample DataFrames ──────────────────────────────────────────────────────

@pytest.fixture
def classification_df():
    """Synthetic binary classification dataset."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":      np.random.randint(18, 70, n),
        "income":   np.random.normal(50000, 15000, n),
        "score":    np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "region":   np.random.choice(["North", "South", "East", "West"], n),
        "target":   np.random.choice([0, 1], n),
    })


@pytest.fixture
def regression_df():
    """Synthetic regression dataset."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 4)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "feature_a": X[:, 0],
        "feature_b": X[:, 1],
        "feature_c": X[:, 2],
        "feature_d": X[:, 3],
        "category":  np.random.choice(["P", "Q"], n),
        "price":     y,
    })


@pytest.fixture
def df_with_missing(classification_df):
    """DataFrame with injected missing values."""
    df = classification_df.copy()
    idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[idx[:10], "age"] = np.nan
    df.loc[idx[10:], "category"] = np.nan
    return df


@pytest.fixture
def tiny_df():
    """Minimal valid DataFrame."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
               1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
               1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })

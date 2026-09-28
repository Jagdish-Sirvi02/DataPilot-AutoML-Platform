"""
DataPilot – Automated EDA Engine.
Generates a full suite of exploratory charts and summary statistics.
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backend.utils.config import settings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Chart theme ───────────────────────────────────────────────────────────────
BG_DARK  = "#0f172a"
BG_PANEL = "#1e293b"
BORDER   = "#334155"
TEXT_LT  = "#e2e8f0"
TEXT_DIM = "#94a3b8"
PALETTE  = ["#6366f1","#22d3ee","#f59e0b","#10b981","#f43f5e","#a78bfa","#fb923c","#34d399"]

_rc = {
    "figure.facecolor": BG_DARK,  "axes.facecolor": BG_PANEL,
    "axes.edgecolor":   BORDER,   "axes.labelcolor": TEXT_LT,
    "text.color":       TEXT_LT,  "xtick.color":     TEXT_DIM,
    "ytick.color":      TEXT_DIM, "grid.color":      BORDER,
    "legend.facecolor": BG_PANEL, "legend.edgecolor": BORDER,
    "figure.titlesize": 14,       "axes.titlesize":   11,
    "axes.labelsize":   9,
}
plt.rcParams.update(_rc)


class EDAEngine:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.viz_dir = Path(settings.VIZ_DIR) / dataset_id
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ─────────────────────────────────────────────────────────────────

    def run_full_eda(
        self, df: pd.DataFrame, target_column: str
    ) -> Dict[str, str]:
        """Return {chart_name: url_path} for all generated charts."""
        results: Dict[str, str] = {}
        tasks = [
            ("correlation_heatmap",    lambda: self._correlation_heatmap(df)),
            ("target_distribution",    lambda: self._target_distribution(df, target_column)),
            ("missing_values",         lambda: self._missing_values(df)),
            ("feature_distributions",  lambda: self._feature_distributions(df, target_column)),
            ("outlier_boxplots",       lambda: self._outlier_boxplots(df, target_column)),
            ("class_balance",          lambda: self._class_balance(df, target_column)),
        ]
        for name, fn in tasks:
            try:
                url = fn()
                if url:
                    results[name] = url
            except Exception as exc:
                logger.warning(f"EDA chart '{name}' failed: {exc}")
        logger.info(f"EDA complete – {len(results)} charts for dataset {self.dataset_id}")
        return results

    def compute_summary_stats(self, df: pd.DataFrame) -> Dict:
        out: Dict = {}
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            out["numeric"] = (
                numeric.describe()
                       .round(4)
                       .to_dict()
            )
        cat = df.select_dtypes(include=["object", "category"])
        if not cat.empty:
            out["categorical"] = {
                col: {
                    "unique": int(df[col].nunique()),
                    "top_5": df[col].value_counts().head(5).to_dict(),
                }
                for col in cat.columns
            }
        return out

    # ── Private chart helpers ──────────────────────────────────────────────────

    def _save(self, name: str) -> str:
        path = self.viz_dir / f"{name}.png"
        plt.savefig(
            str(path), dpi=110, bbox_inches="tight",
            facecolor=BG_DARK, transparent=False,
        )
        plt.close("all")
        return f"/static/visualizations/{self.dataset_id}/{name}.png"

    def _correlation_heatmap(self, df: pd.DataFrame) -> Optional[str]:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return None
        cols = num.columns[:18]
        corr = num[cols].corr()
        size = max(7, len(cols) * 0.55)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask,
            annot=len(cols) <= 12, fmt=".2f", annot_kws={"size": 7},
            cmap="coolwarm", center=0, linewidths=0.4,
            ax=ax, cbar_kws={"shrink": 0.75},
        )
        ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=12)
        plt.tight_layout()
        return self._save("correlation_heatmap")

    def _target_distribution(self, df: pd.DataFrame, target: str) -> Optional[str]:
        if target not in df.columns:
            return None
        series = df[target].dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            ax.hist(series, bins=40, color=PALETTE[0], edgecolor=BG_DARK, alpha=0.85)
            ax.set_ylabel("Count")
        else:
            counts = series.value_counts().head(20)
            bars = ax.bar(
                counts.index.astype(str), counts.values,
                color=PALETTE[: len(counts)], edgecolor=BG_DARK,
            )
            ax.bar_label(bars, fmt="%d", color=TEXT_LT, fontsize=8)
            ax.set_ylabel("Count")
        ax.set_xlabel(target)
        ax.set_title(f"Target Distribution  ·  {target}", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        return self._save("target_distribution")

    def _missing_values(self, df: pd.DataFrame) -> Optional[str]:
        missing = (df.isnull().sum().pipe(lambda s: s[s > 0]).sort_values())
        if missing.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.38)))
        bars = ax.barh(
            missing.index,
            missing.values / len(df) * 100,
            color=PALETTE[4], alpha=0.85,
        )
        ax.bar_label(bars, fmt="%.1f%%", color=TEXT_LT, fontsize=8, padding=3)
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing Values by Column", fontweight="bold")
        ax.set_xlim(0, 115)
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        return self._save("missing_values")

    def _feature_distributions(
        self, df: pd.DataFrame, target: str
    ) -> Optional[str]:
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target
        ][:9]
        if not num_cols:
            return None
        ncols = 3
        nrows = (len(num_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
        axes_flat = np.array(axes).flatten()
        for i, col in enumerate(num_cols):
            ax = axes_flat[i]
            ax.hist(df[col].dropna(), bins=30, color=PALETTE[i % len(PALETTE)],
                    alpha=0.8, edgecolor=BG_DARK)
            ax.set_title(col, fontsize=8, fontweight="bold")
            ax.grid(alpha=0.25)
        for j in range(len(num_cols), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("Numeric Feature Distributions", fontweight="bold", fontsize=12, y=1.01)
        plt.tight_layout()
        return self._save("feature_distributions")

    def _outlier_boxplots(self, df: pd.DataFrame, target: str) -> Optional[str]:
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target
        ][:12]
        if not num_cols:
            return None
        fig, ax = plt.subplots(figsize=(max(10, len(num_cols) * 1.1), 5))
        bp = ax.boxplot(
            [df[c].dropna().values for c in num_cols],
            patch_artist=True,
            medianprops=dict(color=PALETTE[2], linewidth=2),
            whiskerprops=dict(color=TEXT_DIM),
            capprops=dict(color=TEXT_DIM),
            flierprops=dict(marker="o", color=PALETTE[4], alpha=0.4, markersize=3),
        )
        for patch, color in zip(bp["boxes"], PALETTE * 3):
            patch.set_facecolor(color); patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(num_cols) + 1))
        ax.set_xticklabels(num_cols, rotation=30, ha="right", fontsize=8)
        ax.set_title("Outlier Detection – Box Plots", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        return self._save("outlier_boxplots")

    def _class_balance(self, df: pd.DataFrame, target: str) -> Optional[str]:
        if target not in df.columns:
            return None
        series = df[target].dropna()
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            return None   # regression target – skip
        vc = series.value_counts()
        if len(vc) < 2 or len(vc) > 30:
            return None
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            vc.values,
            labels=vc.index.astype(str),
            autopct="%1.1f%%",
            colors=PALETTE[: len(vc)],
            startangle=140,
            wedgeprops=dict(edgecolor=BG_DARK, linewidth=1.2),
        )
        for t in texts + autotexts:
            t.set_color(TEXT_LT)
            t.set_fontsize(9)
        ax.set_title("Class Balance", fontweight="bold")
        plt.tight_layout()
        return self._save("class_balance")

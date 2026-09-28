"""
DataPilot – Model Result Visualization Generator.
Produces: model comparison, feature importance, confusion matrix,
ROC curve (classification) or residuals (regression).
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from schemas.models import ModelMetrics, ProblemType
from utils.config import settings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BG_DARK  = "#0f172a"; BG_PANEL = "#1e293b"; BORDER = "#334155"
TEXT_LT  = "#e2e8f0"; TEXT_DIM = "#94a3b8"
PALETTE  = ["#6366f1","#22d3ee","#f59e0b","#10b981","#f43f5e","#a78bfa","#fb923c","#34d399"]

plt.rcParams.update({
    "figure.facecolor": BG_DARK,  "axes.facecolor":  BG_PANEL,
    "axes.edgecolor":   BORDER,   "axes.labelcolor": TEXT_LT,
    "text.color":       TEXT_LT,  "xtick.color":     TEXT_DIM,
    "ytick.color":      TEXT_DIM, "grid.color":      BORDER,
    "legend.facecolor": BG_PANEL, "legend.edgecolor": BORDER,
})


class VisualizationGenerator:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.viz_dir = Path(settings.VIZ_DIR) / dataset_id
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        best_model: Any,
        best_model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_metrics: List[ModelMetrics],
        feature_importance: Dict[str, float],
        problem_type: ProblemType,
        label_encoder=None,
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}
        tasks = [
            ("model_comparison",  lambda: self._model_comparison(model_metrics, problem_type)),
            ("feature_importance", lambda: self._feature_importance(feature_importance, best_model_name)
                                         if feature_importance else None),
        ]
        if problem_type == ProblemType.CLASSIFICATION:
            tasks += [
                ("confusion_matrix", lambda: self._confusion_matrix(
                    best_model, X_test, y_test, best_model_name, label_encoder)),
                ("roc_curve",        lambda: self._roc_curve(
                    best_model, X_test, y_test, best_model_name)),
            ]
        else:
            tasks += [
                ("residuals", lambda: self._residuals(best_model, X_test, y_test, best_model_name)),
            ]

        for name, fn in tasks:
            try:
                url = fn()
                if url:
                    results[name] = url
            except Exception as exc:
                logger.warning(f"Viz '{name}' failed: {exc}")

        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save(self, name: str) -> str:
        path = self.viz_dir / f"{name}.png"
        plt.savefig(str(path), dpi=110, bbox_inches="tight",
                    facecolor=BG_DARK, transparent=False)
        plt.close("all")
        return f"/static/visualizations/{self.dataset_id}/{name}.png"

    # Model comparison bar chart
    def _model_comparison(
        self, metrics: List[ModelMetrics], pt: ProblemType
    ) -> str:
        if pt == ProblemType.CLASSIFICATION:
            vals  = [m.f1_score  or 0 for m in metrics]
            label = "F1 Score (weighted)"
        else:
            vals  = [m.r2 or 0 for m in metrics]
            label = "R² Score"
        names  = [m.model_name for m in metrics]
        best_v = max(vals)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [PALETTE[0] if v == best_v else PALETTE[2] for v in vals]
        bars = ax.barh(names, vals, color=colors, alpha=0.85, height=0.55, edgecolor=BG_DARK)
        ax.bar_label(bars, fmt="%.4f", color=TEXT_LT, fontsize=9, padding=4)
        ax.set_xlabel(label)
        ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
        ax.set_xlim(0, min(1.15, best_v * 1.25 + 0.05))
        ax.axvline(best_v, color=PALETTE[0], linestyle="--", alpha=0.35, lw=1)
        ax.grid(axis="x", alpha=0.25)
        # Annotate best
        best_idx = vals.index(best_v)
        ax.annotate("⭐ Best", xy=(best_v, best_idx),
                    xytext=(best_v + 0.01, best_idx),
                    color=PALETTE[2], fontsize=8, va="center")
        plt.tight_layout()
        return self._save("model_comparison")

    # Feature importance horizontal bar chart
    def _feature_importance(self, importance: Dict[str, float], name: str) -> str:
        top = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
        )
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.42)))
        bars = ax.barh(
            list(top.keys()), list(top.values()),
            color=PALETTE[0], alpha=0.85, edgecolor=BG_DARK,
        )
        ax.bar_label(bars, fmt="%.4f", color=TEXT_LT, fontsize=8, padding=3)
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance  ·  {name}", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        return self._save("feature_importance")

    # Confusion matrix heatmap
    def _confusion_matrix(
        self, model, X_test, y_test, name, label_encoder
    ) -> str:
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        labels = (
            [str(c) for c in label_encoder.classes_]
            if label_encoder else None
        )
        sz = max(6, cm.shape[0] * 0.9)
        fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap="Blues", linewidths=0.5,
            xticklabels=labels, yticklabels=labels,
            ax=ax, annot_kws={"size": 10, "color": "white"},
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix  ·  {name}", fontweight="bold")
        plt.tight_layout()
        return self._save("confusion_matrix")

    # ROC curve (binary + multi-class)
    def _roc_curve(self, model, X_test, y_test, name) -> Optional[str]:
        if not hasattr(model, "predict_proba"):
            return None
        proba    = model.predict_proba(X_test)
        n_cls    = proba.shape[1]
        classes  = np.unique(y_test)

        fig, ax = plt.subplots(figsize=(7, 6))
        if n_cls == 2:
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[0], lw=2.2,
                    label=f"AUC = {roc_auc_val:.4f}")
        else:
            y_bin = label_binarize(y_test, classes=classes)
            for i, cls in enumerate(classes[:8]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.5,
                        label=f"Class {cls}  AUC={roc_auc_val:.3f}")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve  ·  {name}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        return self._save("roc_curve")

    # Residuals (regression)
    def _residuals(self, model, X_test, y_test, name) -> str:
        y_pred    = model.predict(X_test)
        residuals = y_test - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(y_pred, residuals, color=PALETTE[0], alpha=0.45, s=18)
        ax1.axhline(0, color=PALETTE[4], lw=1.5, linestyle="--")
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("Residual")
        ax1.set_title("Residuals vs Predicted", fontweight="bold")
        ax1.grid(alpha=0.25)

        ax2.hist(residuals, bins=40, color=PALETTE[1], alpha=0.8, edgecolor=BG_DARK)
        ax2.axvline(0, color=PALETTE[4], lw=1.5, linestyle="--")
        ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")
        ax2.set_title("Residual Distribution", fontweight="bold")
        ax2.grid(alpha=0.25)

        fig.suptitle(f"Regression Diagnostics  ·  {name}", fontweight="bold", fontsize=12)
        plt.tight_layout()
        return self._save("residuals")

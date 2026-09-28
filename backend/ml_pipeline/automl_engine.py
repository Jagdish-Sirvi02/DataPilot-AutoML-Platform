"""
DataPilot – AutoML Training Engine.
Trains multiple classifiers / regressors; uses Optuna for the best two.
"""
from __future__ import annotations
import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

from schemas.models import ModelMetrics, ProblemType
from utils.config import settings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# Cap Optuna trials in CI / fast mode
_TRIALS = settings.OPTUNA_TRIALS
_CV     = settings.CV_FOLDS
_RS     = settings.RANDOM_STATE


class AutoMLEngine:
    """Trains all models and selects the best one."""

    def __init__(self, problem_type: ProblemType):
        self.problem_type   = problem_type
        self.best_model     = None
        self.best_model_name: str = ""
        self.best_score: float = -np.inf
        self.trained_models: Dict[str, Any] = {}

    # ── Public ─────────────────────────────────────────────────────────────────

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> List[ModelMetrics]:

        factories = (
            self._clf_factories(X_train, y_train)
            if self.problem_type == ProblemType.CLASSIFICATION
            else self._reg_factories(X_train, y_train)
        )
        total = len(factories)
        results: List[ModelMetrics] = []

        for idx, (name, make_model) in enumerate(factories.items()):
            if progress_callback:
                pct = 20 + int((idx / total) * 60)
                progress_callback(f"Training {name} …", pct)

            logger.info(f"Training: {name}")
            try:
                t0 = time.perf_counter()
                model = make_model()
                model.fit(X_train, y_train)
                elapsed = round(time.perf_counter() - t0, 2)

                scoring = "f1_weighted" if self.problem_type == ProblemType.CLASSIFICATION else "r2"
                n_splits = min(_CV, int(np.unique(y_train).size))
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=n_splits, scoring=scoring, n_jobs=-1,
                )
                metrics = self._evaluate(name, model, X_test, y_test, cv_scores, elapsed)
                results.append(metrics)
                self.trained_models[name] = model

                primary = (
                    metrics.f1_score if self.problem_type == ProblemType.CLASSIFICATION
                    else metrics.r2
                ) or 0.0
                if primary > self.best_score:
                    self.best_score      = primary
                    self.best_model      = model
                    self.best_model_name = name

                logger.info(
                    f"  ✓ {name}: cv={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
                    f"time={elapsed}s"
                )
            except Exception as exc:
                logger.warning(f"  ✗ {name} failed: {exc}")

        # Sort by primary metric descending
        key = (
            (lambda m: m.f1_score or 0)
            if self.problem_type == ProblemType.CLASSIFICATION
            else (lambda m: m.r2 or -999)
        )
        results.sort(key=key, reverse=True)
        return results

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        if self.best_model is None:
            return {}
        try:
            if hasattr(self.best_model, "feature_importances_"):
                imps = self.best_model.feature_importances_
            elif hasattr(self.best_model, "coef_"):
                coef = self.best_model.coef_
                imps = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                return {}
            n = min(len(feature_names), len(imps))
            paired = sorted(
                zip(feature_names[:n], imps[:n]),
                key=lambda x: abs(x[1]), reverse=True,
            )
            return {k: round(float(v), 6) for k, v in paired[:25]}
        except Exception as exc:
            logger.warning(f"Feature importance extraction failed: {exc}")
            return {}

    # ── Model factories ────────────────────────────────────────────────────────

    def _clf_factories(self, X, y) -> Dict[str, Callable]:
        return {
            "Logistic Regression": lambda: LogisticRegression(
                max_iter=2000, random_state=_RS, n_jobs=-1
            ),
            "Random Forest": lambda: self._tune_rf_clf(X, y),
            "Gradient Boosting": lambda: self._tune_gb_clf(X, y),
            "SVM": lambda: SVC(
                kernel="rbf", probability=True, random_state=_RS
            ),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        }

    def _reg_factories(self, X, y) -> Dict[str, Callable]:
        return {
            "Ridge Regression": lambda: Ridge(random_state=_RS),
            "Random Forest": lambda: self._tune_rf_reg(X, y),
            "Gradient Boosting": lambda: self._tune_gb_reg(X, y),
            "SVR": lambda: SVR(kernel="rbf"),
        }

    # ── Optuna tuning helpers ──────────────────────────────────────────────────

    def _tune_rf_clf(self, X, y) -> RandomForestClassifier:
        def obj(trial):
            m = RandomForestClassifier(
                n_estimators  = trial.suggest_int("n_est",  50, 300),
                max_depth     = trial.suggest_int("depth",  3,  20),
                min_samples_split = trial.suggest_int("mss", 2, 10),
                random_state  = _RS, n_jobs=-1,
            )
            return cross_val_score(m, X, y, cv=3, scoring="f1_weighted", n_jobs=-1).mean()
        s = optuna.create_study(direction="maximize")
        s.optimize(obj, n_trials=min(_TRIALS, 25), show_progress_bar=False)
        return RandomForestClassifier(**s.best_params, random_state=_RS, n_jobs=-1)

    def _tune_gb_clf(self, X, y) -> GradientBoostingClassifier:
        def obj(trial):
            m = GradientBoostingClassifier(
                n_estimators  = trial.suggest_int("n_est",  50, 300),
                learning_rate = trial.suggest_float("lr",   0.01, 0.3, log=True),
                max_depth     = trial.suggest_int("depth",  2,   8),
                random_state  = _RS,
            )
            return cross_val_score(m, X, y, cv=3, scoring="f1_weighted", n_jobs=-1).mean()
        s = optuna.create_study(direction="maximize")
        s.optimize(obj, n_trials=min(_TRIALS, 25), show_progress_bar=False)
        return GradientBoostingClassifier(**s.best_params, random_state=_RS)

    def _tune_rf_reg(self, X, y) -> RandomForestRegressor:
        def obj(trial):
            m = RandomForestRegressor(
                n_estimators = trial.suggest_int("n_est", 50, 300),
                max_depth    = trial.suggest_int("depth",  3,  20),
                random_state = _RS, n_jobs=-1,
            )
            return cross_val_score(m, X, y, cv=3, scoring="r2", n_jobs=-1).mean()
        s = optuna.create_study(direction="maximize")
        s.optimize(obj, n_trials=min(_TRIALS, 25), show_progress_bar=False)
        return RandomForestRegressor(**s.best_params, random_state=_RS, n_jobs=-1)

    def _tune_gb_reg(self, X, y) -> GradientBoostingRegressor:
        def obj(trial):
            m = GradientBoostingRegressor(
                n_estimators  = trial.suggest_int("n_est",  50, 300),
                learning_rate = trial.suggest_float("lr",   0.01, 0.3, log=True),
                max_depth     = trial.suggest_int("depth",  2,   8),
                random_state  = _RS,
            )
            return cross_val_score(m, X, y, cv=3, scoring="r2", n_jobs=-1).mean()
        s = optuna.create_study(direction="maximize")
        s.optimize(obj, n_trials=min(_TRIALS, 25), show_progress_bar=False)
        return GradientBoostingRegressor(**s.best_params, random_state=_RS)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(
        self, name: str, model, X_test, y_test, cv_scores, elapsed: float
    ) -> ModelMetrics:
        y_pred = model.predict(X_test)
        base = dict(
            model_name      = name,
            cv_score_mean   = round(float(cv_scores.mean()), 4),
            cv_score_std    = round(float(cv_scores.std()),  4),
            training_time_sec = elapsed,
        )

        if self.problem_type == ProblemType.CLASSIFICATION:
            base.update(
                accuracy  = round(float(accuracy_score(y_test, y_pred)),                              4),
                precision = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                recall    = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),    4),
                f1_score  = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),       4),
            )
            # ROC-AUC (works for binary and multiclass)
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    n_cls = proba.shape[1]
                    if n_cls == 2:
                        auc = roc_auc_score(y_test, proba[:, 1])
                    else:
                        auc = roc_auc_score(
                            y_test, proba, multi_class="ovr", average="weighted"
                        )
                    base["roc_auc"] = round(float(auc), 4)
            except Exception:
                pass
        else:
            mse = mean_squared_error(y_test, y_pred)
            base.update(
                rmse = round(float(np.sqrt(mse)),                      4),
                mae  = round(float(mean_absolute_error(y_test, y_pred)), 4),
                r2   = round(float(r2_score(y_test, y_pred)),           4),
            )

        return ModelMetrics(**base)

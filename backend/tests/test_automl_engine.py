"""
Tests for AutoMLEngine
"""
import pytest
import numpy as np
from unittest.mock import patch

from ml_pipeline.automl_engine import AutoMLEngine
from schemas.models import ProblemType


@pytest.fixture
def clf_data(classification_df):
    from services.preprocessing import PreprocessingService
    svc = PreprocessingService()
    return svc.prepare(classification_df, "target", ProblemType.CLASSIFICATION)


@pytest.fixture
def reg_data(regression_df):
    from services.preprocessing import PreprocessingService
    svc = PreprocessingService()
    return svc.prepare(regression_df, "price", ProblemType.REGRESSION)


class TestClassificationEngine:
    def test_trains_multiple_models(self, clf_data):
        X_train, X_test, y_train, y_test, _ = clf_data
        engine = AutoMLEngine(ProblemType.CLASSIFICATION)

        # Patch optuna to use fewer trials for speed
        with patch("ml_pipeline.automl_engine.AutoMLEngine._tune_random_forest_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(n_estimators=10, random_state=42)), \
             patch("ml_pipeline.automl_engine.AutoMLEngine._tune_gradient_boosting_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["GradientBoostingClassifier"]).GradientBoostingClassifier(n_estimators=10, random_state=42)):

            results = engine.train_all(X_train, y_train, X_test, y_test)

        assert len(results) >= 3

    def test_best_model_set(self, clf_data):
        X_train, X_test, y_train, y_test, _ = clf_data
        engine = AutoMLEngine(ProblemType.CLASSIFICATION)

        with patch("ml_pipeline.automl_engine.AutoMLEngine._tune_random_forest_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(n_estimators=5, random_state=42)), \
             patch("ml_pipeline.automl_engine.AutoMLEngine._tune_gradient_boosting_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["GradientBoostingClassifier"]).GradientBoostingClassifier(n_estimators=5, random_state=42)):
            engine.train_all(X_train, y_train, X_test, y_test)

        assert engine.best_model is not None
        assert engine.best_model_name is not None
        assert engine.best_score > -np.inf

    def test_classification_metrics_present(self, clf_data):
        X_train, X_test, y_train, y_test, _ = clf_data
        engine = AutoMLEngine(ProblemType.CLASSIFICATION)

        with patch("ml_pipeline.automl_engine.AutoMLEngine._tune_random_forest_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(n_estimators=5, random_state=42)), \
             patch("ml_pipeline.automl_engine.AutoMLEngine._tune_gradient_boosting_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["GradientBoostingClassifier"]).GradientBoostingClassifier(n_estimators=5, random_state=42)):
            results = engine.train_all(X_train, y_train, X_test, y_test)

        for r in results:
            assert r.accuracy  is not None
            assert r.f1_score  is not None
            assert r.precision is not None
            assert r.recall    is not None
            assert r.cv_score_mean is not None
            # Regression metrics should be None
            assert r.rmse is None
            assert r.mae  is None
            assert r.r2   is None

    def test_results_sorted_by_f1(self, clf_data):
        X_train, X_test, y_train, y_test, _ = clf_data
        engine = AutoMLEngine(ProblemType.CLASSIFICATION)

        with patch("ml_pipeline.automl_engine.AutoMLEngine._tune_random_forest_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(n_estimators=5, random_state=42)), \
             patch("ml_pipeline.automl_engine.AutoMLEngine._tune_gradient_boosting_clf",
                   return_value=__import__("sklearn.ensemble", fromlist=["GradientBoostingClassifier"]).GradientBoostingClassifier(n_estimators=5, random_state=42)):
            results = engine.train_all(X_train, y_train, X_test, y_test)

        f1s = [r.f1_score for r in results if r.f1_score is not None]
        assert f1s == sorted(f1s, reverse=True)


class TestRegressionEngine:
    def test_regression_metrics_present(self, reg_data):
        X_train, X_test, y_train, y_test, _ = reg_data
        engine = AutoMLEngine(ProblemType.REGRESSION)

        with patch("ml_pipeline.automl_engine.AutoMLEngine._tune_random_forest_reg",
                   return_value=__import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor(n_estimators=5, random_state=42)), \
             patch("ml_pipeline.automl_engine.AutoMLEngine._tune_gradient_boosting_reg",
                   return_value=__import__("sklearn.ensemble", fromlist=["GradientBoostingRegressor"]).GradientBoostingRegressor(n_estimators=5, random_state=42)):
            results = engine.train_all(X_train, y_train, X_test, y_test)

        for r in results:
            assert r.rmse is not None
            assert r.mae  is not None
            assert r.r2   is not None
            # Classification metrics should be None
            assert r.accuracy  is None
            assert r.f1_score  is None


class TestFeatureImportance:
    def test_returns_dict(self, clf_data):
        X_train, X_test, y_train, y_test, _ = clf_data
        engine = AutoMLEngine(ProblemType.CLASSIFICATION)
        from sklearn.ensemble import RandomForestClassifier
        engine.best_model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
        engine.best_model_name = "Random Forest"

        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
        importance = engine.get_feature_importance(feature_names)

        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_returns_empty_when_no_model(self):
        engine = AutoMLEngine(ProblemType.CLASSIFICATION)
        result = engine.get_feature_importance(["a", "b"])
        assert result == {}

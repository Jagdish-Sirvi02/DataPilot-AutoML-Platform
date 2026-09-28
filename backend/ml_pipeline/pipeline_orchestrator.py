"""
DataPilot – Pipeline Orchestrator
Coordinates every stage of the AutoML pipeline in sequence,
updating job progress after each step.
"""
import logging
from typing import Optional, Callable, Dict, Any

from backend.services.data_ingestion import DataIngestionService
from backend.services.eda_engine import EDAEngine
from backend.services.preprocessing import PreprocessingService
from ml_pipeline.automl_engine import AutoMLEngine
from ml_pipeline.visualization_generator import VisualizationGenerator
from workflow_generator.orange_generator import OrangeWorkflowGenerator
from backend.schemas.models import AutoMLRequest, AutoMLResult, ProblemType
from backend.utils.job_state import job_registry, JobStatus

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Runs the full DataPilot pipeline in a single blocking call
    (intended to be executed inside a background thread).

    Steps
    -----
    1.  Load & validate dataset
    2.  Detect problem type
    3.  Automated EDA
    4.  Preprocessing & feature engineering
    5.  AutoML training (with Optuna tuning)
    6.  Feature importance extraction
    7.  Result visualisations
    8.  Orange3 .ows workflow generation
    9.  Persist results to job registry
    """

    def __init__(self, job_id: str, request: AutoMLRequest):
        self.job_id    = job_id
        self.request   = request
        self.dataset_id = request.dataset_id
        self._ingestion = DataIngestionService()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the pipeline; return the serialised result dict."""
        try:
            return self._run()
        except Exception as exc:
            logger.error(f"Pipeline {self.job_id} failed: {exc}", exc_info=True)
            job_registry.update_job(
                self.job_id,
                status=JobStatus.FAILED,
                current_step="Pipeline failed",
                error=str(exc),
            )
            raise

    # ── Private implementation ────────────────────────────────────────────────

    def _progress(self, step: str, pct: int, completed: Optional[str] = None):
        job_registry.update_job(
            self.job_id,
            status=JobStatus.RUNNING,
            progress=pct,
            current_step=step,
            step_completed=completed,
        )
        logger.info(f"[{pct:3d}%] {step}")

    def _run(self) -> Dict[str, Any]:
        p = self._progress  # alias for brevity

        # ── 1. Load ────────────────────────────────────────────────────────
        p("Loading dataset…", 5)
        df, filepath = self._ingestion.load_dataset(self.dataset_id)
        p("Dataset loaded", 8, "Dataset loaded")

        # ── 2. Validate ───────────────────────────────────────────────────
        p("Validating dataset…", 10)
        valid, errors = self._ingestion.validate_dataset(df, self.request.target_column)
        if not valid:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")
        p("Validation passed", 12, "Validation passed")

        # ── 3. Problem type ───────────────────────────────────────────────
        p("Detecting problem type…", 13)
        if self.request.problem_type:
            problem_type = self.request.problem_type
        else:
            meta = self._ingestion.extract_metadata(
                df, self.dataset_id, "dataset.csv", filepath
            )
            problem_type = meta.problem_type or ProblemType.CLASSIFICATION
        p(f"Problem type: {problem_type.value}", 15, f"Problem type: {problem_type.value}")

        # ── 4. EDA ────────────────────────────────────────────────────────
        p("Running exploratory data analysis…", 16)
        eda = EDAEngine(self.dataset_id)
        eda_vizs  = eda.run_full_eda(df, self.request.target_column)
        p("EDA complete", 22, "Exploratory Data Analysis")

        # ── 5. Preprocessing ──────────────────────────────────────────────
        p("Preprocessing data…", 23)
        preprocessor = PreprocessingService()
        X_train, X_test, y_train, y_test, prep_steps = preprocessor.prepare(
            df,
            self.request.target_column,
            problem_type,
            self.request.exclude_columns or [],
        )
        p("Preprocessing complete", 30, "Preprocessing & Feature Engineering")

        # ── 6. AutoML training ────────────────────────────────────────────
        p("Training ML models…", 31)

        def _training_cb(msg: str, pct: int):
            p(msg, pct)

        engine = AutoMLEngine(problem_type)
        model_metrics = engine.train_all(
            X_train, y_train, X_test, y_test,
            progress_callback=_training_cb,
        )
        if not model_metrics:
            raise RuntimeError("No models trained successfully.")
        p("Training complete", 82, "AutoML Model Training")

        # ── 7. Feature importance ─────────────────────────────────────────
        p("Extracting feature importance…", 83)
        feature_importance = engine.get_feature_importance(preprocessor.feature_names)
        p("Feature importance ready", 85, "Feature Importance Analysis")

        # ── 8. Result visualisations ──────────────────────────────────────
        p("Generating visualisations…", 86)
        viz_gen = VisualizationGenerator(self.dataset_id)
        result_vizs = viz_gen.generate_all(
            best_model      = engine.best_model,
            best_model_name = engine.best_model_name,
            X_test          = X_test,
            y_test          = y_test,
            model_metrics   = model_metrics,
            feature_importance = feature_importance,
            problem_type    = problem_type,
            label_encoder   = preprocessor.label_encoder,
        )
        all_vizs = {k: v for k, v in {**eda_vizs, **result_vizs}.items() if v}
        p("Visualisations ready", 92, "Visualization Generation")

        # ── 9. Orange workflow ────────────────────────────────────────────
        p("Generating Orange3 workflow…", 93)
        orange_gen = OrangeWorkflowGenerator(self.dataset_id)
        workflow_path = orange_gen.generate(
            dataset_path       = filepath,
            best_model_name    = engine.best_model_name,
            all_models         = model_metrics,
            problem_type       = problem_type,
            preprocessing_steps = prep_steps,
            target_column      = self.request.target_column,
        )
        p("Orange3 workflow ready", 98, "Orange3 Workflow Generation")

        # ── 10. Finalise ──────────────────────────────────────────────────
        result = AutoMLResult(
            job_id              = self.job_id,
            dataset_id          = self.dataset_id,
            problem_type        = problem_type,
            target_column       = self.request.target_column,
            best_model          = engine.best_model_name,
            best_score          = round(float(engine.best_score), 4),
            models              = model_metrics,
            feature_importance  = feature_importance,
            preprocessing_steps = prep_steps,
            workflow_path       = workflow_path,
            visualizations      = all_vizs,
        ).model_dump()

        job_registry.update_job(
            self.job_id,
            status        = JobStatus.COMPLETED,
            progress      = 100,
            current_step  = "Pipeline complete!",
            step_completed= "Pipeline complete",
            results       = result,
        )
        logger.info(
            f"Pipeline {self.job_id} complete. "
            f"Best: {engine.best_model_name} (score={engine.best_score:.4f})"
        )
        return result

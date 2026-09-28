"""
DataPilot – AutoML API routes
POST /run-automl      → launches pipeline as a background task, returns job_id
GET  /job-status/{id} → poll progress (0-100 %) and step log
GET  /jobs            → list all jobs
"""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.schemas.models import AutoMLRequest, JobStatusResponse
from backend.utils.job_state import job_registry, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_pipeline_bg(job_id: str, request: AutoMLRequest):
    """Target function executed in FastAPI's background thread pool."""
    from ml_pipeline.pipeline_orchestrator import PipelineOrchestrator
    try:
        PipelineOrchestrator(job_id, request).run()
    except Exception as exc:
        logger.error(f"Background pipeline {job_id} crashed: {exc}", exc_info=True)


@router.post("/run-automl", status_code=202)
async def run_automl(request: AutoMLRequest, background_tasks: BackgroundTasks):
    """
    Start the AutoML pipeline.
    Returns immediately with a job_id; poll /job-status/{job_id} for progress.
    """
    from services.data_ingestion import DataIngestionService
    try:
        DataIngestionService().load_dataset(request.dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")

    job_id = job_registry.create_job(request.dataset_id)
    logger.info(f"Job {job_id} created for dataset {request.dataset_id}")
    background_tasks.add_task(_run_pipeline_bg, job_id, request)

    return {"job_id": job_id, "message": "AutoML pipeline started.", "status": "pending"}


@router.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the progress and status of a running or completed pipeline job."""
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(**job)


@router.get("/jobs")
async def list_jobs():
    """Return all jobs (newest-first)."""
    return sorted(job_registry.list_jobs(), key=lambda j: j["created_at"], reverse=True)

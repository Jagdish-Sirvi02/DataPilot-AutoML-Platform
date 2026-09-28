"""
DataPilot – Results API routes
GET /model-results/{job_id}    → full AutoML result payload
GET /visualizations/{job_id}   → chart name → URL map
"""
import logging
from fastapi import APIRouter, HTTPException

from backend.utils.job_state import job_registry, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_completed(job_id: str) -> dict:
    """Shared guard: raise 404/409 when job isn't ready."""
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job status is '{job['status']}'. "
                "Results are only available once the job has completed."
            ),
        )
    return job


@router.get("/model-results/{job_id}")
async def get_model_results(job_id: str):
    """
    Return the complete AutoML result object for a finished job.
    Includes best model name, all model metrics, feature importance,
    preprocessing steps, and paths to generated visualisations.
    """
    job = _require_completed(job_id)
    return job["results"]


@router.get("/visualizations/{job_id}")
async def get_visualizations(job_id: str):
    """
    Return a dictionary mapping chart names to their static image URLs.
    Charts include: correlation_heatmap, target_distribution, confusion_matrix,
    roc_curve, feature_importance, model_comparison, etc.
    """
    job = _require_completed(job_id)
    res = job.get("results", {})
    return {
        "job_id":       job_id,
        "best_model":   res.get("best_model"),
        "problem_type": res.get("problem_type"),
        "visualizations": res.get("visualizations", {}),
    }

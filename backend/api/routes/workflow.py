"""
DataPilot - Workflow API Routes
Download Orange3 .ows workflow files.
"""
import os
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.utils.job_state import job_registry, JobStatus
from backend.utils.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/download-workflow/{job_id}")
async def download_workflow(job_id: str):
    """
    Download the Orange3 .ows workflow file generated for a completed job.
    Open this file directly in Orange3 to explore your ML pipeline visually.
    """
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job status is '{job['status']}'. Workflow only available after completion."
        )

    results = job.get("results", {})
    workflow_path = results.get("workflow_path")

    if not workflow_path or not os.path.exists(workflow_path):
        raise HTTPException(status_code=404, detail="Workflow file not found. Pipeline may not have generated it.")

    filename = Path(workflow_path).name
    return FileResponse(
        path=workflow_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Workflow-Info": f"Best model: {results.get('best_model', 'N/A')}",
        }
    )


@router.get("/workflow-info/{job_id}")
async def get_workflow_info(job_id: str):
    """Get metadata about the generated workflow without downloading it."""
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not yet completed")

    results = job.get("results", {})
    workflow_path = results.get("workflow_path", "")
    file_size_kb = 0

    if workflow_path and os.path.exists(workflow_path):
        file_size_kb = round(os.path.getsize(workflow_path) / 1024, 2)

    return {
        "job_id": job_id,
        "workflow_available": bool(workflow_path and os.path.exists(workflow_path)),
        "filename": Path(workflow_path).name if workflow_path else None,
        "file_size_kb": file_size_kb,
        "best_model": results.get("best_model"),
        "problem_type": results.get("problem_type"),
        "download_url": f"/api/download-workflow/{job_id}",
        "instructions": (
            "Download this .ows file and open it in Orange3 (https://orangedatamining.com) "
            "to visually explore and modify your AutoML pipeline."
        ),
    }

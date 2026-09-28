"""
DataPilot – Dataset API routes
POST /upload-dataset   → validates, saves, returns metadata
GET  /dataset-summary  → re-derive metadata from saved file
"""
import io
import logging
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.data_ingestion import DataIngestionService
from backend.schemas.models import DatasetMetadata
from backend.utils.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
_svc = DataIngestionService()


@router.post("/upload-dataset", response_model=DatasetMetadata, status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file.
    Returns full metadata including auto-detected target column and problem type.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {settings.MAX_FILE_SIZE_MB} MB.",
        )

    try:
        dataset_id, filepath = _svc.save_upload(file.filename, content)
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV contains no data rows.")
        meta = _svc.extract_metadata(df, dataset_id, file.filename, filepath)
        logger.info(f"Uploaded {dataset_id}: {meta.rows}r × {meta.columns}c")
        return meta
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Upload failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not process dataset: {exc}")


@router.get("/dataset-summary/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset_summary(dataset_id: str):
    """Return metadata for a previously uploaded dataset."""
    try:
        df, filepath = _svc.load_dataset(dataset_id)
        return _svc.extract_metadata(df, dataset_id, "dataset.csv", filepath)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

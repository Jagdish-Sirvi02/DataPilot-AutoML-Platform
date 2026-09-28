"""
DataPilot – AutoML Platform
FastAPI application entry point
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import dataset, automl, results, workflow
from backend.utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("datapilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.WORKFLOW_DIR, exist_ok=True)
    os.makedirs(settings.VIZ_DIR, exist_ok=True)
    logger.info("DataPilot started. Directories initialised.")
    yield
    logger.info("DataPilot shutting down.")


app = FastAPI(
    title="DataPilot AutoML Platform",
    description="Upload a dataset → get a full ML pipeline + Orange3 workflow.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated visualisation PNGs as static files
app.mount("/static", StaticFiles(directory=settings.OUTPUT_DIR), name="static")

app.include_router(dataset.router,  prefix="/api", tags=["Dataset"])
app.include_router(automl.router,   prefix="/api", tags=["AutoML"])
app.include_router(results.router,  prefix="/api", tags=["Results"])
app.include_router(workflow.router, prefix="/api", tags=["Workflow"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "DataPilot AutoML Platform"}

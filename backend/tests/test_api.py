"""
DataPilot FastAPI Integration Tests
Uses TestClient to exercise the full request/response cycle.
"""
import io
import os
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Point config to temp dirs before importing app
import tempfile
_tmpdir = tempfile.mkdtemp()
os.environ.setdefault("UPLOAD_DIR",   os.path.join(_tmpdir, "uploads"))
os.environ.setdefault("OUTPUT_DIR",   os.path.join(_tmpdir, "outputs"))
os.environ.setdefault("WORKFLOW_DIR", os.path.join(_tmpdir, "workflows"))

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_csv_bytes():
    np.random.seed(99)
    n = 120
    df = pd.DataFrame({
        "age":    np.random.randint(18, 70, n),
        "income": np.random.normal(50_000, 12_000, n),
        "score":  np.random.uniform(0, 100, n),
        "region": np.random.choice(["North", "South"], n),
        "target": np.random.choice([0, 1], n),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ── /health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ── /api/upload-dataset ────────────────────────────────────────────────────

class TestUploadDataset:
    def test_upload_valid_csv(self, client, sample_csv_bytes):
        r = client.post(
            "/api/upload-dataset",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        assert r.status_code == 201
        data = r.json()
        assert "dataset_id" in data
        assert data["rows"] == 120
        assert data["columns"] == 5
        assert data["has_missing"] is False

    def test_upload_rejects_non_csv(self, client):
        r = client.post(
            "/api/upload-dataset",
            files={"file": ("data.xlsx", b"fake content", "application/vnd.ms-excel")},
        )
        assert r.status_code == 400
        assert "CSV" in r.json()["detail"]

    def test_upload_rejects_empty_file(self, client):
        r = client.post(
            "/api/upload-dataset",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert r.status_code in (400, 500)

    def test_upload_returns_target_column(self, client, sample_csv_bytes):
        r = client.post(
            "/api/upload-dataset",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        data = r.json()
        assert data["target_column"] == "target"

    def test_upload_returns_problem_type(self, client, sample_csv_bytes):
        r = client.post(
            "/api/upload-dataset",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        data = r.json()
        assert data["problem_type"] in ("classification", "regression", None)


# ── /api/dataset-summary ──────────────────────────────────────────────────

class TestDatasetSummary:
    @pytest.fixture(scope="class")
    def dataset_id(self, client, sample_csv_bytes):
        r = client.post(
            "/api/upload-dataset",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        return r.json()["dataset_id"]

    def test_get_summary(self, client, dataset_id):
        r = client.get(f"/api/dataset-summary/{dataset_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["dataset_id"] == dataset_id

    def test_unknown_dataset_404(self, client):
        r = client.get("/api/dataset-summary/nonexistent-id")
        assert r.status_code == 404


# ── /api/run-automl ────────────────────────────────────────────────────────

class TestRunAutoML:
    @pytest.fixture(scope="class")
    def dataset_id(self, client, sample_csv_bytes):
        r = client.post(
            "/api/upload-dataset",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        return r.json()["dataset_id"]

    def test_run_returns_job_id(self, client, dataset_id):
        r = client.post("/api/run-automl", json={
            "dataset_id": dataset_id,
            "target_column": "target",
        })
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_unknown_dataset_returns_404(self, client):
        r = client.post("/api/run-automl", json={
            "dataset_id": "no-such-dataset",
            "target_column": "target",
        })
        assert r.status_code == 404

    def test_job_status_endpoint(self, client, dataset_id):
        # Launch
        r = client.post("/api/run-automl", json={
            "dataset_id": dataset_id,
            "target_column": "target",
        })
        job_id = r.json()["job_id"]

        # Poll immediately — should be pending or running
        r2 = client.get(f"/api/job-status/{job_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "completed", "failed")

    def test_unknown_job_status_404(self, client):
        r = client.get("/api/job-status/fake-job-id")
        assert r.status_code == 404


# ── /api/workflow-info ─────────────────────────────────────────────────────

class TestWorkflowInfo:
    def test_nonexistent_job(self, client):
        r = client.get("/api/workflow-info/fake-job-id")
        assert r.status_code == 404

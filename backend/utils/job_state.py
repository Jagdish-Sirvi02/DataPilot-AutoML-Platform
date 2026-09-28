"""
DataPilot – Job State Manager.
Tries Redis first; falls back to in-process dict when Redis is unavailable.
"""
import json
import uuid
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class JobRegistry:
    """
    Dual-backend job store.
    • Redis available  → persists across restarts, survives multi-worker deploys.
    • Redis missing    → plain dict, fine for single-process dev.
    """

    def __init__(self):
        self._mem:   Dict[str, Dict] = {}
        self._redis  = None
        self._ttl    = 60 * 60 * 24   # 24 h
        self._try_connect()

    def _try_connect(self):
        try:
            import redis as redis_lib
            from utils.config import settings
            r = redis_lib.from_url(settings.REDIS_URL, socket_connect_timeout=2)
            r.ping()
            self._redis = r
            logger.info("JobRegistry: using Redis backend")
        except Exception as exc:
            logger.warning(f"JobRegistry: Redis unavailable ({exc}) – using in-memory store")

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_job(self, dataset_id: str) -> str:
        job_id = str(uuid.uuid4())
        now    = datetime.utcnow().isoformat()
        job: Dict[str, Any] = {
            "job_id":          job_id,
            "dataset_id":      dataset_id,
            "status":          JobStatus.PENDING,
            "progress":        0,
            "current_step":    "Queued",
            "steps_completed": [],
            "error":           None,
            "created_at":      now,
            "updated_at":      now,
            "results":         None,
        }
        self._write(job_id, job)
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status:         Optional[JobStatus] = None,
        progress:       Optional[int]       = None,
        current_step:   Optional[str]       = None,
        step_completed: Optional[str]       = None,
        error:          Optional[str]       = None,
        results:        Optional[Dict]      = None,
    ):
        job = self._read(job_id)
        if not job:
            return
        if status         is not None: job["status"]       = status
        if progress       is not None: job["progress"]     = progress
        if current_step   is not None: job["current_step"] = current_step
        if step_completed is not None and step_completed not in job["steps_completed"]:
            job["steps_completed"].append(step_completed)
        if error   is not None: job["error"]   = error
        if results is not None: job["results"] = results
        job["updated_at"] = datetime.utcnow().isoformat()
        self._write(job_id, job)

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self._read(job_id)

    def list_jobs(self) -> List[Dict]:
        """Return all known jobs (newest-first by created_at)."""
        jobs = self._all()
        return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _key(self, job_id: str) -> str:
        return f"datapilot:job:{job_id}"

    def _write(self, job_id: str, job: Dict):
        if self._redis:
            try:
                self._redis.setex(self._key(job_id), self._ttl, json.dumps(job, default=str))
                return
            except Exception:
                pass
        self._mem[job_id] = job

    def _read(self, job_id: str) -> Optional[Dict]:
        if self._redis:
            try:
                raw = self._redis.get(self._key(job_id))
                return json.loads(raw) if raw else None
            except Exception:
                pass
        return self._mem.get(job_id)

    def _all(self) -> List[Dict]:
        if self._redis:
            try:
                keys = self._redis.keys("datapilot:job:*")
                jobs = []
                for k in keys:
                    raw = self._redis.get(k)
                    if raw:
                        jobs.append(json.loads(raw))
                return jobs
            except Exception:
                pass
        return list(self._mem.values())


# Singleton
job_registry = JobRegistry()

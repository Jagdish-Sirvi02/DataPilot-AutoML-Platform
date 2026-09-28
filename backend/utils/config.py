"""
DataPilot – Application settings via Pydantic Settings v2.
All values can be overridden with environment variables or a .env file.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str  = "DataPilot"
    DEBUG:    bool = False

    # ── Paths (auto-derived from BASE_DIR if not set in env) ───────────────
    BASE_DIR:     str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR:   str = ""
    OUTPUT_DIR:   str = ""
    WORKFLOW_DIR: str = ""

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL:             str = "redis://localhost:6379/0"
    CELERY_BROKER_URL:     str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # ── ML ─────────────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int   = 100
    CV_FOLDS:         int   = 5
    OPTUNA_TRIALS:    int   = 20
    TEST_SIZE:        float = 0.2
    RANDOM_STATE:     int   = 42

    def model_post_init(self, _ctx):
        data = os.path.join(self.BASE_DIR, "data")
        if not self.UPLOAD_DIR:
            object.__setattr__(self, "UPLOAD_DIR",   os.path.join(data, "uploads"))
        if not self.OUTPUT_DIR:
            object.__setattr__(self, "OUTPUT_DIR",   os.path.join(data, "outputs"))
        if not self.WORKFLOW_DIR:
            object.__setattr__(self, "WORKFLOW_DIR", os.path.join(data, "workflows"))

    @property
    def VIZ_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "visualizations")


settings = Settings()

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class ApiSettings:
    app_name: str = field(default_factory=lambda: os.getenv("RETENTION_API_NAME", "Retention ROI Backend API"))
    api_prefix: str = field(default_factory=lambda: os.getenv("RETENTION_API_PREFIX", "/api/v1"))
    host: str = field(default_factory=lambda: os.getenv("RETENTION_API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("RETENTION_API_PORT", "8000")))
    reload: bool = field(default_factory=lambda: os.getenv("RETENTION_API_RELOAD", "false").lower() == "true")
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("RETENTION_API_DATA_DIR", "data/raw")))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("RETENTION_MODEL_DIR", "models")))
    result_dir: Path = field(default_factory=lambda: Path(os.getenv("RETENTION_RESULT_DIR", "results")))
    feature_store_dir: Path = field(default_factory=lambda: Path(os.getenv("RETENTION_FEATURE_STORE_DIR", "data/feature_store")))
    default_budget: int = field(default_factory=lambda: int(os.getenv("RETENTION_API_DEFAULT_BUDGET", "5000000")))
    default_threshold: float = field(default_factory=lambda: float(os.getenv("RETENTION_API_DEFAULT_THRESHOLD", "0.50")))
    redis_url: str = field(default_factory=lambda: os.getenv("RETENTION_REDIS_URL", "redis://redis:6379/0"))
    realtime_stream_key: str = field(default_factory=lambda: os.getenv("RETENTION_REALTIME_STREAM_KEY", "retention:events"))
    realtime_consumer_group: str = field(default_factory=lambda: os.getenv("RETENTION_REALTIME_CONSUMER_GROUP", "retention-risk-scorers"))
    realtime_consumer_name: str = field(default_factory=lambda: os.getenv("RETENTION_REALTIME_CONSUMER_NAME", "retention-risk-worker-1"))
    survival_horizon_days: int = field(default_factory=lambda: int(os.getenv("RETENTION_SURVIVAL_HORIZON_DAYS", "90")))
    survival_test_size: float = field(default_factory=lambda: float(os.getenv("RETENTION_SURVIVAL_TEST_SIZE", "0.20")))
    survival_random_state: int = field(default_factory=lambda: int(os.getenv("RETENTION_SURVIVAL_RANDOM_STATE", "42")))
    survival_penalizer: float = field(default_factory=lambda: float(os.getenv("RETENTION_SURVIVAL_PENALIZER", "0.10")))
    allowed_origins: List[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv(
                "RETENTION_API_ALLOWED_ORIGINS",
                "http://localhost:8501,http://127.0.0.1:8501,http://dashboard:8501,http://localhost:3000,http://127.0.0.1:3000",
            )
        )
    )

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.resolve()

    @property
    def resolved_model_dir(self) -> Path:
        return self.model_dir.resolve()

    @property
    def resolved_result_dir(self) -> Path:
        return self.result_dir.resolve()

    @property
    def resolved_feature_store_dir(self) -> Path:
        return self.feature_store_dir.resolve()


SETTINGS = ApiSettings()

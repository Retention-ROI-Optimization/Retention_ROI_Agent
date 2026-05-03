from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UniversalAutoOpsConfig:
    """Filesystem and runtime settings for the additive Universal AutoOps layer."""

    project_root: Path = Path(os.getenv("RETENTION_PROJECT_ROOT", ".")).resolve()
    data_dir_name: str = os.getenv("RETENTION_DATA_DIR", "data")
    result_dir_name: str = os.getenv("RETENTION_RESULT_DIR", "results")
    model_dir_name: str = os.getenv("RETENTION_MODEL_DIR", "models")
    feature_store_dir_name: str = os.getenv("RETENTION_FEATURE_STORE_DIR", "data/feature_store")
    upload_dir_name: str = os.getenv("RETENTION_UPLOAD_DIR", "data/uploads")
    job_dir_name: str = os.getenv("RETENTION_JOB_DIR", "results/autoops_jobs")
    random_state: int = int(os.getenv("RETENTION_RANDOM_STATE", "42"))
    default_budget: int = int(os.getenv("RETENTION_DEFAULT_BUDGET", "50000000"))
    default_threshold: float = float(os.getenv("RETENTION_DEFAULT_THRESHOLD", "0.50"))
    default_max_customers: int = int(os.getenv("RETENTION_DEFAULT_MAX_CUSTOMERS", "1000"))
    max_train_rows: int = int(os.getenv("RETENTION_AUTOOPS_MAX_TRAIN_ROWS", "200000"))
    max_dashboard_synthetic_events: int = int(os.getenv("RETENTION_AUTOOPS_MAX_SYNTHETIC_EVENTS", "300000"))

    @property
    def data_dir(self) -> Path:
        return self.project_root / self.data_dir_name

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def result_dir(self) -> Path:
        return self.project_root / self.result_dir_name

    @property
    def model_dir(self) -> Path:
        return self.project_root / self.model_dir_name

    @property
    def feature_store_dir(self) -> Path:
        return self.project_root / self.feature_store_dir_name

    @property
    def upload_dir(self) -> Path:
        return self.project_root / self.upload_dir_name

    @property
    def job_dir(self) -> Path:
        return self.project_root / self.job_dir_name

    def ensure_dirs(self) -> None:
        for path in [
            self.raw_dir,
            self.result_dir,
            self.model_dir,
            self.feature_store_dir,
            self.upload_dir,
            self.job_dir,
            self.feature_store_dir / "survival",
        ]:
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = UniversalAutoOpsConfig()

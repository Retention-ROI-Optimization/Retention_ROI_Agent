from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import SETTINGS, UniversalAutoOpsConfig
from .io_utils import read_json, write_json


@dataclass
class JobState:
    job_id: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    result: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class JobManager:
    def __init__(self, config: UniversalAutoOpsConfig = SETTINGS) -> None:
        self.config = config
        self.config.ensure_dirs()
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}

    def _path(self, job_id: str) -> Path:
        return self.config.job_dir / f"{job_id}.json"

    def _persist(self, state: JobState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        write_json(self._path(state.job_id), state.to_dict())
        write_json(self.config.result_dir / "universal_autoops_latest_job.json", state.to_dict())

    def create(self, func: Callable[[], dict[str, Any]]) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        state = JobState(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = state
            self._persist(state)

        def runner() -> None:
            with self._lock:
                state.status = "running"
                self._persist(state)
            try:
                result = func()
                with self._lock:
                    state.status = "ready"
                    state.result = result
                    self._persist(state)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    state.status = "failed"
                    state.error = str(exc)
                    state.traceback = traceback.format_exc()
                    self._persist(state)

        thread = threading.Thread(target=runner, name=f"universal-autoops-{job_id}", daemon=True)
        thread.start()
        return state

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id].to_dict()
        path = self._path(job_id)
        if path.exists():
            return read_json(path)
        return {"job_id": job_id, "status": "not_found"}

    def latest(self) -> dict[str, Any]:
        path = self.config.result_dir / "universal_autoops_latest_job.json"
        if path.exists():
            return read_json(path)
        return {"status": "not_started"}


JOB_MANAGER = JobManager()

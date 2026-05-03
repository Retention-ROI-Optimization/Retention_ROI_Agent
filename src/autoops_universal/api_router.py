from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from .abtest import run_campaign_result_analysis
from .config import SETTINGS
from .io_utils import read_csv_safely, read_json
from .jobs import JOB_MANAGER
from .pipeline import run_universal_onboarding_pipeline
from .validation import InvalidAutoOpsDataset, validate_uploaded_dataset

router = APIRouter(prefix="/api/v1/universal-autoops", tags=["universal-autoops"])


def _save_body_csv(body: bytes, filename: str, prefix: str) -> Path:
    if not body:
        raise HTTPException(status_code=400, detail="CSV body is empty.")
    safe_name = Path(filename or f"{prefix}.csv").name
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"
    SETTINGS.ensure_dirs()
    path = SETTINGS.upload_dir / safe_name
    path.write_bytes(body)
    try:
        read_csv_safely(path, nrows=10)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"CSV parsing failed: {exc}") from exc
    return path


def _parse_mapping_json(mapping_json: str | None) -> dict[str, str] | None:
    if not mapping_json:
        return None
    try:
        payload = json.loads(mapping_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"mapping_json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="mapping_json must be an object like {\"customer_id\": \"회원번호\"}.")
    return {str(k): str(v) for k, v in payload.items() if v is not None and str(v).strip()}


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "retention-roi-universal-autoops", "mode": "additive"}


@router.post("/onboard-csv")
async def onboard_csv(
    request: Request,
    filename: str = Query("uploaded_customers.csv"),
    budget: int = Query(50_000_000, ge=1),
    threshold: float = Query(0.50, ge=0.0, le=1.0),
    max_customers: int = Query(1000, ge=1),
    background: bool = Query(True),
    mapping_json: str | None = Query(None),
) -> dict[str, Any]:
    path = _save_body_csv(await request.body(), filename, "uploaded_customers")
    manual_mapping = _parse_mapping_json(mapping_json)

    try:
        preview_df = read_csv_safely(path, nrows=5000)
        validate_uploaded_dataset(preview_df, manual_mapping=manual_mapping)
    except InvalidAutoOpsDataset as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc), "diagnostics": exc.diagnostics}) from exc

    def run() -> dict[str, Any]:
        return run_universal_onboarding_pipeline(
            path,
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
            manual_mapping=manual_mapping,
        )

    if background:
        job = JOB_MANAGER.create(run)
        return {"status": "accepted", "job_id": job.job_id, "message": "Universal AutoOps job started in background."}
    try:
        return run()["summary"]
    except InvalidAutoOpsDataset as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc), "diagnostics": exc.diagnostics}) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return JOB_MANAGER.get(job_id)


@router.get("/status")
def status() -> dict[str, Any]:
    latest = JOB_MANAGER.latest()
    status_path = SETTINGS.result_dir / "platform_pipeline_status.json"
    payload = read_json(status_path) if status_path.exists() else {}
    if not payload:
        return {"status": latest.get("status", "not_started"), "latest_job": latest, "message": "No universal AutoOps output has been generated yet."}
    return {"status": payload.get("status", "ready"), "latest_job": latest, "pipeline": payload.get("summary", payload)}


@router.get("/artifacts")
def artifacts() -> dict[str, Any]:
    return {
        "status": "ready" if (SETTINGS.result_dir / "platform_pipeline_status.json").exists() else "not_ready",
        "pipeline_status": read_json(SETTINGS.result_dir / "platform_pipeline_status.json"),
        "schema_mapping": read_json(SETTINGS.result_dir / "schema_mapping_report.json"),
        "diagnostics": read_json(SETTINGS.result_dir / "onboarding_diagnostics.json"),
        "latest_job": JOB_MANAGER.latest(),
    }


@router.post("/campaign-results-csv")
async def campaign_results_csv(
    request: Request,
    filename: str = Query("campaign_results.csv"),
    assignment_csv: str | None = Query(None),
) -> dict[str, Any]:
    path = _save_body_csv(await request.body(), filename, "campaign_results")
    assignment_path = Path(assignment_csv) if assignment_csv else SETTINGS.result_dir / "campaign_assignment.csv"
    try:
        return run_campaign_result_analysis(path, assignment_csv=assignment_path if assignment_path.exists() else None)["summary"]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

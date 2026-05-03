from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = os.getenv("RETENTION_UNIVERSAL_AUTOOPS_API_BASE_URL") or os.getenv("RETENTION_AUTOOPS_API_BASE_URL") or "http://localhost:8010"
BASE_URL = BASE_URL.rstrip("/")


def _url(path: str) -> str:
    return f"{BASE_URL}{path}"


def _raise_for_status(r: requests.Response) -> None:
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        try:
            payload = r.json()
            detail = payload.get("detail", payload)
        except Exception:
            detail = r.text
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("error") or str(detail)
            diagnostics = detail.get("diagnostics")
            mapped = diagnostics.get("mapped_fields") if isinstance(diagnostics, dict) else None
            if mapped:
                message += f" | mapped_fields={mapped}"
        else:
            message = str(detail)
        raise RuntimeError(message) from exc


def health(timeout: int = 10) -> dict[str, Any]:
    r = requests.get(_url("/api/v1/universal-autoops/health"), timeout=timeout)
    _raise_for_status(r)
    return r.json()


def status(timeout: int = 15) -> dict[str, Any]:
    r = requests.get(_url("/api/v1/universal-autoops/status"), timeout=timeout)
    _raise_for_status(r)
    return r.json()


def artifacts(timeout: int = 20) -> dict[str, Any]:
    r = requests.get(_url("/api/v1/universal-autoops/artifacts"), timeout=timeout)
    _raise_for_status(r)
    return r.json()


def submit_onboarding_csv(
    data: bytes,
    *,
    filename: str,
    budget: int,
    threshold: float,
    max_customers: int,
    mapping_json: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "filename": filename,
        "budget": budget,
        "threshold": threshold,
        "max_customers": max_customers,
        "background": "true",
    }
    if mapping_json and mapping_json.strip():
        params["mapping_json"] = mapping_json
    r = requests.post(
        _url("/api/v1/universal-autoops/onboard-csv"),
        params=params,
        data=data,
        headers={"Content-Type": "text/csv"},
        timeout=timeout,
    )
    _raise_for_status(r)
    return r.json()


def job(job_id: str, timeout: int = 15) -> dict[str, Any]:
    r = requests.get(_url(f"/api/v1/universal-autoops/jobs/{job_id}"), timeout=timeout)
    _raise_for_status(r)
    return r.json()

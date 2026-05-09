from __future__ import annotations

import json
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from src.api.services.user_live_db import user_live_session
from src.api.services.user_live_actions import update_live_actions_for_customers


def _json_safe(value: Any) -> Any:
    """
    PostgreSQL/SQLAlchemy 결과를 JSONB와 FastAPI 응답으로 안전하게 변환한다.

    drift_check 결과에는 datetime/date/Decimal 등이 포함될 수 있으므로
    json.dumps() 전에 반드시 이 함수를 통과시킨다.
    """
    if value is None:
        return None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def ensure_user_live_job_tables(db_url: str) -> None:
    """
    user-live batch/microbatch job 실행 기록 테이블을 생성한다.

    이벤트 단위 실시간 처리와 분리된 drift check, recent action refresh,
    재학습/재계산 job의 성공/실패 상태를 추적하기 위한 테이블이다.
    """
    with user_live_session(db_url) as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS user_live_job_runs (
            id BIGSERIAL PRIMARY KEY,
            job_name TEXT NOT NULL,
            job_status TEXT NOT NULL,
            started_at TIMESTAMPTZ DEFAULT now(),
            finished_at TIMESTAMPTZ,
            duration_sec FLOAT,
            message TEXT,
            result_payload JSONB DEFAULT '{}'::jsonb
        )
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_user_live_job_runs_started
        ON user_live_job_runs (started_at DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_user_live_job_runs_name_status
        ON user_live_job_runs (job_name, job_status)
        """))


def _start_job(conn, job_name: str) -> int:
    row = conn.execute(
        text("""
        INSERT INTO user_live_job_runs (job_name, job_status)
        VALUES (:job_name, 'running')
        RETURNING id
        """),
        {"job_name": job_name},
    ).mappings().first()

    if row is None:
        raise RuntimeError(f"failed to start user-live job: {job_name}")

    return int(row["id"])


def _finish_job(
    conn,
    job_id: int,
    status: str,
    message: str,
    result: dict[str, Any],
    started_monotonic: float,
) -> None:
    conn.execute(
        text("""
        UPDATE user_live_job_runs
        SET job_status = :status,
            finished_at = now(),
            duration_sec = :duration_sec,
            message = :message,
            result_payload = CAST(:result_payload AS JSONB)
        WHERE id = :job_id
        """),
        {
            "job_id": job_id,
            "status": status,
            "duration_sec": time.monotonic() - started_monotonic,
            "message": message,
            "result_payload": json.dumps(_json_safe(result), ensure_ascii=False),
        },
    )


def _open_job(db_url: str, job_name: str) -> tuple[int, float]:
    """
    job 시작 기록을 별도 트랜잭션으로 즉시 저장한다.
    이후 job 본문이 실패해도 running/failed 기록을 남기기 위함이다.
    """
    ensure_user_live_job_tables(db_url)
    started = time.monotonic()

    with user_live_session(db_url) as conn:
        job_id = _start_job(conn, job_name)

    return job_id, started


def _mark_job_success(
    db_url: str,
    job_id: int,
    message: str,
    result: dict[str, Any],
    started_monotonic: float,
) -> None:
    with user_live_session(db_url) as conn:
        _finish_job(
            conn,
            job_id,
            "success",
            message,
            result,
            started_monotonic,
        )


def _mark_job_failed(
    db_url: str,
    job_id: int,
    message: str,
    result: dict[str, Any],
    started_monotonic: float,
) -> None:
    with user_live_session(db_url) as conn:
        _finish_job(
            conn,
            job_id,
            "failed",
            message,
            result,
            started_monotonic,
        )


def run_live_drift_check(db_url: str) -> dict[str, Any]:
    """
    가벼운 drift check.

    모델 재학습은 하지 않고, 현재 customer_scores/customer_events 분포만 점검한다.
    datetime/Decimal 값은 JSON-safe 값으로 변환해서 반환·저장한다.
    """
    job_id, started = _open_job(db_url, "drift_check")

    try:
        with user_live_session(db_url) as conn:
            current = conn.execute(
                text("""
                SELECT
                    COUNT(*) AS scored_customers,
                    AVG(churn_score) AS avg_churn_score,
                    STDDEV(churn_score) AS std_churn_score,
                    SUM(CASE WHEN churn_score >= 0.7 THEN 1 ELSE 0 END) AS high_risk_customers,
                    MAX(scored_at) AS latest_scored_at
                FROM customer_scores
                """)
            ).mappings().first()

            events = conn.execute(
                text("""
                SELECT
                    COUNT(*) AS event_count,
                    MAX(event_time) AS latest_event_time
                FROM customer_events
                """)
            ).mappings().first()

        result = {
            "scores": dict(current or {}),
            "events": dict(events or {}),
        }
        safe_result = _json_safe(result)

        _mark_job_success(
            db_url,
            job_id,
            "drift check completed",
            safe_result,
            started,
        )

        return {
            "success": True,
            "job_id": job_id,
            "result": safe_result,
        }

    except Exception as exc:
        error_result = {
            "error": str(exc),
            "job_name": "drift_check",
        }
        _mark_job_failed(
            db_url,
            job_id,
            str(exc),
            error_result,
            started,
        )
        raise


def run_recent_action_refresh(
    db_url: str,
    limit: int = 100,
    action_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    최근 scored_at이 갱신된 고객만 action_queue를 다시 갱신한다.

    주의:
    - 전체 최적화 full recompute가 아니다.
    - HTTP 응답과 job_runs.result_payload가 지나치게 커지지 않도록
      records 전체가 아니라 summary와 preview 일부만 반환·저장한다.
    """
    safe_limit = max(int(limit), 0)
    job_id, started = _open_job(db_url, "recent_action_refresh")

    try:
        with user_live_session(db_url) as conn:
            rows = conn.execute(
                text("""
                SELECT customer_id
                FROM customer_scores
                ORDER BY scored_at DESC NULLS LAST
                LIMIT :limit
                """),
                {"limit": safe_limit},
            ).mappings().all()

        customer_ids = [int(row["customer_id"]) for row in rows]

        if not customer_ids:
            result_summary = {
                "requested_customers": 0,
                "scored_customers_found": 0,
                "recommendation_candidates_updated": 0,
                "action_queue_updated": 0,
                "threshold": action_threshold,
            }

            _mark_job_success(
                db_url,
                job_id,
                "no recent customers found",
                result_summary,
                started,
            )

            return {
                "success": True,
                "job_id": job_id,
                "customer_count": 0,
                "result": result_summary,
                "preview_records": [],
            }

        refresh_result = update_live_actions_for_customers(
            db_url=db_url,
            customer_ids=customer_ids,
            threshold=action_threshold,
        )

        result_summary = {
            "requested_customers": refresh_result.get("requested_customers", len(customer_ids)),
            "scored_customers_found": refresh_result.get("scored_customers_found"),
            "recommendation_candidates_updated": refresh_result.get("recommendation_candidates_updated"),
            "action_queue_updated": refresh_result.get("action_queue_updated"),
            "threshold": refresh_result.get("threshold", action_threshold),
            "min_expected_roi": refresh_result.get("min_expected_roi"),
            "min_expected_profit": refresh_result.get("min_expected_profit"),
        }

        preview_records = (refresh_result.get("records") or [])[:20]
        safe_summary = _json_safe(result_summary)
        safe_preview = _json_safe(preview_records)

        _mark_job_success(
            db_url,
            job_id,
            "recent action refresh completed",
            safe_summary,
            started,
        )

        return {
            "success": True,
            "job_id": job_id,
            "customer_count": len(customer_ids),
            "result": safe_summary,
            "preview_records": safe_preview,
        }

    except Exception as exc:
        error_result = {
            "error": str(exc),
            "limit": safe_limit,
            "action_threshold": action_threshold,
            "job_name": "recent_action_refresh",
        }
        _mark_job_failed(
            db_url,
            job_id,
            str(exc),
            error_result,
            started,
        )
        raise


def get_user_live_job_status(db_url: str, limit: int = 20) -> dict[str, Any]:
    ensure_user_live_job_tables(db_url)

    with user_live_session(db_url) as conn:
        rows = conn.execute(
            text("""
            SELECT *
            FROM user_live_job_runs
            ORDER BY started_at DESC
            LIMIT :limit
            """),
            {"limit": int(limit)},
        ).mappings().all()

    return {
        "success": True,
        "records": [_json_safe(dict(row)) for row in rows],
    }

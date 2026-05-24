from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.api.dependencies import get_settings
from src.api.services.user_live_db import (
    init_user_live_tables,
    user_live_session,
)
from src.api.settings import ApiSettings
from src.api.services.cache import invalidate_user_live_cache
from pathlib import Path

from src.api.services.user_live_seed import (
    get_user_live_seed_status,
    seed_user_live_from_artifacts,
)
from src.api.services.user_live_scoring import (
    get_user_live_scores,
    score_changed_customers,
)
from src.api.services.user_live_actions import (
    get_live_action_queue,
    get_live_recommendation_candidates,
    update_live_actions_for_customers,
)
from src.api.services.user_live_jobs import (
    get_user_live_job_status,
    run_live_drift_check,
    run_recent_action_refresh,
)

router = APIRouter(prefix="/user-live", tags=["user-live"])


StandardEventType = Literal[
    "visit",
    "page_view",
    "browse",
    "search",
    "add_to_cart",
    "remove_from_cart",
    "purchase",
    "support_contact",
    "refund",
    "coupon_open",
    "coupon_redeem",
    "login",
    "logout",
    "other",
]


class UserEventIn(BaseModel):
    """
    자사 서비스에서 들어오는 고객 행동 이벤트 1건.

    source_event_id:
        외부 시스템의 이벤트 ID. 있으면 중복 적재 방지에 사용한다.
    customer_id:
        내부 고객 ID. 2단계에서는 int 기준으로 받는다.
    event_type:
        표준화된 이벤트 타입.
    event_time:
        이벤트 발생 시각.
    amount:
        구매/환불/장바구니 금액 등. 없으면 0.
    raw_payload:
        원본 이벤트 전체. 디버깅/추후 feature 확장용.
    """
    customer_id: int = Field(..., ge=1)
    event_type: StandardEventType
    event_time: datetime
    amount: float = 0.0
    source_event_id: str | None = None
    item_category: str | None = None
    channel: str | None = None
    session_id: str | None = None
    raw_payload: dict[str, Any] | None = None


class UserEventBatchIn(BaseModel):
    events: list[UserEventIn] = Field(default_factory=list)


def _normalize_event_time(value: datetime) -> datetime:
    """
    timezone이 없는 datetime이 들어오면 UTC로 간주한다.
    실제 운영에서는 KST/UTC 정책을 하나로 정해야 한다.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _counter_column_for_event(event_type: str) -> str | None:
    """
    이벤트 타입별로 customer_feature_state에서 증가시킬 컬럼.
    """
    mapping = {
        "visit": "visit_7d",
        "login": "visit_7d",
        "page_view": "browse_7d",
        "browse": "browse_7d",
        "search": "search_7d",
        "add_to_cart": "add_to_cart_7d",
        "remove_from_cart": "cart_remove_7d",
        "purchase": "purchase_30d",
        "support_contact": "support_30d",
        "refund": "refund_30d",
        "coupon_open": "coupon_open_30d",
        "coupon_redeem": "coupon_redeem_30d",
    }
    return mapping.get(event_type)


def _trigger_reason_for_event(event_type: str, amount: float) -> str:
    """
    2단계에서는 모델 추론 없이 feature 변화 이유만 기록한다.
    5단계 action_queue에서 trigger_reason으로 재사용 가능하다.
    """
    if event_type == "purchase":
        return f"purchase event amount={amount:.2f}"
    if event_type == "refund":
        return f"refund event amount={amount:.2f}"
    if event_type == "remove_from_cart":
        return "cart removal event"
    if event_type == "coupon_open":
        return "coupon opened"
    if event_type == "coupon_redeem":
        return "coupon redeemed"
    if event_type == "support_contact":
        return "support contact event"
    return f"{event_type} event"


def _insert_event_and_update_feature_state(
    *,
    db_url: str,
    event: UserEventIn,
) -> dict[str, Any]:
    """
    핵심 처리:
    1. customer_events append
    2. customer_feature_state upsert
    3. 이벤트 타입별 카운터 증가
    """
    event_time = _normalize_event_time(event.event_time)
    raw_payload_json = json.dumps(event.raw_payload or {}, ensure_ascii=False)
    counter_column = _counter_column_for_event(event.event_type)

    try:
        with user_live_session(db_url) as conn:
            inserted = conn.execute(
                text("""
                INSERT INTO customer_events (
                    source_event_id,
                    customer_id,
                    event_type,
                    event_time,
                    amount,
                    item_category,
                    channel,
                    session_id,
                    raw_payload,
                    processed
                )
                VALUES (
                    :source_event_id,
                    :customer_id,
                    :event_type,
                    :event_time,
                    :amount,
                    :item_category,
                    :channel,
                    :session_id,
                    CAST(:raw_payload AS JSONB),
                    FALSE
                )
                ON CONFLICT (source_event_id)
                WHERE source_event_id IS NOT NULL
                DO NOTHING
                RETURNING event_id
                """),
                {
                    "source_event_id": event.source_event_id,
                    "customer_id": event.customer_id,
                    "event_type": event.event_type,
                    "event_time": event_time,
                    "amount": event.amount,
                    "item_category": event.item_category,
                    "channel": event.channel,
                    "session_id": event.session_id,
                    "raw_payload": raw_payload_json,
                },
            ).mappings().first()

            if inserted is None and event.source_event_id:
                return {
                    "customer_id": event.customer_id,
                    "event_type": event.event_type,
                    "inserted": False,
                    "duplicate": True,
                    "message": "duplicate source_event_id; ignored",
                }
            # 고객 row가 없으면 생성, 있으면 last_event_time만 최신값으로 갱신

            existing = conn.execute(
                text("SELECT customer_id FROM customer_feature_state WHERE customer_id = :cid"),
                {"cid": event.customer_id},
            ).scalar_one_or_none()

            is_new = existing is None

            if is_new:
                conn.execute(
                    text("""
                    INSERT INTO customer_feature_state (
                        customer_id, last_event_time,
                        is_new_customer, first_seen_at, event_count_total,
                        persona, acquisition_channel, updated_at
                    ) VALUES (
                        :customer_id, :event_time,
                        TRUE, :event_time, 1,
                        'new_signup', :channel, now()
                    )
                    """),
                    {
                        "customer_id": event.customer_id,
                        "event_time": event_time,
                        "channel": event.channel or "unknown",
                    },
                )
            else:
                conn.execute(
                    text("""
                    UPDATE customer_feature_state
                    SET last_event_time = CASE
                            WHEN last_event_time IS NULL OR :event_time > last_event_time
                            THEN :event_time ELSE last_event_time
                        END,
                        event_count_total = COALESCE(event_count_total, 0) + 1,
                        updated_at = now()
                    WHERE customer_id = :customer_id
                    """),
                    {
                        "customer_id": event.customer_id,
                        "event_time": event_time,
                    },
                )

            if counter_column:
                conn.execute(
                    text(f"""
                    UPDATE customer_feature_state
                    SET {counter_column} = COALESCE({counter_column}, 0) + 1,
                        revenue_30d = CASE
                            WHEN :event_type = 'purchase'
                            THEN COALESCE(revenue_30d, 0) + :amount
                            ELSE COALESCE(revenue_30d, 0)
                        END,
                        updated_at = now()
                    WHERE customer_id = :customer_id
                    """),
                    {
                        "customer_id": event.customer_id,
                        "event_type": event.event_type,
                        "amount": event.amount,
                    },
                )

            conn.execute(
                text("""
                UPDATE customer_events
                SET processed = TRUE
                WHERE event_id = :event_id
                """),
                {"event_id": inserted["event_id"]},
            )

            latest_state = conn.execute(
                text("""
                SELECT *
                FROM customer_feature_state
                WHERE customer_id = :customer_id
                """),
                {"customer_id": event.customer_id},
            ).mappings().first()

        return {
            "customer_id": event.customer_id,
            "event_id": int(inserted["event_id"]),
            "event_type": event.event_type,
            "is_new_customer": is_new,
            "counter_updated": counter_column,
            "inserted": True,
            "duplicate": False,
            "trigger_reason": _trigger_reason_for_event(event.event_type, event.amount),
            "feature_state": dict(latest_state or {}),
        }

    except IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"event insert conflict: {exc}",
        ) from exc


@router.post("/events")
def ingest_user_event(
    event: UserEventIn,
    score_after_event: bool = True,
    update_actions: bool = True,
    action_threshold: float = 0.50,
    min_expected_roi: float = 0.0,
    min_expected_profit: float = 0.0,
    settings: ApiSettings = Depends(get_settings),
):
    """
    고객 행동 이벤트 1건 적재.

    5단계 흐름:
    1. customer_events 적재
    2. customer_feature_state 갱신
    3. 해당 customer_id만 customer_scores 재추론
    4. recommendation_candidates/action_queue 갱신
    """
    init_user_live_tables(settings.user_db_url)

    result = _insert_event_and_update_feature_state(
        db_url=settings.user_db_url,
        event=event,
    )

    scoring_result: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None

    if score_after_event and result.get("inserted", True):
        try:
            scoring_result = score_changed_customers(
                db_url=settings.user_db_url,
                model_dir=Path.cwd() / "models_user",
                customer_ids=[event.customer_id],
            )
        except Exception as exc:
            scoring_result = {
                "success": False,
                "error": str(exc),
                "customer_id": event.customer_id,
            }

    if (
        update_actions
        and result.get("inserted", True)
        and scoring_result is not None
        and scoring_result.get("success")
    ):
        try:
            action_result = update_live_actions_for_customers(
                db_url=settings.user_db_url,
                customer_ids=[event.customer_id],
                threshold=action_threshold,
                min_expected_roi=min_expected_roi,
                min_expected_profit=min_expected_profit,
            )
        except Exception as exc:
            action_result = {
                "success": False,
                "error": str(exc),
                "customer_id": event.customer_id,
            }

    invalidate_user_live_cache(settings.redis_url)

    return {
        "success": True,
        "mode": "user-live",
        "result": result,
        "scoring": scoring_result,
        "actions": action_result,
    }

@router.post("/events/batch")
def ingest_user_events_batch(
    payload: UserEventBatchIn,
    score_after_event: bool = True,
    update_actions: bool = True,
    action_threshold: float = 0.50,
    min_expected_roi: float = 0.0,
    min_expected_profit: float = 0.0,
    settings: ApiSettings = Depends(get_settings),
):
    """
    고객 행동 이벤트 여러 건 적재.

    5단계부터는 batch 안에서 실제 insert된 고객 ID만 모아서:
    - 한 번에 재추론
    - 한 번에 추천/action queue 갱신
    """
    init_user_live_tables(settings.user_db_url)

    if not payload.events:
        return {
            "success": True,
            "mode": "user-live",
            "received": 0,
            "inserted": 0,
            "duplicates": 0,
            "results": [],
            "scoring": None,
            "actions": None,
        }

    results: list[dict[str, Any]] = []
    inserted_count = 0
    duplicate_count = 0
    changed_customer_ids: set[int] = set()

    for event in payload.events:
        result = _insert_event_and_update_feature_state(
            db_url=settings.user_db_url,
            event=event,
        )
        results.append(result)

        if result.get("inserted"):
            inserted_count += 1
            changed_customer_ids.add(int(event.customer_id))

        if result.get("duplicate"):
            duplicate_count += 1

    scoring_result: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None

    if score_after_event and changed_customer_ids:
        try:
            scoring_result = score_changed_customers(
                db_url=settings.user_db_url,
                model_dir=Path.cwd() / "models_user",
                customer_ids=sorted(changed_customer_ids),
            )
        except Exception as exc:
            scoring_result = {
                "success": False,
                "error": str(exc),
                "customer_ids": sorted(changed_customer_ids),
            }

    if (
        update_actions
        and changed_customer_ids
        and scoring_result is not None
        and scoring_result.get("success")
    ):
        try:
            action_result = update_live_actions_for_customers(
                db_url=settings.user_db_url,
                customer_ids=sorted(changed_customer_ids),
                threshold=action_threshold,
                min_expected_roi=min_expected_roi,
                min_expected_profit=min_expected_profit,
            )
        except Exception as exc:
            action_result = {
                "success": False,
                "error": str(exc),
                "customer_ids": sorted(changed_customer_ids),
            }

    if inserted_count:
        invalidate_user_live_cache(settings.redis_url)

    return {
        "success": True,
        "mode": "user-live",
        "received": len(payload.events),
        "inserted": inserted_count,
        "duplicates": duplicate_count,
        "changed_customer_ids": sorted(changed_customer_ids),
        "results": results,
        "scoring": scoring_result,
        "actions": action_result,
    }

@router.get("/feature-state")
def get_feature_state(
    limit: int = Query(default=100, ge=1, le=100000),
    customer_id: int | None = Query(default=None, ge=1),
    settings: ApiSettings = Depends(get_settings),
):
    """
    최신 고객 feature_state 조회.
    2단계 테스트에서 가장 많이 쓰는 확인용 API다.
    """
    init_user_live_tables(settings.user_db_url)

    with user_live_session(settings.user_db_url) as conn:
        if customer_id is not None:
            rows = conn.execute(
                text("""
                SELECT *
                FROM customer_feature_state
                WHERE customer_id = :customer_id
                """),
                {"customer_id": customer_id},
            ).mappings().all()
        else:
            rows = conn.execute(
                text("""
                SELECT *
                FROM customer_feature_state
                ORDER BY updated_at DESC
                LIMIT :limit
                """),
                {"limit": limit},
            ).mappings().all()

    return {
        "success": True,
        "records": [dict(row) for row in rows],
    }


@router.get("/events")
def get_recent_events(
    limit: int = Query(default=100, ge=1, le=100000),
    customer_id: int | None = Query(default=None, ge=1),
    settings: ApiSettings = Depends(get_settings),
):
    """
    최근 적재된 customer_events 조회.
    이벤트가 실제로 DB에 append되는지 확인한다.
    """
    init_user_live_tables(settings.user_db_url)

    with user_live_session(settings.user_db_url) as conn:
        if customer_id is not None:
            rows = conn.execute(
                text("""
                SELECT *
                FROM customer_events
                WHERE customer_id = :customer_id
                ORDER BY event_time DESC, event_id DESC
                LIMIT :limit
                """),
                {
                    "customer_id": customer_id,
                    "limit": limit,
                },
            ).mappings().all()
        else:
            rows = conn.execute(
                text("""
                SELECT *
                FROM customer_events
                ORDER BY event_time DESC, event_id DESC
                LIMIT :limit
                """),
                {"limit": limit},
            ).mappings().all()

    return {
        "success": True,
        "records": [dict(row) for row in rows],
    }


def _has_column(conn, table_name: str, column_name: str) -> bool:
    return bool(conn.execute(
        text("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = :table_name
              AND column_name = :column_name
        )
        """),
        {"table_name": table_name, "column_name": column_name},
    ).scalar())


@router.get("/health")
def user_live_health(
    settings: ApiSettings = Depends(get_settings),
):
    """
    user-live DB 상태 확인.

    Dashboard cache invalidation depends on this endpoint.  It therefore exposes
    not only event/feature counts but also the latest insert, score,
    recommendation, and action timestamps.  Without those timestamps the
    Streamlit layer can keep showing stale churn KPIs or stale action queues
    even after a live event has been ingested and rescored.
    """
    init_user_live_tables(settings.user_db_url)

    with user_live_session(settings.user_db_url) as conn:
        event_count = conn.execute(text("SELECT COUNT(*) FROM customer_events")).scalar_one()
        feature_state_count = conn.execute(text("SELECT COUNT(*) FROM customer_feature_state")).scalar_one()
        processed_count = conn.execute(text("SELECT COUNT(*) FROM customer_events WHERE processed = TRUE")).scalar_one()
        latest_event_time = conn.execute(text("SELECT MAX(event_time) FROM customer_events")).scalar_one()
        latest_event_created_at = conn.execute(text("SELECT MAX(created_at) FROM customer_events")).scalar_one()
        latest_update_time = conn.execute(text("SELECT MAX(updated_at) FROM customer_feature_state")).scalar_one()

        score_count = conn.execute(text("SELECT COUNT(*) FROM customer_scores")).scalar_one()
        latest_score_time = conn.execute(text("SELECT MAX(scored_at) FROM customer_scores")).scalar_one()
        avg_churn_score = conn.execute(text("SELECT AVG(churn_score) FROM customer_scores")).scalar_one()
        high_risk_customers_default = conn.execute(
            text("SELECT COUNT(*) FROM customer_scores WHERE churn_score >= 0.50")
        ).scalar_one()
        critical_risk_customers = conn.execute(
            text("SELECT COUNT(*) FROM customer_scores WHERE churn_score >= 0.85")
        ).scalar_one()

        recommendation_count = conn.execute(text("SELECT COUNT(*) FROM recommendation_candidates")).scalar_one()
        live_recommendation_count = conn.execute(
            text("""
            SELECT COUNT(*)
            FROM recommendation_candidates
            WHERE source_type = 'live_policy_v1'
            """)
        ).scalar_one() if _has_column(conn, "recommendation_candidates", "source_type") else 0
        rec_latest_col = "updated_at" if _has_column(conn, "recommendation_candidates", "updated_at") else "generated_at"
        latest_recommendation_update_time = conn.execute(
            text(f"SELECT MAX({rec_latest_col}) FROM recommendation_candidates")
        ).scalar_one()

        action_count = conn.execute(text("SELECT COUNT(*) FROM action_queue")).scalar_one()
        queued_action_count = conn.execute(
            text("SELECT COUNT(*) FROM action_queue WHERE action_status = 'queued'")
        ).scalar_one()
        live_action_count = conn.execute(
            text("""
            SELECT COUNT(*)
            FROM action_queue
            WHERE source_type = 'live_policy_v1'
            """)
        ).scalar_one() if _has_column(conn, "action_queue", "source_type") else 0
        action_latest_col = "updated_at" if _has_column(conn, "action_queue", "updated_at") else "queued_at"
        latest_action_update_time = conn.execute(
            text(f"SELECT MAX({action_latest_col}) FROM action_queue")
        ).scalar_one()

    return {
        "status": "ok",
        "mode": "user-live",
        "event_count": int(event_count),
        "processed_event_count": int(processed_count),
        "feature_state_count": int(feature_state_count),
        "score_count": int(score_count),
        "avg_churn_score": float(avg_churn_score or 0.0),
        "high_risk_customers_default": int(high_risk_customers_default or 0),
        "critical_risk_customers": int(critical_risk_customers or 0),
        "recommendation_count": int(recommendation_count or 0),
        "live_recommendation_count": int(live_recommendation_count or 0),
        "action_count": int(action_count or 0),
        "queued_action_count": int(queued_action_count or 0),
        "live_action_count": int(live_action_count or 0),
        "latest_event_time": latest_event_time,
        "latest_event_created_at": latest_event_created_at,
        "latest_feature_update_time": latest_update_time,
        "latest_score_time": latest_score_time,
        "latest_recommendation_update_time": latest_recommendation_update_time,
        "latest_action_update_time": latest_action_update_time,
    }


@router.post("/reset")
def reset_user_live_tables(
    confirm: bool = Query(default=False),
    settings: ApiSettings = Depends(get_settings),
):
    """
    개발/테스트용 초기화 API.
    운영에서는 막아야 한다.

    사용:
    POST /api/v1/user-live/reset?confirm=true
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="reset requires confirm=true",
        )

    init_user_live_tables(settings.user_db_url)

    with user_live_session(settings.user_db_url) as conn:
        conn.execute(text("TRUNCATE TABLE customer_events RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE customer_feature_state RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE customer_scores RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE recommendation_candidates RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE action_queue RESTART IDENTITY CASCADE"))

    invalidate_user_live_cache(settings.redis_url)

    return {
        "success": True,
        "message": "user-live tables reset",
    }
@router.post("/seed-from-user-artifacts")
def seed_from_user_artifacts(
    reset: bool = True,
    rescore_after_seed: bool = True,
    refresh_actions_after_rescore: bool = True,
    rescore_batch_size: int = 2000,
    action_threshold: float = 0.50,
    settings: ApiSettings = Depends(get_settings),
):
    """
    3단계 API.

    이미 존재하는 user 산출물:
    - data/raw_user/customer_summary.csv
    - data/feature_store_user/customer_features.csv
    - results_user/uplift_segmentation.csv
    - results_user/optimization_selected_customers.csv
    - results_user/personalized_recommendations.csv

    위 파일들을 PostgreSQL live serving table에 초기 적재한다.

    reset=True:
        기존 live table을 비우고 현재 파일 기준으로 다시 seed한다.
    """
    result = seed_user_live_from_artifacts(
        db_url=settings.user_db_url,
        project_root=Path.cwd(),
        reset=reset,
        data_dir="data/raw_user",
        feature_store_dir="data/feature_store_user",
        result_dir="results_user",
        model_dir="models_user",
        rescore_after_seed=rescore_after_seed,
        refresh_actions_after_rescore=refresh_actions_after_rescore,
        rescore_batch_size=int(rescore_batch_size),
        action_threshold=float(action_threshold),
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result,
        )

    invalidate_user_live_cache(settings.redis_url)
    return result


@router.get("/seed-status")
def seed_status(
    settings: ApiSettings = Depends(get_settings),
):
    """
    PostgreSQL live serving table이 현재 얼마나 seed되어 있는지 확인한다.
    """
    return get_user_live_seed_status(
        db_url=settings.user_db_url,
    )

@router.post("/score-customers")
def score_customers(
    customer_ids: list[int],
    update_actions: bool = True,
    action_threshold: float = 0.50,
    min_expected_roi: float = 0.0,
    min_expected_profit: float = 0.0,
    settings: ApiSettings = Depends(get_settings),
):
    """
    특정 고객 ID 목록만 수동 재추론한다.

    5단계부터는 옵션에 따라 recommendation_candidates/action_queue도 같이 갱신한다.
    """
    if not customer_ids:
        raise HTTPException(
            status_code=400,
            detail="customer_ids must not be empty",
        )

    scoring_result = score_changed_customers(
        db_url=settings.user_db_url,
        model_dir=Path.cwd() / "models_user",
        customer_ids=customer_ids,
    )

    action_result: dict[str, Any] | None = None

    if update_actions and scoring_result.get("success"):
        action_result = update_live_actions_for_customers(
            db_url=settings.user_db_url,
            customer_ids=customer_ids,
            threshold=action_threshold,
            min_expected_roi=min_expected_roi,
            min_expected_profit=min_expected_profit,
        )

    return {
        "success": True,
        "scoring": scoring_result,
        "actions": action_result,
    }

@router.get("/scores")
def live_scores(
    limit: int | None = Query(default=None, ge=1),
    customer_id: int | None = None,
    risk_threshold: float = Query(default=0.70, ge=0.0, le=1.0),
    settings: ApiSettings = Depends(get_settings),
):
    """
    PostgreSQL customer_scores 최신 점수 조회.

    limit을 생략하면 전체 customer_scores를 반환한다.
    예:
    - /api/v1/user-live/scores              → 전체 조회
    - /api/v1/user-live/scores?limit=100    → 100명만 조회
    - /api/v1/user-live/scores?customer_id=1001 → 특정 고객 조회
    """
    return get_user_live_scores(
        db_url=settings.user_db_url,
        limit=limit,
        customer_id=customer_id,
        risk_threshold=risk_threshold,
        redis_url=settings.redis_url,
    )

@router.post("/refresh-actions")
def refresh_actions(
    customer_ids: list[int],
    action_threshold: float = 0.50,
    min_expected_roi: float = 0.0,
    min_expected_profit: float = 0.0,
    settings: ApiSettings = Depends(get_settings),
):
    """
    customer_scores는 이미 갱신되어 있다고 보고,
    특정 고객들의 recommendation_candidates/action_queue만 다시 갱신한다.
    """
    if not customer_ids:
        raise HTTPException(
            status_code=400,
            detail="customer_ids must not be empty",
        )

    result = update_live_actions_for_customers(
        db_url=settings.user_db_url,
        customer_ids=customer_ids,
        threshold=action_threshold,
        min_expected_roi=min_expected_roi,
        min_expected_profit=min_expected_profit,
    )
    invalidate_user_live_cache(settings.redis_url)
    return result


@router.get("/recommendations")
def live_recommendations(
    limit: int = 100,
    customer_id: int | None = None,
    source_type: str | None = None,
    settings: ApiSettings = Depends(get_settings),
):
    """
    PostgreSQL recommendation_candidates 조회.
    """
    return get_live_recommendation_candidates(
        db_url=settings.user_db_url,
        limit=limit,
        customer_id=customer_id,
        source_type=source_type,
        redis_url=settings.redis_url,
    )


@router.get("/actions")
def live_actions(
    limit: int = 100,
    customer_id: int | None = None,
    source_type: str | None = None,
    status: str | None = None,
    settings: ApiSettings = Depends(get_settings),
):
    """
    PostgreSQL action_queue 조회.
    """
    return get_live_action_queue(
        db_url=settings.user_db_url,
        limit=limit,
        customer_id=customer_id,
        source_type=source_type,
        status=status,
        redis_url=settings.redis_url,
    )
@router.post("/jobs/drift-check")
def run_drift_check_job(
    settings: ApiSettings = Depends(get_settings),
):
    """
    가벼운 drift check job.
    모델 재학습은 하지 않는다.
    """
    return run_live_drift_check(
        db_url=settings.user_db_url,
    )


@router.post("/jobs/recent-action-refresh")
def run_recent_action_refresh_job(
    limit: int = 1000,
    action_threshold: float = 0.5,
    settings: ApiSettings = Depends(get_settings),
):
    """
    최근 갱신 고객 중심 action queue microbatch refresh.
    전체 최적화 full recompute가 아니다.
    """
    return run_recent_action_refresh(
        db_url=settings.user_db_url,
        limit=limit,
        action_threshold=action_threshold,
    )


@router.get("/jobs/status")
def user_live_jobs_status(
    limit: int = 20,
    settings: ApiSettings = Depends(get_settings),
):
    """
    user-live batch job 실행 기록 조회.
    """
    return get_user_live_job_status(
        db_url=settings.user_db_url,
        limit=limit,
    )


# ── Demo endpoints ──

from src.api.services.user_live_demo import (
    DemoConfig,
    get_demo_status,
    reset_demo,
    start_demo,
    stop_demo,
)


@router.post("/demo/start")
async def demo_start(
    interval_seconds: float = Query(default=2.0, ge=0.5, le=30.0),
    new_customer_ratio: float = Query(default=0.3, ge=0.0, le=1.0),
    action_threshold: float = Query(default=0.30, ge=0.0, le=1.0),
    settings: ApiSettings = Depends(get_settings),
):
    init_user_live_tables(settings.user_db_url)
    config = DemoConfig(
        interval_seconds=interval_seconds,
        new_customer_ratio=new_customer_ratio,
        action_threshold=action_threshold,
    )
    result = start_demo(
        db_url=settings.user_db_url,
        model_dir=Path.cwd() / "models_user",
        config=config,
    )
    invalidate_user_live_cache(settings.redis_url)
    return result


@router.post("/demo/stop")
def demo_stop():
    return stop_demo()


@router.post("/demo/reset")
def demo_reset(settings: ApiSettings = Depends(get_settings)):
    result = reset_demo(db_url=settings.user_db_url)
    invalidate_user_live_cache(settings.redis_url)
    return result


@router.get("/demo/status")
def demo_status():
    return get_demo_status()


@router.get("/new-customers")
def get_new_customers(
    limit: int = Query(default=50, ge=1, le=500),
    settings: ApiSettings = Depends(get_settings),
):
    init_user_live_tables(settings.user_db_url)

    with user_live_session(settings.user_db_url) as conn:
        rows = conn.execute(
            text("""
            SELECT
                fs.customer_id,
                fs.first_seen_at,
                fs.event_count_total,
                fs.is_new_customer,
                fs.persona,
                fs.acquisition_channel,
                fs.last_event_time,
                cs.churn_score,
                cs.risk_segment
            FROM customer_feature_state fs
            LEFT JOIN customer_scores cs ON cs.customer_id = fs.customer_id
            WHERE fs.is_new_customer = TRUE
            ORDER BY fs.first_seen_at DESC
            LIMIT :limit
            """),
            {"limit": limit},
        ).mappings().all()

        total_new = conn.execute(
            text("SELECT COUNT(*) FROM customer_feature_state WHERE is_new_customer = TRUE")
        ).scalar_one()

    return {
        "success": True,
        "total_new_customers": int(total_new),
        "records": [dict(row) for row in rows],
    }
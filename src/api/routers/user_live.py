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
            conn.execute(
                text("""
                INSERT INTO customer_feature_state (
                    customer_id,
                    last_event_time,
                    updated_at
                )
                VALUES (
                    :customer_id,
                    :event_time,
                    now()
                )
                ON CONFLICT (customer_id)
                DO UPDATE SET
                    last_event_time = CASE
                        WHEN customer_feature_state.last_event_time IS NULL
                             OR EXCLUDED.last_event_time > customer_feature_state.last_event_time
                        THEN EXCLUDED.last_event_time
                        ELSE customer_feature_state.last_event_time
                    END,
                    updated_at = now()
                """),
                {
                    "customer_id": event.customer_id,
                    "event_time": event_time,
                },
            )

            # counter_column은 내부 mapping에서만 나오므로 SQL injection 위험 없음
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
    settings: ApiSettings = Depends(get_settings),
):
    """
    고객 행동 이벤트 1건 적재.

    2단계 완료 기준:
    - customer_events에 row가 쌓인다.
    - customer_feature_state의 해당 customer_id row가 갱신된다.
    """
    init_user_live_tables(settings.user_db_url)

    result = _insert_event_and_update_feature_state(
        db_url=settings.user_db_url,
        event=event,
    )

    return {
        "success": True,
        "mode": "user-live",
        "result": result,
    }


@router.post("/events/batch")
def ingest_user_events_batch(
    payload: UserEventBatchIn,
    settings: ApiSettings = Depends(get_settings),
):
    """
    고객 행동 이벤트 여러 건 적재.
    로컬 테스트, CSV 일부 샘플 적재, 외부 시스템 micro-batch 연동에 사용한다.
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
        }

    results: list[dict[str, Any]] = []
    inserted_count = 0
    duplicate_count = 0

    for event in payload.events:
        result = _insert_event_and_update_feature_state(
            db_url=settings.user_db_url,
            event=event,
        )
        results.append(result)
        if result.get("inserted"):
            inserted_count += 1
        if result.get("duplicate"):
            duplicate_count += 1

    return {
        "success": True,
        "mode": "user-live",
        "received": len(payload.events),
        "inserted": inserted_count,
        "duplicates": duplicate_count,
        "results": results,
    }


@router.get("/feature-state")
def get_feature_state(
    limit: int = Query(default=100, ge=1, le=5000),
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
    limit: int = Query(default=100, ge=1, le=5000),
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


@router.get("/health")
def user_live_health(
    settings: ApiSettings = Depends(get_settings),
):
    """
    user-live DB 상태 확인.
    """
    init_user_live_tables(settings.user_db_url)

    with user_live_session(settings.user_db_url) as conn:
        event_count = conn.execute(
            text("SELECT COUNT(*) FROM customer_events")
        ).scalar_one()

        feature_state_count = conn.execute(
            text("SELECT COUNT(*) FROM customer_feature_state")
        ).scalar_one()

        processed_count = conn.execute(
            text("SELECT COUNT(*) FROM customer_events WHERE processed = TRUE")
        ).scalar_one()

        latest_event_time = conn.execute(
            text("SELECT MAX(event_time) FROM customer_events")
        ).scalar_one()

        latest_update_time = conn.execute(
            text("SELECT MAX(updated_at) FROM customer_feature_state")
        ).scalar_one()

    return {
        "status": "ok",
        "mode": "user-live",
        "event_count": int(event_count),
        "processed_event_count": int(processed_count),
        "feature_state_count": int(feature_state_count),
        "latest_event_time": latest_event_time,
        "latest_feature_update_time": latest_update_time,
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

    return {
        "success": True,
        "message": "user-live tables reset",
    }
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from src.api.services.user_live_db import (
    ensure_user_live_seed_columns,
    user_live_session,
)


LIVE_SOURCE_TYPE = "live_policy_v1"


def _jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
        return value

    return str(value)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _safe_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return str(value)


def ensure_user_live_action_columns(db_url: str) -> None:
    """
    5단계 live recommendation/action queue 갱신에 필요한 보강 컬럼을 추가한다.

    기존 3단계 seed 데이터와 구분하기 위해 source_type을 둔다.
    - seed: 기존 results_user에서 들어온 초기 row
    - live_policy_v1: 이벤트 기반으로 갱신된 최신 live row
    """
    ensure_user_live_seed_columns(db_url)

    with user_live_session(db_url) as conn:
        conn.execute(text("""
        ALTER TABLE recommendation_candidates
        ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'seed'
        """))

        conn.execute(text("""
        ALTER TABLE recommendation_candidates
        ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE
        """))

        conn.execute(text("""
        ALTER TABLE recommendation_candidates
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()
        """))

        conn.execute(text("""
        ALTER TABLE action_queue
        ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'seed'
        """))

        conn.execute(text("""
        ALTER TABLE action_queue
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_recommendation_candidates_customer_source
        ON recommendation_candidates (customer_id, source_type)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_recommendation_candidates_active_priority
        ON recommendation_candidates (active, priority_score DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_action_queue_customer_source
        ON action_queue (customer_id, source_type)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_action_queue_live_priority
        ON action_queue (source_type, action_status, priority_score DESC)
        """))


def _priority_score(
    *,
    churn_score: float | None,
    clv: float | None,
    uplift_score: float | None,
    expected_roi: float | None,
    expected_incremental_profit: float | None,
) -> float:
    """
    live queue 우선순위 점수.

    전체 예산 최적화가 아니라 변경 고객의 실행 우선순위를 정하는 점수다.
    가능한 경우 expected_incremental_profit을 가장 신뢰하고,
    없으면 churn * clv * uplift 또는 expected_roi를 사용한다.
    """
    if expected_incremental_profit is not None:
        return float(expected_incremental_profit)

    if churn_score is not None and clv is not None and uplift_score is not None:
        return max(float(churn_score), 0.0) * max(float(clv), 0.0) * max(float(uplift_score), 0.0)

    if expected_roi is not None:
        return float(expected_roi)

    if churn_score is not None:
        return float(churn_score)

    return 0.0


def _recommend_action(
    *,
    churn_score: float | None,
    uplift_score: float | None,
    expected_roi: float | None,
    risk_segment: str | None,
) -> tuple[str, str, str]:
    """
    추천 액션, 추천 카테고리, 개입 강도를 결정한다.

    복잡한 개인화 모델은 6단계 이후 붙이고,
    5단계에서는 점수 기반 정책 rule로 충분하다.
    """
    churn = churn_score or 0.0
    uplift = uplift_score or 0.0
    roi = expected_roi or 0.0
    risk = (risk_segment or "").lower()

    if churn >= 0.85 or risk == "critical":
        if uplift >= 0.03 and roi >= 1.0:
            return "high_value_retention_coupon", "coupon", "high"
        return "priority_human_followup", "crm", "high"

    if churn >= 0.70 or risk == "high":
        if uplift >= 0.02:
            return "personalized_retention_offer", "coupon", "medium"
        return "retention_message", "message", "medium"

    if churn >= 0.50 or risk == "medium":
        return "light_retention_message", "message", "low"

    if uplift >= 0.05 and roi >= 1.0:
        return "low_risk_upsell_offer", "upsell", "low"

    return "monitor_only", "monitoring", "none"


def _should_queue_action(
    *,
    churn_score: float | None,
    expected_roi: float | None,
    expected_incremental_profit: float | None,
    threshold: float,
    min_expected_roi: float,
    min_expected_profit: float,
) -> bool:
    """
    action_queue에 넣을지 여부.

    기본 정책:
    - churn_score가 threshold 이상
    - expected_roi가 최소 기준 이상
    - expected_incremental_profit이 최소 기준 이상

    단, expected_roi/profit이 없는 경우에도 churn_score가 충분히 높으면 큐에 넣을 수 있게 한다.
    """
    churn = churn_score or 0.0
    roi = expected_roi
    profit = expected_incremental_profit

    if churn < threshold:
        return False

    if roi is not None and roi < min_expected_roi:
        return False

    if profit is not None and profit < min_expected_profit:
        return False

    return True


def _load_scores_for_customers(
    *,
    conn,
    customer_ids: list[int],
) -> list[dict[str, Any]]:
    if not customer_ids:
        return []

    stmt = text("""
        SELECT
            customer_id,
            churn_score,
            clv,
            uplift_score,
            expected_roi,
            expected_incremental_profit,
            risk_segment,
            uplift_segment,
            model_version,
            scored_at,
            score_payload
        FROM customer_scores
        WHERE customer_id IN :customer_ids
    """).bindparams(bindparam("customer_ids", expanding=True))

    rows = conn.execute(
        stmt,
        {"customer_ids": customer_ids},
    ).mappings().all()

    return [dict(row) for row in rows]


def _delete_previous_live_rows(
    *,
    conn,
    customer_ids: list[int],
) -> None:
    """
    같은 고객의 이전 live_policy_v1 row를 지운 뒤 최신 row를 다시 넣는다.

    seed row는 source_type='seed'로 남겨둔다.
    """
    if not customer_ids:
        return

    stmt_candidates = text("""
        DELETE FROM recommendation_candidates
        WHERE source_type = :source_type
          AND customer_id IN :customer_ids
    """).bindparams(bindparam("customer_ids", expanding=True))

    conn.execute(
        stmt_candidates,
        {
            "source_type": LIVE_SOURCE_TYPE,
            "customer_ids": customer_ids,
        },
    )

    stmt_queue = text("""
        DELETE FROM action_queue
        WHERE source_type = :source_type
          AND customer_id IN :customer_ids
    """).bindparams(bindparam("customer_ids", expanding=True))

    conn.execute(
        stmt_queue,
        {
            "source_type": LIVE_SOURCE_TYPE,
            "customer_ids": customer_ids,
        },
    )


def update_live_actions_for_customers(
    *,
    db_url: str,
    customer_ids: list[int],
    threshold: float = 0.50,
    min_expected_roi: float = 0.0,
    min_expected_profit: float = 0.0,
) -> dict[str, Any]:
    """
    5단계 핵심 함수.

    customer_scores가 갱신된 고객만 대상으로:
    1. recommendation_candidates 최신 row 생성
    2. 조건을 만족하면 action_queue에 queued row 생성
    3. 조건 미충족이면 action_queue에서는 제외

    전체 budget optimization full recompute를 수행하지 않는다.
    """
    unique_customer_ids = sorted({int(cid) for cid in customer_ids if cid is not None})

    if not unique_customer_ids:
        return {
            "success": True,
            "requested_customers": 0,
            "recommendation_candidates_updated": 0,
            "action_queue_updated": 0,
            "records": [],
        }

    ensure_user_live_action_columns(db_url)

    with user_live_session(db_url) as conn:
        score_rows = _load_scores_for_customers(
            conn=conn,
            customer_ids=unique_customer_ids,
        )

        if not score_rows:
            return {
                "success": False,
                "requested_customers": len(unique_customer_ids),
                "recommendation_candidates_updated": 0,
                "action_queue_updated": 0,
                "message": "no customer_scores found for requested customers",
            }

        _delete_previous_live_rows(
            conn=conn,
            customer_ids=unique_customer_ids,
        )

        recommendation_count = 0
        queue_count = 0
        records: list[dict[str, Any]] = []

        for row in score_rows:
            customer_id = int(row["customer_id"])

            churn_score = _safe_float(row.get("churn_score"), None)
            clv = _safe_float(row.get("clv"), None)
            uplift_score = _safe_float(row.get("uplift_score"), None)
            expected_roi = _safe_float(row.get("expected_roi"), None)
            expected_profit = _safe_float(row.get("expected_incremental_profit"), None)
            risk_segment = _safe_str(row.get("risk_segment"), None)

            priority = _priority_score(
                churn_score=churn_score,
                clv=clv,
                uplift_score=uplift_score,
                expected_roi=expected_roi,
                expected_incremental_profit=expected_profit,
            )

            recommended_action, recommended_category, intensity = _recommend_action(
                churn_score=churn_score,
                uplift_score=uplift_score,
                expected_roi=expected_roi,
                risk_segment=risk_segment,
            )

            should_queue = _should_queue_action(
                churn_score=churn_score,
                expected_roi=expected_roi,
                expected_incremental_profit=expected_profit,
                threshold=threshold,
                min_expected_roi=min_expected_roi,
                min_expected_profit=min_expected_profit,
            )

            reason_tags = {
                "source": LIVE_SOURCE_TYPE,
                "rule": "score_based_live_policy",
                "churn_score": churn_score,
                "clv": clv,
                "uplift_score": uplift_score,
                "expected_roi": expected_roi,
                "expected_incremental_profit": expected_profit,
                "risk_segment": risk_segment,
                "threshold": threshold,
                "should_queue": should_queue,
            }

            source_payload = {
                "customer_score": {
                    key: _jsonable(value)
                    for key, value in row.items()
                    if key != "score_payload"
                },
                "policy": reason_tags,
            }

            conn.execute(
                text("""
                INSERT INTO recommendation_candidates (
                    customer_id,
                    recommended_action,
                    recommended_category,
                    coupon_cost,
                    expected_roi,
                    expected_incremental_profit,
                    priority_score,
                    reason_tags,
                    source_payload,
                    source_type,
                    active,
                    seeded_at,
                    generated_at,
                    updated_at
                )
                VALUES (
                    :customer_id,
                    :recommended_action,
                    :recommended_category,
                    :coupon_cost,
                    :expected_roi,
                    :expected_incremental_profit,
                    :priority_score,
                    :reason_tags,
                    CAST(:source_payload AS JSONB),
                    :source_type,
                    TRUE,
                    now(),
                    now(),
                    now()
                )
                """),
                {
                    "customer_id": customer_id,
                    "recommended_action": recommended_action,
                    "recommended_category": recommended_category,
                    "coupon_cost": 0.0,
                    "expected_roi": expected_roi,
                    "expected_incremental_profit": expected_profit,
                    "priority_score": priority,
                    "reason_tags": json.dumps(reason_tags, ensure_ascii=False),
                    "source_payload": json.dumps(source_payload, ensure_ascii=False),
                    "source_type": LIVE_SOURCE_TYPE,
                },
            )

            recommendation_count += 1

            if should_queue:
                trigger_reason = (
                    f"live policy queued: churn={churn_score}, "
                    f"roi={expected_roi}, profit={expected_profit}, "
                    f"risk={risk_segment}"
                )

                conn.execute(
                    text("""
                    INSERT INTO action_queue (
                        customer_id,
                        action_status,
                        recommended_action,
                        intervention_intensity,
                        coupon_cost,
                        expected_profit,
                        expected_roi,
                        priority_score,
                        trigger_reason,
                        source_payload,
                        source_type,
                        seeded_at,
                        queued_at,
                        updated_at
                    )
                    VALUES (
                        :customer_id,
                        'queued',
                        :recommended_action,
                        :intervention_intensity,
                        :coupon_cost,
                        :expected_profit,
                        :expected_roi,
                        :priority_score,
                        :trigger_reason,
                        CAST(:source_payload AS JSONB),
                        :source_type,
                        now(),
                        now(),
                        now()
                    )
                    """),
                    {
                        "customer_id": customer_id,
                        "recommended_action": recommended_action,
                        "intervention_intensity": intensity,
                        "coupon_cost": 0.0,
                        "expected_profit": expected_profit,
                        "expected_roi": expected_roi,
                        "priority_score": priority,
                        "trigger_reason": trigger_reason,
                        "source_payload": json.dumps(source_payload, ensure_ascii=False),
                        "source_type": LIVE_SOURCE_TYPE,
                    },
                )

                queue_count += 1

            records.append({
                "customer_id": customer_id,
                "churn_score": churn_score,
                "clv": clv,
                "uplift_score": uplift_score,
                "expected_roi": expected_roi,
                "expected_incremental_profit": expected_profit,
                "risk_segment": risk_segment,
                "priority_score": priority,
                "recommended_action": recommended_action,
                "recommended_category": recommended_category,
                "intervention_intensity": intensity,
                "queued": should_queue,
            })

    return {
        "success": True,
        "source_type": LIVE_SOURCE_TYPE,
        "requested_customers": len(unique_customer_ids),
        "scored_customers_found": len(score_rows),
        "recommendation_candidates_updated": recommendation_count,
        "action_queue_updated": queue_count,
        "threshold": threshold,
        "min_expected_roi": min_expected_roi,
        "min_expected_profit": min_expected_profit,
        "records": records,
    }


def get_live_recommendation_candidates(
    *,
    db_url: str,
    limit: int = 100,
    customer_id: int | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    ensure_user_live_action_columns(db_url)

    with user_live_session(db_url) as conn:
        if customer_id is not None:
            rows = conn.execute(
                text("""
                SELECT *
                FROM recommendation_candidates
                WHERE customer_id = :customer_id
                ORDER BY updated_at DESC NULLS LAST, generated_at DESC
                LIMIT :limit
                """),
                {
                    "customer_id": customer_id,
                    "limit": limit,
                },
            ).mappings().all()
        elif source_type is not None:
            rows = conn.execute(
                text("""
                SELECT *
                FROM recommendation_candidates
                WHERE source_type = :source_type
                ORDER BY priority_score DESC NULLS LAST, updated_at DESC NULLS LAST
                LIMIT :limit
                """),
                {
                    "source_type": source_type,
                    "limit": limit,
                },
            ).mappings().all()
        else:
            rows = conn.execute(
                text("""
                SELECT *
                FROM recommendation_candidates
                ORDER BY priority_score DESC NULLS LAST, updated_at DESC NULLS LAST
                LIMIT :limit
                """),
                {"limit": limit},
            ).mappings().all()

        summary = conn.execute(
            text("""
            SELECT
                COUNT(*) AS total_recommendations,
                SUM(CASE WHEN source_type = 'live_policy_v1' THEN 1 ELSE 0 END) AS live_recommendations,
                SUM(CASE WHEN active = TRUE THEN 1 ELSE 0 END) AS active_recommendations,
                MAX(updated_at) AS latest_updated_at
            FROM recommendation_candidates
            """)
        ).mappings().first()

    return {
        "success": True,
        "summary": dict(summary or {}),
        "records": [dict(row) for row in rows],
    }


def get_live_action_queue(
    *,
    db_url: str,
    limit: int = 100,
    customer_id: int | None = None,
    source_type: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    ensure_user_live_action_columns(db_url)

    with user_live_session(db_url) as conn:
        conditions = []
        params: dict[str, Any] = {"limit": limit}

        if customer_id is not None:
            conditions.append("customer_id = :customer_id")
            params["customer_id"] = customer_id

        if source_type is not None:
            conditions.append("source_type = :source_type")
            params["source_type"] = source_type

        if status is not None:
            conditions.append("action_status = :status")
            params["status"] = status

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        rows = conn.execute(
            text(f"""
            SELECT *
            FROM action_queue
            {where_clause}
            ORDER BY priority_score DESC NULLS LAST, updated_at DESC NULLS LAST, queued_at DESC
            LIMIT :limit
            """),
            params,
        ).mappings().all()

        summary = conn.execute(
            text("""
            SELECT
                COUNT(*) AS total_actions,
                SUM(CASE WHEN source_type = 'live_policy_v1' THEN 1 ELSE 0 END) AS live_actions,
                SUM(CASE WHEN action_status = 'queued' THEN 1 ELSE 0 END) AS queued_actions,
                MAX(updated_at) AS latest_updated_at
            FROM action_queue
            """)
        ).mappings().first()

    return {
        "success": True,
        "summary": dict(summary or {}),
        "records": [dict(row) for row in rows],
    }
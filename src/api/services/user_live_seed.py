from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.api.services.user_live_db import (
    ensure_user_live_seed_columns,
    user_live_session,
)


CUSTOMER_ID_ALIASES = [
    "customer_id",
    "user_id",
    "member_id",
    "client_id",
    "id",
]

CHURN_SCORE_ALIASES = [
    "churn_score",
    "churn_probability",
    "churn_prob",
    "risk_score",
    "predicted_churn_probability",
]

CLV_ALIASES = [
    "clv",
    "customer_lifetime_value",
    "predicted_clv",
    "ltv",
]

UPLIFT_ALIASES = [
    "uplift_score",
    "uplift",
    "predicted_uplift",
    "treatment_effect",
]

EXPECTED_ROI_ALIASES = [
    "expected_roi",
    "roi",
    "predicted_roi",
]

EXPECTED_PROFIT_ALIASES = [
    "expected_incremental_profit",
    "expected_profit",
    "incremental_profit",
    "expected_value",
]

RISK_SEGMENT_ALIASES = [
    "risk_segment",
    "churn_segment",
    "risk_group",
]

UPLIFT_SEGMENT_ALIASES = [
    "uplift_segment",
    "segment",
    "treatment_segment",
]

LAST_EVENT_TIME_ALIASES = [
    "last_event_time",
    "last_activity_time",
    "last_order_time",
    "last_purchase_time",
    "event_time",
    "timestamp",
    "datetime",
]

RECOMMENDED_ACTION_ALIASES = [
    "recommended_action",
    "action",
    "recommendation",
    "message_type",
    "offer_type",
]

RECOMMENDED_CATEGORY_ALIASES = [
    "recommended_category",
    "category",
    "item_category",
    "product_category",
]

COUPON_COST_ALIASES = [
    "coupon_cost",
    "cost",
    "discount_amount",
    "offer_cost",
]

PRIORITY_SCORE_ALIASES = [
    "priority_score",
    "selection_score",
    "score",
    "expected_incremental_profit",
    "expected_profit",
    "expected_roi",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_first_existing_csv(paths: list[Path]) -> tuple[pd.DataFrame, str | None]:
    for path in paths:
        df = _read_csv_if_exists(path)
        if not df.empty:
            return df, str(path)

    return pd.DataFrame(), None


def _pick_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    lower_to_original = {str(col).lower(): str(col) for col in df.columns}

    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]

    for alias in aliases:
        alias_lower = alias.lower()
        for col in df.columns:
            if alias_lower in str(col).lower():
                return str(col)

    return None


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


def _safe_int(value: Any, default: int | None = None) -> int | None:
    number = _safe_float(value, None)
    if number is None:
        return default

    try:
        return int(number)
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

    if isinstance(value, (dict, list, tuple, str, int, float, bool)):
        return value

    return str(value)


def _row_payload(row: pd.Series) -> str:
    payload = {
        str(key): _jsonable(value)
        for key, value in row.to_dict().items()
    }
    return json.dumps(payload, ensure_ascii=False)


def _get_first_value(row: pd.Series, aliases: list[str], default: Any = None) -> Any:
    lower_to_original = {str(col).lower(): str(col) for col in row.index}

    for alias in aliases:
        col = lower_to_original.get(alias.lower())
        if col is not None:
            value = row.get(col)
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value

    for alias in aliases:
        alias_lower = alias.lower()
        for col in row.index:
            if alias_lower in str(col).lower():
                value = row.get(col)
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                return value

    return default


def _risk_segment_from_score(churn_score: float | None) -> str | None:
    if churn_score is None:
        return None

    if churn_score >= 0.85:
        return "critical"
    if churn_score >= 0.70:
        return "high"
    if churn_score >= 0.50:
        return "medium"
    return "low"


def _normalize_customer_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    cid_col = _pick_column(df, CUSTOMER_ID_ALIASES)
    normalized = df.copy()

    if cid_col is None:
        normalized["customer_id"] = range(1, len(normalized) + 1)
    elif cid_col != "customer_id":
        normalized = normalized.rename(columns={cid_col: "customer_id"})

    normalized["customer_id"] = pd.to_numeric(
        normalized["customer_id"],
        errors="coerce",
    )

    normalized = normalized.dropna(subset=["customer_id"]).copy()
    normalized["customer_id"] = normalized["customer_id"].astype(int)
    normalized = normalized.drop_duplicates(subset=["customer_id"], keep="last")

    return normalized


def _merge_on_customer_id(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames: list[pd.DataFrame] = []

    for df in frames:
        normalized = _normalize_customer_id_column(df)
        if not normalized.empty and "customer_id" in normalized.columns:
            valid_frames.append(normalized)

    if not valid_frames:
        return pd.DataFrame()

    merged = valid_frames[0]

    for other in valid_frames[1:]:
        merged = merged.merge(
            other,
            on="customer_id",
            how="outer",
            suffixes=("", "_dup"),
        )

        duplicate_cols = [col for col in merged.columns if col.endswith("_dup")]
        for dup_col in duplicate_cols:
            base_col = dup_col[:-4]
            if base_col in merged.columns:
                merged[base_col] = merged[base_col].combine_first(merged[dup_col])
            else:
                merged[base_col] = merged[dup_col]
            merged = merged.drop(columns=[dup_col])

    merged = merged.drop_duplicates(subset=["customer_id"], keep="last")
    return merged


def _load_user_artifact_frames(
    *,
    project_root: Path,
    data_dir: str = "data/raw_user",
    feature_store_dir: str = "data/feature_store_user",
    result_dir: str = "results_user",
) -> dict[str, tuple[pd.DataFrame, str | None]]:
    data_base = project_root / data_dir
    feature_base = project_root / feature_store_dir
    result_base = project_root / result_dir

    feature_df, feature_source = _read_first_existing_csv([
        feature_base / "customer_features.csv",
        feature_base / "features.csv",
        data_base / "customer_summary.csv",
        data_base / "customers.csv",
    ])

    customer_summary_df, customer_summary_source = _read_first_existing_csv([
        data_base / "customer_summary.csv",
        data_base / "customers.csv",
    ])

    uplift_df, uplift_source = _read_first_existing_csv([
        result_base / "uplift_segmentation.csv",
    ])

    selected_df, selected_source = _read_first_existing_csv([
        result_base / "optimization_selected_customers.csv",
    ])

    recommendation_df, recommendation_source = _read_first_existing_csv([
        result_base / "personalized_recommendations.csv",
    ])

    return {
        "features": (feature_df, feature_source),
        "customer_summary": (customer_summary_df, customer_summary_source),
        "uplift": (uplift_df, uplift_source),
        "selected": (selected_df, selected_source),
        "recommendations": (recommendation_df, recommendation_source),
    }


def _seed_feature_state(
    *,
    conn,
    feature_df: pd.DataFrame,
) -> int:
    if feature_df.empty:
        return 0

    feature_df = _normalize_customer_id_column(feature_df)
    seeded = 0

    for _, row in feature_df.iterrows():
        customer_id = _safe_int(row.get("customer_id"))
        if customer_id is None:
            continue

        last_event_time = _get_first_value(row, LAST_EVENT_TIME_ALIASES)
        parsed_last_event_time = None

        if last_event_time is not None:
            try:
                parsed_last_event_time = pd.to_datetime(last_event_time, errors="coerce")
                if pd.isna(parsed_last_event_time):
                    parsed_last_event_time = None
            except Exception:
                parsed_last_event_time = None

        conn.execute(
            text("""
            INSERT INTO customer_feature_state (
                customer_id,
                last_event_time,
                visit_7d,
                browse_7d,
                search_7d,
                add_to_cart_7d,
                cart_remove_7d,
                purchase_30d,
                revenue_30d,
                support_30d,
                refund_30d,
                coupon_open_30d,
                coupon_redeem_30d,
                inactivity_days,
                feature_payload,
                seeded_at,
                source_updated_at,
                updated_at
            )
            VALUES (
                :customer_id,
                :last_event_time,
                :visit_7d,
                :browse_7d,
                :search_7d,
                :add_to_cart_7d,
                :cart_remove_7d,
                :purchase_30d,
                :revenue_30d,
                :support_30d,
                :refund_30d,
                :coupon_open_30d,
                :coupon_redeem_30d,
                :inactivity_days,
                CAST(:feature_payload AS JSONB),
                now(),
                now(),
                now()
            )
            ON CONFLICT (customer_id)
            DO UPDATE SET
                last_event_time = COALESCE(EXCLUDED.last_event_time, customer_feature_state.last_event_time),
                visit_7d = EXCLUDED.visit_7d,
                browse_7d = EXCLUDED.browse_7d,
                search_7d = EXCLUDED.search_7d,
                add_to_cart_7d = EXCLUDED.add_to_cart_7d,
                cart_remove_7d = EXCLUDED.cart_remove_7d,
                purchase_30d = EXCLUDED.purchase_30d,
                revenue_30d = EXCLUDED.revenue_30d,
                support_30d = EXCLUDED.support_30d,
                refund_30d = EXCLUDED.refund_30d,
                coupon_open_30d = EXCLUDED.coupon_open_30d,
                coupon_redeem_30d = EXCLUDED.coupon_redeem_30d,
                inactivity_days = EXCLUDED.inactivity_days,
                feature_payload = EXCLUDED.feature_payload,
                seeded_at = now(),
                source_updated_at = now(),
                updated_at = now()
            """),
            {
                "customer_id": customer_id,
                "last_event_time": parsed_last_event_time,
                "visit_7d": _safe_int(_get_first_value(row, ["visit_7d", "visits_7d", "session_7d"]), 0),
                "browse_7d": _safe_int(_get_first_value(row, ["browse_7d", "page_view_7d", "views_7d"]), 0),
                "search_7d": _safe_int(_get_first_value(row, ["search_7d", "searches_7d"]), 0),
                "add_to_cart_7d": _safe_int(_get_first_value(row, ["add_to_cart_7d", "cart_add_7d"]), 0),
                "cart_remove_7d": _safe_int(_get_first_value(row, ["cart_remove_7d", "remove_from_cart_7d"]), 0),
                "purchase_30d": _safe_int(_get_first_value(row, ["purchase_30d", "orders_30d", "order_count_30d"]), 0),
                "revenue_30d": _safe_float(_get_first_value(row, ["revenue_30d", "sales_30d", "amount_30d"]), 0.0),
                "support_30d": _safe_int(_get_first_value(row, ["support_30d", "cs_30d", "support_contact_30d"]), 0),
                "refund_30d": _safe_int(_get_first_value(row, ["refund_30d", "refund_count_30d"]), 0),
                "coupon_open_30d": _safe_int(_get_first_value(row, ["coupon_open_30d", "coupon_opens_30d"]), 0),
                "coupon_redeem_30d": _safe_int(_get_first_value(row, ["coupon_redeem_30d", "coupon_redeems_30d"]), 0),
                "inactivity_days": _safe_float(_get_first_value(row, ["inactivity_days", "days_since_last_activity", "recency_days"]), 0.0),
                "feature_payload": _row_payload(row),
            },
        )

        seeded += 1

    return seeded


def _seed_customer_scores(
    *,
    conn,
    score_df: pd.DataFrame,
) -> int:
    if score_df.empty:
        return 0

    score_df = _normalize_customer_id_column(score_df)
    seeded = 0

    for _, row in score_df.iterrows():
        customer_id = _safe_int(row.get("customer_id"))
        if customer_id is None:
            continue

        churn_score = _safe_float(_get_first_value(row, CHURN_SCORE_ALIASES), None)
        clv = _safe_float(_get_first_value(row, CLV_ALIASES), None)
        uplift_score = _safe_float(_get_first_value(row, UPLIFT_ALIASES), None)
        expected_roi = _safe_float(_get_first_value(row, EXPECTED_ROI_ALIASES), None)
        expected_profit = _safe_float(_get_first_value(row, EXPECTED_PROFIT_ALIASES), None)

        risk_segment = _safe_str(_get_first_value(row, RISK_SEGMENT_ALIASES), None)
        if risk_segment is None:
            risk_segment = _risk_segment_from_score(churn_score)

        uplift_segment = _safe_str(_get_first_value(row, UPLIFT_SEGMENT_ALIASES), None)

        persona_value = _safe_str(_get_first_value(row, ["persona", "customer_persona", "customer_segment", "lifecycle_segment"]), None)

        conn.execute(
            text("""
            INSERT INTO customer_scores (
                customer_id,
                churn_score,
                clv,
                uplift_score,
                expected_roi,
                expected_incremental_profit,
                risk_segment,
                uplift_segment,
                persona,
                model_version,
                score_payload,
                seeded_at,
                scored_at
            )
            VALUES (
                :customer_id,
                :churn_score,
                :clv,
                :uplift_score,
                :expected_roi,
                :expected_incremental_profit,
                :risk_segment,
                :uplift_segment,
                :persona,
                :model_version,
                CAST(:score_payload AS JSONB),
                now(),
                now()
            )
            ON CONFLICT (customer_id)
            DO UPDATE SET
                churn_score = EXCLUDED.churn_score,
                clv = EXCLUDED.clv,
                uplift_score = EXCLUDED.uplift_score,
                expected_roi = EXCLUDED.expected_roi,
                expected_incremental_profit = EXCLUDED.expected_incremental_profit,
                risk_segment = EXCLUDED.risk_segment,
                uplift_segment = EXCLUDED.uplift_segment,
                persona = COALESCE(EXCLUDED.persona, customer_scores.persona),                
                model_version = EXCLUDED.model_version,
                score_payload = EXCLUDED.score_payload,
                seeded_at = now(),
                scored_at = now()
            """),
            {
                "customer_id": customer_id,
                "churn_score": churn_score,
                "clv": clv,
                "uplift_score": uplift_score,
                "expected_roi": expected_roi,
                "expected_incremental_profit": expected_profit,
                "risk_segment": risk_segment,
                "uplift_segment": uplift_segment,
                "persona": persona_value,                
                "model_version": "seeded_from_user_artifacts",
                "score_payload": _row_payload(row),
            },
        )

        seeded += 1

    return seeded


def _seed_recommendation_candidates(
    *,
    conn,
    recommendation_df: pd.DataFrame,
) -> int:
    if recommendation_df.empty:
        return 0

    recommendation_df = _normalize_customer_id_column(recommendation_df)
    seeded = 0

    for _, row in recommendation_df.iterrows():
        customer_id = _safe_int(row.get("customer_id"))
        if customer_id is None:
            continue

        recommended_action = _safe_str(
            _get_first_value(row, RECOMMENDED_ACTION_ALIASES),
            "retention_message",
        )

        recommended_category = _safe_str(
            _get_first_value(row, RECOMMENDED_CATEGORY_ALIASES),
            None,
        )

        coupon_cost = _safe_float(_get_first_value(row, COUPON_COST_ALIASES), 0.0)
        expected_roi = _safe_float(_get_first_value(row, EXPECTED_ROI_ALIASES), None)
        expected_profit = _safe_float(_get_first_value(row, EXPECTED_PROFIT_ALIASES), None)
        priority_score = _safe_float(_get_first_value(row, PRIORITY_SCORE_ALIASES), 0.0)

        reason_tags = _safe_str(
            _get_first_value(row, ["reason_tags", "reason", "recommendation_reason"]),
            "seeded_from_user_artifacts",
        )

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
                seeded_at,
                generated_at
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
                now(),
                now()
            )
            """),
            {
                "customer_id": customer_id,
                "recommended_action": recommended_action,
                "recommended_category": recommended_category,
                "coupon_cost": coupon_cost,
                "expected_roi": expected_roi,
                "expected_incremental_profit": expected_profit,
                "priority_score": priority_score,
                "reason_tags": reason_tags,
                "source_payload": _row_payload(row),
            },
        )

        seeded += 1

    return seeded


def _seed_action_queue(
    *,
    conn,
    selected_df: pd.DataFrame,
) -> int:
    if selected_df.empty:
        return 0

    selected_df = _normalize_customer_id_column(selected_df)
    seeded = 0

    for _, row in selected_df.iterrows():
        customer_id = _safe_int(row.get("customer_id"))
        if customer_id is None:
            continue

        recommended_action = _safe_str(
            _get_first_value(row, RECOMMENDED_ACTION_ALIASES),
            "retention_offer",
        )

        coupon_cost = _safe_float(_get_first_value(row, COUPON_COST_ALIASES), 0.0)
        expected_roi = _safe_float(_get_first_value(row, EXPECTED_ROI_ALIASES), None)
        expected_profit = _safe_float(_get_first_value(row, EXPECTED_PROFIT_ALIASES), None)
        priority_score = _safe_float(_get_first_value(row, PRIORITY_SCORE_ALIASES), 0.0)

        trigger_reason = _safe_str(
            _get_first_value(row, ["trigger_reason", "reason", "selection_reason"]),
            "seeded from optimization_selected_customers",
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
                seeded_at,
                queued_at
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
                now(),
                now()
            )
            """),
            {
                "customer_id": customer_id,
                "recommended_action": recommended_action,
                "intervention_intensity": _safe_str(
                    _get_first_value(row, ["intervention_intensity", "intensity"]),
                    "medium",
                ),
                "coupon_cost": coupon_cost,
                "expected_profit": expected_profit,
                "expected_roi": expected_roi,
                "priority_score": priority_score,
                "trigger_reason": trigger_reason,
                "source_payload": _row_payload(row),
            },
        )

        seeded += 1

    return seeded


def seed_user_live_from_artifacts(
    *,
    db_url: str,
    project_root: Path | None = None,
    reset: bool = True,
    data_dir: str = "data/raw_user",
    feature_store_dir: str = "data/feature_store_user",
    result_dir: str = "results_user",
    model_dir: str = "models_user",
    rescore_after_seed: bool = True,
    refresh_actions_after_rescore: bool = True,
    rescore_batch_size: int = 2000,
    action_threshold: float = 0.50,
    min_expected_roi: float = 0.0,
    min_expected_profit: float = 0.0,
) -> dict[str, Any]:
    """
    3단계 핵심 함수.

    이미 생성된 user 산출물들을 PostgreSQL live serving table에 초기 적재한다.

    reset=True:
        기존 live table 내용을 비우고 현재 user 산출물 기준으로 다시 seed한다.
        개발/검증 단계에서는 True 권장.
    """
    root = project_root or _project_root()

    ensure_user_live_seed_columns(db_url)

    frames = _load_user_artifact_frames(
        project_root=root,
        data_dir=data_dir,
        feature_store_dir=feature_store_dir,
        result_dir=result_dir,
    )

    feature_df, feature_source = frames["features"]
    customer_summary_df, customer_summary_source = frames["customer_summary"]
    uplift_df, uplift_source = frames["uplift"]
    selected_df, selected_source = frames["selected"]
    recommendation_df, recommendation_source = frames["recommendations"]

    score_df = _merge_on_customer_id([
        customer_summary_df,
        feature_df,
        uplift_df,
        selected_df,
    ])

    if feature_df.empty and customer_summary_df.empty:
        return {
            "success": False,
            "message": "No user feature/customer summary artifact found. Check data/raw_user or data/feature_store_user.",
            "sources": {
                "feature_source": feature_source,
                "customer_summary_source": customer_summary_source,
                "uplift_source": uplift_source,
                "selected_source": selected_source,
                "recommendation_source": recommendation_source,
            },
        }

    if feature_df.empty:
        feature_df = customer_summary_df.copy()

    with user_live_session(db_url) as conn:
        if reset:
            conn.execute(text("TRUNCATE TABLE recommendation_candidates RESTART IDENTITY"))
            conn.execute(text("TRUNCATE TABLE action_queue RESTART IDENTITY"))
            conn.execute(text("TRUNCATE TABLE customer_scores"))
            conn.execute(text("TRUNCATE TABLE customer_feature_state"))
            conn.execute(text("TRUNCATE TABLE customer_events RESTART IDENTITY"))

        feature_count = _seed_feature_state(
            conn=conn,
            feature_df=feature_df,
        )

        score_count = _seed_customer_scores(
            conn=conn,
            score_df=score_df,
        )

        recommendation_count = _seed_recommendation_candidates(
            conn=conn,
            recommendation_df=recommendation_df,
        )

        action_queue_count = _seed_action_queue(
            conn=conn,
            selected_df=selected_df,
        )

        summary = conn.execute(
            text("""
            SELECT
                (SELECT COUNT(*) FROM customer_feature_state) AS feature_state_count,
                (SELECT COUNT(*) FROM customer_scores) AS score_count,
                (SELECT COUNT(*) FROM recommendation_candidates) AS recommendation_count,
                (SELECT COUNT(*) FROM action_queue) AS action_queue_count
            """)
        ).mappings().first()

    rescore_result: dict[str, Any] | None = None
    action_refresh_result: dict[str, Any] | None = None

    if rescore_after_seed and feature_count > 0:
        # Import lazily to avoid a module import cycle.
        from src.api.services.user_live_scoring import score_all_customers

        rescore_result = score_all_customers(
            db_url=db_url,
            model_dir=root / model_dir,
            batch_size=int(rescore_batch_size),
        )

        if refresh_actions_after_rescore and rescore_result.get("success"):
            from src.api.services.user_live_actions import update_live_actions_for_customers

            customer_ids = _normalize_customer_id_column(feature_df)["customer_id"].astype(int).tolist()
            action_refresh_result = update_live_actions_for_customers(
                db_url=db_url,
                customer_ids=customer_ids,
                threshold=float(action_threshold),
                min_expected_roi=float(min_expected_roi),
                min_expected_profit=float(min_expected_profit),
            )

    with user_live_session(db_url) as conn:
        final_summary = conn.execute(
            text("""
            SELECT
                (SELECT COUNT(*) FROM customer_feature_state) AS feature_state_count,
                (SELECT COUNT(*) FROM customer_scores) AS score_count,
                (SELECT COUNT(*) FROM recommendation_candidates) AS recommendation_count,
                (SELECT COUNT(*) FROM action_queue) AS action_queue_count,
                (SELECT AVG(churn_score) FROM customer_scores) AS avg_churn_score,
                (SELECT COUNT(*) FROM customer_scores WHERE churn_score >= 0.5) AS high_risk_customers,
                (SELECT MIN(churn_score) FROM customer_scores) AS min_churn_score,
                (SELECT MAX(churn_score) FROM customer_scores) AS max_churn_score,
                (SELECT MAX(scored_at) FROM customer_scores) AS latest_scored_at
            """)
        ).mappings().first()

    return {
        "success": True,
        "reset": reset,
        "seeded": {
            "customer_feature_state": feature_count,
            "customer_scores": score_count,
            "recommendation_candidates": recommendation_count,
            "action_queue": action_queue_count,
        },
        "rescore_after_seed": bool(rescore_after_seed),
        "rescore": rescore_result,
        "action_refresh_after_rescore": bool(refresh_actions_after_rescore),
        "action_refresh": action_refresh_result,
        "db_counts": dict(summary or {}),
        "final_db_counts": dict(final_summary or {}),
        "sources": {
            "feature_source": feature_source,
            "customer_summary_source": customer_summary_source,
            "uplift_source": uplift_source,
            "selected_source": selected_source,
            "recommendation_source": recommendation_source,
            "model_dir": str(root / model_dir),
        },
    }


def get_user_live_seed_status(
    *,
    db_url: str,
) -> dict[str, Any]:
    ensure_user_live_seed_columns(db_url)

    with user_live_session(db_url) as conn:
        row = conn.execute(
            text("""
            SELECT
                (SELECT COUNT(*) FROM customer_events) AS event_count,
                (SELECT COUNT(*) FROM customer_feature_state) AS feature_state_count,
                (SELECT COUNT(*) FROM customer_scores) AS score_count,
                (SELECT COUNT(*) FROM recommendation_candidates) AS recommendation_count,
                (SELECT COUNT(*) FROM action_queue) AS action_queue_count,
                (SELECT MAX(seeded_at) FROM customer_feature_state) AS latest_feature_seeded_at,
                (SELECT MAX(seeded_at) FROM customer_scores) AS latest_score_seeded_at,
                (SELECT MAX(seeded_at) FROM recommendation_candidates) AS latest_recommendation_seeded_at,
                (SELECT MAX(seeded_at) FROM action_queue) AS latest_action_queue_seeded_at
            """)
        ).mappings().first()

    return {
        "success": True,
        "status": dict(row or {}),
    }
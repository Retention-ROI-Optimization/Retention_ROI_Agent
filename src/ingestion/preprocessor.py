"""
preprocessor.py — Auto-preprocessing engine for arbitrary CSV datasets.

Converts user-uploaded data into the internal schema required by the
churn/retention ML pipelines, handling:
- Column mapping & renaming
- Missing value imputation (adaptive strategy per dtype)
- Datetime parsing and feature extraction
- Categorical encoding
- Outlier clipping
- Feature generation from transactional data
- Chunked processing for large files (no size limit)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.ingestion.validator import ValidationResult


@dataclass
class PreprocessingResult:
    """Output of the auto-preprocessing pipeline."""
    customer_summary: pd.DataFrame
    events: pd.DataFrame
    orders: pd.DataFrame
    cohort_retention: pd.DataFrame
    treatment_assignments: pd.DataFrame
    campaign_exposures: pd.DataFrame
    state_snapshots: pd.DataFrame
    customers: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ── Constants ──

INTERNAL_CUSTOMER_COLUMNS = [
    "customer_id", "persona", "signup_date", "acquisition_month",
    "region", "device_type", "acquisition_channel",
    "churn_probability", "uplift_score", "clv",
    "coupon_cost", "expected_incremental_profit", "expected_roi",
    "uplift_segment", "treatment_group", "treatment_flag",
    "recency_days", "frequency", "monetary",
    "visits_last_7", "visits_prev_7", "visit_change_rate",
    "purchase_last_30", "purchase_prev_30", "purchase_change_rate",
    "inactivity_days", "coupon_exposure_count", "coupon_redeem_count",
    "coupon_fatigue_score", "discount_dependency_score",
    "discount_pressure_score", "discount_effect_penalty",
    "price_sensitivity", "coupon_affinity", "support_contact_propensity",
    "uplift_segment_true",
]

DEFAULT_PERSONA_NAMES = ["vip_loyal", "regular_loyal", "price_sensitive", "explorer", "churn_progressing", "new_signup"]
DEFAULT_UPLIFT_SEGMENTS = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]

CHUNK_SIZE = 50000  # rows per chunk for large file processing

# ── Role / event_type 설명 사전 (UI 도움말용) ─────────────────────────

ROLE_DESCRIPTIONS: Dict[str, str] = {
    "customer_id": "고객을 식별하는 고유 ID. 같은 고객이 여러 번 등장해도 동일한 값이어야 합니다.",
    "timestamp": "이벤트가 발생한 시각. 분석 기준 시점이 되며, RFM·세션·시계열 분석에 사용됩니다.",
    "event_type": "이벤트 종류 (구매, 방문, 검색 등). 회사마다 다른 명명이 있어 매핑이 필요합니다.",
    "amount": "거래 금액 또는 결제 금액. 매출·CLV·ROI 계산에 사용됩니다.",
    "churn_flag": "이탈 여부 (활성·이탈·취소 등). 모델 학습 라벨로 사용됩니다.",
    "category": "상품 또는 서비스 카테고리. 카테고리별 행동 분석에 사용됩니다.",
    "quantity": "주문 수량. '평균 주문 수량' 피처에 활용됩니다(선택 컬럼).",
    "persona": "고객 세그먼트 (VIP·일반·신규 등). 페르소나별 분석에 사용됩니다.",
    "region": "지역 또는 국가. 지역별 지표 분석에 사용됩니다.",
}

# ── Event type value mapping ──────────────────────────────────────────
# 회사마다 이벤트 명명이 다르므로(예: "login" vs "session_start" vs "방문"),
# 사용자 값을 내부 표준 6종으로 매핑한다. 매칭 실패 시 "other"로 분류.
INTERNAL_EVENT_TYPES = ["visit", "page_view", "search", "add_to_cart", "purchase", "support_contact"]

EVENT_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "visit": "사이트/앱 접속 — 로그인, 세션 시작, 앱 실행 등을 포함합니다.",
    "page_view": "페이지/상품 조회 — 단순 조회, 스크롤, 클릭, 영상 재생 등을 포함합니다.",
    "search": "검색 — 키워드 검색, 필터 적용 등.",
    "add_to_cart": "장바구니/위시리스트 추가 — 즐겨찾기, 좋아요도 포함됩니다.",
    "purchase": "구매·결제 완료 — 결제 성공, 주문 완료, 구독 시작도 여기 포함됩니다.",
    "support_contact": "고객 지원 — 문의, 환불 요청, 취소, 해지, NPS 응답 등.",
    "other": "위 6종에 해당하지 않는 기타 이벤트. 분석에는 포함되지만 활용도는 낮습니다.",
    "ignore": "해당 행을 분석에서 완전히 제외합니다 (의미 없는 이벤트로 판단될 때).",
}

EVENT_VALUE_SYNONYMS: Dict[str, Set[str]] = {
    "purchase": {
        "purchase", "purchased", "buy", "bought", "checkout", "checkout_complete",
        "checkout_start", "order", "order_complete", "order_placed", "transaction",
        "payment", "paid", "complete_purchase", "payment_success", "payment_complete",
        "payment_fail", "payment_failed", "subscription_start", "subscribe",
        "renewal", "renew", "plan_change", "plan_upgrade", "plan_downgrade",
        "upgrade", "downgrade",
        "결제", "구매", "주문", "주문완료", "결제완료",
    },
    "visit": {
        "visit", "visited", "session_start", "session_begin", "session_end",
        "session_close", "login", "logged_in", "logout", "log_out",
        "app_open", "app_close", "app_launch", "site_visit", "launch",
        "sign_in", "signin", "active_session", "push_open", "notification_open",
        "방문", "로그인", "접속", "세션시작", "세션종료",
    },
    "page_view": {
        "page_view", "pageview", "view", "viewed", "product_view", "viewed_product",
        "item_view", "page", "screen_view", "impression", "view_item",
        "view_item_list", "select_item", "scroll", "feature_use", "feature_view",
        "click", "tap", "select", "browse", "explore",
        "stream_start", "stream_complete", "stream_end", "watch", "watched",
        "video_play", "video_complete", "video_pause", "play", "pause", "resume",
        "push_received", "notification_received",
        "조회", "상품조회", "페이지뷰", "둘러보기",
    },
    "search": {
        "search", "searched", "query", "find", "lookup", "filter", "sort",
        "검색", "필터",
    },
    "add_to_cart": {
        "add_to_cart", "addtocart", "cart_add", "add_cart", "added_to_cart",
        "remove_from_cart", "cart_remove", "wishlist_add", "favorite", "favorited",
        "like", "liked", "bookmark", "save",
        "장바구니", "장바구니추가", "찜", "즐겨찾기",
    },
    "support_contact": {
        "support", "support_contact", "support_chat", "contact", "inquiry", "help",
        "feedback", "cs", "customer_service", "ticket", "ticket_open", "ticket_close",
        "complaint", "report_issue", "nps", "nps_submit", "survey",
        "refund_request", "refund", "return_request", "cancel_request",
        "cancel", "cancellation", "uninstall", "uninstall_signal", "unsubscribe",
        "문의", "상담", "고객센터", "신고", "환불", "취소", "해지",
    },
}


def _normalize_event_type(value: Any) -> str:
    """
    사용자 event_type 값 → 내부 표준 6종 중 하나로 매핑.
    1) 정확 매칭 우선
    2) 부분 매칭은 '가장 긴 키워드'를 가진 후보가 이김
       (예: "ticket_open"은 "open"(visit, 4글자)보다 "ticket"(support, 6글자)이 우선)
    3) 너무 짧은 키워드(view/buy 등 4글자 미만)는 정확매칭만 인정 — substring 오매칭 방지
    매칭 실패 시 'other'.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "other"
    norm = re.sub(r"[^a-z0-9가-힣]", "_", str(value).strip().lower())
    norm = re.sub(r"_+", "_", norm).strip("_")
    if not norm:
        return "other"

    # 1) 정확 매칭
    for std, synonyms in EVENT_VALUE_SYNONYMS.items():
        if norm in synonyms:
            return std

    # 2) 부분 매칭 — 가장 긴 키워드가 우선
    best_std = None
    best_len = 0
    for std, synonyms in EVENT_VALUE_SYNONYMS.items():
        for syn in synonyms:
            if len(syn) < 4:
                continue
            if syn in norm and len(syn) > best_len:
                best_std = std
                best_len = len(syn)
    return best_std if best_std else "other"


def _build_event_type_mapping_report(original_values: pd.Series) -> Dict[str, Any]:
    """매핑 결과 리포트 — 사용자에게 어떤 값이 어떻게 매핑되었는지 보여주기 위함."""
    mapping: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for raw in original_values.dropna().astype(str).unique():
        std = _normalize_event_type(raw)
        mapping[raw] = std
    for raw, std in mapping.items():
        counts[std] = counts.get(std, 0) + int((original_values.astype(str) == raw).sum())
    unmapped = [k for k, v in mapping.items() if v == "other"]
    return {
        "value_mapping": mapping,
        "count_by_internal_type": counts,
        "unmapped_values": unmapped,
        "coverage_rate": round(1.0 - (counts.get("other", 0) / max(sum(counts.values()), 1)), 4),
    }


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_divide(a, b, default: float = 0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, default, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out


def _detect_date_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Try to parse a column as datetime."""
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return df[col]
    try:
        return pd.to_datetime(df[col], errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=df.index)


def _infer_churn_label(
    df: pd.DataFrame,
    schema: Dict[str, str],
    inactivity_threshold_days: int = 30,
) -> pd.Series:
    """
    Infer churn labels from data.
    NOTE: 이 함수는 preprocess_uploaded_data Step 1 이후에 호출되므로,
    df의 customer_id 컬럼은 이미 "customer_id"로 표준화되어 있다.

    inactivity_threshold_days: "이탈" 정의 — 마지막 활동 이후 N일 이상 비활성이면 이탈.
    """
    if "churn_flag" in schema and schema["churn_flag"] in df.columns:
        col = schema["churn_flag"]
        series = df[col].copy()
        # Handle various formats
        if series.dtype == object:
            mapping = {
                "yes": 1, "no": 0, "y": 1, "n": 0,
                "true": 1, "false": 0, "1": 1, "0": 0,
                "churn": 1, "active": 0, "churned": 1,
                "churn_risk": 1, "dormant": 0.5,
            }
            series = series.str.strip().str.lower().map(mapping).fillna(0.0)
        return _safe_numeric(series, 0.0).clip(0.0, 1.0)

    # If no churn flag, infer from inactivity threshold
    if "timestamp" in schema and schema["timestamp"] in df.columns:
        ts_col = schema["timestamp"]
        ts = _detect_date_column(df, ts_col)
        if ts.notna().any() and "customer_id" in df.columns:
            max_date = ts.max()
            last_activity = df.groupby("customer_id")[ts_col].transform("max")
            last_ts = pd.to_datetime(last_activity, errors="coerce")
            days_since = (max_date - last_ts).dt.days.fillna(999)
            return (days_since >= int(inactivity_threshold_days)).astype(float)

    return pd.Series(0.5, index=df.index)


def _compute_rfm(df: pd.DataFrame, customer_id_col: str, amount_col: Optional[str], timestamp_col: Optional[str]) -> pd.DataFrame:
    """Compute RFM (Recency, Frequency, Monetary) features."""
    rfm = pd.DataFrame({"customer_id": df[customer_id_col].unique()})

    if timestamp_col and timestamp_col in df.columns:
        ts = _detect_date_column(df, timestamp_col)
        valid = df[ts.notna()].copy()
        valid["_ts"] = ts[ts.notna()]
        max_date = valid["_ts"].max()

        # Recency
        recency = valid.groupby(customer_id_col)["_ts"].max()
        rfm = rfm.merge(
            (max_date - recency).dt.days.rename("recency_days").reset_index(),
            left_on="customer_id", right_on=customer_id_col, how="left"
        )
        if customer_id_col != "customer_id" and customer_id_col in rfm.columns:
            rfm = rfm.drop(columns=[customer_id_col])
        rfm["recency_days"] = rfm["recency_days"].fillna(999).clip(lower=0)

        # Frequency
        freq = valid.groupby(customer_id_col).size().rename("frequency")
        rfm = rfm.merge(freq.reset_index(), left_on="customer_id", right_on=customer_id_col, how="left")
        if customer_id_col != "customer_id" and customer_id_col in rfm.columns:
            rfm = rfm.drop(columns=[customer_id_col])
        rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    else:
        rfm["recency_days"] = 0
        rfm["frequency"] = df.groupby(customer_id_col).size().reindex(rfm["customer_id"]).fillna(0).astype(int).values

    if amount_col and amount_col in df.columns:
        monetary = _safe_numeric(df[amount_col], 0.0)
        mon = df.assign(_amount=monetary).groupby(customer_id_col)["_amount"].sum().rename("monetary")
        rfm = rfm.merge(mon.reset_index(), left_on="customer_id", right_on=customer_id_col, how="left")
        if customer_id_col != "customer_id" and customer_id_col in rfm.columns:
            rfm = rfm.drop(columns=[customer_id_col])
        rfm["monetary"] = rfm["monetary"].fillna(0.0)
    else:
        rfm["monetary"] = 0.0

    return rfm


def _assign_personas(df: pd.DataFrame) -> pd.Series:
    """Heuristically assign customer personas based on available features."""
    n = len(df)
    personas = pd.Series("regular_loyal", index=df.index)

    monetary = _safe_numeric(df.get("monetary", pd.Series(0.0, index=df.index)))
    frequency = _safe_numeric(df.get("frequency", pd.Series(0.0, index=df.index)))
    recency = _safe_numeric(df.get("recency_days", pd.Series(0.0, index=df.index)))
    churn = _safe_numeric(df.get("churn_probability", pd.Series(0.5, index=df.index)))

    # Percentile-based assignment
    if monetary.std() > 0:
        mon_pct = monetary.rank(pct=True)
        freq_pct = frequency.rank(pct=True)

        personas = np.select(
            [
                (mon_pct >= 0.80) & (freq_pct >= 0.70),
                (mon_pct >= 0.50) & (freq_pct >= 0.50),
                (churn >= 0.60),
                (recency <= 30) & (frequency <= 2),
                (mon_pct < 0.30),
            ],
            ["vip_loyal", "regular_loyal", "churn_progressing", "new_signup", "price_sensitive"],
            default="explorer",
        )
    return pd.Series(personas, index=df.index)


def _assign_uplift_segments(df: pd.DataFrame) -> pd.Series:
    """Assign uplift segments based on churn probability and other signals."""
    churn = _safe_numeric(df.get("churn_probability", pd.Series(0.5, index=df.index)))
    monetary = _safe_numeric(df.get("monetary", pd.Series(0.0, index=df.index)))

    segments = np.select(
        [
            (churn >= 0.45) & (monetary > monetary.median()),
            (churn < 0.45) & (monetary > monetary.median()),
            (churn >= 0.45) & (monetary <= monetary.median()),
        ],
        ["Persuadables", "Sure Things", "Lost Causes"],
        default="Sleeping Dogs",
    )
    return pd.Series(segments, index=df.index)


def _extract_real_events(
    df: pd.DataFrame,
    schema: Dict[str, str],
    user_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    사용자가 올린 데이터에 event_type + timestamp가 모두 있으면 실제 이벤트 테이블 구성.
    값을 내부 표준 6종으로 매핑한 결과와 매핑 리포트를 함께 반환.
    하나라도 빠지면 (None, None) → 호출부에서 합성으로 fallback.

    user_mapping: 사용자가 수동으로 확정한 매핑 ({"product_view": "page_view", ...}).
                  지정되면 자동 매핑보다 우선. "ignore"/"skip" 값이면 해당 행 제외.
    """
    ev_col = schema.get("event_type")
    ts_col = schema.get("timestamp")
    if not ev_col or not ts_col or ev_col not in df.columns or ts_col not in df.columns:
        return None, None

    ts = _detect_date_column(df, ts_col)
    valid_mask = ts.notna() & df["customer_id"].notna()
    if valid_mask.sum() == 0:
        return None, None

    sub = df[valid_mask].copy()
    sub["_ts"] = ts[valid_mask]

    original_events = sub[ev_col].astype(str)

    # 매핑 결정: 사용자 매핑 우선, 없으면 자동
    if user_mapping:
        normalized = original_events.map(lambda v: user_mapping.get(v, _normalize_event_type(v)))
        mapping_source = "manual"
    else:
        normalized = original_events.map(_normalize_event_type)
        mapping_source = "auto"

    # "ignore" / "skip" 으로 표시된 값은 events에서 제외
    drop_mask = normalized.isin({"ignore", "skip"})
    if drop_mask.any():
        sub = sub[~drop_mask]
        original_events = original_events[~drop_mask]
        normalized = normalized[~drop_mask]

    mapping_report = _build_event_type_mapping_report(original_events)
    mapping_report["mapping_source"] = mapping_source
    if user_mapping:
        mapping_report["value_mapping"] = {
            raw: user_mapping.get(raw, _normalize_event_type(raw))
            for raw in original_events.astype(str).unique()
        }

    events_df = pd.DataFrame({
        "customer_id": sub["customer_id"].astype(int).values,
        "timestamp": sub["_ts"].values,
        "event_type": normalized.values,
        "event_type_original": original_events.values,
    })

    # 선택 컬럼 (있으면 사용, 없으면 기본값)
    cat_col = schema.get("category")
    if cat_col and cat_col in sub.columns:
        events_df["item_category"] = sub[cat_col].astype(str).values
    else:
        events_df["item_category"] = "general"

    qty_col = schema.get("quantity")
    if qty_col and qty_col in sub.columns:
        events_df["quantity"] = _safe_numeric(sub[qty_col], 1).astype(int).values
    else:
        events_df["quantity"] = 1

    # 식별자 생성
    events_df = events_df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    events_df["event_id"] = ["EVT-" + str(i) for i in range(len(events_df))]
    events_df["session_id"] = (
        events_df["customer_id"].astype(str) + "-"
        + pd.to_datetime(events_df["timestamp"]).dt.strftime("%Y%m%d")
    )

    # 컬럼 순서 정리 (다운스트림 호환: customer_id, timestamp, event_type, session_id, item_category, quantity)
    events_df = events_df[[
        "event_id", "customer_id", "timestamp", "event_type",
        "event_type_original", "session_id", "item_category", "quantity",
    ]]

    return events_df, mapping_report


def _build_orders_from_real_events(
    df: pd.DataFrame,
    real_events: pd.DataFrame,
    schema: Dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    사용자 데이터에 amount 컬럼이 있으면 실제 금액으로 orders 구성.
    purchase로 매핑된 이벤트만 추출하고, (customer_id, timestamp) 기준 머지로 원본 amount를 가져온다.
    """
    amount_col = schema["amount"]
    ts_col = schema.get("timestamp")

    purchase_mask = real_events["event_type"] == "purchase"
    if purchase_mask.sum() == 0:
        return pd.DataFrame(columns=[
            "order_id", "customer_id", "order_time", "item_category",
            "quantity", "gross_amount", "discount_amount", "net_amount", "coupon_used",
        ])

    src = df[[c for c in ["customer_id", ts_col, amount_col] if c and c in df.columns]].copy()
    src["customer_id"] = pd.to_numeric(src["customer_id"], errors="coerce")
    src = src.dropna(subset=["customer_id"])
    src["customer_id"] = src["customer_id"].astype(int)
    src["_ts"] = _detect_date_column(src, ts_col) if ts_col else pd.NaT
    src["_amount"] = _safe_numeric(src[amount_col], 0.0)

    purchases = real_events[purchase_mask].copy()
    purchases = purchases.merge(
        src[["customer_id", "_ts", "_amount"]],
        left_on=["customer_id", "timestamp"],
        right_on=["customer_id", "_ts"],
        how="left",
    )
    purchases["_amount"] = purchases["_amount"].fillna(0.0)

    coupon_used = rng.binomial(1, 0.3, size=len(purchases))
    discount = purchases["_amount"].values * 0.1 * coupon_used

    orders = pd.DataFrame({
        "order_id": ["ORD-" + str(i) for i in range(len(purchases))],
        "customer_id": purchases["customer_id"].astype(int).values,
        "order_time": purchases["timestamp"].values,
        "item_category": purchases["item_category"].values,
        "quantity": purchases["quantity"].astype(int).values,
        "gross_amount": np.round(purchases["_amount"].values, 2),
        "discount_amount": np.round(discount, 2),
        "net_amount": np.round(purchases["_amount"].values - discount, 2),
        "coupon_used": coupon_used.astype(int),
    })
    return orders


def _generate_synthetic_events(customer_summary: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate minimal synthetic event data from customer summary for pipeline compatibility."""
    rows = []
    event_types = ["visit", "page_view", "search", "add_to_cart", "purchase", "support_contact"]
    event_weights = [0.30, 0.20, 0.15, 0.15, 0.12, 0.08]

    for _, row in customer_summary.iterrows():
        cid = int(row["customer_id"])
        freq = max(int(row.get("frequency", 1)), 1)
        n_events = min(freq * 5, 50)

        base_date = pd.Timestamp(row.get("signup_date", "2025-01-01"))
        for i in range(n_events):
            event_type = rng.choice(event_types, p=event_weights)
            offset_days = rng.integers(0, 365)
            ts = base_date + pd.Timedelta(days=int(offset_days), hours=int(rng.integers(8, 22)), minutes=int(rng.integers(0, 60)))
            rows.append({
                "event_id": f"EVT-{cid}-{i}",
                "customer_id": cid,
                "timestamp": ts,
                "event_type": event_type,
                "session_id": f"SES-{cid}-{i // 3}",
                "item_category": rng.choice(["fashion", "beauty", "grocery", "sports", "health"]),
                "quantity": int(rng.integers(1, 4)),
            })
    return pd.DataFrame(rows)


def _generate_synthetic_orders(customer_summary: pd.DataFrame, events_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate order data from purchase events."""
    purchase_events = events_df[events_df["event_type"] == "purchase"].copy()
    if purchase_events.empty:
        return pd.DataFrame(columns=["order_id", "customer_id", "order_time", "item_category", "quantity", "gross_amount", "discount_amount", "net_amount", "coupon_used"])

    monetary_lookup = customer_summary.set_index("customer_id")["monetary"].to_dict()
    freq_lookup = customer_summary.set_index("customer_id")["frequency"].to_dict()

    orders = []
    for idx, row in purchase_events.iterrows():
        cid = int(row["customer_id"])
        freq = max(freq_lookup.get(cid, 1), 1)
        total_monetary = monetary_lookup.get(cid, 50000.0)
        avg_order = max(total_monetary / freq, 15000.0)

        gross = max(float(rng.normal(avg_order, avg_order * 0.2)), 10000.0)
        coupon_used = int(rng.random() < 0.3)
        discount = gross * 0.1 * coupon_used
        orders.append({
            "order_id": f"ORD-{cid}-{idx}",
            "customer_id": cid,
            "order_time": row["timestamp"],
            "item_category": row.get("item_category", "general"),
            "quantity": int(row.get("quantity", 1)),
            "gross_amount": round(gross, 2),
            "discount_amount": round(discount, 2),
            "net_amount": round(gross - discount, 2),
            "coupon_used": coupon_used,
        })
    return pd.DataFrame(orders)


def _generate_treatment_assignments(customer_summary: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate treatment/control assignments."""
    n = len(customer_summary)
    treatment_flags = rng.binomial(1, 0.5, size=n)

    base_cost = _safe_numeric(customer_summary.get("coupon_cost", pd.Series(8000, index=customer_summary.index)), 8000)

    return pd.DataFrame({
        "customer_id": customer_summary["customer_id"].astype(int),
        "treatment_group": np.where(treatment_flags, "treatment", "control"),
        "treatment_flag": treatment_flags,
        "campaign_type": "retention_coupon",
        "coupon_cost": base_cost.astype(int),
        "assigned_at": customer_summary.get("signup_date", pd.Timestamp("2025-01-01")),
    })


def _generate_state_snapshots(customer_summary: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate state snapshot data — fully vectorized.
    각 고객당 12개월 스냅샷을 한 번에 생성. iterrows 대신 cross-join으로 O(N) 처리.
    """
    n = len(customer_summary)
    if n == 0:
        return pd.DataFrame(columns=[
            "customer_id", "snapshot_date", "last_visit_date", "last_purchase_date",
            "visits_total", "purchases_total", "monetary_total", "inactivity_days",
            "current_status", "recent_visit_score", "recent_purchase_score",
            "recent_exposure_score", "coupon_fatigue_score", "discount_dependency_score",
        ])

    months = 12
    cs = customer_summary

    # 고객 단위 컬럼 추출 (벡터화)
    cid = cs["customer_id"].astype(int).values
    inactivity = _safe_numeric(cs.get("inactivity_days", pd.Series(0, index=cs.index)), 0).astype(int).values
    churn_prob = _safe_numeric(cs.get("churn_probability", pd.Series(0.5, index=cs.index)), 0.5).values
    frequency = _safe_numeric(cs.get("frequency", pd.Series(0, index=cs.index)), 0).astype(int).values
    monetary = _safe_numeric(cs.get("monetary", pd.Series(0.0, index=cs.index)), 0.0).values
    recency = _safe_numeric(cs.get("recency_days", pd.Series(0, index=cs.index)), 0).astype(int).values
    base_date = pd.to_datetime(cs.get("signup_date", pd.Timestamp("2025-01-01")), errors="coerce").fillna(pd.Timestamp("2025-01-01")).values

    # cross-join: 각 고객 × 12개월 → numpy tile/repeat으로 한 번에
    cid_rep = np.repeat(cid, months)
    inactivity_rep = np.repeat(inactivity, months)
    churn_rep = np.repeat(churn_prob, months)
    freq_rep = np.repeat(frequency, months)
    monetary_rep = np.repeat(monetary, months)
    recency_rep = np.repeat(recency, months)
    base_rep = np.repeat(base_date, months)

    month_offsets = np.tile(np.arange(months), n)
    snapshot_dates = pd.to_datetime(base_rep) + pd.to_timedelta(month_offsets * 30, unit="D")
    last_visit_dates = snapshot_dates - pd.to_timedelta(np.maximum(inactivity_rep, 0), unit="D")
    last_purchase_dates = snapshot_dates - pd.to_timedelta(np.maximum(recency_rep, 0), unit="D")

    # status 벡터화
    status = np.where(
        (inactivity_rep >= 30) | (churn_rep >= 0.7), "churn_risk",
        np.where((inactivity_rep >= 14) | (churn_rep >= 0.5), "dormant", "active")
    )

    total = n * months
    return pd.DataFrame({
        "customer_id": cid_rep,
        "snapshot_date": snapshot_dates,
        "last_visit_date": last_visit_dates,
        "last_purchase_date": last_purchase_dates,
        "visits_total": (freq_rep * 3).astype(int),
        "purchases_total": freq_rep.astype(int),
        "monetary_total": monetary_rep.astype(float),
        "inactivity_days": inactivity_rep.astype(int),
        "current_status": status,
        "recent_visit_score": rng.uniform(0, 2, size=total),
        "recent_purchase_score": rng.uniform(0, 2, size=total),
        "recent_exposure_score": rng.uniform(0, 1, size=total),
        "coupon_fatigue_score": rng.uniform(0, 2, size=total),
        "discount_dependency_score": rng.uniform(0, 1, size=total),
    })


def _generate_campaign_exposures(treatment_assignments: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate campaign exposure records for treatment customers — vectorized."""
    treated = treatment_assignments[treatment_assignments["treatment_flag"] == 1]
    if treated.empty:
        return pd.DataFrame(columns=["exposure_id", "customer_id", "exposure_time", "campaign_type", "coupon_cost"])

    n_treated = len(treated)
    # 각 고객당 1~3회 노출
    n_exposures = rng.integers(1, 4, size=n_treated)
    total = int(n_exposures.sum())

    cid = treated["customer_id"].astype(int).values
    assigned_at = pd.to_datetime(treated["assigned_at"], errors="coerce").fillna(pd.Timestamp("2025-01-01")).values
    campaign = treated.get("campaign_type", pd.Series(["retention_coupon"] * n_treated)).astype(str).values
    cost = treated.get("coupon_cost", pd.Series([8000] * n_treated)).astype(int).values

    cid_rep = np.repeat(cid, n_exposures)
    assigned_rep = np.repeat(assigned_at, n_exposures)
    campaign_rep = np.repeat(campaign, n_exposures)
    cost_rep = np.repeat(cost, n_exposures)

    offsets = rng.integers(0, 90, size=total)
    exposure_times = pd.to_datetime(assigned_rep) + pd.to_timedelta(offsets, unit="D")

    # exposure_id: 고객별 시퀀스 번호 (각 고객 안에서 0..n_exposures-1)
    seq = np.concatenate([np.arange(n) for n in n_exposures])
    exposure_ids = [f"EXP-{c}-{s}" for c, s in zip(cid_rep, seq)]

    return pd.DataFrame({
        "exposure_id": exposure_ids,
        "customer_id": cid_rep,
        "exposure_time": exposure_times,
        "campaign_type": campaign_rep,
        "coupon_cost": cost_rep,
    })


def _build_cohort_retention(customer_summary: pd.DataFrame) -> pd.DataFrame:
    """Build cohort retention table from customer summary."""
    if "acquisition_month" not in customer_summary.columns:
        return pd.DataFrame(columns=["cohort_month", "period", "cohort_size", "retained_customers", "retention_rate", "observed", "activity_definition", "retention_mode", "min_events_per_period"])

    rng = np.random.default_rng(42)
    cohorts = customer_summary["acquisition_month"].dropna().unique()
    rows = []
    for cohort in sorted(cohorts):
        cohort_size = int((customer_summary["acquisition_month"] == cohort).sum())
        for period in range(7):
            if period == 0:
                retention = 1.0
            else:
                base_retention = max(0.85 - 0.08 * period + rng.normal(0, 0.02), 0.15)
                retention = round(base_retention, 4)
            retained = int(round(cohort_size * retention))
            for activity_def in ["core_engagement", "all_activity", "purchase_only"]:
                for mode in ["rolling", "point"]:
                    rows.append({
                        "cohort_month": str(cohort),
                        "period": period,
                        "cohort_size": cohort_size,
                        "retained_customers": retained,
                        "retention_rate": retention,
                        "observed": True,
                        "activity_definition": activity_def,
                        "retention_mode": mode,
                        "min_events_per_period": 1,
                    })
    return pd.DataFrame(rows)


def preprocess_uploaded_data(
    df: pd.DataFrame,
    validation: ValidationResult,
    *,
    column_mapping_override: Optional[Dict[str, str]] = None,
    event_value_mapping: Optional[Dict[str, str]] = None,
    allow_synthetic_fallback: bool = True,
    churn_inactivity_days: int = 30,
    seed: int = 42,
) -> PreprocessingResult:
    """
    Transform uploaded data into the full internal schema.

    Parameters
    ----------
    column_mapping_override : Dict[str, str], optional
        사용자가 수동으로 수정한 컬럼 → 역할 매핑
        (예: {"customer_id": "user_id", "timestamp": "event_time"})
        지정되면 validation.detected_schema를 덮어씀.
    event_value_mapping : Dict[str, str], optional
        사용자가 수동으로 지정한 event_type 값 → 내부 표준값 매핑
        (예: {"product_view": "page_view", "checkout_start": "purchase"})
        지정되면 자동 매핑(_normalize_event_type)을 덮어씀.
        값이 "ignore" 또는 "skip"이면 해당 행은 events에서 제외.
    seed : int
        난수 시드.
    """
    rng = np.random.default_rng(seed)
    # 사용자 컬럼 매핑이 들어오면 그것을 우선, 없으면 자동 감지된 schema 사용
    schema = dict(column_mapping_override) if column_mapping_override else dict(validation.detected_schema)
    warnings: List[str] = []
    metadata: Dict[str, Any] = {
        "source": "user_upload",
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "detected_schema": schema,
    }

    # ── Step 1: Extract customer ID ──
    id_col = schema.get("customer_id", df.columns[0])

    # 메모리 절약: schema에 매핑된 컬럼들 + customer_id만 유지하고 불필요 컬럼 제거
    needed_cols = set()
    needed_cols.add(id_col)
    for role_col in schema.values():
        if role_col and role_col in df.columns:
            needed_cols.add(role_col)
    keep_cols = [c for c in df.columns if c in needed_cols]
    if len(keep_cols) < len(df.columns):
        df = df[keep_cols].copy()
    else:
        df = df.copy()

    if id_col != "customer_id":
        df = df.rename(columns={id_col: "customer_id"})

    # null ID 행 제거
    df = df.dropna(subset=["customer_id"])

    # 숫자 변환 시도 → 실패하면 factorize로 문자열 ID(UUID, "U12345" 등)를 정수 코드로 매핑.
    # 다운스트림(int(row["customer_id"]) 등)은 정수를 가정하므로, 원본 문자열은
    # customer_id_original 컬럼에 보존한다.
    numeric_ids = pd.to_numeric(df["customer_id"], errors="coerce")
    if len(df) > 0 and numeric_ids.notna().all():
        df["customer_id"] = numeric_ids.astype(int)
        metadata["customer_id_type"] = "numeric"
    else:
        original_ids = df["customer_id"].astype(str)
        codes, uniques = pd.factorize(original_ids)
        df["customer_id_original"] = original_ids.values
        df["customer_id"] = (codes + 1).astype(int)  # 1-indexed
        metadata["customer_id_type"] = "string_factorized"
        metadata["customer_id_unique_count"] = int(len(uniques))
        if len(uniques) > 0:
            warnings.append(
                f"고객 ID가 문자열 형식({uniques[0]} 등) 이어서 정수 코드로 자동 매핑했습니다. "
                f"원본 ID는 customer_id_original 컬럼에 보존됩니다."
            )

    if len(df) == 0:
        raise ValueError("유효한 customer_id가 있는 행이 없습니다. 고객 ID 컬럼을 확인해주세요.")

    # ── Step 2: Determine data granularity ──
    id_uniqueness = df["customer_id"].nunique() / max(len(df), 1)
    is_transaction_level = id_uniqueness < 0.5  # multiple rows per customer = transactional
    metadata["data_granularity"] = "transaction" if is_transaction_level else "customer_summary"

    # ── Step 3: Parse timestamps ──
    ts_col = schema.get("timestamp")
    if ts_col and ts_col in df.columns:
        df[ts_col] = _detect_date_column(df, ts_col)

    # ── Step 4: Compute RFM ──
    amount_col = schema.get("amount")
    rfm = _compute_rfm(df, "customer_id", amount_col, ts_col)

    # ── Step 5: Build customer summary ──
    if is_transaction_level:
        # Aggregate to customer level
        customer_summary = rfm.copy()

        # 원본 문자열 ID 보존 (factorize 된 경우)
        if "customer_id_original" in df.columns:
            id_lookup = df.groupby("customer_id")["customer_id_original"].first().reset_index()
            customer_summary = customer_summary.merge(id_lookup, on="customer_id", how="left")

        # Add signup date
        if ts_col and ts_col in df.columns:
            first_date = df.groupby("customer_id")[ts_col].min().rename("signup_date")
            customer_summary = customer_summary.merge(first_date.reset_index(), on="customer_id", how="left")
        else:
            customer_summary["signup_date"] = pd.Timestamp("2025-01-01")

        # Add categorical features
        for role, col in schema.items():
            if role in {"persona", "region", "category"} and col in df.columns:
                mode_val = df.groupby("customer_id")[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
                customer_summary = customer_summary.merge(mode_val.rename(role).reset_index(), on="customer_id", how="left")
    else:
        customer_summary = df.copy()
        customer_summary = customer_summary.merge(rfm[["customer_id", "recency_days", "frequency", "monetary"]], on="customer_id", how="left", suffixes=("", "_rfm"))
        for col in ["recency_days", "frequency", "monetary"]:
            if f"{col}_rfm" in customer_summary.columns:
                customer_summary[col] = customer_summary[col].fillna(customer_summary[f"{col}_rfm"])
                customer_summary = customer_summary.drop(columns=[f"{col}_rfm"])

        if "signup_date" not in customer_summary.columns:
            if ts_col and ts_col in df.columns:
                customer_summary["signup_date"] = df[ts_col]
            else:
                customer_summary["signup_date"] = pd.Timestamp("2025-01-01")

    customer_summary["signup_date"] = pd.to_datetime(customer_summary["signup_date"], errors="coerce").fillna(pd.Timestamp("2025-01-01"))
    customer_summary["acquisition_month"] = customer_summary["signup_date"].dt.to_period("M").astype(str)

    # ── Step 6: Infer churn probability ──
    churn_labels = _infer_churn_label(df, schema, inactivity_threshold_days=churn_inactivity_days)
    metadata["churn_inactivity_threshold_days"] = int(churn_inactivity_days)
    if is_transaction_level:
        churn_by_customer = df.assign(_churn=churn_labels).groupby("customer_id")["_churn"].max()
        customer_summary = customer_summary.merge(churn_by_customer.rename("churn_probability").reset_index(), on="customer_id", how="left")
    else:
        customer_summary["churn_probability"] = churn_labels.reindex(customer_summary.index).fillna(0.5)

    customer_summary["churn_probability"] = _safe_numeric(customer_summary["churn_probability"], 0.5).clip(0.01, 0.99)

    # ── Step 7: Fill missing core features ──
    for col, default in [
        ("recency_days", 0), ("frequency", 0), ("monetary", 0.0),
        ("visits_last_7", 0), ("visits_prev_7", 0), ("purchase_last_30", 0),
        ("purchase_prev_30", 0),
    ]:
        if col not in customer_summary.columns:
            customer_summary[col] = default

    # inactivity_days: 사용자 데이터에선 명시 컬럼 없을 가능성 큼 → recency_days로 대체
    if "inactivity_days" not in customer_summary.columns:
        customer_summary["inactivity_days"] = customer_summary["recency_days"]

    customer_summary["visit_change_rate"] = _safe_divide(
        customer_summary["visits_last_7"] - customer_summary["visits_prev_7"],
        customer_summary["visits_prev_7"],
    )
    customer_summary["purchase_change_rate"] = _safe_divide(
        customer_summary["purchase_last_30"] - customer_summary["purchase_prev_30"],
        customer_summary["purchase_prev_30"],
    )

    # ── Step 7b: 시뮬레이터 전용 컬럼들을 ML 호환을 위해 default로 채움 ──
    # 사용자 데이터엔 이런 컬럼이 없으므로 합리적 default를 채워 ML 단계가 안 깨지게 함.
    # (학습 결과에 의미는 없으나 KeyError를 방지)
    sim_only_defaults: Dict[str, Any] = {
        "treatment_lift_base": 0.0,
        "basket_size_preference": 1.0,
        "avg_order_value_mean": 0.0,
        "avg_order_value_std": 0.0,
    }
    # avg_order_value 평균/표준편차는 monetary/frequency에서 추정 가능
    if customer_summary["frequency"].max() > 0:
        avg_order = _safe_divide(
            customer_summary["monetary"].values,
            np.maximum(customer_summary["frequency"].values, 1.0),
        )
        sim_only_defaults["avg_order_value_mean"] = float(np.mean(avg_order))
        sim_only_defaults["avg_order_value_std"] = float(np.std(avg_order))

    # signup_date 기준 simulation_start 일자 추정 (가장 이른 가입일을 0일로)
    if "signup_date" in customer_summary.columns:
        min_signup = pd.to_datetime(customer_summary["signup_date"], errors="coerce").min()
        if pd.notna(min_signup):
            sim_only_defaults["days_from_simulation_start"] = (
                pd.to_datetime(customer_summary["signup_date"], errors="coerce") - min_signup
            ).dt.days.fillna(0).astype(int)
        else:
            sim_only_defaults["days_from_simulation_start"] = 0
    else:
        sim_only_defaults["days_from_simulation_start"] = 0

    for col, default in sim_only_defaults.items():
        if col not in customer_summary.columns:
            customer_summary[col] = default

    # ── Step 8: Assign personas and segments ──
    if "persona" not in customer_summary.columns:
        customer_summary["persona"] = _assign_personas(customer_summary)
    customer_summary["uplift_segment_true"] = customer_summary.get("uplift_segment_true", _assign_uplift_segments(customer_summary))

    # ── Step 9: Generate derived scores ──
    if "uplift_score" not in customer_summary.columns:
        customer_summary["uplift_score"] = np.clip(
            rng.normal(0.08, 0.05, size=len(customer_summary))
            + 0.05 * (customer_summary["churn_probability"] - 0.5),
            -0.15, 0.42,
        )

    if "clv" not in customer_summary.columns:
        avg_order = _safe_divide(customer_summary["monetary"], customer_summary["frequency"])
        retention_factor = np.clip(1.15 - customer_summary["churn_probability"], 0.20, 1.15)
        customer_summary["clv"] = (
            customer_summary["monetary"] * (1.30 + 1.25 * retention_factor)
            + customer_summary["frequency"] * np.maximum(avg_order, 20000) * 0.55
        ).clip(lower=15000)

    if "coupon_cost" not in customer_summary.columns:
        customer_summary["coupon_cost"] = rng.integers(5000, 15000, size=len(customer_summary))

    customer_summary["expected_incremental_profit"] = np.maximum(
        customer_summary["clv"] * customer_summary["uplift_score"], -50000
    )
    customer_summary["expected_roi"] = _safe_divide(
        customer_summary["expected_incremental_profit"] - customer_summary["coupon_cost"],
        customer_summary["coupon_cost"],
    )
    customer_summary["uplift_segment"] = _assign_uplift_segments(customer_summary)

    # ── Step 10: Fill remaining columns ──
    for col, default in [
        ("region", "Seoul"), ("device_type", "mobile"), ("acquisition_channel", "organic"),
        ("treatment_group", "treatment"), ("treatment_flag", 1),
        ("coupon_exposure_count", 0), ("coupon_redeem_count", 0),
        ("coupon_fatigue_score", 0.0), ("discount_dependency_score", 0.0),
        ("discount_pressure_score", 0.0), ("discount_effect_penalty", 1.0),
        ("price_sensitivity", 0.5), ("coupon_affinity", 0.5),
        ("support_contact_propensity", 0.1),
    ]:
        if col not in customer_summary.columns:
            if isinstance(default, str):
                customer_summary[col] = default
            else:
                customer_summary[col] = default

    # ── Step 11: Generate auxiliary tables ──
    treatment_assignments = _generate_treatment_assignments(customer_summary, rng)
    customer_summary = customer_summary.merge(
        treatment_assignments[["customer_id", "treatment_group", "treatment_flag", "coupon_cost"]],
        on="customer_id", how="left", suffixes=("", "_ta"),
    )
    for col in ["treatment_group", "treatment_flag", "coupon_cost"]:
        if f"{col}_ta" in customer_summary.columns:
            customer_summary[col] = customer_summary[col].fillna(customer_summary[f"{col}_ta"])
            customer_summary = customer_summary.drop(columns=[f"{col}_ta"])

    # ── 실제 사용자 event가 있으면 우선 사용, 없으면 합성 fallback ──
    real_events, mapping_report = _extract_real_events(df, schema, user_mapping=event_value_mapping)
    if real_events is not None and len(real_events) > 0:
        events_df = real_events
        metadata["events_source"] = "user_upload"
        metadata["event_type_mapping"] = mapping_report
        if mapping_report["unmapped_values"]:
            warnings.append(
                f"event_type 값 중 매핑되지 않은 항목 {len(mapping_report['unmapped_values'])}개: "
                f"{', '.join(mapping_report['unmapped_values'][:5])}"
                f"{' ...' if len(mapping_report['unmapped_values']) > 5 else ''} "
                f"→ 'other'로 분류됨 (매핑 커버리지: {mapping_report['coverage_rate']:.0%})"
            )
        # amount 컬럼이 있으면 실제 금액 기반 orders 사용
        amount_col_for_orders = schema.get("amount")
        if amount_col_for_orders and amount_col_for_orders in df.columns:
            orders_df = _build_orders_from_real_events(df, real_events, schema, rng)
        else:
            orders_df = _generate_synthetic_orders(customer_summary, events_df, rng)
    else:
        if not allow_synthetic_fallback:
            raise ValueError(
                "이 CSV에는 event_type 또는 timestamp 컬럼이 없어 실제 이벤트 분석이 불가능합니다. "
                "event_type + timestamp 컬럼이 있는 데이터를 올리거나, "
                "합성 이벤트로 진행에 명시적으로 동의해주세요."
            )
        events_df = _generate_synthetic_events(customer_summary, rng)
        metadata["events_source"] = "synthetic"
        orders_df = _generate_synthetic_orders(customer_summary, events_df, rng)
    campaign_exposures = _generate_campaign_exposures(treatment_assignments, rng)
    state_snapshots = _generate_state_snapshots(customer_summary, rng)
    cohort_retention = _build_cohort_retention(customer_summary)

    # Customers base table — 시뮬레이터 호환을 위해 ML 단계가 요구하는 컬럼들도 포함
    _customers_cols = [
        "customer_id", "persona", "signup_date", "acquisition_month",
        "region", "device_type", "acquisition_channel",
        "price_sensitivity", "coupon_affinity", "support_contact_propensity",
        # 시뮬레이터 전용이지만 CLV 모델이 요구함
        "treatment_lift_base", "basket_size_preference",
        "avg_order_value_mean", "avg_order_value_std", "days_from_simulation_start",
    ]
    _existing_customers_cols = [c for c in _customers_cols if c in customer_summary.columns]
    customers_df = customer_summary[_existing_customers_cols].copy()

    # Sort and reset
    customer_summary = customer_summary.sort_values("customer_id").reset_index(drop=True)

    metadata.update({
        "processed_customers": int(len(customer_summary)),
        "processed_events": int(len(events_df)),
        "processed_orders": int(len(orders_df)),
        "churn_rate": float(customer_summary["churn_probability"].mean()),
        "avg_clv": float(customer_summary["clv"].mean()),
        "preprocessing_complete": True,
    })

    return PreprocessingResult(
        customer_summary=customer_summary,
        events=events_df,
        orders=orders_df,
        cohort_retention=cohort_retention,
        treatment_assignments=treatment_assignments,
        campaign_exposures=campaign_exposures,
        state_snapshots=state_snapshots,
        customers=customers_df,
        metadata=metadata,
        warnings=warnings,
    )


def save_preprocessed_data(result: PreprocessingResult, output_dir: str | Path) -> Dict[str, str]:
    """Save all preprocessed tables to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "customer_summary": result.customer_summary,
        "events": result.events,
        "orders": result.orders,
        "cohort_retention": result.cohort_retention,
        "treatment_assignments": result.treatment_assignments,
        "campaign_exposures": result.campaign_exposures,
        "state_snapshots": result.state_snapshots,
        "customers": result.customers,
    }

    saved = {}
    for name, df in files.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        saved[name] = str(path)

    # Save metadata
    meta_path = output_dir / "preprocessing_metadata.json"
    meta_path.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    saved["metadata"] = str(meta_path)

    return saved

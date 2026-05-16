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
        "click", "tap", "select", "browse", "explore", "review_write", "review",
        "stream_start", "stream_complete", "stream_end", "watch", "watched",
        "video_play", "video_complete", "video_pause", "play", "pause", "resume",
        "push_received", "notification_received", "coupon_use", "coupon", "point_use", "points_use",
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
        "refund_request", "refund", "return", "returned", "return_request", "cancel_request",
        "cancel", "cancellation", "uninstall", "uninstall_signal", "unsubscribe",
        "문의", "상담", "고객센터", "신고", "환불", "반품", "취소", "해지",
    },
}


def _normalize_event_type(value: Any) -> str:

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


def _slugify_column(name: Any, max_len: int = 64) -> str:
    """Return a safe, stable feature-name fragment for arbitrary CSV columns."""
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", str(name).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_") or "col"
    return text[:max_len]


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized_lookup = {_slugify_column(c): c for c in df.columns}
    for cand in candidates:
        key = _slugify_column(cand)
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None


def _mode_or_unknown(series: pd.Series) -> Any:
    s = series.dropna()
    if s.empty:
        return "unknown"
    mode = s.astype(str).mode(dropna=True)
    return mode.iloc[0] if not mode.empty else str(s.iloc[0])


def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(80)
    if sample.empty:
        return False
    # Avoid slow dateutil fallback and noisy warnings for arbitrary category
    # strings such as gender/persona/payment_method.
    date_like = sample.str.contains(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        regex=True,
        na=False,
    ).mean()
    if date_like < 0.60:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.70)


def _build_external_customer_features(df: pd.DataFrame, schema: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate all non-core uploaded columns to customer-level ext_* features.

    The implementation builds one feature block per uploaded column and concatenates
    once. This avoids repeated wide DataFrame merges, which become very slow for
    external CSVs with almost one row per customer and many categorical columns.
    """
    if "customer_id" not in df.columns or df.empty:
        return pd.DataFrame(columns=["customer_id"]), {"used_columns": [], "feature_columns": []}

    protected = {"customer_id"}
    protected.update(c for c in schema.values() if c in df.columns)
    keep_even_if_mapped = {
        schema.get("persona"), schema.get("region"), schema.get("category"),
        schema.get("quantity"), schema.get("amount"), schema.get("churn_flag"),
    }
    keep_even_if_mapped = {c for c in keep_even_if_mapped if c}

    customer_ids = pd.Index(sorted(df["customer_id"].dropna().unique()), name="customer_id")
    feature_frames: List[pd.DataFrame] = []
    feature_columns: List[str] = []
    used_columns: List[Dict[str, Any]] = []
    group_ids = df["customer_id"]

    for col in [c for c in df.columns if c != "customer_id"]:
        if col in protected and col not in keep_even_if_mapped and col != "customer_id_original":
            continue
        slug = _slugify_column(col)

        if col == "customer_id_original":
            s = df.groupby(group_ids)[col].first().reindex(customer_ids).rename("customer_id_original")
            feature_frames.append(s.to_frame())
            feature_columns.append("customer_id_original")
            used_columns.append({"column": col, "kind": "id_lookup", "features": ["customer_id_original"]})
            continue

        series = df[col]
        if _looks_datetime(series):
            ts = _detect_date_column(df, col)
            tmp = pd.DataFrame({"customer_id": group_ids, "_ts": ts})
            agg = tmp.groupby("customer_id")["_ts"].agg(["min", "max"]).reindex(customer_ids)
            agg = agg.rename(columns={"min": f"ext_date__{slug}_first", "max": f"ext_date__{slug}_last"})
            max_ts = ts.max() if ts.notna().any() else pd.NaT
            if pd.notna(max_ts):
                agg[f"ext_num__{slug}_days_since_last"] = (max_ts - agg[f"ext_date__{slug}_last"]).dt.total_seconds().div(86400).fillna(999)
            feature_frames.append(agg)
            feats = list(agg.columns)
            feature_columns.extend(feats)
            used_columns.append({"column": col, "kind": "datetime", "features": feats})
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if float(numeric.notna().mean()) >= 0.85:
            tmp = pd.DataFrame({"customer_id": group_ids, "_v": numeric})
            agg = tmp.groupby("customer_id")["_v"].agg(["mean", "sum", "min", "max"]).reindex(customer_ids)
            rename_map = {
                "mean": f"ext_num__{slug}_mean",
                "sum": f"ext_num__{slug}_sum",
                "min": f"ext_num__{slug}_min",
                "max": f"ext_num__{slug}_max",
            }
            agg = agg.rename(columns=rename_map)
            feature_frames.append(agg)
            feats = list(rename_map.values())
            feature_columns.extend(feats)
            used_columns.append({"column": col, "kind": "numeric", "features": feats})
            continue

        tmp = pd.DataFrame({"customer_id": group_ids, "_v": series.astype("object")})
        nunique = int(series.nunique(dropna=True))
        mode_like = tmp.groupby("customer_id")["_v"].first().reindex(customer_ids).rename(f"ext_cat__{slug}_mode")
        diversity = tmp.groupby("customer_id")["_v"].nunique(dropna=True).reindex(customer_ids).rename(f"ext_num__{slug}_nunique")
        block = pd.concat([mode_like, diversity], axis=1)
        feature_frames.append(block)
        feats = [f"ext_cat__{slug}_mode", f"ext_num__{slug}_nunique"]
        feature_columns.extend(feats)
        used_columns.append({"column": col, "kind": "categorical", "unique_values": nunique, "features": feats})

    if feature_frames:
        out = pd.concat(feature_frames, axis=1).reset_index()
    else:
        out = pd.DataFrame({"customer_id": customer_ids})
    return out, {"used_columns": used_columns, "feature_columns": feature_columns}


def _attach_external_features(customer_summary: pd.DataFrame, external_features: pd.DataFrame) -> pd.DataFrame:
    if external_features.empty or "customer_id" not in external_features.columns:
        return customer_summary
    dedup = external_features.drop_duplicates(subset=["customer_id"]).copy()
    overlap = [c for c in dedup.columns if c != "customer_id" and c in customer_summary.columns]
    rename_map = {c: f"ext_uploaded__{c}" for c in overlap if c != "customer_id_original"}
    if rename_map:
        dedup = dedup.rename(columns=rename_map)
    return customer_summary.merge(dedup, on="customer_id", how="left")


def _coalesce_known_external_columns(customer_summary: pd.DataFrame, df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Map common external retail columns to existing dashboard/core features."""
    out = customer_summary.copy()

    def _customer_numeric(col: Optional[str], agg: str = "mean") -> Optional[pd.Series]:
        if not col or col not in df.columns:
            return None
        values = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({"customer_id": df["customer_id"], "_v": values})
        if agg == "sum":
            return tmp.groupby("customer_id")["_v"].sum()
        if agg == "max":
            return tmp.groupby("customer_id")["_v"].max()
        if agg == "first":
            return tmp.groupby("customer_id")["_v"].first()
        return tmp.groupby("customer_id")["_v"].mean()

    def _customer_mode(col: Optional[str]) -> Optional[pd.Series]:
        if not col or col not in df.columns:
            return None
        return df.groupby("customer_id")[col].agg(_mode_or_unknown)

    amount_col = schema.get("amount")
    if amount_col and amount_col in df.columns:
        amount_sum = _customer_numeric(amount_col, "sum")
        if amount_sum is not None:
            out["monetary"] = out["customer_id"].map(amount_sum).fillna(out.get("monetary", 0.0))

    total_order_col = _first_existing_column(df, ["total_order_count", "order_count", "orders_count", "purchase_count"])
    if total_order_col:
        freq = _customer_numeric(total_order_col, "max")
        if freq is not None:
            out["frequency"] = np.maximum(pd.to_numeric(out.get("frequency", 0), errors="coerce").fillna(0), out["customer_id"].map(freq).fillna(0)).astype(int)

    days_reg_col = _first_existing_column(df, ["days_since_registration", "customer_age_days", "days_from_signup"])
    if days_reg_col:
        days = _customer_numeric(days_reg_col, "max")
        if days is not None:
            ts_col = schema.get("timestamp")
            max_ts = _detect_date_column(df, ts_col).max() if ts_col and ts_col in df.columns else pd.NaT
            base_date = pd.Timestamp(max_ts).normalize() if pd.notna(max_ts) else pd.Timestamp("2025-01-01")
            reg_days = out["customer_id"].map(days).fillna(0).clip(lower=0)
            out["signup_date"] = base_date - pd.to_timedelta(reg_days, unit="D")
            out["customer_age_days_uploaded"] = reg_days

    access_col = _first_existing_column(df, ["access_channel", "channel", "device", "platform"])
    if access_col:
        channel = _customer_mode(access_col)
        if channel is not None:
            mapped = out["customer_id"].map(channel).fillna("unknown").astype(str)
            if "acquisition_channel" not in out.columns or out["acquisition_channel"].eq("organic").all():
                out["acquisition_channel"] = mapped
            if "device_type" not in out.columns or out["device_type"].eq("mobile").all():
                out["device_type"] = np.where(mapped.str.contains("mobile|app|ios|android", case=False, regex=True), "mobile", np.where(mapped.str.contains("web|pc|desktop", case=False, regex=True), "web", mapped))

    for src_names, dst in [
        (["gender"], "gender"),
        (["age_group", "age_band", "age_range"], "age_group"),
        (["payment_method"], "payment_method"),
        (["delivery_type", "shipping_type"], "delivery_type"),
    ]:
        src = _first_existing_column(df, src_names)
        mode = _customer_mode(src)
        if mode is not None:
            out[dst] = out["customer_id"].map(mode).fillna("unknown")

    session_col = _first_existing_column(df, ["session_duration_sec", "session_duration", "duration_sec"])
    if session_col:
        session = _customer_numeric(session_col, "mean")
        if session is not None:
            out["avg_session_duration_sec_uploaded"] = out["customer_id"].map(session).fillna(0)

    page_col = _first_existing_column(df, ["page_views", "pageviews", "views"])
    if page_col:
        pages = _customer_numeric(page_col, "mean")
        if pages is not None:
            out["pageviews_per_session_uploaded"] = out["customer_id"].map(pages).fillna(0)

    discount_col = _first_existing_column(df, ["discount_amount", "discount", "coupon_discount"])
    point_col = _first_existing_column(df, ["point_used", "points_used", "mileage_used"])
    if discount_col:
        disc_sum = _customer_numeric(discount_col, "sum")
        amount_sum = _customer_numeric(amount_col, "sum") if amount_col else None
        if disc_sum is not None:
            out["discount_amount_total"] = out["customer_id"].map(disc_sum).fillna(0)
            denom = out["customer_id"].map(amount_sum).fillna(0) if amount_sum is not None else pd.Series(0, index=out.index)
            dep = _safe_divide(out["discount_amount_total"], np.maximum(denom, 1.0), default=0.0)
            out["discount_dependency_score"] = np.clip(dep, 0, 1)
            out["price_sensitivity"] = np.clip(0.35 + 0.8 * out["discount_dependency_score"], 0, 1)
            out["coupon_affinity"] = np.clip(0.30 + 0.7 * out["discount_dependency_score"], 0, 1)
    if point_col:
        points = _customer_numeric(point_col, "sum")
        if points is not None:
            out["point_used_total"] = out["customer_id"].map(points).fillna(0)

    # [PATCH] uploaded coupon exposure/redeem fields
    # 업로드 CSV의 coupon_exposure / coupon_used / campaign_id / discount_amount를
    # 할인·쿠폰 운영 리스크 화면이 사용하는 고객 단위 표준 컬럼으로 집계한다.
    exposure_col = _first_existing_column(
        df,
        [
            "coupon_exposure",
            "coupon_exposed",
            "coupon_sent",
            "coupon_offer",
            "promotion_exposure",
            "campaign_exposure",
        ],
    )
    redeem_col = _first_existing_column(
        df,
        [
            "coupon_used",
            "coupon_use",
            "coupon_redeemed",
            "coupon_redemption",
            "redeemed_coupon",
        ],
    )
    campaign_col = _first_existing_column(
        df,
        [
            "campaign_id",
            "campaign",
            "campaign_name",
            "promotion_id",
            "coupon_id",
        ],
    )

    def _customer_flag_sum(src_col: Optional[str], *, nonblank: bool = False) -> Optional[pd.Series]:
        if not src_col or src_col not in df.columns:
            return None

        tmp = df[["customer_id", src_col]].copy()
        raw = tmp[src_col]

        if nonblank:
            flag = raw.notna() & raw.astype(str).str.strip().ne("")
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            truthy = raw.fillna("").astype(str).str.strip().str.lower().isin(
                ["1", "true", "t", "y", "yes", "used", "redeemed", "open", "opened", "exposed"]
            )
            flag = pd.Series(
                np.where(numeric.notna(), numeric > 0, truthy),
                index=tmp.index,
            )

        tmp["_flag"] = flag.astype(int)
        return tmp.groupby("customer_id")["_flag"].sum()

    exposure_sum = _customer_flag_sum(exposure_col)
    if exposure_sum is None and campaign_col:
        exposure_sum = _customer_flag_sum(campaign_col, nonblank=True)

    redeem_sum = _customer_flag_sum(redeem_col)
    if redeem_sum is None and discount_col:
        redeem_sum = _customer_flag_sum(discount_col)

    if exposure_sum is not None:
        out["coupon_exposure_count"] = (
            out["customer_id"].map(exposure_sum).fillna(0).clip(lower=0).astype(int)
        )

    if redeem_sum is not None:
        out["coupon_redeem_count"] = (
            out["customer_id"].map(redeem_sum).fillna(0).clip(lower=0).astype(int)
        )

    if "coupon_exposure_count" in out.columns and "coupon_redeem_count" in out.columns:
        out["coupon_exposure_count"] = np.maximum(
            pd.to_numeric(out["coupon_exposure_count"], errors="coerce").fillna(0),
            pd.to_numeric(out["coupon_redeem_count"], errors="coerce").fillna(0),
        ).astype(int)

    if "coupon_exposure_count" in out.columns:
        exposure_count = pd.to_numeric(out["coupon_exposure_count"], errors="coerce").fillna(0)
        out["coupon_fatigue_score"] = np.clip(exposure_count / 3.0, 0, 2)

    if "coupon_redeem_count" in out.columns:
        redeem_count = pd.to_numeric(out["coupon_redeem_count"], errors="coerce").fillna(0)
        exposure_count = pd.to_numeric(
            out.get("coupon_exposure_count", redeem_count),
            errors="coerce",
        ).fillna(0)
        out["coupon_redeem_rate_uploaded"] = _safe_divide(
            redeem_count,
            np.maximum(exposure_count, 1.0),
            default=0.0,
        )

    discount_risk = pd.to_numeric(
        out.get("discount_dependency_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0).clip(0, 1)

    fatigue_risk = pd.to_numeric(
        out.get("coupon_fatigue_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0).clip(0, 2) / 2.0

    out["discount_pressure_score"] = np.clip(
        0.65 * discount_risk + 0.35 * fatigue_risk,
        0,
        1,
    )
    out["discount_effect_penalty"] = np.clip(
        1.0 - 0.25 * out["discount_pressure_score"],
        0.50,
        1.0,
    )

    refund_col = _first_existing_column(df, ["refund_reason", "return_reason", "cancel_reason"])
    if refund_col:
        refund_mode = _customer_mode(refund_col)
        if refund_mode is not None:
            reason = out["customer_id"].map(refund_mode).fillna("none").astype(str)
            out["refund_reason"] = reason
            has_issue = ~reason.str.lower().isin(["", "none", "nan", "no", "없음"])
            if "support_contact_propensity" in out.columns:
                base_support = pd.to_numeric(out["support_contact_propensity"], errors="coerce").fillna(0.1)
            else:
                base_support = pd.Series(0.1, index=out.index)
            out["support_contact_propensity"] = np.where(has_issue, np.maximum(base_support, 0.65), base_support)

    return out


def _deduplicate_customer_summary(customer_summary: pd.DataFrame) -> pd.DataFrame:
    """Guarantee one row per customer to avoid many-to-many downstream joins.

    External CSVs often have almost one row per customer with only a handful of
    accidental duplicate IDs. A full custom groupby aggregation over every object
    column is needlessly slow, so only duplicated IDs are reconciled and the rest
    of the table is kept as-is.
    """
    if customer_summary.empty or "customer_id" not in customer_summary.columns or customer_summary["customer_id"].is_unique:
        return customer_summary

    unique_part = customer_summary.loc[~customer_summary["customer_id"].duplicated(keep=False)].copy()
    dup_part = customer_summary.loc[customer_summary["customer_id"].duplicated(keep=False)].copy()
    agg: Dict[str, Any] = {}
    for col in dup_part.columns:
        if col == "customer_id":
            continue
        if pd.api.types.is_numeric_dtype(dup_part[col]):
            if any(key in col for key in ["probability", "score", "rate", "roi", "affinity", "sensitivity", "propensity"]):
                agg[col] = "mean"
            elif any(key in col for key in ["monetary", "amount", "clv", "cost", "profit", "frequency", "count", "total"]):
                agg[col] = "max"
            else:
                agg[col] = "last"
        else:
            agg[col] = "last"
    reconciled = dup_part.groupby("customer_id", as_index=False, sort=False).agg(agg)
    out = pd.concat([unique_part, reconciled], ignore_index=True, sort=False)
    return out.sort_values("customer_id").reset_index(drop=True)


def _estimate_churn_probability(customer_summary: pd.DataFrame, observed_label: Optional[pd.Series] = None, *, seed: int = 42) -> pd.Series:
    """Create a continuous, rank-calibrated churn probability instead of echoing 0/1 labels.

    External files often contain coarse recency/frequency values or an uploaded churn flag.
    A raw weighted average then collapses many customers into the same narrow band.  The
    dashboard needs a probability-like operating score, so we keep the business rank signal
    but spread it through a logit calibration around the observed/base churn rate.
    """
    cs = customer_summary.copy()
    n = len(cs)
    if n == 0:
        return pd.Series(dtype=float)

    rng = np.random.default_rng(seed)

    def _rank01(values: pd.Series, ascending: bool = True) -> pd.Series:
        v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if v.notna().nunique() <= 1:
            # Break ties deterministically so a flat uploaded column does not create
            # thousands of identical churn probabilities in the dashboard histogram.
            return pd.Series(rng.random(len(values)), index=values.index).rank(pct=True, ascending=ascending)
        jitter = pd.Series(rng.normal(0.0, 1e-7, size=len(values)), index=values.index)
        return (v + jitter).rank(method="first", pct=True, ascending=ascending).fillna(0.5).clip(0.001, 0.999)

    def _sigmoid(x: pd.Series | np.ndarray | float) -> pd.Series:
        arr = np.asarray(x, dtype=float)
        arr = np.clip(arr, -18, 18)
        return pd.Series(1.0 / (1.0 + np.exp(-arr)), index=cs.index)

    def _logit(p: float) -> float:
        p = float(np.clip(p, 0.02, 0.98))
        return float(np.log(p / (1.0 - p)))

    recency_risk = _rank01(cs.get("recency_days", pd.Series(0, index=cs.index)), ascending=True)
    inactivity_risk = _rank01(cs.get("inactivity_days", cs.get("recency_days", pd.Series(0, index=cs.index))), ascending=True)
    low_frequency_risk = 1.0 - _rank01(cs.get("frequency", pd.Series(0, index=cs.index)), ascending=True)
    low_monetary_risk = 1.0 - _rank01(cs.get("monetary", pd.Series(0, index=cs.index)), ascending=True)
    support_risk = pd.to_numeric(cs.get("support_contact_propensity", pd.Series(0.1, index=cs.index)), errors="coerce").fillna(0.1).clip(0, 1)
    discount_risk = pd.to_numeric(cs.get("discount_dependency_score", pd.Series(0.0, index=cs.index)), errors="coerce").fillna(0).clip(0, 1)
    session = pd.to_numeric(cs.get("avg_session_duration_sec_uploaded", pd.Series(np.nan, index=cs.index)), errors="coerce")
    pageviews = pd.to_numeric(cs.get("pageviews_per_session_uploaded", pd.Series(np.nan, index=cs.index)), errors="coerce")
    engagement_risk = pd.Series(0.5, index=cs.index)
    if session.notna().nunique() > 1:
        engagement_risk = 0.5 * engagement_risk + 0.5 * (1.0 - _rank01(session, ascending=True))
    if pageviews.notna().nunique() > 1:
        engagement_risk = 0.5 * engagement_risk + 0.5 * (1.0 - _rank01(pageviews, ascending=True))

    behavior_score = (
        0.28 * recency_risk
        + 0.20 * inactivity_risk
        + 0.16 * low_frequency_risk
        + 0.12 * low_monetary_risk
        + 0.12 * support_risk
        + 0.06 * discount_risk
        + 0.06 * engagement_risk
    ).clip(0.01, 0.99)
    centered_behavior = behavior_score - float(behavior_score.mean())
    tie_breaker = pd.Series(rng.normal(0.0, 0.025, size=n), index=cs.index)

    if observed_label is not None:
        label = pd.to_numeric(observed_label.reindex(cs.index), errors="coerce")
        if label.notna().any():
            label = label.fillna(label.mean()).clip(0, 1)
            rate = float(np.clip(label.mean(), 0.03, 0.97))
            rank_signal = _rank01(behavior_score + 0.35 * (label - rate), ascending=True) - 0.5
            logit = _logit(rate) + 2.35 * rank_signal + 1.10 * (label - rate) + 0.75 * centered_behavior + tie_breaker
            return _sigmoid(logit).clip(0.02, 0.98)

    # No explicit churn flag was supplied. Do not center the dashboard score at
    # 45~50% simply because behavior_score is rank-based and therefore averages
    # around 0.5. Use a conservative operating prior while preserving the risk
    # ordering; the real binary training label is built later from a held-out
    # future activity horizon in feature engineering.
    base_rate = float(np.clip(0.06 + 0.34 * float(behavior_score.mean()), 0.08, 0.36))
    rank_signal = _rank01(behavior_score, ascending=True) - 0.5
    logit = _logit(base_rate) + 2.15 * rank_signal + 0.65 * centered_behavior + tie_breaker
    return _sigmoid(logit).clip(0.02, 0.82)


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
        "source_row_id": sub.index.astype(int).values,
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
        "event_id", "source_row_id", "customer_id", "timestamp", "event_type",
        "event_type_original", "session_id", "item_category", "quantity",
    ]]

    return events_df, mapping_report


def _build_orders_from_real_events(
    df: pd.DataFrame,
    real_events: pd.DataFrame,
    schema: Dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:

    amount_col = schema["amount"]
    ts_col = schema.get("timestamp")

    purchase_mask = real_events["event_type"] == "purchase"
    if purchase_mask.sum() == 0:
        return pd.DataFrame(columns=[
            "order_id", "customer_id", "order_time", "item_category",
            "quantity", "gross_amount", "discount_amount", "net_amount", "coupon_used",
        ])

    discount_col = _first_existing_column(df, ["discount_amount", "discount", "coupon_discount"])
    coupon_col = _first_existing_column(df, ["coupon_used", "coupon_use", "coupon_redeemed"])
    src_cols = [c for c in ["customer_id", ts_col, amount_col, discount_col, coupon_col] if c and c in df.columns]
    src = df[src_cols].copy()
    src["source_row_id"] = df.index.astype(int)
    src["customer_id"] = pd.to_numeric(src["customer_id"], errors="coerce")
    src = src.dropna(subset=["customer_id"])
    src["customer_id"] = src["customer_id"].astype(int)
    src["_ts"] = _detect_date_column(src, ts_col) if ts_col else pd.NaT
    src["_amount"] = _safe_numeric(src[amount_col], 0.0)
    src["_discount"] = _safe_numeric(src[discount_col], 0.0) if discount_col else 0.0
    if coupon_col:
        src["_coupon_used"] = (_safe_numeric(src[coupon_col], 0.0) > 0).astype(int)
    elif discount_col:
        src["_coupon_used"] = (_safe_numeric(src[discount_col], 0.0) > 0).astype(int)
    else:
        src["_coupon_used"] = rng.binomial(1, 0.3, size=len(src))

    purchases = real_events[purchase_mask].copy()
    if "source_row_id" in purchases.columns:
        purchases = purchases.merge(
            src[["source_row_id", "customer_id", "_ts", "_amount", "_discount", "_coupon_used"]],
            on=["source_row_id", "customer_id"],
            how="left",
        )
    else:
        purchases = purchases.merge(
            src[["customer_id", "_ts", "_amount", "_discount", "_coupon_used"]],
            left_on=["customer_id", "timestamp"],
            right_on=["customer_id", "_ts"],
            how="left",
        )
    purchases["_amount"] = purchases["_amount"].fillna(0.0)
    purchases["_discount"] = pd.to_numeric(purchases.get("_discount", 0.0), errors="coerce").fillna(0.0)
    purchases["_coupon_used"] = pd.to_numeric(purchases.get("_coupon_used", 0), errors="coerce").fillna(0).astype(int)

    coupon_used = purchases["_coupon_used"].astype(int).values
    discount = np.minimum(purchases["_discount"].values, purchases["_amount"].values)

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


def _generate_state_snapshots(
    customer_summary: pd.DataFrame,
    rng: np.random.Generator,
    inactivity_threshold_days: int = 30,
    events_df: Optional[pd.DataFrame] = None,
    orders_df: Optional[pd.DataFrame] = None,
    snapshot_frequency_days: int = 7,
) -> pd.DataFrame:

    n = len(customer_summary)
    if n == 0:
        return pd.DataFrame(columns=[
            "customer_id", "snapshot_date", "last_visit_date", "last_purchase_date",
            "visits_total", "purchases_total", "monetary_total", "inactivity_days",
            "current_status", "recent_visit_score", "recent_purchase_score",
            "recent_exposure_score", "coupon_fatigue_score", "discount_dependency_score",
        ])

    activity_snapshots = _generate_activity_state_snapshots(
        customer_summary,
        events_df=events_df,
        orders_df=orders_df,
        rng=rng,
        inactivity_threshold_days=inactivity_threshold_days,
        snapshot_frequency_days=snapshot_frequency_days,
    )
    if not activity_snapshots.empty:
        activity_snapshots.attrs["state_snapshot_strategy"] = "activity_history_sparse_7d"
        return activity_snapshots

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

    churn_risk_days = max(int(inactivity_threshold_days), 1)
    dormant_days = max(int(round(churn_risk_days / 2)), 1)

    # status 벡터화
    status = np.where(
        (inactivity_rep >= churn_risk_days) | (churn_rep >= 0.7), "churn_risk",
        np.where((inactivity_rep >= dormant_days) | (churn_rep >= 0.5), "dormant", "active")
    )

    total = n * months
    fallback = pd.DataFrame({
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
    fallback.attrs["state_snapshot_strategy"] = "customer_summary_repeated_30d"
    return fallback


def _generate_activity_state_snapshots(
    customer_summary: pd.DataFrame,
    *,
    events_df: Optional[pd.DataFrame],
    orders_df: Optional[pd.DataFrame],
    rng: np.random.Generator,
    inactivity_threshold_days: int,
    snapshot_frequency_days: int,
) -> pd.DataFrame:
    """Build sparse state snapshots from uploaded activity history when available."""
    pieces: List[pd.DataFrame] = []
    if events_df is not None and not events_df.empty and {"customer_id", "timestamp"}.issubset(events_df.columns):
        cols = ["customer_id", "timestamp"] + (["event_type"] if "event_type" in events_df.columns else [])
        events = events_df[cols].copy()
        events["activity_time"] = pd.to_datetime(events["timestamp"], errors="coerce")
        if "event_type" in events.columns:
            events["activity_kind"] = events["event_type"].astype(str).str.lower()
            if events["activity_kind"].isin(["visit", "purchase"]).any():
                events = events[events["activity_kind"].isin(["visit", "purchase"])]
        else:
            events["activity_kind"] = "activity"
        pieces.append(events[["customer_id", "activity_time", "activity_kind"]])
    if orders_df is not None and not orders_df.empty and {"customer_id", "order_time"}.issubset(orders_df.columns):
        orders = orders_df[["customer_id", "order_time"]].copy()
        orders["activity_time"] = pd.to_datetime(orders["order_time"], errors="coerce")
        orders["activity_kind"] = "purchase"
        pieces.append(orders[["customer_id", "activity_time", "activity_kind"]])

    if not pieces:
        return pd.DataFrame()

    activity = pd.concat(pieces, ignore_index=True).dropna(subset=["customer_id", "activity_time"])
    if activity.empty:
        return pd.DataFrame()
    activity["customer_id"] = pd.to_numeric(activity["customer_id"], errors="coerce")
    activity = activity.dropna(subset=["customer_id"])
    if activity.empty:
        return pd.DataFrame()
    activity["customer_id"] = activity["customer_id"].astype(int)
    activity = activity.sort_values(["customer_id", "activity_time"])

    cs = customer_summary.copy()
    cs["customer_id"] = pd.to_numeric(cs["customer_id"], errors="coerce")
    cs = cs.dropna(subset=["customer_id"])
    if cs.empty:
        return pd.DataFrame()
    cs["customer_id"] = cs["customer_id"].astype(int)
    cs["signup_date"] = pd.to_datetime(cs.get("signup_date", pd.Timestamp("2025-01-01")), errors="coerce")

    first_activity = activity.groupby("customer_id")["activity_time"].min()
    last_activity = activity.groupby("customer_id")["activity_time"].max()
    global_end = activity["activity_time"].max().floor("D")
    if pd.isna(global_end):
        return pd.DataFrame()

    freq_days = max(int(snapshot_frequency_days), 1)
    grid_parts: List[pd.DataFrame] = []
    for row in cs[["customer_id", "signup_date"]].itertuples(index=False):
        cid = int(row.customer_id)
        signup = pd.Timestamp(row.signup_date) if pd.notna(row.signup_date) else first_activity.get(cid, global_end)
        start = min(pd.Timestamp(signup), pd.Timestamp(first_activity.get(cid, signup))).floor("D")
        end = max(pd.Timestamp(last_activity.get(cid, start)), global_end).floor("D")
        if end < start:
            end = start
        dates = pd.date_range(start=start, end=end, freq=f"{freq_days}D")
        if dates.empty or dates[-1] != end:
            dates = dates.append(pd.DatetimeIndex([end]))
        grid_parts.append(pd.DataFrame({"customer_id": cid, "snapshot_date": dates}))
    if not grid_parts:
        return pd.DataFrame()

    grid = pd.concat(grid_parts, ignore_index=True).sort_values(["snapshot_date", "customer_id"])
    activity_for_asof = activity[["customer_id", "activity_time"]].rename(columns={"activity_time": "last_visit_date"})
    grid = pd.merge_asof(
        grid,
        activity_for_asof.sort_values(["last_visit_date", "customer_id"]),
        left_on="snapshot_date",
        right_on="last_visit_date",
        by="customer_id",
        direction="backward",
    )
    purchases = activity.loc[activity["activity_kind"].eq("purchase"), ["customer_id", "activity_time"]].rename(
        columns={"activity_time": "last_purchase_date"}
    )
    if purchases.empty:
        grid["last_purchase_date"] = pd.NaT
    else:
        grid = pd.merge_asof(
            grid.sort_values(["snapshot_date", "customer_id"]),
            purchases.sort_values(["last_purchase_date", "customer_id"]),
            left_on="snapshot_date",
            right_on="last_purchase_date",
            by="customer_id",
            direction="backward",
        )

    signup_lookup = cs.set_index("customer_id")["signup_date"]
    fallback_last = grid["customer_id"].map(signup_lookup)
    grid["last_visit_date"] = pd.to_datetime(grid["last_visit_date"]).fillna(fallback_last)
    grid["inactivity_days"] = (grid["snapshot_date"] - grid["last_visit_date"]).dt.days.clip(lower=0).fillna(0).astype(int)
    grid["last_purchase_date"] = pd.to_datetime(grid["last_purchase_date"]).fillna(grid["last_visit_date"])

    churn_risk_days = max(int(inactivity_threshold_days), 1)
    dormant_days = max(int(round(churn_risk_days / 2)), 1)
    grid["current_status"] = np.where(
        grid["inactivity_days"] >= churn_risk_days,
        "churn_risk",
        np.where(grid["inactivity_days"] >= dormant_days, "dormant", "active"),
    )

    activity["activity_date"] = activity["activity_time"].dt.floor("D")
    visits = activity[activity["activity_kind"].eq("visit")].groupby(["customer_id", "activity_date"]).size().rename("visits")
    purchases_daily = activity[activity["activity_kind"].eq("purchase")].groupby(["customer_id", "activity_date"]).size().rename("purchases")
    daily = pd.concat([visits, purchases_daily], axis=1).fillna(0).reset_index()
    if daily.empty:
        grid["visits_total"] = 0
        grid["purchases_total"] = 0
    else:
        daily = daily.sort_values(["customer_id", "activity_date"])
        daily["visits_total"] = daily.groupby("customer_id")["visits"].cumsum().astype(int)
        daily["purchases_total"] = daily.groupby("customer_id")["purchases"].cumsum().astype(int)
        grid = pd.merge_asof(
            grid.sort_values(["snapshot_date", "customer_id"]),
            daily[["customer_id", "activity_date", "visits_total", "purchases_total"]].sort_values(["activity_date", "customer_id"]),
            left_on="snapshot_date",
            right_on="activity_date",
            by="customer_id",
            direction="backward",
        )
        grid[["visits_total", "purchases_total"]] = grid[["visits_total", "purchases_total"]].fillna(0).astype(int)
        grid = grid.drop(columns=["activity_date"])

    grid = grid.sort_values(["customer_id", "snapshot_date"])
    monetary_lookup = _safe_numeric(cs.get("monetary", pd.Series(0.0, index=cs.index)), 0.0)
    freq_lookup = _safe_numeric(cs.get("frequency", pd.Series(0, index=cs.index)), 0).replace(0, np.nan)
    per_purchase_amount = pd.Series(_safe_divide(monetary_lookup, freq_lookup, default=0.0), index=cs.index)
    amount_by_customer = per_purchase_amount.groupby(cs["customer_id"]).first()
    grid["monetary_total"] = grid["purchases_total"] * grid["customer_id"].map(amount_by_customer).fillna(0.0)
    grid["recent_visit_score"] = np.minimum(grid.groupby("customer_id")["visits_total"].diff().fillna(grid["visits_total"]), 2.0)
    grid["recent_purchase_score"] = np.minimum(
        grid.groupby("customer_id")["purchases_total"].diff().fillna(grid["purchases_total"]), 2.0
    )
    grid["recent_exposure_score"] = 0.0
    grid["coupon_fatigue_score"] = rng.uniform(0, 0.5, size=len(grid))
    grid["discount_dependency_score"] = _safe_numeric(
        grid["customer_id"].map(cs.set_index("customer_id").get("discount_dependency_score", pd.Series(dtype=float))),
        0.0,
    )

    return grid[
        [
            "customer_id", "snapshot_date", "last_visit_date", "last_purchase_date",
            "visits_total", "purchases_total", "monetary_total", "inactivity_days",
            "current_status", "recent_visit_score", "recent_purchase_score",
            "recent_exposure_score", "coupon_fatigue_score", "discount_dependency_score",
        ]
    ]


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


def _build_cohort_retention(
    customer_summary: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    orders_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build cohort retention table, using uploaded event/order activity when available."""
    columns = [
        "cohort_month", "period", "cohort_size", "retained_customers", "retention_rate",
        "observed", "activity_definition", "retention_mode", "min_events_per_period",
    ]
    if "acquisition_month" not in customer_summary.columns or customer_summary.empty:
        return pd.DataFrame(columns=columns)

    pieces = []
    if events_df is not None and not events_df.empty and {"customer_id", "timestamp", "event_type"}.issubset(events_df.columns):
        e = events_df[["customer_id", "timestamp", "event_type"]].copy()
        e["activity_time"] = pd.to_datetime(e["timestamp"], errors="coerce")
        e["activity_type"] = e["event_type"].astype(str)
        pieces.append(e[["customer_id", "activity_time", "activity_type"]])
    if orders_df is not None and not orders_df.empty and {"customer_id", "order_time"}.issubset(orders_df.columns):
        o = orders_df[["customer_id", "order_time"]].copy()
        o["activity_time"] = pd.to_datetime(o["order_time"], errors="coerce")
        o["activity_type"] = "purchase"
        pieces.append(o[["customer_id", "activity_time", "activity_type"]])
    activity = pd.concat(pieces, ignore_index=True).dropna(subset=["activity_time"]) if pieces else pd.DataFrame(columns=["customer_id", "activity_time", "activity_type"])

    cs = customer_summary[["customer_id", "acquisition_month", "signup_date"]].copy()
    cs["signup_date"] = pd.to_datetime(cs["signup_date"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    use_actual = not activity.empty and activity["customer_id"].nunique() >= max(10, int(0.05 * len(cs)))
    rng = np.random.default_rng(42)

    for cohort in sorted(cs["acquisition_month"].dropna().astype(str).unique()):
        cohort_customers = cs.loc[cs["acquisition_month"].astype(str) == cohort, ["customer_id", "signup_date"]]
        cohort_ids = set(cohort_customers["customer_id"].tolist())
        cohort_size = int(len(cohort_customers))
        if cohort_size == 0:
            continue
        cohort_start = pd.to_datetime(cohort_customers["signup_date"], errors="coerce").min()
        for period in range(7):
            if use_actual and pd.notna(cohort_start):
                period_start = cohort_start + pd.DateOffset(months=period)
                period_end = cohort_start + pd.DateOffset(months=period + 1)
                period_activity = activity[activity["activity_time"].between(period_start, period_end, inclusive="left")]
                active_ids_all = set(period_activity["customer_id"].tolist()) & cohort_ids
                active_ids_purchase = set(period_activity.loc[period_activity["activity_type"].eq("purchase"), "customer_id"].tolist()) & cohort_ids
                active_ids_core = set(period_activity.loc[period_activity["activity_type"].isin(["visit", "page_view", "search", "add_to_cart", "purchase"]), "customer_id"].tolist()) & cohort_ids
                counts = {
                    "all_activity": len(active_ids_all) if period > 0 else cohort_size,
                    "purchase_only": len(active_ids_purchase) if period > 0 else cohort_size,
                    "core_engagement": len(active_ids_core) if period > 0 else cohort_size,
                }
            else:
                base_retention = 1.0 if period == 0 else max(0.85 - 0.08 * period + rng.normal(0, 0.02), 0.15)
                counts = {k: int(round(cohort_size * base_retention)) for k in ["all_activity", "purchase_only", "core_engagement"]}

            for activity_def, retained in counts.items():
                retained = int(min(max(retained, 0), cohort_size))
                retention = retained / max(cohort_size, 1)
                for mode in ["rolling", "point"]:
                    rows.append({
                        "cohort_month": str(cohort),
                        "period": period,
                        "cohort_size": cohort_size,
                        "retained_customers": retained,
                        "retention_rate": round(float(retention), 4),
                        "observed": bool(use_actual),
                        "activity_definition": activity_def,
                        "retention_mode": mode,
                        "min_events_per_period": 1,
                    })
    return pd.DataFrame(rows, columns=columns)


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
    """Transform uploaded data into the full internal schema."""
    rng = np.random.default_rng(seed)
    # 사용자 컬럼 매핑이 들어오면 그것을 우선, 없으면 자동 감지된 schema 사용
    schema = dict(column_mapping_override) if column_mapping_override else dict(validation.detected_schema)
    warnings: List[str] = []
    metadata: Dict[str, Any] = {
        "source": "user_upload",
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "original_column_names": list(df.columns),
        "detected_schema": schema,
    }

    # ── Step 1: Extract customer ID ──
    id_col = schema.get("customer_id", df.columns[0])

    # 외부 CSV의 모든 열을 보존한다. 예전 코드는 schema에 매핑된 9개 역할 열만
    # 남겨 gender/age_group/session/page_views/discount 등 대부분의 자사 열을 버렸다.
    # 이후 단계에서 비표준 열은 ext_* 피처로 고객 단위 집계된다.
    df = df.copy()

    if id_col != "customer_id":
        df = df.rename(columns={id_col: "customer_id"})
        schema = {role: ("customer_id" if col == id_col else col) for role, col in schema.items()}

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
    metadata["customer_id_unique_ratio"] = float(id_uniqueness)

    # 모든 비표준 업로드 열을 고객 단위 피처로 선집계한다.
    external_customer_features, external_feature_meta = _build_external_customer_features(df, schema)
    metadata["external_feature_usage"] = external_feature_meta

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

    customer_summary = _attach_external_features(customer_summary, external_customer_features)
    customer_summary = _coalesce_known_external_columns(customer_summary, df, schema)
    customer_summary = _deduplicate_customer_summary(customer_summary)

    customer_summary["signup_date"] = pd.to_datetime(customer_summary["signup_date"], errors="coerce").fillna(pd.Timestamp("2025-01-01"))
    customer_summary["acquisition_month"] = customer_summary["signup_date"].dt.to_period("M").astype(str)

    # ── Step 6: Infer churn label and continuous churn probability ──
    churn_labels = _infer_churn_label(df, schema, inactivity_threshold_days=churn_inactivity_days)
    metadata["churn_inactivity_threshold_days"] = int(churn_inactivity_days)
    has_explicit_churn_label = "churn_flag" in schema and schema.get("churn_flag") in df.columns
    metadata["churn_label_source"] = "uploaded_churn_flag" if has_explicit_churn_label else "inactivity_rule"

    churn_by_customer = df.assign(_churn=churn_labels).groupby("customer_id")["_churn"].max()
    observed = customer_summary["customer_id"].map(churn_by_customer).fillna(0.5)
    customer_summary["churn_label_observed"] = _safe_numeric(observed, 0.5).clip(0.0, 1.0)
    customer_summary["churn_probability"] = _estimate_churn_probability(
        customer_summary,
        observed_label=customer_summary["churn_label_observed"] if has_explicit_churn_label else None,
        seed=seed,
    )
    metadata["observed_churn_label_rate"] = float(customer_summary["churn_label_observed"].mean())
    metadata["churn_probability_mean"] = float(customer_summary["churn_probability"].mean())
    metadata["churn_probability_strategy"] = (
        "rank_calibrated_continuous_proxy_blended_with_uploaded_label" if has_explicit_churn_label
        else "conservative_rank_calibrated_behavior_prior_from_recency_frequency_value_engagement"
    )

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

    # 보조 테이블 생성 전에 한 번 더 고객 단위 유일성을 보장한다.
    customer_summary = _deduplicate_customer_summary(customer_summary)

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
    state_snapshots = _generate_state_snapshots(
        customer_summary,
        rng,
        inactivity_threshold_days=churn_inactivity_days,
        events_df=events_df if metadata.get("events_source") == "user_upload" else None,
        orders_df=orders_df if metadata.get("events_source") == "user_upload" else None,
    )
    metadata["state_snapshot_strategy"] = (
        state_snapshots.attrs.get("state_snapshot_strategy", "customer_summary_repeated_30d")
        if not state_snapshots.empty else "empty"
    )
    cohort_retention = _build_cohort_retention(customer_summary, events_df=events_df, orders_df=orders_df)

    _customers_cols = [
        "customer_id", "customer_id_original", "persona", "signup_date", "acquisition_month",
        "region", "device_type", "acquisition_channel", "gender", "age_group",
        "payment_method", "delivery_type", "refund_reason", "churn_label_observed",
        "price_sensitivity", "coupon_affinity", "support_contact_propensity",
        # 시뮬레이터 전용이지만 CLV 모델이 요구함
        "treatment_lift_base", "basket_size_preference",
        "avg_order_value_mean", "avg_order_value_std", "days_from_simulation_start",
    ]
    external_cols = [
        c for c in customer_summary.columns
        if c.startswith(("ext_num__", "ext_cat__", "ext_date__"))
        or c in {"avg_session_duration_sec_uploaded", "pageviews_per_session_uploaded", "discount_amount_total", "point_used_total", "customer_age_days_uploaded"}
    ]
    _existing_customers_cols = []
    for c in _customers_cols + external_cols:
        if c in customer_summary.columns and c not in _existing_customers_cols:
            _existing_customers_cols.append(c)
    customers_df = customer_summary[_existing_customers_cols].copy()

    # Sort and reset
    customer_summary = customer_summary.sort_values("customer_id").reset_index(drop=True)

    metadata.update({
        "processed_customers": int(len(customer_summary)),
        "processed_events": int(len(events_df)),
        "processed_orders": int(len(orders_df)),
        "churn_rate": float(customer_summary["churn_probability"].mean()),
        "observed_churn_label_rate": float(customer_summary["churn_label_observed"].mean()),
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

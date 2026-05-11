import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from dashboard.services.api_client import (
    advance_realtime_stream,
    fetch_personalized_recommendations,
    fetch_realtime_scores,
    fetch_saved_results_artifacts,
    fetch_survival_summary,
    fetch_training_artifacts,
    fetch_user_live_actions,
    fetch_user_live_health,
    fetch_user_live_recommendations,
    fetch_user_live_scores,
    fetch_user_live_seed_status,
    seed_user_live_from_artifacts,
)
from dashboard.services.churn_service import get_churn_status
from dashboard.services.cohort_service import (
    get_activity_definition_label,
    get_available_activity_definitions,
    get_available_retention_modes,
    get_cohort_curve,
    get_cohort_display_table,
    get_cohort_pivot,
    get_cohort_summary,
    get_retention_mode_label,
)
from dashboard.services.data_loader import load_dashboard_bundle
from dashboard.services.artifact_loader import load_dashboard_artifacts
from dashboard.services.insight_service import (
    build_coupon_risk_overview,
    build_customer_explanations,
    build_data_diagnostics,
    build_experiment_overview,
    build_global_feature_table,
    build_operational_overview,
    build_realtime_monitor_overview,
    load_dashboard_insight_bundle,
)
from dashboard.services.llm_service import (
    DEFAULT_MODEL_NAME,
    answer_dashboard_question,
    build_payload_json,
    dataframe_snapshot,
    generate_dashboard_summary,
    get_llm_status,
    numeric_summary,
    series_distribution,
)
from dashboard.services.optimize_service import get_budget_result
from dashboard.services.uplift_service import (
    get_retention_targets,
    get_top_high_value_customers,
)
from dashboard.utils.formatters import money, pct


DASHBOARD_VIEW_ITEMS: tuple[tuple[str, str], ...] = (
    # 고객 현황
    ("1", "이탈현황"),
    ("2", "코호트 리텐션 곡선"),

    # 타겟팅·예산
    ("3", "Uplift·CLV 세그먼트 분석"),
    ("4", "예산 최적화 및 리텐션 타겟"),
    ("5", "개인화 추천"),

    # 운영·리스크
    ("6", "실시간 운영 모니터"),
    ("7", "할인·쿠폰 운영 리스크"),

    # 모델 검증·진단
    ("8", "학습 결과 아티팩트"),
    ("9", "이탈 시점 예측 (Survival Analysis)"),
    ("10", "증분 성과 / A-B 실험"),
    ("11", "설명가능성 / 고객별 개입 이유"),
)
DASHBOARD_VIEW_OPTIONS: tuple[str, ...] = tuple(f"{n}. {t}" for n, t in DASHBOARD_VIEW_ITEMS)
VIEW_OPTION_BY_NUM: dict[str, str] = {num: f"{num}. {title}" for num, title in DASHBOARD_VIEW_ITEMS}

DASHBOARD_VIEW_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("고객 현황", ("1", "2")),
    ("타겟팅·예산", ("3", "4", "5")),
    ("운영·리스크", ("6", "7")),
    ("모델 검증·진단", ("8", "9", "10", "11")),
)

GROUP_TO_VIEW_OPTIONS: dict[str, tuple[str, ...]] = {
    group: tuple(VIEW_OPTION_BY_NUM[num] for num in nums if num in VIEW_OPTION_BY_NUM)
    for group, nums in DASHBOARD_VIEW_GROUPS
}

VIEW_TO_GROUP: dict[str, str] = {
    option: group
    for group, options in GROUP_TO_VIEW_OPTIONS.items()
    for option in options
}

LEGACY_VIEW_REDIRECTS: dict[str, str] = {
    # 병합/삭제 전 메뉴명
    "6. 의사결정 엔진 비교": "4. 예산 최적화 및 리텐션 타겟",
    "3. Uplift + CLV 상위 고객": "3. Uplift·CLV 세그먼트 분석",
    "4. 예산 배분 결과": "4. 예산 최적화 및 리텐션 타겟",
    "5. 예상 최적화 ROI": "4. 예산 최적화 및 리텐션 타겟",
    "6. 리텐션 대상 고객 목록": "4. 예산 최적화 및 리텐션 타겟",
    "7. 학습 결과 아티팩트": "8. 학습 결과 아티팩트",
    "8. Uplift/최적화 결과": "3. Uplift·CLV 세그먼트 분석",
    "8. Uplift/최적화 결과 (실시간)": "3. Uplift·CLV 세그먼트 분석",
    "9. 개인화 추천": "5. 개인화 추천",
    "10. 실시간 운영 모니터": "6. 실시간 운영 모니터",
    "10. 실시간 위험 스코어링 / 운영 모니터": "6. 실시간 운영 모니터",
    "11. 이탈 시점 예측 (Survival Analysis)": "9. 이탈 시점 예측 (Survival Analysis)",
    "12. 의사결정 엔진 비교": "4. 예산 최적화 및 리텐션 타겟",
    "13. 운영 한눈에 보기": "6. 실시간 운영 모니터",
    "14. 증분 성과 / A-B 실험": "10. 증분 성과 / A-B 실험",
    "15. 설명가능성 / 고객별 개입 이유": "11. 설명가능성 / 고객별 개입 이유",
    "17. 할인·쿠폰 운영 리스크": "7. 할인·쿠폰 운영 리스크",

    # 의사결정 엔진 비교 삭제 직후 13개 구조에서 새 번호로 이동
    "7. 실시간 운영 모니터": "6. 실시간 운영 모니터",
    "8. 할인·쿠폰 운영 리스크": "7. 할인·쿠폰 운영 리스크",
    "9. 학습 결과 아티팩트": "8. 학습 결과 아티팩트",
    "10. 이탈 시점 예측 (Survival Analysis)": "9. 이탈 시점 예측 (Survival Analysis)",
    "11. 증분 성과 / A-B 실험": "10. 증분 성과 / A-B 실험",
    "12. 설명가능성 / 고객별 개입 이유": "11. 설명가능성 / 고객별 개입 이유",

    # 더 오래된 user-mode 메뉴명
    "6. 개인화 추천": "5. 개인화 추천",
    "8. 이탈 시점 예측 (Survival Analysis)": "9. 이탈 시점 예측 (Survival Analysis)",
    "9. 의사결정 엔진 비교": "4. 예산 최적화 및 리텐션 타겟",
    "10. 증분 성과 / A-B 실험": "10. 증분 성과 / A-B 실험",
    "11. 설명가능성 / 고객별 개입 이유": "11. 설명가능성 / 고객별 개입 이유",
    "13. 할인·쿠폰 운영 리스크": "7. 할인·쿠폰 운영 리스크",
}
REALTIME_REFRESH_VIEWS: set[str] = {
    "6. 실시간 운영 모니터",
}

INSIGHT_HEAVY_VIEWS: set[str] = {
    "6. 실시간 운영 모니터",
    "7. 할인·쿠폰 운영 리스크",
    "10. 증분 성과 / A-B 실험",
    "11. 설명가능성 / 고객별 개입 이유",
}

def parse_unlimited_nonnegative_int(raw_value: str, default: int = 0) -> int:
    cleaned = str(raw_value).replace(",", "").strip()

    if cleaned == "":
        return default

    if not cleaned.isdigit():
        raise ValueError("0 이상의 정수만 입력할 수 있습니다.")

    return int(cleaned)

# ============================================================
# [PATCH] 자사 데이터(user) 모드에서 Treatment/Control 의존
# 화면을 "해당 데이터 없음" 으로 처리하는 헬퍼.
# 외부 데이터(UCI / Retailrocket 등)에는 처치/대조 정보가 없어
# Uplift, A/B 테스트, 예산 최적화 등을 산출할 수 없기 때문.
# ============================================================
def _user_mode_unavailable(feature_name: str, reason: str = "") -> bool:
    """사용자 CSV가 아직 처리되지 않았을 때만 user 전용 화면을 막는다.
    업로드 산출물이 있으면 기존 화면을 그대로 사용하되, treatment/control이 없는
    CSV는 전처리 단계의 자동 배정·휴리스틱 추정값으로 표시된다는 점만 안내한다."""
    import streamlit as _st
    from pathlib import Path as _P

    _mode = _st.session_state.get("data_mode", "simulator")
    if _mode != "user":
        return False

    _has_user_data = (_P("data/raw_user") / "customer_summary.csv").exists()
    _has_user_results = _P("results_user").exists() and any(_P("results_user").iterdir())
    if _has_user_data or _has_user_results:
        _st.info(
            "현재 화면은 업로드 CSV에서 생성된 user 산출물을 기준으로 표시합니다. "
            "원본 CSV에 Treatment/Control이 없으면 전처리 단계의 자동 배정 및 "
            "휴리스틱 Uplift/ROI 추정값이 사용됩니다."
        )
        return False

    _default_reason = (
        "아직 업로드 CSV로 생성된 user 산출물이 없습니다. 먼저 사이드바에서 CSV를 업로드하고 "
        "매핑 확정 후 학습을 실행하세요."
    )
    _reason = reason or _default_reason
    _st.markdown(
        f"""
        <div style="
            background-color: #F3F4F6;
            border: 1px dashed #9CA3AF;
            border-radius: 12px;
            padding: 32px 24px;
            margin: 16px 0;
            text-align: center;
        ">
            <div style="font-size: 40px; opacity: 0.5;">🔒</div>
            <div style="font-size: 20px; font-weight: 700; color: #374151; margin-top: 8px;">
                해당 데이터 없음
            </div>
            <div style="font-size: 14px; color: #6B7280; margin-top: 8px;">
                {feature_name}
            </div>
            <div style="font-size: 13px; color: #9CA3AF; margin-top: 12px; line-height: 1.5;">
                {_reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return True

# ============================================================
# [PATCH] user mode PostgreSQL live serving helpers.
# simulator 모드는 기존 CSV/results/Redis replay 구조를 그대로 사용하고,
# user 모드에서만 /api/v1/user-live/* API를 우선 조회한다.
# ============================================================
def _is_user_live_mode() -> bool:
    return st.session_state.get("data_mode", "simulator") == "user"



def _is_missing_live_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _parse_live_payload(value: Any) -> dict[str, Any]:
    """score_payload/source_payload처럼 JSON 문자열 또는 dict로 온 payload를 dict로 변환한다."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _nested_payload_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """seed payload와 live_scoring payload의 중첩 구조를 모두 검색 후보로 만든다."""
    candidates: list[dict[str, Any]] = []
    if payload:
        candidates.append(payload)
        for key in [
            "feature_snapshot",
            "customer_score",
            "score_payload",
            "feature_payload",
            "source_payload",
            "raw_payload",
            "previous_scores",
        ]:
            nested = payload.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
    return candidates


def _lookup_payload_value(payload: dict[str, Any], aliases: list[str]) -> Any:
    """payload 안에서 alias와 일치하는 값을 대소문자 무시하고 찾는다."""
    if not payload:
        return None

    for candidate in _nested_payload_candidates(payload):
        lower_to_key = {str(key).lower(): key for key in candidate.keys()}
        for alias in aliases:
            key = lower_to_key.get(alias.lower())
            if key is not None:
                value = candidate.get(key)
                if not _is_missing_live_value(value):
                    return value

    for candidate in _nested_payload_candidates(payload):
        for value in candidate.values():
            if isinstance(value, dict):
                nested = _lookup_payload_value(value, aliases)
                if not _is_missing_live_value(nested):
                    return nested

    return None


def _lookup_live_row_value(row: pd.Series, aliases: list[str]) -> Any:
    """DataFrame row의 top-level 컬럼과 JSON payload에서 값을 찾는다."""
    lower_to_key = {str(key).lower(): key for key in row.index}
    for alias in aliases:
        key = lower_to_key.get(alias.lower())
        if key is not None:
            value = row.get(key)
            if not _is_missing_live_value(value):
                return value

    for payload_col in ["score_payload", "feature_payload", "source_payload"]:
        if payload_col in row.index:
            payload = _parse_live_payload(row.get(payload_col))
            value = _lookup_payload_value(payload, aliases)
            if not _is_missing_live_value(value):
                return value

    return None


def _derive_uplift_segment_from_score(value: Any) -> str:
    """payload에 세그먼트명이 없을 때 uplift_score로 안정적인 대체 세그먼트를 만든다."""
    try:
        score = float(value)
    except Exception:
        return "unknown_segment"

    if math.isnan(score) or math.isinf(score):
        return "unknown_segment"
    if score >= 0.08:
        return "very_high_uplift"
    if score >= 0.05:
        return "high_uplift"
    if score >= 0.02:
        return "medium_uplift"
    if score >= 0.0:
        return "low_uplift"
    return "negative_uplift"


def _is_placeholder_segment(value: Any) -> bool:
    if _is_missing_live_value(value):
        return True
    normalized = str(value).strip().lower()
    return normalized in {
        "",
        "live",
        "live_user",
        "unknown",
        "unknown_segment",
        "nan",
        "none",
        "null",
    }


def _restore_live_dimension_columns(fixed: pd.DataFrame) -> pd.DataFrame:
    """score_payload/feature_payload/source_payload에서 persona·uplift segment 계열 컬럼을 복원한다."""
    if fixed.empty:
        return fixed

    persona_aliases = [
        "persona",
        "customer_persona",
        "customer_segment",
        "lifecycle_segment",
        "marketing_segment",
        "segment_name",
        "membership_tier",
        "member_tier",
        "membership_grade",
        "tier",
        "grade",
    ]
    uplift_aliases = [
        "uplift_segment",
        "uplift_group",
        "uplift_bucket",
        "treatment_segment",
        "campaign_segment",
        "response_segment",
        "persuadable_segment",
    ]
    region_aliases = ["region", "area", "city", "province"]
    age_aliases = ["age_group", "age_band", "age_segment"]
    gender_aliases = ["gender", "sex"]

    restored_persona: list[str] = []
    restored_uplift: list[str] = []
    persona_source: list[str] = []
    uplift_source: list[str] = []

    for _, row in fixed.iterrows():
        persona_value = _lookup_live_row_value(row, persona_aliases)
        p_source = "payload"

        if _is_placeholder_segment(persona_value):
            tier = _lookup_live_row_value(row, ["membership_tier", "member_tier", "membership_grade", "tier", "grade"])
            age_group = _lookup_live_row_value(row, age_aliases)
            region = _lookup_live_row_value(row, region_aliases)
            gender = _lookup_live_row_value(row, gender_aliases)

            parts = [
                str(value).strip()
                for value in [tier, age_group, region, gender]
                if not _is_placeholder_segment(value)
            ]
            if parts:
                persona_value = " / ".join(parts[:3])
                p_source = "derived"
            else:
                persona_value = "unknown_persona"
                p_source = "fallback"

        uplift_value = _lookup_live_row_value(row, uplift_aliases)
        u_source = "payload"

        # action_queue row는 top-level에 uplift_segment가 없을 수 있으므로 source_payload.customer_score를 먼저 본다.
        # 그래도 없으면 uplift_score 기준으로 bucket을 만들어 'live' 단일 막대가 생기지 않게 한다.
        if _is_placeholder_segment(uplift_value):
            uplift_score = _lookup_live_row_value(row, ["uplift_score", "uplift", "predicted_uplift", "treatment_effect"])
            uplift_value = _derive_uplift_segment_from_score(uplift_score)
            u_source = "derived_from_uplift_score" if uplift_value != "unknown_segment" else "fallback"

        restored_persona.append(str(persona_value))
        restored_uplift.append(str(uplift_value))
        persona_source.append(p_source)
        uplift_source.append(u_source)

    fixed["persona"] = restored_persona
    fixed["persona_source"] = persona_source
    fixed["uplift_segment"] = restored_uplift
    fixed["uplift_segment_source"] = uplift_source
    return fixed


@st.cache_data(show_spinner=False, ttl=15)
def _fetch_user_live_scores_cached(cache_key: str) -> tuple[dict, pd.DataFrame]:
    """전체 customer_scores는 크므로 동일 이벤트 상태에서는 15초 동안 재사용한다."""
    return fetch_user_live_scores(limit=None)

def _rename_live_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """customer_scores API 결과를 기존 대시보드 렌더링 컬럼과 맞춘다.

    persona는 더 이상 live_user로 덮어쓰지 않는다. score_payload/feature_payload에
    보존된 원본 persona·segment·membership_tier 계열 값을 우선 복원한다.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    fixed = df.copy()
    fixed = _restore_live_dimension_columns(fixed)

    if "churn_probability" not in fixed.columns and "churn_score" in fixed.columns:
        fixed["churn_probability"] = pd.to_numeric(fixed["churn_score"], errors="coerce").fillna(0.0)

    defaults = {
        "persona": "unknown_persona",
        "uplift_segment": "live",
        "risk_segment": "unknown",
        "expected_roi": 0.0,
        "expected_incremental_profit": 0.0,
        "clv": 0.0,
        "uplift_score": 0.0,
        "coupon_cost": 0.0,
    }
    for col, default in defaults.items():
        if col not in fixed.columns:
            fixed[col] = default
        elif col in {"persona", "uplift_segment", "risk_segment"}:
            fixed[col] = fixed[col].fillna(default).astype(str).replace({"": default, "nan": default, "None": default})

    for numeric_col in [
        "churn_probability",
        "churn_score",
        "expected_roi",
        "expected_incremental_profit",
        "clv",
        "uplift_score",
        "coupon_cost",
    ]:
        if numeric_col in fixed.columns:
            fixed[numeric_col] = pd.to_numeric(fixed[numeric_col], errors="coerce").fillna(0.0)

    if "priority_score" not in fixed.columns:
        fixed["priority_score"] = fixed["expected_incremental_profit"]
    else:
        fixed["priority_score"] = pd.to_numeric(fixed["priority_score"], errors="coerce").fillna(0.0)

    if "selection_score" not in fixed.columns:
        fixed["selection_score"] = fixed["priority_score"]

    heavy_cols = [col for col in ["score_payload", "feature_payload", "source_payload"] if col in fixed.columns]
    if heavy_cols:
        fixed = fixed.drop(columns=heavy_cols)

    return fixed


def _normalize_live_actions_df(df: pd.DataFrame) -> pd.DataFrame:
    """action_queue API 결과를 기존 타겟/추천 화면 컬럼과 맞춘다."""
    if df is None or df.empty:
        return pd.DataFrame()

    fixed = df.copy()
    # source_payload.customer_score 안의 persona/uplift_segment를 먼저 복원한다.
    fixed = _restore_live_dimension_columns(fixed)

    if "expected_incremental_profit" not in fixed.columns and "expected_profit" in fixed.columns:
        fixed["expected_incremental_profit"] = fixed["expected_profit"]
    if "coupon_cost" not in fixed.columns:
        fixed["coupon_cost"] = 0.0
    if "churn_probability" not in fixed.columns:
        fixed["churn_probability"] = 0.0
    if "priority_score" not in fixed.columns:
        if "expected_incremental_profit" in fixed.columns:
            fixed["priority_score"] = pd.to_numeric(fixed["expected_incremental_profit"], errors="coerce").fillna(0.0)
        else:
            fixed["priority_score"] = 0.0
    if "selection_score" not in fixed.columns:
        fixed["selection_score"] = fixed["priority_score"]

    for numeric_col in [
        "expected_roi",
        "expected_incremental_profit",
        "expected_profit",
        "coupon_cost",
        "priority_score",
        "selection_score",
        "churn_probability",
    ]:
        if numeric_col in fixed.columns:
            fixed[numeric_col] = pd.to_numeric(fixed[numeric_col], errors="coerce").fillna(0.0)

    heavy_cols = [col for col in ["score_payload", "feature_payload", "source_payload"] if col in fixed.columns]
    normalized = _ensure_retention_target_schema(fixed)
    if heavy_cols:
        normalized = normalized.drop(columns=[col for col in heavy_cols if col in normalized.columns])
    return normalized


def _live_scores_to_realtime_df(scores_df: pd.DataFrame, actions_df: pd.DataFrame) -> pd.DataFrame:
    """user live scores/actions를 기존 실시간 운영 모니터가 기대하는 컬럼으로 변환한다."""
    scores = _rename_live_score_columns(scores_df)
    if scores.empty:
        return pd.DataFrame()

    live = scores.copy()
    live["realtime_churn_score"] = live.get("churn_score", live.get("churn_probability", 0.0))
    live["base_churn_probability"] = live.get("churn_probability", live["realtime_churn_score"])
    live["score_delta"] = live["realtime_churn_score"] - live["base_churn_probability"]
    live["last_event_type"] = "user_live_event"
    live["latest_trigger_reason"] = "PostgreSQL user-live score"
    live["action_queue_status"] = "not_queued"
    live["queued_recommended_action"] = None
    live["queued_intervention_intensity"] = None
    live["queued_coupon_cost"] = 0.0
    live["queued_expected_profit"] = live.get("expected_incremental_profit", 0.0)
    live["queued_expected_roi"] = live.get("expected_roi", 0.0)
    live["reoptimization_count"] = 0

    if actions_df is not None and not actions_df.empty and "customer_id" in actions_df.columns:
        action_cols = [
            "customer_id",
            "action_status",
            "recommended_action",
            "intervention_intensity",
            "coupon_cost",
            "expected_profit",
            "expected_roi",
            "trigger_reason",
        ]
        action_lookup = actions_df[[col for col in action_cols if col in actions_df.columns]].copy()
        action_lookup = action_lookup.drop_duplicates("customer_id", keep="first")
        live = live.merge(action_lookup, on="customer_id", how="left", suffixes=("", "_action"))
        if "action_status" in live.columns:
            live["action_queue_status"] = live["action_status"].fillna("not_queued")
        if "recommended_action" in live.columns:
            live["queued_recommended_action"] = live["recommended_action"]
        if "intervention_intensity" in live.columns:
            live["queued_intervention_intensity"] = live["intervention_intensity"]
        if "coupon_cost_action" in live.columns:
            live["queued_coupon_cost"] = pd.to_numeric(live["coupon_cost_action"], errors="coerce").fillna(0.0)
        elif "coupon_cost" in live.columns:
            live["queued_coupon_cost"] = pd.to_numeric(live["coupon_cost"], errors="coerce").fillna(0.0)
        if "expected_profit" in live.columns:
            live["queued_expected_profit"] = pd.to_numeric(live["expected_profit"], errors="coerce").fillna(live["queued_expected_profit"])
        if "expected_roi_action" in live.columns:
            live["queued_expected_roi"] = pd.to_numeric(live["expected_roi_action"], errors="coerce").fillna(live["queued_expected_roi"])
        if "trigger_reason" in live.columns:
            live["latest_trigger_reason"] = live["trigger_reason"].fillna(live["latest_trigger_reason"])

    return live


def _merge_live_score_dimensions(actions_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    """action_queue row에 score table의 persona/uplift/risk 차원을 보강한다."""
    if actions_df is None or actions_df.empty or scores_df is None or scores_df.empty:
        return actions_df if actions_df is not None else pd.DataFrame()
    if "customer_id" not in actions_df.columns or "customer_id" not in scores_df.columns:
        return actions_df

    scores = _rename_live_score_columns(scores_df).copy()
    dim_cols = [
        col for col in [
            "customer_id",
            "persona",
            "persona_source",
            "uplift_segment",
            "uplift_segment_source",
            "risk_segment",
            "churn_probability",
            "churn_score",
            "clv",
            "uplift_score",
        ]
        if col in scores.columns
    ]
    if len(dim_cols) <= 1:
        return actions_df

    merged = actions_df.merge(
        scores[dim_cols].drop_duplicates("customer_id", keep="first"),
        on="customer_id",
        how="left",
        suffixes=("", "_score"),
    )

    for col in [
        "persona",
        "persona_source",
        "uplift_segment",
        "uplift_segment_source",
        "risk_segment",
        "churn_probability",
        "churn_score",
        "clv",
        "uplift_score",
    ]:
        score_col = f"{col}_score"
        if score_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[score_col]
        else:
            missing_mask = merged[col].map(_is_placeholder_segment)
            merged.loc[missing_mask, col] = merged.loc[missing_mask, score_col]
        merged = merged.drop(columns=[score_col])

    return merged



def _build_score_based_live_budget_payload(
    scores_df: pd.DataFrame | None,
    *,
    budget: int,
    threshold: float,
    max_customers: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """action_queue 행이 현재 필터에 걸리지 않을 때 score table로 예산 타겟을 재계산한다.

    user-live action_queue에는 과거 조건으로 생성된 action만 들어 있을 수 있고,
    일부 row는 churn/profit/cost 컬럼이 비어 있을 수 있다. 이 경우 action_queue만
    기준으로 필터링하면 고객 score는 존재하는데 최종 타겟이 0명으로 떨어져
    개인화 추천 화면까지 비는 문제가 생긴다. score table을 고객 후보 풀로 삼아
    기존 예산 최적화 로직을 한 번 더 태워 화면 컨트롤과 일관된 타겟을 만든다.
    """
    if scores_df is None or scores_df.empty or budget <= 0:
        return pd.DataFrame(), {}, pd.DataFrame()

    score_customers = _rename_live_score_columns(scores_df).copy()
    if score_customers.empty or "customer_id" not in score_customers.columns:
        return pd.DataFrame(), {}, pd.DataFrame()

    if "churn_probability" not in score_customers.columns and "churn_score" in score_customers.columns:
        score_customers["churn_probability"] = score_customers["churn_score"]
    if "churn_probability" in score_customers.columns:
        score_customers["churn_probability"] = pd.to_numeric(score_customers["churn_probability"], errors="coerce").fillna(0.0)

    # 필수 표시/최적화 컬럼이 없으면 보수적인 기본값을 둔다. 실제 비용/수익 산식은
    # get_budget_result 내부의 build_intensity_action_candidates에서 다시 계산된다.
    defaults: dict[str, Any] = {
        "persona": "live_user",
        "uplift_segment": "live",
        "risk_segment": "live",
        "uplift_score": 0.12,
        "clv": 0.0,
        "coupon_cost": 0.0,
        "expected_incremental_profit": 0.0,
        "expected_roi": 0.0,
    }
    for col, default in defaults.items():
        if col not in score_customers.columns:
            score_customers[col] = default
    for col in ["uplift_score", "clv", "coupon_cost", "expected_incremental_profit", "expected_roi"]:
        score_customers[col] = pd.to_numeric(score_customers[col], errors="coerce").fillna(float(defaults.get(col, 0.0)))

    selected, summary, allocation = get_budget_result(
        score_customers,
        budget=int(budget),
        threshold=float(threshold),
        max_customers=max_customers,
    )
    if not selected.empty:
        summary = dict(summary or {})
        summary["source"] = "postgresql_user_live_score_fallback_reoptimized"
        summary["fallback_reason"] = "action_queue 후보가 현재 예산/임계값 조건에서 비어 score table로 재계산"
    return selected, summary or {}, allocation

def _build_live_optimize_payload(
    actions_df: pd.DataFrame,
    budget: int,
    threshold: float = 0.50,
    max_customers: int | None = None,
    scores_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """user live action_queue를 현재 분석 컨트롤에 맞춰 재선정한다.

    기존 구현은 live action_queue에 저장된 모든 추천을 그대로 합산했기 때문에
    사이드바의 예산/이탈 임계값/최대 고객 수를 바꿔도 집행 예산과 추천 대상이
    고정되어 보였다. 여기서는 action_queue를 후보 풀로만 사용하고, 현재 컨트롤
    값으로 다시 필터링·정렬·예산 컷을 적용한다.
    """
    budget = max(int(budget or 0), 0)
    threshold = float(threshold or 0.0)
    max_customers = int(max_customers) if max_customers is not None else None
    if max_customers is not None:
        max_customers = max(max_customers, 0)

    empty_summary = {
        "budget": int(budget),
        "spent": 0.0,
        "remaining": float(budget),
        "num_targeted": 0,
        "expected_incremental_profit": 0.0,
        "overall_roi": 0.0,
        "candidate_segment_counts": {},
        "eligible_actions": 0,
        "eligible_customers": 0,
        "threshold": threshold,
        "max_customers_cap": max_customers,
        "source": "postgresql_user_live_action_queue_reoptimized",
    }
    if actions_df is None or actions_df.empty or budget <= 0 or max_customers == 0:
        return pd.DataFrame(), empty_summary, pd.DataFrame()

    enriched_actions = _merge_live_score_dimensions(actions_df, scores_df if scores_df is not None else pd.DataFrame())
    candidates = _normalize_live_actions_df(enriched_actions)
    if candidates.empty:
        return pd.DataFrame(), empty_summary, pd.DataFrame()

    for col in [
        "coupon_cost",
        "expected_incremental_profit",
        "expected_profit",
        "expected_roi",
        "priority_score",
        "selection_score",
        "churn_probability",
        "clv",
        "uplift_score",
    ]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")

    if "expected_incremental_profit" not in candidates.columns:
        candidates["expected_incremental_profit"] = pd.to_numeric(
            candidates.get("expected_profit", pd.Series(0.0, index=candidates.index)),
            errors="coerce",
        ).fillna(0.0)
    else:
        candidates["expected_incremental_profit"] = candidates["expected_incremental_profit"].fillna(
            pd.to_numeric(candidates.get("expected_profit", pd.Series(0.0, index=candidates.index)), errors="coerce")
        ).fillna(0.0)

    if "coupon_cost" not in candidates.columns:
        candidates["coupon_cost"] = 0.0
    candidates["coupon_cost"] = candidates["coupon_cost"].fillna(0.0)

    if "churn_probability" not in candidates.columns:
        candidates["churn_probability"] = pd.to_numeric(
            candidates.get("churn_score", pd.Series(0.0, index=candidates.index)),
            errors="coerce",
        ).fillna(0.0)
    else:
        candidates["churn_probability"] = candidates["churn_probability"].fillna(
            pd.to_numeric(candidates.get("churn_score", pd.Series(0.0, index=candidates.index)), errors="coerce")
        ).fillna(0.0)

    if "expected_roi" not in candidates.columns:
        candidates["expected_roi"] = np.where(
            candidates["coupon_cost"] > 0,
            candidates["expected_incremental_profit"] / candidates["coupon_cost"],
            0.0,
        )
    candidates["expected_roi"] = pd.to_numeric(candidates["expected_roi"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "selection_score" not in candidates.columns:
        candidates["selection_score"] = (
            candidates["expected_incremental_profit"].rank(pct=True).fillna(0.0) * 0.50
            + candidates["expected_roi"].rank(pct=True).fillna(0.0) * 0.25
            + candidates["churn_probability"].rank(pct=True).fillna(0.0) * 0.25
        )
    if "priority_score" not in candidates.columns:
        candidates["priority_score"] = candidates["selection_score"]

    eligible = candidates[
        (candidates["churn_probability"] >= threshold)
        & (candidates["coupon_cost"] > 0)
        & (candidates["expected_incremental_profit"] > 0)
    ].copy()
    if eligible.empty:
        fallback_selected, fallback_summary, fallback_allocation = _build_score_based_live_budget_payload(
            scores_df,
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
        )
        if not fallback_selected.empty:
            fallback_summary = dict(fallback_summary or {})
            fallback_summary.update({
                "candidate_actions": int(len(candidates)),
                "candidate_customers": int(candidates["customer_id"].nunique()) if "customer_id" in candidates.columns else 0,
                "action_queue_eligible_actions": 0,
            })
            return fallback_selected, fallback_summary, fallback_allocation

        summary = empty_summary.copy()
        summary.update({
            "candidate_actions": int(len(candidates)),
            "candidate_customers": int(candidates["customer_id"].nunique()) if "customer_id" in candidates.columns else 0,
        })
        return pd.DataFrame(), summary, pd.DataFrame()

    sort_cols = [
        col for col in [
            "selection_score",
            "priority_score",
            "expected_incremental_profit",
            "expected_roi",
            "churn_probability",
        ] if col in eligible.columns
    ]
    eligible = eligible.sort_values(
        sort_cols + (["coupon_cost"] if "coupon_cost" in eligible.columns else []),
        ascending=[False] * len(sort_cols) + ([True] if "coupon_cost" in eligible.columns else []),
        kind="mergesort",
    )

    selected_rows: list[pd.Series] = []
    seen_customers: set[Any] = set()
    spent = 0.0
    for _, row in eligible.iterrows():
        customer_id = row.get("customer_id")
        if customer_id in seen_customers:
            continue
        cost = float(row.get("coupon_cost", 0.0) or 0.0)
        if cost <= 0 or spent + cost > budget:
            continue
        selected_rows.append(row)
        seen_customers.add(customer_id)
        spent += cost
        if max_customers is not None and len(selected_rows) >= max_customers:
            break

    if not selected_rows:
        fallback_selected, fallback_summary, fallback_allocation = _build_score_based_live_budget_payload(
            scores_df,
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
        )
        if not fallback_selected.empty:
            fallback_summary = dict(fallback_summary or {})
            fallback_summary.update({
                "candidate_actions": int(len(candidates)),
                "candidate_customers": int(candidates["customer_id"].nunique()) if "customer_id" in candidates.columns else 0,
                "action_queue_eligible_actions": int(len(eligible)),
                "action_queue_eligible_customers": int(eligible["customer_id"].nunique()) if "customer_id" in eligible.columns else 0,
            })
            return fallback_selected, fallback_summary, fallback_allocation

        summary = empty_summary.copy()
        summary.update({
            "candidate_actions": int(len(candidates)),
            "candidate_customers": int(candidates["customer_id"].nunique()) if "customer_id" in candidates.columns else 0,
            "eligible_actions": int(len(eligible)),
            "eligible_customers": int(eligible["customer_id"].nunique()) if "customer_id" in eligible.columns else 0,
        })
        return pd.DataFrame(), summary, pd.DataFrame()

    targets = pd.DataFrame(selected_rows).reset_index(drop=True)
    expected_profit = float(pd.to_numeric(targets["expected_incremental_profit"], errors="coerce").fillna(0.0).sum())
    spent = float(pd.to_numeric(targets["coupon_cost"], errors="coerce").fillna(0.0).sum())
    overall_roi = expected_profit / spent if spent > 0 else 0.0

    segment_col = "uplift_segment"
    candidate_segment_counts = (
        eligible[segment_col].fillna("unknown_segment").replace({"live": "unknown_segment"}).value_counts().to_dict()
        if segment_col in eligible.columns
        else {}
    )
    optimize_summary = {
        "budget": int(budget),
        "spent": spent,
        "remaining": max(float(budget) - spent, 0.0),
        "num_targeted": int(len(targets)),
        "expected_incremental_profit": expected_profit,
        "overall_roi": overall_roi,
        "candidate_segment_counts": candidate_segment_counts,
        "candidate_actions": int(len(candidates)),
        "candidate_customers": int(candidates["customer_id"].nunique()) if "customer_id" in candidates.columns else 0,
        "eligible_actions": int(len(eligible)),
        "eligible_customers": int(eligible["customer_id"].nunique()) if "customer_id" in eligible.columns else 0,
        "threshold": threshold,
        "max_customers_cap": max_customers,
        "source": "postgresql_user_live_action_queue_reoptimized",
    }

    if segment_col not in targets.columns:
        segment_allocation = pd.DataFrame()
    else:
        segment_allocation = (
            targets.groupby(segment_col, as_index=False)
            .agg(
                customer_count=("customer_id", "nunique"),
                allocated_budget=("coupon_cost", "sum"),
                expected_profit=("expected_incremental_profit", "sum"),
            )
            .rename(columns={segment_col: "uplift_segment"})
        )
        if "intervention_intensity" in targets.columns:
            intensity = (
                targets.groupby(segment_col)["intervention_intensity"]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "medium")
                .reset_index(drop=True)
            )
            segment_allocation["intervention_intensity"] = intensity

    return targets, optimize_summary, segment_allocation



def _normalize_live_recommendations_for_display(df: pd.DataFrame, per_customer: int) -> pd.DataFrame:
    """DB 저장 추천 후보를 5번 화면이 기대하는 컬럼명과 고객당 추천 수로 정리한다."""
    if df is None or df.empty:
        return pd.DataFrame()
    fixed = df.copy()
    if "recommended_category" not in fixed.columns:
        for alias in ["category", "product_category", "recommended_action", "action", "item_category"]:
            if alias in fixed.columns:
                fixed["recommended_category"] = fixed[alias]
                break
        else:
            fixed["recommended_category"] = "retention_action"
    if "recommendation_score" not in fixed.columns:
        for alias in ["score", "priority_score", "selection_score", "recommendation_priority"]:
            if alias in fixed.columns:
                fixed["recommendation_score"] = pd.to_numeric(fixed[alias], errors="coerce").fillna(0.0)
                break
        else:
            fixed["recommendation_score"] = 0.0
    if "customer_id" in fixed.columns:
        fixed["recommendation_score"] = pd.to_numeric(fixed["recommendation_score"], errors="coerce").fillna(0.0)
        fixed = fixed.sort_values(["customer_id", "recommendation_score"], ascending=[True, False], kind="mergesort")
        fixed["recommendation_rank"] = fixed.groupby("customer_id").cumcount() + 1
        fixed = fixed[fixed["recommendation_rank"] <= max(1, int(per_customer))].reset_index(drop=True)
    elif "recommendation_rank" not in fixed.columns:
        fixed["recommendation_rank"] = range(1, len(fixed) + 1)
    return fixed


def _fallback_existing_live_recommendations(
    *,
    per_customer: int,
    max_customers: int,
    optimize_summary: dict[str, Any] | None,
    reason: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """현재 타겟 재생성이 실패해도 저장된 live 추천 후보를 화면에 계속 보여준다."""
    limit = max(100, int(max_customers or 0) * max(1, int(per_customer)))
    try:
        live_summary, live_df = fetch_user_live_recommendations(limit=limit)
    except Exception as exc:
        return {
            "rows": 0,
            "customers_covered": 0,
            "per_customer": int(per_customer),
            "candidate_limit": int(max_customers or 0),
            "budget_context": dict(optimize_summary or {}),
            "error": f"{reason} 저장된 live 추천 후보 조회도 실패했습니다: {exc}",
        }, pd.DataFrame()

    live_df = _normalize_live_recommendations_for_display(live_df, per_customer=per_customer)
    if live_df.empty:
        return {
            "rows": 0,
            "customers_covered": 0,
            "per_customer": int(per_customer),
            "candidate_limit": int(max_customers or 0),
            "budget_context": dict(optimize_summary or {}),
            "error": reason,
        }, pd.DataFrame()

    summary = dict(live_summary or {})
    summary.update({
        "rows": int(len(live_df)),
        "customers_covered": int(live_df["customer_id"].nunique()) if "customer_id" in live_df.columns else 0,
        "per_customer": int(per_customer),
        "candidate_limit": int(max_customers or 0),
        "budget_context": dict(optimize_summary or {}),
        "source": "postgresql_user_live_saved_recommendation_fallback",
        "warning": reason + " 저장된 live 추천 후보를 대신 표시합니다.",
    })
    return summary, live_df

def _build_dynamic_user_recommendations(
    selected_customers: pd.DataFrame,
    optimize_summary: dict[str, Any],
    *,
    per_customer: int,
    budget: int,
    threshold: float,
    max_customers: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """현재 최적화 결과를 기준으로 user mode 개인화 추천을 즉시 재생성한다."""
    if selected_customers is None or selected_customers.empty:
        return _fallback_existing_live_recommendations(
            per_customer=per_customer,
            max_customers=max_customers,
            optimize_summary=optimize_summary,
            reason="현재 예산/임계값 조건에서 새 추천을 만들 최종 타겟 고객이 없습니다.",
        )

    data_dir = Path("data/raw_user")
    result_dir = Path("results_user")
    required_files = [data_dir / "customer_summary.csv", data_dir / "orders.csv", data_dir / "events.csv"]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        fallback_path = result_dir / "personalized_recommendations.csv"
        fallback_df = pd.read_csv(fallback_path) if fallback_path.exists() else pd.DataFrame()
        summary = {
            "rows": int(len(fallback_df)),
            "customers_covered": int(fallback_df["customer_id"].nunique()) if not fallback_df.empty and "customer_id" in fallback_df.columns else 0,
            "per_customer": int(per_customer),
            "candidate_limit": int(max_customers),
            "budget_context": dict(optimize_summary or {}),
            "error": "user raw data 파일이 없어 저장된 추천 결과만 표시합니다: " + ", ".join(missing_files),
        }
        return summary, fallback_df

    try:
        from src.recommendations.modeling import run_personalized_recommendation_pipeline

        candidate_limit = max(1, min(int(max_customers), int(len(selected_customers))))
        target_df = selected_customers.copy().head(candidate_limit)
        artifacts = run_personalized_recommendation_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            per_customer=max(1, int(per_customer)),
            candidate_limit=candidate_limit,
            target_customers=target_df,
            target_source="current_budget_threshold_targets",
        )
        rec_df = pd.read_csv(artifacts.recommendations_path) if Path(artifacts.recommendations_path).exists() else pd.DataFrame()
        summary = dict(artifacts.summary)
    except Exception as exc:
        return _fallback_existing_live_recommendations(
            per_customer=per_customer,
            max_customers=max_customers,
            optimize_summary=optimize_summary,
            reason=f"새 추천 재생성에 실패했습니다({exc}).",
        )

    budget_context = dict(optimize_summary or {})
    budget_context.update({
        "budget": int(budget),
        "threshold": float(threshold),
        "max_customers_cap": int(max_customers),
    })
    summary.update({
        "rows": int(len(rec_df)),
        "customers_covered": int(rec_df["customer_id"].nunique()) if not rec_df.empty and "customer_id" in rec_df.columns else 0,
        "per_customer": int(per_customer),
        "candidate_limit": int(max_customers),
        "eligible_target_customers": int(len(selected_customers)),
        "budget_context": budget_context,
    })
    try:
        (result_dir / "personalized_recommendations_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return summary, rec_df


def _load_user_live_tables(*, top_n: int, target_cap: int) -> dict[str, Any]:
    """user mode 전용 live API 조회 묶음. 실패 시 빈 DataFrame fallback.

    성능 최적화:
    - health/seed는 가벼우므로 매번 조회한다.
    - 전체 scores 19,999행은 무거우므로 latest_event_time/seed 상태가 같으면 cache를 재사용한다.
    - recommendations/actions는 화면 표시용이므로 safe_limit만 조회한다.
    """
    payload: dict[str, Any] = {
        "enabled": _is_user_live_mode(),
        "health": {},
        "seed_status": {},
        "score_summary": {},
        "scores": pd.DataFrame(),
        "recommendation_summary": {},
        "recommendations": pd.DataFrame(),
        "action_summary": {},
        "actions": pd.DataFrame(),
    }
    if not payload["enabled"]:
        return payload

    safe_limit = max(int(top_n), int(target_cap), 100)
    try:
        payload["health"] = fetch_user_live_health()
    except Exception as exc:
        payload["health"] = {"status": "error", "error": str(exc)}
    try:
        payload["seed_status"] = fetch_user_live_seed_status()
    except Exception as exc:
        payload["seed_status"] = {"success": False, "error": str(exc)}

    try:
        seed_status = payload.get("seed_status", {}) or {}
        seed_inner = seed_status.get("status", {}) if isinstance(seed_status, dict) else {}
        score_cache_key = "|".join([
            str((payload.get("health", {}) or {}).get("latest_event_time") or "no_event"),
            str(seed_inner.get("score_count") or 0),
            str(seed_inner.get("latest_score_seeded_at") or "no_seed"),
        ])
        summary, scores = _fetch_user_live_scores_cached(score_cache_key)
        payload["score_summary"] = summary
        payload["scores"] = _rename_live_score_columns(scores)
    except Exception as exc:
        payload["score_summary"] = {"error": str(exc)}

    try:
        summary, recommendations = fetch_user_live_recommendations(limit=safe_limit)
        payload["recommendation_summary"] = summary
        payload["recommendations"] = recommendations
    except Exception as exc:
        payload["recommendation_summary"] = {"error": str(exc)}
    try:
        summary, actions = fetch_user_live_actions(limit=safe_limit, status="queued")
        payload["action_summary"] = summary
        payload["actions"] = _normalize_live_actions_df(actions)
    except Exception as exc:
        payload["action_summary"] = {"error": str(exc)}
    return payload


def _render_user_live_status(live_payload: dict[str, Any]) -> None:
    if not live_payload.get("enabled"):
        return
    health = live_payload.get("health", {}) or {}
    if health.get("status") == "ok":
        st.success(
            f"자사 데이터 Live DB 연결됨 · 이벤트 {int(health.get('event_count') or 0):,}건 · "
            f"실시간 고객 상태 {int(health.get('feature_state_count') or 0):,}명 · "
            f"최신 이벤트 {health.get('latest_event_time') or '-'}"
        )
    else:
        st.warning(f"자사 데이터 Live DB 상태 확인 실패: {health.get('error', 'unknown error')}")

    seed_status = live_payload.get("seed_status", {}) or {}
    status = seed_status.get("status", {}) if isinstance(seed_status, dict) else {}
    if status:
        st.caption(
            "Live seed 상태 · "
            f"scores={int(status.get('score_count') or 0):,}, "
            f"recommendations={int(status.get('recommendation_count') or 0):,}, "
            f"actions={int(status.get('action_queue_count') or 0):,}"
        )
# ============================================================
# [/PATCH]
# ============================================================

def _path_exists(path_value: Any) -> bool:
    """컨테이너/로컬 양쪽에서 산출물 경로가 실제로 존재하는지 확인한다."""
    if not path_value:
        return False
    try:
        path = Path(str(path_value))
    except Exception:
        return False

    candidates = [path]
    if not path.is_absolute():
        candidates.append(_project_root() / path)
    return any(candidate.exists() for candidate in candidates)


def _render_missing_data_box(feature_name: str, reason: str = "", action_hint: str = "") -> None:
    """산출물이 아직 없을 때 렌더링 실패 대신 일관된 '해당 데이터 없음' 박스를 보여준다."""
    default_reason = (
        "이 화면에 필요한 산출물이 아직 생성되지 않았습니다. "
        "Docker 컨테이너만 실행한 상태라면 학습/생존분석/실험/실시간 리플레이 관련 결과 파일이 없을 수 있습니다."
    )
    default_hint = (
        "필요한 경우 시뮬레이터 파이프라인 명령을 먼저 실행한 뒤 대시보드를 새로고침하세요."
    )
    safe_feature = html.escape(str(feature_name))
    safe_reason = html.escape(str(reason or default_reason))
    safe_hint = html.escape(str(action_hint or default_hint))
    st.markdown(
        f"""
        <div style="
            background-color: #F3F4F6;
            border: 1px dashed #9CA3AF;
            border-radius: 12px;
            padding: 32px 24px;
            margin: 16px 0;
            text-align: center;
        ">
            <div style="font-size: 40px; opacity: 0.5;">📭</div>
            <div style="font-size: 20px; font-weight: 700; color: #374151; margin-top: 8px;">
                해당 데이터 없음
            </div>
            <div style="font-size: 14px; color: #6B7280; margin-top: 8px;">
                {safe_feature}
            </div>
            <div style="font-size: 13px; color: #9CA3AF; margin-top: 12px; line-height: 1.5;">
                {safe_reason}
            </div>
            <div style="font-size: 12px; color: #9CA3AF; margin-top: 12px; font-style: italic;">
                💡 {safe_hint}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _simulator_missing_result_box(feature_name: str, reason: str = "", action_hint: str = "") -> None:
    """시뮬레이터 데모에서 산출물이 없을 때 사용할 안내 박스."""
    _render_missing_data_box(
        feature_name,
        reason or "시뮬레이터 데모 산출물이 아직 없습니다. docker compose up만 실행하면 일부 모델 검증/생존분석/실험 산출물은 생성되지 않습니다.",
        action_hint or "python src/main.py --mode train, survival, abtest, fidelity 등 필요한 시뮬레이터 산출 명령을 먼저 실행하세요.",
    )


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and len(value) > 0


def _simulator_mode_unavailable(feature_name: str, has_data: bool, reason: str = "", action_hint: str = "") -> bool:
    """simulator 모드에서 필요한 데이터가 없을 때 일관된 안내를 보여준다."""
    if st.session_state.get("data_mode", "simulator") != "simulator":
        return False
    if has_data:
        return False
    _simulator_missing_result_box(feature_name, reason=reason, action_hint=action_hint)
    return True


def _ensure_retention_target_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    외부 CSV/user 산출물에서 리텐션 대상 고객 목록 렌더링에 필요한 컬럼이
    누락되어도 화면이 깨지지 않도록 공통 스키마를 보정한다.

    특히 6번 화면은 priority_score, selection_score,
    expected_incremental_profit, customer_id 기준으로 정렬하므로,
    이 컬럼들이 없으면 사용 가능한 대체 점수로 생성한다.
    """
    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    # empty DataFrame도 아래에서 필요한 컬럼을 만들어야 한다.
    # 여기서 바로 return하면 sort_values(["selection_score", ...])가 다시 KeyError를 낸다.
    fixed = df.copy()

    def _series_from(col: str, default: float = 0.0) -> pd.Series:
        if col in fixed.columns:
            return pd.to_numeric(fixed[col], errors="coerce").fillna(default)
        return pd.Series(default, index=fixed.index, dtype="float64")

    # customer_id가 없는 외부 산출물도 정렬/표시 가능하게 보정
    if "customer_id" not in fixed.columns:
        fixed["customer_id"] = range(1, len(fixed) + 1)

    # 화면·hover·요약에서 자주 쓰는 수치 컬럼 기본값 보장
    for numeric_col in [
        "churn_probability",
        "uplift_score",
        "clv",
        "coupon_cost",
        "expected_roi",
    ]:
        if numeric_col not in fixed.columns:
            fixed[numeric_col] = 0.0
        else:
            fixed[numeric_col] = _series_from(numeric_col)

    # expected_incremental_profit 보정
    if "expected_incremental_profit" not in fixed.columns:
        if "expected_profit" in fixed.columns:
            fixed["expected_incremental_profit"] = _series_from("expected_profit")
        elif "incremental_profit" in fixed.columns:
            fixed["expected_incremental_profit"] = _series_from("incremental_profit")
        elif "expected_roi" in fixed.columns and "coupon_cost" in fixed.columns:
            fixed["expected_incremental_profit"] = _series_from("expected_roi") * _series_from("coupon_cost")
        elif "uplift_score" in fixed.columns and "clv" in fixed.columns:
            fixed["expected_incremental_profit"] = _series_from("uplift_score") * _series_from("clv")
        else:
            fixed["expected_incremental_profit"] = 0.0
    else:
        fixed["expected_incremental_profit"] = _series_from("expected_incremental_profit")

    # priority_score 보정: 가장 추천 우선순위에 가까운 컬럼부터 사용
    if "priority_score" not in fixed.columns:
        if "selection_score" in fixed.columns:
            fixed["priority_score"] = _series_from("selection_score")
        elif "value_score" in fixed.columns:
            fixed["priority_score"] = _series_from("value_score")
        elif "expected_incremental_profit" in fixed.columns:
            fixed["priority_score"] = _series_from("expected_incremental_profit")
        elif "uplift_score" in fixed.columns and "clv" in fixed.columns:
            fixed["priority_score"] = _series_from("uplift_score") * _series_from("clv")
        elif "expected_roi" in fixed.columns:
            fixed["priority_score"] = _series_from("expected_roi")
        elif "uplift_score" in fixed.columns:
            fixed["priority_score"] = _series_from("uplift_score")
        elif "churn_probability" in fixed.columns:
            fixed["priority_score"] = _series_from("churn_probability")
        elif "churn_prob" in fixed.columns:
            fixed["priority_score"] = _series_from("churn_prob")
        elif "risk_score" in fixed.columns:
            fixed["priority_score"] = _series_from("risk_score")
        else:
            fixed["priority_score"] = 0.0
    else:
        fixed["priority_score"] = _series_from("priority_score")

    # selection_score 보정
    if "selection_score" not in fixed.columns:
        fixed["selection_score"] = _series_from("priority_score")
    else:
        fixed["selection_score"] = _series_from("selection_score")

    return fixed


def _circled_num(n: str) -> str:
    try:
        i = int(n)
        if 1 <= i <= 20:
            return chr(0x245F + i)  # ① = 0x2460
    except Exception:
        pass
    return f"{n}."


def _view_title_from_option(option: str) -> str:
    for num, title in DASHBOARD_VIEW_ITEMS:
        if f"{num}. {title}" == option:
            return f"{_circled_num(num)}  {title}"
    return option

st.set_page_config(
    page_title="Retention ROI Dashboard",
    page_icon="📊",
    layout="wide",
)



def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --bg-grad-1: #0f172a;
            --bg-grad-2: #111827;
            --card-bg: rgba(255,255,255,0.88);
            --card-border: rgba(15, 23, 42, 0.08);
            --accent: #2563eb;
            --accent-2: #7c3aed;
            --text-main: #0f172a;
            --text-soft: #475569;
            --success-bg: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(16,185,129,0.10));
            --warn-bg: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(251,191,36,0.10));
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent 28%),
                radial-gradient(circle at top right, rgba(124,58,237,0.08), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            color: var(--text-main);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* 사이드바 기본 텍스트 */
        section[data-testid="stSidebar"] {
            color: #e5eefc !important;
        }
        section[data-testid="stSidebar"] * {
            text-shadow: none !important;
        }

        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div div {
            background-color: rgba(255,255,255,0.18);
        }

        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(135deg, rgba(37,99,235,0.24), rgba(124,58,237,0.24));
            color: white !important;
            font-weight: 600;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
            color: #e5eefc !important;
            font-weight: 700 !important;
            font-size: 1.2rem !important;
            margin-bottom: 4px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {
            gap: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label {
            display: flex !important;
            align-items: center !important;
            padding: 3px 6px !important;
            margin: 0 !important;
            border: none !important;
            background: transparent !important;
            border-radius: 4px !important;
            box-shadow: none !important;
            cursor: pointer !important;
            transition: background 0.15s ease, color 0.15s ease !important;
            width: 100% !important;
        }
        /* 라벨 텍스트: 번호(①..⑪)가 또렷하게 보이도록 크기/굵기 확보 */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label div {
            color: #e5eefc !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
            margin: 0 !important;
            white-space: normal !important;
            word-break: keep-all !important;
        }
        /* hover: 배경색만 은은하게 변경, 크기/위치 변화 없음 */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
            background: rgba(37,99,235,0.22) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:hover p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:hover div {
            color: #ffffff !important;
        }
        /* 선택됨: 해당 항목의 input이 checked 상태인 label을 찾아 진하게 표시 */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
            background: rgba(37,99,235,0.42) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) div {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked):hover {
            background: rgba(37,99,235,0.55) !important;
        }
        
        /* radio / toggle / slider 글자 고정 */
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stRadio label p,
        section[data-testid="stSidebar"] .stToggle label,
        section[data-testid="stSidebar"] .stToggle label p,
        section[data-testid="stSidebar"] .stCheckbox label,
        section[data-testid="stSidebar"] .stCheckbox label p,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stSlider span,
        section[data-testid="stSidebar"] .stSlider p,
        section[data-testid="stSidebar"] [role="radiogroup"] label,
        section[data-testid="stSidebar"] [role="radiogroup"] label p {
            color: #e5eefc !important;
            -webkit-text-fill-color: #e5eefc !important;
            opacity: 1 !important;
        }

        /* 사이드바 입력칸은 흰 배경 + 진한 글씨로 원복 */
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stTextArea textarea,
        section[data-testid="stSidebar"] input[type="password"],
        section[data-testid="stSidebar"] input[type="text"],
        section[data-testid="stSidebar"] [data-baseweb="input"] input,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] input,
        section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
            border-radius: 14px !important;
            background: #ffffff !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            caret-color: #111827 !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] .stTextInput input::placeholder,
        section[data-testid="stSidebar"] .stNumberInput input::placeholder,
        section[data-testid="stSidebar"] .stTextArea textarea::placeholder,
        section[data-testid="stSidebar"] input[type="password"]::placeholder,
        section[data-testid="stSidebar"] input[type="text"]::placeholder,
        section[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] input::placeholder,
        section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] .stNumberInput button,
        section[data-testid="stSidebar"] [data-baseweb="input"] button,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] button {
            color: #111827 !important;
            background: #ffffff !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
        }

        section[data-testid="stSidebar"] .stNumberInput button svg,
        section[data-testid="stSidebar"] [data-baseweb="input"] button svg,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] button svg {
            fill: #111827 !important;
            color: #111827 !important;
        }


        /* selectbox / combobox는 흰 배경 위에서 진한 글씨로 고정 */
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
        section[data-testid="stSidebar"] .stSelectbox [role="combobox"],
        section[data-testid="stSidebar"] .stSelectbox svg {
            background: #ffffff !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            fill: #111827 !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            border-radius: 14px !important;
        }
        section[data-testid="stSidebar"] .stSelectbox [role="listbox"],
        section[data-testid="stSidebar"] .stSelectbox [role="option"] {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            padding: 32px 32px 26px 32px;
            margin-bottom: 18px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(37,99,235,0.92) 60%, rgba(124,58,237,0.88));
            box-shadow: 0 24px 60px rgba(15,23,42,0.22);
            color: white;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -70px -90px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(255,255,255,0.22), transparent 65%);
            pointer-events: none;
        }

        .hero-kicker {
            font-size: 0.9rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            opacity: 0.78;
            margin-bottom: 10px;
        }

        .hero-title {
            font-size: 2.5rem;
            line-height: 1.08;
            font-weight: 800;
            margin: 0 0 12px 0;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: rgba(255,255,255,0.82);
            max-width: 900px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            border-radius: 999px;
            padding: 10px 16px;
            margin: 10px 0 18px 0;
            font-weight: 600;
            font-size: 0.96rem;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 12px 30px rgba(15,23,42,0.06);
        }

        .status-pill.success {
            background: var(--success-bg);
            color: #166534;
        }

        .status-pill.warn {
            background: var(--warn-bg);
            color: #92400e;
        }

        .section-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 24px 24px 10px 24px;
            box-shadow: 0 12px 30px rgba(15,23,42,0.06);
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }

        .section-card h2, .section-card h3 {
            margin-top: 0;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 24px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 14px 28px rgba(15,23,42,0.06);
        }

        [data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: #111827;
            font-weight: 800;
            font-size: clamp(1.35rem, 1.1vw + 0.75rem, 2.05rem);
            line-height: 1.08;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
            max-width: 100%;
        }
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] p {
            font-size: clamp(1.1rem, 0.75vw + 0.65rem, 2.05rem) !important;
            line-height: 1.12 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
            overflow: visible !important;
            text-overflow: clip !important;
            margin: 0 !important;
            max-width: 100% !important;
        }

        .stPlotlyChart, .stDataFrame, [data-testid="stImage"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 24px;
            padding: 10px;
            box-shadow: 0 14px 28px rgba(15,23,42,0.05);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.65);
            border-radius: 14px 14px 0 0;
            padding-left: 16px;
            padding-right: 16px;
        }

        .stAlert {
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.16);
            box-shadow: 0 10px 24px rgba(15,23,42,0.05);
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(37,99,235,0.14);
            background: linear-gradient(135deg, rgba(37,99,235,0.96), rgba(124,58,237,0.92));
            color: white;
            font-weight: 700;
            box-shadow: 0 12px 22px rgba(37,99,235,0.22);
        }

        .stTextArea textarea, .stTextInput input, .stNumberInput input {
            border-radius: 14px;
        }

        hr {
            margin-top: 1.6rem !important;
            margin-bottom: 1.3rem !important;
            border-color: rgba(148,163,184,0.18);
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }

        .sidebar-chatbot-card {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 22px;
            padding: 18px 16px;
            background: linear-gradient(135deg, rgba(37,99,235,0.20), rgba(124,58,237,0.22));
            box-shadow: 0 14px 30px rgba(15,23,42,0.18);
            text-align: center;
            margin-bottom: 10px;
        }

        .sidebar-chatbot-emoji {
            font-size: 3rem;
            line-height: 1;
            margin-bottom: 10px;
        }

        .sidebar-chatbot-title {
            color: #ffffff;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .sidebar-chatbot-desc {
            color: rgba(229,238,252,0.84);
            font-size: 0.90rem;
            line-height: 1.45;
        }

        .chatbot-view-chip {
            display: inline-block;
            margin-top: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: #dbeafe;
            font-size: 0.80rem;
            font-weight: 600;
        }

        .chatbot-dialog-note {
            background: rgba(37,99,235,0.08);
            border: 1px solid rgba(37,99,235,0.12);
            border-radius: 14px;
            padding: 10px 12px;
            color: #334155;
            margin-bottom: 12px;
        }

        .chatbot-drag-handle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
            padding: 10px 12px;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(37,99,235,0.92));
            color: #ffffff;
            font-weight: 700;
            cursor: move;
            user-select: none;
        }

        .chatbot-drag-handle small {
            color: rgba(255,255,255,0.78);
            font-weight: 600;
        }

        .oai-table-wrapper {
            overflow: auto;
            border: 1px solid rgba(148,163,184,0.28);
            border-radius: 16px;
            background: rgba(255,255,255,0.96);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
        }

        .oai-table-wrapper table {
            width: max-content;
            min-width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            line-height: 1.45;
            color: #0f172a;
        }

        .oai-table-wrapper thead th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: #f8fafc !important;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            text-align: left;
            font-weight: 800;
            border-bottom: 1px solid #cbd5e1;
        }

        .oai-table-wrapper th,
        .oai-table-wrapper td {
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
            white-space: nowrap;
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            background: transparent;
        }
        .oai-table-wrapper td * {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        .oai-table-wrapper tbody tr:nth-child(even) {
            background: rgba(248,250,252,0.8);
        }

        .oai-table-wrapper tbody tr:hover {
            background: rgba(219,234,254,0.35);
        }

        .oai-table-controls {
            margin: 4px 0 8px 0;
        }

        /* ── 메인 영역 radio 가시성 보장 (사이드바와 완전히 분리) ── */
        /* 사이드바(stSidebar)는 어두운 남색 배경 + 흰 글씨 → 기존 CSS 유지
           메인(stMain)은 흰 배경 + 진한 글씨로만 적용 */
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label {
            background: rgba(243,244,246,0.6) !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            border-radius: 8px !important;
            padding: 6px 12px !important;
            margin: 2px !important;
            color: #1f2937 !important;
        }
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label p,
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label div {
            color: #1f2937 !important;
            -webkit-text-fill-color: #1f2937 !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
            background: rgba(219,234,254,0.7) !important;
            border-color: #2563eb !important;
        }
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
            background: #2563eb !important;
            border-color: #2563eb !important;
        }
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p,
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) div {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 700 !important;
        }
        /* horizontal radio 줄바꿈 정렬 */
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] {
            flex-wrap: wrap !important;
            gap: 4px !important;
        }

        /* ── 추가 가독성 보장: 라디오 옵션 안의 모든 텍스트 노드 강제 진하게 ── */
        /* 메인 영역 라디오 - 미선택 상태 */
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label *,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label * {
            color: #1f2937 !important;
            -webkit-text-fill-color: #1f2937 !important;
        }
        /* 메인 영역 라디오 - 선택된 상태 (파란 배경 위 흰 글씨) */
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) *,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        /* 라디오의 라벨 텍스트 (st.radio의 메인 label) — 메인 영역에서 진하게 */
        section[data-testid="stMain"] div[data-testid="stRadio"] > label,
        section[data-testid="stMain"] div[data-testid="stRadio"] > label *,
        .main .block-container div[data-testid="stRadio"] > label,
        .main .block-container div[data-testid="stRadio"] > label * {
            color: #1e293b !important;
            -webkit-text-fill-color: #1e293b !important;
            font-weight: 600 !important;
        }
        /* 사이드바 라디오 옵션 안의 모든 자식 노드 — 흰 글씨 강제 (남색 배경 대비) */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label *,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label {
            color: #e5eefc !important;
            -webkit-text-fill-color: #e5eefc !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) *,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 700 !important;
        }
        /* 사이드바의 markdown 텍스트, subheader, caption 등 흰색 계열로 강제 */
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] .stCaption {
            color: #e5eefc !important;
            -webkit-text-fill-color: #e5eefc !important;
        }

        /* ── 사이드바 metric (st.metric) — 흰 배경 카드 제거 + 글자 흰 계열 ── */
        section[data-testid="stSidebar"] [data-testid="stMetric"],
        section[data-testid="stSidebar"] [data-testid="stMetricContainer"],
        section[data-testid="stSidebar"] div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
            padding: 6px 8px !important;
            margin: 2px 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] *,
        section[data-testid="stSidebar"] [data-testid="stMetricValue"],
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] *,
        section[data-testid="stSidebar"] [data-testid="stMetricDelta"],
        section[data-testid="stSidebar"] [data-testid="stMetricDelta"] * {
            color: #e5eefc !important;
            -webkit-text-fill-color: #e5eefc !important;
            opacity: 1 !important;
        }
        /* 좁은 블록에서는 글자 크기를 줄이되 줄바꿈은 막음 */
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        /* ── 사이드바 좁은 컬럼 안의 텍스트도 줄바꿈 방지 + 자동 축소 ── */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"] *,
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"] * {
            word-break: keep-all !important;
        }
        /* 사이드바 안의 일반 markdown bold/strong (예: "**매핑 후 분포 (예상)**") */
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] b {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 700 !important;
        }
        /* 사이드바 안의 dataframe / data_editor 헤더·내용 — 흰 배경 위 진한 글씨 */
        section[data-testid="stSidebar"] [data-testid="stDataFrame"] thead th,
        section[data-testid="stSidebar"] [data-testid="stDataFrame"] tbody td,
        section[data-testid="stSidebar"] [data-testid="stDataEditor"] thead th,
        section[data-testid="stSidebar"] [data-testid="stDataEditor"] tbody td {
            color: #1f2937 !important;
            -webkit-text-fill-color: #1f2937 !important;
        }
        /* 사이드바 dataframe 셀 내용이 좁을 때 줄바꿈 막고 글자 작게 */
        section[data-testid="stSidebar"] [data-testid="stDataFrame"] td,
        section[data-testid="stSidebar"] [data-testid="stDataEditor"] td {
            font-size: 0.78rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        section[data-testid="stSidebar"] [data-testid="stDataFrame"] th,
        section[data-testid="stSidebar"] [data-testid="stDataEditor"] th {
            font-size: 0.78rem !important;
            font-weight: 700 !important;
        }
        /* fallback: stMain 셀렉터가 없는 Streamlit 버전 대비
           — main의 block-container만 타겟 (사이드바는 section selector라 매치 안 됨) */
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label {
            background: rgba(243,244,246,0.6) !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            border-radius: 8px !important;
            padding: 6px 12px !important;
            margin: 2px !important;
            color: #1f2937 !important;
        }
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label p,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label div {
            color: #1f2937 !important;
            -webkit-text-fill-color: #1f2937 !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
            background: #2563eb !important;
            border-color: #2563eb !important;
        }
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) div {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* ── 메인 영역 컨트라스트 일괄 보정 (하늘색 위 하늘색 글씨 방지) ── */
        /* 1) Alert 박스(info/success/warning/error)의 본문 글자 진하게 고정 */
        section[data-testid="stMain"] .stAlert,
        section[data-testid="stMain"] [data-testid="stAlert"],
        .main .block-container .stAlert {
            color: #0f172a !important;
        }
        section[data-testid="stMain"] .stAlert p,
        section[data-testid="stMain"] .stAlert div,
        section[data-testid="stMain"] .stAlert span,
        section[data-testid="stMain"] [data-testid="stAlert"] p,
        section[data-testid="stMain"] [data-testid="stAlert"] div,
        .main .block-container .stAlert p,
        .main .block-container .stAlert div {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }
        /* 2) caption / 작은 보조 텍스트 — 너무 흐리지 않게 */
        section[data-testid="stMain"] [data-testid="stCaptionContainer"],
        section[data-testid="stMain"] .stCaption,
        .main .block-container [data-testid="stCaptionContainer"],
        .main .block-container .stCaption {
            color: #475569 !important;
        }
        /* 3) Metric 카드의 라벨·수치 — 진한 글자 강제 */
        section[data-testid="stMain"] [data-testid="stMetricLabel"],
        section[data-testid="stMain"] [data-testid="stMetricLabel"] p,
        section[data-testid="stMain"] [data-testid="stMetricLabel"] div,
        .main .block-container [data-testid="stMetricLabel"],
        .main .block-container [data-testid="stMetricLabel"] p {
            color: #475569 !important;
            -webkit-text-fill-color: #475569 !important;
            font-weight: 600 !important;
        }
        section[data-testid="stMain"] [data-testid="stMetricValue"],
        section[data-testid="stMain"] [data-testid="stMetricValue"] p,
        section[data-testid="stMain"] [data-testid="stMetricValue"] div,
        .main .block-container [data-testid="stMetricValue"],
        .main .block-container [data-testid="stMetricValue"] p {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            font-weight: 700 !important;
        }
        /* 4) DataFrame 헤더 — 흐린 파란 위에 흐린 파란 글씨 방지 */
        section[data-testid="stMain"] [data-testid="stDataFrame"] thead th,
        .main .block-container [data-testid="stDataFrame"] thead th {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            font-weight: 700 !important;
        }
        /* 5) Tab 라벨 — 비활성/활성 모두 잘 보이게 */
        section[data-testid="stMain"] [data-baseweb="tab"],
        section[data-testid="stMain"] [data-baseweb="tab"] p,
        .main .block-container [data-baseweb="tab"] {
            color: #475569 !important;
        }
        section[data-testid="stMain"] [data-baseweb="tab"][aria-selected="true"],
        section[data-testid="stMain"] [data-baseweb="tab"][aria-selected="true"] p,
        .main .block-container [data-baseweb="tab"][aria-selected="true"] {
            color: #2563eb !important;
            font-weight: 700 !important;
        }
        /* 6) Selectbox 본문 글자 — 흰 배경에 진한 글씨 */
        section[data-testid="stMain"] .stSelectbox [data-baseweb="select"] > div,
        .main .block-container .stSelectbox [data-baseweb="select"] > div {
            color: #0f172a !important;
        }
        /* 7) Markdown 안의 모든 일반 텍스트 (Streamlit이 가끔 light grey로 렌더) */
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] span {
            color: inherit;
        }
        /* 메인 영역 기본 글자색 */
        section[data-testid="stMain"], .main .block-container {
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">Retention Intelligence Copilot</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(message: str, variant: str = "success"):
    st.markdown(
        f'<div class="status-pill {variant}">{message}</div>',
        unsafe_allow_html=True,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _file_version_token(relative_paths: list[str]) -> str:
    parts: list[str] = []
    root = _project_root()
    for relative_path in relative_paths:
        resolved = (root / relative_path).resolve()
        if resolved.exists():
            stat = resolved.stat()
            parts.append(f"{relative_path}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            parts.append(f"{relative_path}:missing")
    return "|".join(parts)


def _raw_data_token() -> str:
    # 모드 인지: 모드에 해당하는 디렉토리 변경시 토큰 변경 → 캐시 자동 invalidation
    mode = st.session_state.get("data_mode", "simulator") if hasattr(st, "session_state") else "simulator"
    base = {
        "simulator": "data/raw_simulator",
        "user": "data/raw_user",
    }.get(mode, "data/raw")
    return _file_version_token([
        f"{base}/customer_summary.csv",
        f"{base}/cohort_retention.csv",
    ])


def _result_data_token() -> str:
    # 모드 인지: 모드별 results 디렉토리 변경 시 토큰 변경 → 캐시 자동 invalidation
    mode = st.session_state.get("data_mode", "simulator") if hasattr(st, "session_state") else "simulator"
    base = {
        "simulator": "results_simulator",
        "user":      "results_user",
    }.get(mode, "results")
    # user 모드에서는 results_user가 아직 비어 있어도 기본 results로 fallback하지 않는다.
    from pathlib import Path as _P
    if mode != "user" and not _P(base).exists():
        base = "results"
    return _file_version_token([
        f"{base}/churn_top10_feature_importance.json",
        f"{base}/optimization_selected_customers.csv",
        f"{base}/personalized_recommendations.csv",
        f"{base}/realtime_scores_snapshot.csv",
        f"{base}/realtime_scores_summary.json",
        f"{base}/realtime_action_queue_snapshot.csv",
        f"{base}/realtime_action_queue_summary.json",
        f"{base}/survival_predictions.csv",
        f"{base}/uplift_segmentation.csv",
        f"{base}/ab_test_results.json",
        f"{base}/dose_response_summary.json",
        f"{base}/customer_segment_summary.json",
        f"{base}/persuadables_analysis.json",
        f"{base}/optimization_summary.json",
        f"{base}/personalized_recommendation_summary.json",
        f"{base}/clv_validation_metrics.json",
        f"{base}/feature_engineering_summary.json",
        f"{base}/churn_metrics.json",
    ])


@st.cache_data(show_spinner=False)
def _load_app_bundle_cached(token: str, data_dir: str = "data/raw"):
    return load_dashboard_bundle(data_dir=data_dir, include_optional=False)


def _resolve_data_dir_for_mode(mode: str) -> str:
    """모드별 data 디렉토리. 부재 시 기본 data/raw로 fallback."""
    from pathlib import Path as _P
    mapping = {
        "simulator": "data/raw_simulator",
        "user":      "data/raw_user",
    }
    target = mapping.get(mode, "data/raw")
    if not (_P(target) / "customer_summary.csv").exists():
        return "data/raw"
    return target


def _resolve_result_dir_for_mode(mode: str) -> str:
    """모드별 results 디렉토리.

    user 모드에서는 results_user가 비어 있어도 기본 results로 떨어지지 않는다.
    그래야 업로드 CSV 화면이 시뮬레이터 결과를 섞어 보여주는 문제를 막을 수 있다.
    """
    from pathlib import Path as _P
    mapping = {
        "simulator": "results_simulator",
        "user":      "results_user",
    }
    target = mapping.get(mode, "results")
    if mode == "user":
        return target
    if not _P(target).exists() or not any(_P(target).iterdir()):
        return "results"
    return target


@st.cache_data(show_spinner=False)
def _load_insight_bundle_cached(raw_token: str, result_token: str, data_dir: str = "data/raw", result_dir: str = "results"):
    return load_dashboard_insight_bundle(data_dir=data_dir, result_dir=result_dir)


def load_app_data():
    mode = st.session_state.get("data_mode", "simulator")
    data_dir = _resolve_data_dir_for_mode(mode)
    return _load_app_bundle_cached(_raw_data_token(), data_dir=data_dir)


def load_insight_data():
    mode = st.session_state.get("data_mode", "simulator")
    data_dir = _resolve_data_dir_for_mode(mode)
    result_dir = _resolve_result_dir_for_mode(mode)
    return _load_insight_bundle_cached(
        _raw_data_token(), _result_data_token(),
        data_dir=data_dir, result_dir=result_dir,
    )


def clear_dashboard_caches() -> None:
    _load_app_bundle_cached.clear()
    _load_insight_bundle_cached.clear()
    # user-live score 전체 조회는 별도 cache를 쓰므로, 업로드/학습/seed 직후 함께 비운다.
    try:
        _fetch_user_live_scores_cached.clear()
    except Exception:
        pass


def load_training_artifacts_api():
    mode = st.session_state.get("data_mode", "simulator")
    if mode == "user":
        artifacts = load_dashboard_artifacts(
            result_dir=_resolve_result_dir_for_mode("user"),
            model_dir="models_user",
            feature_store_dir="data/feature_store_user",
        )
        return {
            "churn_metrics": artifacts.churn_metrics or {},
            "threshold_analysis": artifacts.threshold_analysis or {},
            "top_feature_importance": artifacts.top_feature_importance.to_dict(orient="records"),
            "customer_features": artifacts.customer_features.head(500).to_dict(orient="records"),
            "image_paths": artifacts.image_paths,
            "model_paths": artifacts.model_paths,
            "training_parameters": (artifacts.churn_metrics or {}).get("training_parameters", {}),
            "feature_engineering_summary": artifacts.feature_summary or {},
            "customer_features_metadata": artifacts.customer_features_metadata or {},
        }
    return fetch_training_artifacts()


def load_saved_results_artifacts_api(
    budget: int,
    threshold: float,
    max_customers: int | None,
    rebuild: bool = False,
):
    """
    모드 인지 — 사용자 모드면 results_user/에서 직접 파일 읽기 (API 우회).
    시뮬레이터 모드면 기존 API 호출 (results/는 시뮬레이터 결과로 채워져 있음).
    """
    mode = st.session_state.get("data_mode", "simulator")
    if mode == "user":
        # API는 results/만 보므로, 사용자 모드에선 results_user/를 직접 로드
        return _load_saved_results_from_dir("results_user")
    # 시뮬레이터 모드는 기존 API 호출 그대로 (rebuild 등 동적 옵션 활용 가능)
    try:
        return fetch_saved_results_artifacts(
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
            rebuild=rebuild,
        )
    except Exception:
        # API 호출 실패 시 results_simulator/ 또는 results/에서 직접 로드 fallback
        for d in ("results_simulator", "results"):
            try:
                return _load_saved_results_from_dir(d)
            except Exception:
                continue
        return {}


def _load_saved_results_from_dir(result_dir: str) -> Dict[str, Any]:
    """results/, results_simulator/, results_user/ 같은 디렉토리에서 saved-results 페이로드 구성."""
    from pathlib import Path as _P
    base = _P(result_dir)
    payload: Dict[str, Any] = {"parameters": {}}

    def _load_csv(name: str):
        p = base / name
        if p.exists():
            try:
                return pd.read_csv(p).to_dict(orient="records")
            except Exception:
                return []
        return []

    def _load_json(name: str):
        p = base / name
        if p.exists():
            try:
                import json as _j
                with open(p, "r", encoding="utf-8") as f:
                    return _j.load(f)
            except Exception:
                return {}
        return {}

    payload["uplift_segmentation"] = _load_csv("uplift_segmentation.csv")
    payload["uplift_summary"] = _load_json("uplift_summary.json")
    if not payload["uplift_summary"] and payload["uplift_segmentation"]:
        # uplift_summary.json이 없으면 segmentation에서 요약 만들기
        seg_df = pd.DataFrame(payload["uplift_segmentation"])
        if not seg_df.empty and "uplift_segment" in seg_df.columns:
            payload["uplift_summary"] = {
                "rows": int(len(seg_df)),
                "segment_counts": seg_df["uplift_segment"].value_counts().to_dict(),
            }
    payload["optimization_summary"] = _load_json("optimization_summary.json")
    payload["optimization_segment_budget"] = _load_csv("optimization_segment_budget.csv")
    payload["optimization_selected_customers"] = _load_csv("optimization_selected_customers.csv")
    return payload


def _normalize_artifact_value(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (pd.Timedelta, Path)):
        return str(value)

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (pd.Series, pd.Index)):
        value = value.tolist()

    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return ""
        return numeric

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return value


def _sanitize_artifact_dataframe(df: pd.DataFrame, max_columns: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    clean = df.copy()
    clean.columns = [str(col) for col in clean.columns]

    if max_columns is not None:
        clean = clean.loc[:, list(clean.columns[:max_columns])]

    clean = clean.reset_index(drop=True)

    for column in clean.columns:
        clean[column] = clean[column].map(_normalize_artifact_value)

    return clean


def _artifact_frame(records, max_columns: int | None = None) -> pd.DataFrame:
    return _sanitize_artifact_dataframe(pd.DataFrame(records or []), max_columns=max_columns)


def _describe_table_count(df: pd.DataFrame, label: str = "테이블") -> str:
    rows = int(len(df))
    customers = None
    if isinstance(df, pd.DataFrame) and "customer_id" in df.columns:
        customers = int(df["customer_id"].nunique())

    if customers is not None:
        if rows == customers:
            return f"{label}: 고객 {customers:,}명"
        return f"{label}: 고객 {customers:,}명 / 행 {rows:,}개"
    return f"{label}: 행 {rows:,}개"


def _make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for column in columns:
        base = str(column) if column is not None else "column"
        count = seen.get(base, 0)
        seen[base] = count + 1
        unique.append(base if count == 0 else f"{base}_{count + 1}")
    return unique


def _normalize_table_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return ""
        return numeric
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _sanitize_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    safe_df = df.copy().reset_index(drop=True)
    safe_df.columns = _make_unique_columns([str(col) for col in safe_df.columns])

    for column in safe_df.columns:
        normalized = safe_df[column].map(_normalize_table_cell)
        non_empty = [value for value in normalized.tolist() if value not in ("", None)]
        numeric_only = bool(non_empty) and all(isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)) for value in non_empty)
        if numeric_only:
            safe_df[column] = pd.to_numeric(normalized, errors="coerce")
        else:
            safe_df[column] = normalized.map(lambda value: "" if value is None else str(value))

    return safe_df


def _table_widget_key(label: str, suffix: str) -> str:
    digest = hashlib.md5(f"{label}:{suffix}".encode("utf-8")).hexdigest()[:10]
    return f"table_{suffix}_{digest}"


def _render_html_table(
    df: pd.DataFrame,
    *,
    label: str,
    hide_index: bool = True,
    max_height: int = 520,
    prefer_static: bool = False,
) -> None:
    safe_df = _sanitize_display_dataframe(df)
    st.caption(_describe_table_count(safe_df, label=label))

    if safe_df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    total_rows = int(len(safe_df))
    view_df = safe_df
    show_controls = (not prefer_static) and total_rows > 40
    if show_controls:
        controls = st.columns([1.2, 1.2, 4.6])
        size_key = _table_widget_key(label, "page_size")
        page_key = _table_widget_key(label, "page")
        options = [50, 100, 250, 500, 1000]
        options = [opt for opt in options if opt < total_rows]
        options.append(total_rows if total_rows <= 5000 else 1000)
        options = sorted(set(options))
        default_page_size = 100 if total_rows >= 100 else total_rows
        page_size = controls[0].selectbox(
            "행/페이지",
            options=options,
            index=options.index(default_page_size if default_page_size in options else options[-1]),
            key=size_key,
        )
        total_pages = max(1, math.ceil(total_rows / int(page_size)))
        page = controls[1].number_input(
            "페이지",
            min_value=1,
            max_value=total_pages,
            value=min(st.session_state.get(page_key, 1), total_pages),
            step=1,
            key=page_key,
        )
        start = (int(page) - 1) * int(page_size)
        end = min(start + int(page_size), total_rows)
        controls[2].markdown(
            f"<div class='oai-table-controls'>전체 <b>{total_rows:,}</b>행 중 <b>{start + 1:,}</b>–<b>{end:,}</b>행 표시</div>",
            unsafe_allow_html=True,
        )
        view_df = safe_df.iloc[start:end].copy()

    html_table = view_df.to_html(index=not hide_index, classes=["oai-data-table"], border=0, escape=True)
    st.markdown(
        f"<div class='oai-table-wrapper' style='max-height:{max(220, int(max_height))}px'>{html_table}</div>",
        unsafe_allow_html=True,
    )


def _render_dataframe_with_count(
    df: pd.DataFrame,
    *,
    label: str = "테이블",
    use_container_width: bool = True,
    hide_index: bool = True,
    height: int | None = None,
    prefer_static: bool = False,
) -> None:
    max_height = height if isinstance(height, int) and height > 0 else 520
    _render_html_table(
        df,
        label=label,
        hide_index=hide_index,
        max_height=max_height,
        prefer_static=prefer_static,
    )


def _render_artifact_table(
    df: pd.DataFrame,
    *,
    use_dataframe: bool = False,
    height: int | None = None,
    label: str = "테이블",
) -> None:
    safe_df = _sanitize_artifact_dataframe(df)
    if safe_df.empty:
        return
    _render_dataframe_with_count(
        safe_df,
        label=label,
        hide_index=True,
        height=height if isinstance(height, int) and height > 0 else 520,
        prefer_static=not use_dataframe,
    )




def _payload_hash(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def get_session_cached_summary(
    view_title: str,
    payload_json: str,
    api_key: str,
    model_name: str,
) -> str:
    cache_key = f"summary::{_payload_hash(view_title, payload_json, model_name)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = generate_dashboard_summary(
            view_title=view_title,
            payload_json=payload_json,
            user_api_key=api_key,
            model_name=model_name,
        )
    return st.session_state[cache_key]


def get_session_cached_answer(
    view_title: str,
    payload_json: str,
    question: str,
    api_key: str,
    model_name: str,
) -> str:
    cache_key = f"qa::{_payload_hash(view_title, payload_json, question, model_name)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = answer_dashboard_question(
            view_title=view_title,
            payload_json=payload_json,
            question=question,
            user_api_key=api_key,
            model_name=model_name,
        )
    return st.session_state[cache_key]


def get_chat_history_key(view_key: str) -> str:
    return f"llm_chat_history_{view_key}"


def get_chat_input_key(view_key: str) -> str:
    return f"llm_chat_input_{view_key}"


def resolve_chatbot_image() -> Optional[str]:
    candidates = [
        Path(__file__).resolve().parent / "assets" / "chatbot.png",
        Path(__file__).resolve().parent / "assets" / "chatbot.jpg",
        Path(__file__).resolve().parent / "data" / "chatbot.png",
        Path(__file__).resolve().parent / "data" / "chatbot.jpg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def close_llm_chat_dialog():
    st.session_state["llm_chat_open"] = False
    st.session_state["llm_chat_view_key"] = None


def build_contextual_chat_question(
    view_title: str,
    history: list,
    latest_question: str,
    max_messages: int = 6,
) -> str:
    recent_history = history[-max_messages:] if history else []
    if not recent_history:
        return latest_question

    history_lines = []
    for item in recent_history:
        role = "사용자" if item.get("role") == "user" else "AI"
        content = str(item.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")

    if not history_lines:
        return latest_question

    history_block = "\\n".join(history_lines)
    return (
        f"현재 대시보드 화면: {view_title}\\n"
        "아래는 직전 대화 맥락이다. 반드시 이 맥락을 참고해 이어서 답변하라.\\n\\n"
        f"{history_block}\\n\\n"
        f"현재 질문: {latest_question}"
    )


def render_llm_summary(
    view_key: str,
    view_title: str,
    payload: Dict,
    api_key: Optional[str],
    model_name: str,
):
    st.divider()
    st.subheader("LLM 결과 요약")
    st.caption("현재 화면의 지표·표·그래프에서 추린 요약 컨텍스트만 바탕으로 응답합니다.")

    ready, status_message = get_llm_status(api_key)
    payload_json = build_payload_json(payload)

    if not ready:
        st.info(status_message)
        return

    with st.spinner("AI가 현재 화면의 결과를 요약하는 중입니다..."):
        try:
            summary = get_session_cached_summary(
                view_title=view_title,
                payload_json=payload_json,
                api_key=api_key or "",
                model_name=model_name,
            )
        except Exception as exc:
            st.error(f"AI 요약 생성 중 오류가 발생했습니다: {exc}")
            return

    st.markdown(summary)
    st.caption("추가 질문은 사이드바의 AI 챗봇 버튼을 눌러 이어서 대화할 수 있습니다.")


@st.fragment
def render_sidebar_chatbot_launcher(
    view_key: str,
    view_title: str,
    llm_enabled: bool,
    api_key: Optional[str],
    payload: Optional[Dict] = None,
    model_name: str = "gpt-4.1-mini",
):
    """사이드바에 챗봇 UI를 inline으로 렌더링 (별도 dialog 창 없음)."""
    st.divider()
    st.subheader("🤖 AI 챗봇")

    ready, status_message = get_llm_status(api_key)
    is_open = st.session_state.get("llm_chat_open", False) and \
              st.session_state.get("llm_chat_view_key") == view_key

    # 토글 버튼: 닫혀있으면 "열기", 열려있으면 "닫기"
    btn_label = "❌ 챗봇 닫기" if is_open else "💬 챗봇 열기"
    if st.button(
        btn_label,
        key=f"toggle_chatbot_{view_key}",
        use_container_width=True,
        disabled=(not llm_enabled) or (not ready),
    ):
        st.session_state["llm_chat_open"] = not is_open
        st.session_state["llm_chat_view_key"] = view_key
        st.rerun(scope="fragment")

    if not llm_enabled:
        st.caption("⚠️ LLM 기능이 꺼져 있어 챗봇을 열 수 없습니다.")
    elif not ready:
        st.caption(f"⚠️ {status_message}")
    elif not is_open:
        st.caption(f"📍 현재 화면: **{view_title}**")
        st.caption("화면의 표·그래프를 보면서 질문할 수 있습니다.")

    # ── 챗봇이 열려있을 때 사이드바에 inline 렌더 ──
    if is_open and ready and llm_enabled and payload is not None:
        _render_sidebar_chatbot_inline(
            view_key=view_key,
            view_title=view_title,
            payload=payload,
            api_key=api_key,
            model_name=model_name,
        )


def _render_sidebar_chatbot_inline(
    view_key: str,
    view_title: str,
    payload: Dict,
    api_key: Optional[str],
    model_name: str,
):
    """사이드바 안에 챗봇 대화 UI를 inline으로 표시."""
    payload_json = build_payload_json(payload)
    history_key = get_chat_history_key(view_key)
    input_key = get_chat_input_key(view_key)

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    st.caption(f"📍 컨텍스트: **{view_title}**")

    # 대화 지우기 버튼
    if st.button("🗑 대화 지우기", key=f"clear_sidebar_chat_{view_key}", use_container_width=True):
        st.session_state[history_key] = []
        st.rerun(scope="fragment")

    history = st.session_state[history_key]

    # 대화 내역 (스크롤 가능 컨테이너 — height 제한)
    chat_container = st.container(height=400)
    with chat_container:
        if not history:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    "안녕하세요. 현재 보고 있는 화면 기준으로 답해드릴게요.\n\n"
                    "- 왜 이 지표가 높/낮은지\n"
                    "- 어떤 고객/세그먼트가 핵심인지\n"
                    "- 예산·threshold에서 뭘 바꾸면 좋을지"
                )
        for item in history:
            role = item.get("role", "assistant")
            avatar = "🧑" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(item.get("content", ""))

    # 입력창
    prompt = st.chat_input(
        "현재 화면에 대해 질문하세요...",
        key=f"sidebar_chat_input_{view_key}",
    )

    if prompt:
        history.append({"role": "user", "content": prompt})
        st.session_state[history_key] = history

        contextual_question = build_contextual_chat_question(
            view_title=view_title,
            history=history[:-1],
            latest_question=prompt,
        )

        with st.spinner("AI 답변 생성 중..."):
            try:
                answer = get_session_cached_answer(
                    view_title=view_title,
                    payload_json=payload_json,
                    question=contextual_question,
                    api_key=api_key or "",
                    model_name=model_name,
                )
            except Exception as exc:
                answer = f"AI 답변 생성 중 오류가 발생했습니다: {exc}"

        history.append({"role": "assistant", "content": answer})
        st.session_state[history_key] = history
        st.rerun(scope="fragment")


@st.dialog("AI 분석 챗봇")
def open_chatbot_dialog(
    view_key: str,
    view_title: str,
    payload: Dict,
    api_key: Optional[str],
    model_name: str,
):
    ready, status_message = get_llm_status(api_key)
    payload_json = build_payload_json(payload)
    history_key = get_chat_history_key(view_key)
    input_key = get_chat_input_key(view_key)

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    st.markdown(
        """
        <div id="chatbot-drag-handle" class="chatbot-drag-handle">
            <span>🤖 AI 분석 챗봇</span>
            <small>드래그해서 이동</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="chatbot-dialog-note">
            <strong>현재 화면:</strong> {view_title}<br/>
            현재 화면의 지표·표·그래프 요약 컨텍스트를 바탕으로 답변합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_col1, top_col2 = st.columns([1, 1])
    if top_col1.button("대화 지우기", key=f"clear_chat_{view_key}", use_container_width=True):
        st.session_state[history_key] = []
        st.rerun()
    if top_col2.button("닫기", key=f"close_chat_{view_key}", use_container_width=True):
        close_llm_chat_dialog()
        st.rerun()

    if not ready:
        st.info(status_message)
        return

    history = st.session_state[history_key]

    if not history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "안녕하세요. 현재 보고 있는 대시보드 화면을 기준으로 설명해드릴게요.\n\n"
                "- 왜 이 지표가 높거나 낮은지\n"
                "- 어떤 고객/세그먼트가 핵심인지\n"
                "- 지금 예산·threshold에서 무엇을 바꾸면 좋을지\n"
                "같은 질문을 이어서 해보세요."
            )

    for item in history:
        role = item.get("role", "assistant")
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(item.get("content", ""))

    prompt = st.chat_input(
        "현재 화면에 대해 질문하세요.",
        key=input_key,
    )

    if prompt:
        history.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        contextual_question = build_contextual_chat_question(
            view_title=view_title,
            history=history[:-1],
            latest_question=prompt,
        )

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI가 답변하는 중입니다..."):
                try:
                    answer = get_session_cached_answer(
                        view_title=view_title,
                        payload_json=payload_json,
                        question=contextual_question,
                        api_key=api_key or "",
                        model_name=model_name,
                    )
                except Exception as exc:
                    answer = f"AI 답변 생성 중 오류가 발생했습니다: {exc}"

            st.markdown(answer)

        history.append({"role": "assistant", "content": answer})
        st.session_state[history_key] = history



def inject_draggable_chat_dialog():
    components.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;

          function setupDraggableDialog() {
            const handle = doc.getElementById('chatbot-drag-handle');
            if (!handle) return;

            const dialog = handle.closest('[role="dialog"]');
            if (!dialog) return;
            if (dialog.dataset.dragBound === '1') return;

            dialog.dataset.dragBound = '1';
            dialog.style.position = 'fixed';
            dialog.style.margin = '0';
            dialog.style.transform = 'none';
            dialog.style.right = '24px';
            dialog.style.top = '92px';
            dialog.style.left = 'auto';
            dialog.style.width = 'min(460px, 92vw)';
            dialog.style.maxWidth = '92vw';
            dialog.style.maxHeight = '82vh';
            dialog.style.overflow = 'auto';
            dialog.style.zIndex = '999999';

            let dragging = false;
            let startX = 0;
            let startY = 0;
            let startLeft = 0;
            let startTop = 0;

            function clamp(value, minValue, maxValue) {
              return Math.min(Math.max(value, minValue), maxValue);
            }

            function onMouseMove(event) {
              if (!dragging) return;

              const nextLeft = startLeft + (event.clientX - startX);
              const nextTop = startTop + (event.clientY - startY);
              const maxLeft = Math.max(12, window.parent.innerWidth - dialog.offsetWidth - 12);
              const maxTop = Math.max(12, window.parent.innerHeight - dialog.offsetHeight - 12);

              dialog.style.left = clamp(nextLeft, 12, maxLeft) + 'px';
              dialog.style.top = clamp(nextTop, 12, maxTop) + 'px';
              dialog.style.right = 'auto';
            }

            function onMouseUp() {
              dragging = false;
              doc.removeEventListener('mousemove', onMouseMove);
              doc.removeEventListener('mouseup', onMouseUp);
            }

            handle.addEventListener('mousedown', function(event) {
              if (event.target.closest('button, input, textarea, a, label')) return;

              dragging = true;
              const rect = dialog.getBoundingClientRect();
              startLeft = rect.left;
              startTop = rect.top;
              startX = event.clientX;
              startY = event.clientY;

              dialog.style.left = rect.left + 'px';
              dialog.style.top = rect.top + 'px';
              dialog.style.right = 'auto';

              doc.addEventListener('mousemove', onMouseMove);
              doc.addEventListener('mouseup', onMouseUp);
              event.preventDefault();
            });
          }

          setupDraggableDialog();
          const observer = new MutationObserver(setupDraggableDialog);
          observer.observe(doc.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


inject_custom_css()


CONTROL_DEFAULTS = {
    "control_threshold": 0.50,
    "control_budget": 5_000_000,
    "control_top_n": 25,
    "control_target_cap": 1500,
    "control_recommendation_per_customer": 3,
}
for _state_key, _state_value in CONTROL_DEFAULTS.items():
    st.session_state.setdefault(_state_key, _state_value)

bundle = load_app_data()

customers = bundle.customer_summary
cohort_df = bundle.cohort_retention

render_hero(
    "고객 이탈 예측·개입 최적화·ROI 분석 플랫폼",
    "누가 이탈할 가능성이 높은지뿐 아니라, 언제 개입해야 하는지, 누구에게 예산을 우선 배분할지, " \
    "어떤 액션을 추천할지까지 연결해 보여주는 운영형 리텐션 분석 플랫폼입니다.",
)

if bundle.used_mock:
    render_status_pill("실제 data/raw 산출물을 찾지 못해 mock data로 실행 중입니다.", "warn")
elif bundle.source_dir:
    _current_mode = st.session_state.get("data_mode", "simulator")
    if _current_mode == "user":
        render_status_pill(f"실제 자사 고객 CSV 데이터 사용 중: {bundle.source_dir}", "success")
    else:
        render_status_pill(f"실제 시뮬레이터 산출물 사용 중: {bundle.source_dir}", "success")

with st.sidebar:
    st.header("제어 패널")

    # ── Mode Selection (최상단) ──
    st.subheader("🎯 분석 모드")
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = "simulator"

    mode_choice = st.radio(
        "어떤 데이터로 분석할까요?",
        options=[("simulator", "🧪 시뮬레이터 데모"), ("user", "📂 자사 데이터")],
        format_func=lambda x: x[1] if isinstance(x, tuple) else x,
        index=0 if st.session_state["data_mode"] == "simulator" else 1,
        key="data_mode_radio",
    )
    selected_mode = mode_choice[0] if isinstance(mode_choice, tuple) else mode_choice

    if selected_mode != st.session_state["data_mode"]:
        st.session_state["data_mode"] = selected_mode
        clear_dashboard_caches()
        st.rerun()

    # 현재 모드 안내
    if selected_mode == "simulator":
        st.caption("🟢 시뮬레이터 데이터 사용 중 (`data/raw_simulator/`)")
    else:
        st.caption("🔵 자사 데이터 사용 중 (`data/raw_user/`)")

    # ── Data Upload Section (사용자 모드일 때만) ──
    if selected_mode != "user":
        st.info("ℹ️ 데이터 업로드는 **'📂 자사 데이터' 모드**에서만 가능합니다.")
        # 시뮬레이터 모드에선 업로드 UI 자체를 숨김 (변수 초기화만)
        uploaded_file = None
    else:
        st.subheader("📂 데이터 업로드")
        st.caption("자사 CSV 데이터를 업로드하면 자동으로 전처리 → 모델 학습 → 대시보드 반영까지 수행합니다.")

        uploaded_file = st.file_uploader(
            "CSV 파일을 업로드하세요",
            type=["csv", "tsv"],
            key="csv_upload",
            help="고객 데이터, 거래 데이터, 이벤트 데이터 등 분석 가능한 CSV를 업로드하세요. 파일 크기 제한은 없습니다.",
        )

    if uploaded_file is not None:
        import sys
        from pathlib import Path as _UploadPath

        _project_root_for_upload = _UploadPath(__file__).resolve().parents[1]
        if str(_project_root_for_upload) not in sys.path:
            sys.path.insert(0, str(_project_root_for_upload))

        # 업로드 파일 저장
        upload_dir = _project_root_for_upload / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / uploaded_file.name
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 새 파일이면 이전 매핑 상태 초기화
        if st.session_state.get("upload_path_cached") != str(upload_path):
            st.session_state["upload_path_cached"] = str(upload_path)
            st.session_state.pop("mapping_preview", None)

        # ── Step A: 매핑 미리보기 생성 ──
        if "mapping_preview" not in st.session_state:
            from src.ingestion.pipeline import prepare_mapping_preview as _prepare_preview
            import threading, time as _time

            # 파일 크기 → 예상 소요 시간 (MB당 약 0.4초 + 기본 2초)
            try:
                _file_size_mb = max(upload_path.stat().st_size / (1024 * 1024), 0.1)
            except Exception:
                _file_size_mb = 1.0
            _estimated_seconds = max(_file_size_mb * 0.4 + 2.0, 3.0)

            # 백그라운드 스레드에서 prepare_mapping_preview 실행
            _result_box: dict = {"value": None, "error": None}

            def _worker():
                try:
                    _result_box["value"] = _prepare_preview(upload_path)
                except Exception as _e:
                    _result_box["error"] = _e

            _t = threading.Thread(target=_worker, daemon=True)
            _t.start()

            _progress_bar = st.progress(0, text="🚀 시작 중...")
            _elapsed = 0.0
            while _t.is_alive():
                _time.sleep(0.25)
                _elapsed += 0.25
                _pct = min(int((_elapsed / _estimated_seconds) * 90), 92)
                if _pct < 25:
                    _msg = f"📥 CSV 파일 읽는 중 ({_file_size_mb:.1f} MB)"
                elif _pct < 50:
                    _msg = "🔍 컬럼 자동 감지 중 (역할 매칭)"
                elif _pct < 75:
                    _msg = "📊 이벤트 타입 분포 분석 중 (최대 200,000행 샘플링)"
                else:
                    _msg = "🧮 매핑 결과 정리 중..."
                _progress_bar.progress(_pct, text=f"{_msg} · {_elapsed:.1f}s 경과")

            _t.join()

            if _result_box["error"] is not None:
                _progress_bar.progress(100, text="❌ 오류 발생")
                st.error(f"매핑 미리보기 실패: {_result_box['error']}")
                st.stop()
            else:
                _progress_bar.progress(100, text=f"✅ 매핑 미리보기 완료 ({_elapsed:.1f}s)")
                st.session_state["mapping_preview"] = _result_box["value"]

        preview = st.session_state["mapping_preview"]
        validation_result = preview.validation

        if not validation_result.is_valid:
            for err in validation_result.errors:
                st.error(f"⛔ {err}")
            if validation_result.warnings:
                for warn in validation_result.warnings:
                    st.warning(f"⚠️ {warn}")
        else:
            st.success(
                f"✅ 검증 통과 (관련성: {validation_result.relevance_score:.0%}, "
                f"{validation_result.row_count:,}행 × {validation_result.column_count}열)"
            )
            if validation_result.warnings:
                for warn in validation_result.warnings:
                    st.caption(f"⚠️ {warn}")

            from src.ingestion.preprocessor import (
                INTERNAL_EVENT_TYPES as _STD,
                ROLE_DESCRIPTIONS as _ROLE_DESC,
                EVENT_TYPE_DESCRIPTIONS as _EV_DESC,
            )

            # ── Step B: 컬럼 매핑 검토 + 수정 UI ──
            st.markdown("### 📋 컬럼 매핑")
            st.caption(
                "왼쪽은 **시스템 스키마 칼럼**, 오른쪽은 **자사 CSV 컬럼** 입니다. "
                "오른쪽 셀을 더블클릭하면 매핑 컬럼을 변경할 수 있습니다."
            )

            # 9개 역할 의미 안내 (event_type 매핑과 동일 패턴 — expander로 토글)
            with st.expander("💡 시스템 스키마의 9개 역할이 각각 무엇을 의미하나요?", expanded=False):
                _schema_help_html = "<div style='font-size: 0.82rem; line-height: 1.5;'>"
                for _role_key, _role_desc in _ROLE_DESC.items():
                    _schema_help_html += f"<div><b><code>{_role_key}</code></b> — {_role_desc}</div>"
                _schema_help_html += "<div><b><code>(매핑 안 함)</code></b> — 이 컬럼은 분석에 사용하지 않습니다.</div>"
                _schema_help_html += "</div>"
                st.markdown(_schema_help_html, unsafe_allow_html=True)

            # 사용자 CSV의 모든 컬럼 + 자동 매핑 결과
            all_user_columns = list(preview.validation.column_report and
                [c["original_name"] for c in preview.validation.column_report]
                or list(preview.column_mapping.values()))
            auto_role_to_col = dict[preview, preview](preview.column_mapping)
            user_col_options = ["(매핑 안 함)"] + list(all_user_columns)
        
            cm_rows = []
            for role in _ROLE_DESC.keys():
                detected_col = auto_role_to_col.get(role)
                # 자동 매핑 안 됐으면 "(매핑 안 함)"으로
                cm_rows.append({
                    "시스템 스키마": role,
                    "자사 CSV 컬럼": detected_col if detected_col in all_user_columns else "(매핑 안 함)",
                })
            cm_df = pd.DataFrame(cm_rows)        

            edited_cm = st.data_editor(
                cm_df,
                use_container_width=True,
                hide_index=True,
                disabled=["시스템 스키마"],  # 시스템 스키마는 고정 (사용자가 수정 못 함)
                column_config={
                    "시스템 스키마": st.column_config.TextColumn(
                        "시스템 스키마 (고정)",
                        help="시스템에서 사용하는 표준 역할명 — 변경 불가",
                    ),
                    "자사 CSV 컬럼": st.column_config.SelectboxColumn(
                        "자사 CSV 컬럼 ▼",
                        options=user_col_options,
                        required=True,
                        help="자동 감지된 결과 — 잘못 매핑되었으면 ▼ 클릭해서 변경",
                    ),
                },
                key="column_mapping_editor",
            )

            user_column_mapping_override: dict[str, str] = {}
            for _, _r in edited_cm.iterrows():
                _role = str(_r["시스템 스키마"])
                _col = str(_r["자사 CSV 컬럼"])
                if _col and _col != "(매핑 안 함)":
                    user_column_mapping_override[_role] = _col
            
            # ── Step C: event_type 값 매핑 검토 + 수정 UI ──
            user_event_mapping: dict | None = None
            allow_synthetic_fallback = False  

            if preview.has_event_data:
                st.markdown("### 🔁 event_type 값 매핑")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.caption(
                        f"감지된 event_type 고유값 **{len(preview.event_value_mapping)}개**, "
                        f"자동 매핑 커버리지 **{preview.coverage_rate:.0%}**."
                    )
                with col_b:
                    if preview.coverage_rate >= 0.9:
                        st.markdown("🟢 **매핑 양호**")
                    elif preview.coverage_rate >= 0.7:
                        st.markdown("🟡 **검토 권장**")
                    else:
                        st.markdown("🔴 **수정 필요**")
                        
                with st.expander("💡 내부 표준 값 6종이 각각 무엇을 의미하나요?", expanded=False):
                    _ev_help_html = "<div style='font-size: 0.82rem; line-height: 1.5;'>"
                    for _std, _desc in _EV_DESC.items():
                        _ev_help_html += f"<div><b><code>{_std}</code></b> — {_desc}</div>"
                    _ev_help_html += "</div>"
                    st.markdown(_ev_help_html, unsafe_allow_html=True)

                if preview.unmapped_values:
                    st.warning(
                        f"⚠️ 자동 매핑 실패한 {len(preview.unmapped_values)}개 값: "
                        f"`{', '.join(preview.unmapped_values)}` → 'other'로 분류되었습니다. "
                        "필요시 직접 수정해 주세요."
                    )

                std_options = list(_STD) + ["other", "ignore"]

                editor_rows = []
                for raw, std in sorted(
                    preview.event_value_mapping.items(),
                    key=lambda x: -preview.event_value_counts.get(x[0], 0),
                ):
                    editor_rows.append({
                        "원본 값": raw,
                        "빈도": preview.event_value_counts.get(raw, 0),
                        "내부 표준 값": std,
                    })
                editor_df = pd.DataFrame(editor_rows)

                edited = st.data_editor(
                    editor_df,
                    use_container_width=True,
                    hide_index=True,
                    disabled=["원본 값", "빈도"],
                    column_config={
                        "원본 값": st.column_config.TextColumn(
                            "원본 값",
                            help="당신의 CSV에 있는 event_type 값입니다.",
                        ),
                        "빈도": st.column_config.NumberColumn(
                            "빈도",
                            format="%d",
                            help="해당 값이 데이터에 등장한 횟수 (200,000행 샘플 기준).",
                        ),
                        "내부 표준 값": st.column_config.SelectboxColumn(
                            "내부 표준 값",
                            options=std_options,
                            required=True,
                            help=(
                                "이 원본 값을 어떤 표준 이벤트로 분류할지 선택하세요. "
                                "visit=접속, page_view=조회, search=검색, "
                                "add_to_cart=장바구니, purchase=구매·결제, "
                                "support_contact=문의·환불, other=기타, "
                                "ignore=분석에서 제외."
                            ),
                        ),
                    },
                    key="event_mapping_editor",
                )

                user_event_mapping = dict(zip(edited["원본 값"].astype(str), edited["내부 표준 값"].astype(str)))

                std_dist: dict[str, int] = {}
                for raw, std in user_event_mapping.items():
                    std_dist[std] = std_dist.get(std, 0) + preview.event_value_counts.get(raw, 0)

                if std_dist:
                    st.markdown("**매핑 후 분포 (예상)**")
                    dist_cols = st.columns(min(len(std_dist), 4))
                    sorted_dist = sorted(std_dist.items(), key=lambda x: -x[1])
                    for idx, (k, v) in enumerate(sorted_dist):
                        col = dist_cols[idx % len(dist_cols)]
                        with col:
                            st.metric(label=k, value=f"{v:,}")
            else:
                st.error("⛔ event_type 또는 timestamp 컬럼이 감지되지 않았습니다.")
                st.markdown(
                    """
                    이 경우 시스템은 **합성 이벤트 데이터**로 분석을 진행할 수 있지만,
                    아래 항목은 **신뢰할 수 없습니다**:
                    - 이벤트 시퀀스/세션 분석
                    - 시간대별 행동 패턴
                    - 이벤트 다양성 기반 피처

                    가능하면 **event_type + timestamp 컬럼이 있는 CSV**로 다시 올려주세요.
                    """
                )
                allow_synthetic_fallback = st.checkbox(
                    "그래도 합성 이벤트로 진행 (제한된 분석만 신뢰 가능)",
                    value=False,
                    key="allow_synthetic",
                    help="체크하면 시스템이 가짜 이벤트를 생성해서 학습합니다. 결과 해석에 주의하세요.",
                )

            st.markdown("### ⚙️ 학습 설정")
            col1, col2 = st.columns(2)
            with col1:
                upload_budget = st.number_input("학습 예산", value=5_000_000, step=100000, key="upload_budget")
            with col2:
                upload_threshold = st.slider("학습 이탈 Threshold", 0.10, 0.90, 0.50, 0.01, key="upload_threshold")

            st.markdown("### 📛 이탈 고객 정의")
            st.caption(
                "마지막 활동(이벤트/주문) 이후 며칠 동안 활동이 없으면 \"이탈\"로 분류할지 정합니다. "
                "업종에 따라 적절한 값이 다릅니다."
            )
            churn_inactivity_days = st.slider(
                "이탈 기준: N일 이상 비활성",
                min_value=7, max_value=180,
                value=30, step=1,
                key="churn_inactivity_days",
                help=(
                    "예: 30일 → 이커머스 일반\n"
                    "    60~90일 → 구독 서비스 (월간/분기 결제)\n"
                    "    7~14일 → 일일 사용 앱 (게임/소셜)"
                ),
            )
            st.caption(f"현재 설정: **마지막 활동 {churn_inactivity_days}일 후 이탈**로 간주")

            can_proceed = preview.has_event_data or allow_synthetic_fallback
            btn_label = "✅ 매핑 확정 후 학습 시작" if preview.has_event_data else "⚠️ 합성 이벤트로 진행 (제한 분석)"

            if not can_proceed:
                st.button(btn_label, disabled=True, use_container_width=True, help="event_type/timestamp 컬럼이 없어 진행 불가. 위에서 합성 진행에 동의하면 활성화됩니다.")
            elif st.button(btn_label, key="confirm_and_train", use_container_width=True, type="primary"):
                from src.ingestion.pipeline import run_ingestion_pipeline as _run_pipeline
                import threading
                import time as _time

                progress_bar = st.progress(0, text="시작 중...")
                status_text = st.empty()
                try:
                    _result_holder: dict = {}

                    def _run_pipeline_thread():
                        try:
                            _result_holder["result"] = _run_pipeline(
                                file_path=upload_path,
                                data_dir=_project_root_for_upload / "data" / "raw_user",
                                model_dir=_project_root_for_upload / "models_user",
                                result_dir=_project_root_for_upload / "results_user",
                                feature_store_dir=_project_root_for_upload / "data" / "feature_store_user",
                                budget=int(upload_budget),
                                threshold=float(upload_threshold),
                                column_mapping_override=user_column_mapping_override or None,
                                event_value_mapping=user_event_mapping,
                                allow_synthetic_fallback=allow_synthetic_fallback,
                                churn_inactivity_days=int(churn_inactivity_days),
                            )
                        except Exception as _exc:
                            _result_holder["error"] = _exc

                    _thread = threading.Thread(target=_run_pipeline_thread, daemon=True)
                    _thread.start()


                    _stage_msgs = [
                        f"📥 CSV 읽는 중 ({validation_result.row_count:,}행)…",
                        "🔍 데이터 검증 중…",
                        "⚙️ 컬럼 매핑 적용 중…",
                        "🧮 RFM·이탈 라벨 계산 중…",
                        "🧠 피처 엔지니어링 중…",
                        "🏋️ 이탈 예측 모델 학습 중 (XGBoost)…",
                        "🎯 Uplift 모델 학습 중…",
                        "💰 CLV 모델 학습 중…",
                        "⏳ Survival(이탈 시점) 분석 중…",
                        "📊 세그먼테이션 / A·B 테스트 분석 중…",
                        "📈 예산 최적화 / 추천 생성 중…",
                        "🔬 설명가능성·코호트 분석 중…",
                    ]
                    _start = _time.time()
                    _msg_idx = 0
                    while _thread.is_alive():
                        _elapsed = _time.time() - _start
                        _progress = min(int(95 * (1 - 1 / (1 + _elapsed / 25))), 95)
                        _msg_idx = min(int(_elapsed / 8), len(_stage_msgs) - 1)
                        progress_bar.progress(
                            max(_progress, 3),
                            text=f"{_stage_msgs[_msg_idx]}  ({int(_elapsed)}초 경과)",
                        )
                        status_text.caption(
                            f"⏱️ 전체 단계: 검증 → 전처리 → 피처 → ML 학습 (13단계). 큰 파일은 5~10분 소요."
                        )
                        _time.sleep(0.4)

                    _thread.join()
                    if "error" in _result_holder:
                        raise _result_holder["error"]
                    pipeline_result = _result_holder["result"]

                    if pipeline_result.success:
                        progress_bar.progress(96, text="PostgreSQL user-live 테이블 초기 적재 중...")

                        live_seed_result = None
                        live_seed_error = None
                        try:
                            # 학습 산출물(results_user/models_user/feature_store_user)이 생성된 직후
                            # 이를 PostgreSQL user-live serving table에 자동 적재한다.
                            # 이후 curl로 /api/v1/user-live/events를 호출하면 바로 feature/state/score/action이 갱신된다.
                            live_seed_result = seed_user_live_from_artifacts(reset=True)
                            st.session_state["user_live_seed_result"] = live_seed_result
                            st.session_state.pop("user_live_seed_error", None)
                        except Exception as _seed_exc:
                            live_seed_error = _seed_exc
                            st.session_state["user_live_seed_error"] = str(_seed_exc)

                        progress_bar.progress(100, text="완료!")
                        if isinstance(live_seed_result, dict) and live_seed_result.get("success"):
                            st.success(
                                "🎉 전처리, 모델 학습, user-live DB 초기 적재가 완료되었습니다! "
                                "이제 터미널에서 curl 이벤트를 주입하면 실시간 운영 모니터에 반영됩니다."
                            )
                        else:
                            st.success("🎉 전처리 및 모델 학습이 완료되었습니다! 대시보드가 자동으로 새로고침됩니다.")
                            st.warning(
                                "PostgreSQL user-live DB 자동 적재는 실패했습니다. "
                                "시연 전 RETENTION_USER_DB_URL, PostgreSQL 실행 상태, API 로그를 확인하세요. "
                                "필요하면 터미널에서 seed-from-user-artifacts를 수동 호출하면 됩니다."
                            )
                            if live_seed_error is not None:
                                st.caption(f"seed 오류: {live_seed_error}")

                        if pipeline_result.preprocessing:
                            meta = pipeline_result.preprocessing.metadata or {}
                            ev_source = meta.get("events_source")
                            ev_mapping = meta.get("event_type_mapping") or {}
                            id_type = meta.get("customer_id_type", "numeric")

                            badge_cols = st.columns(3)
                            with badge_cols[0]:
                                if ev_source == "user_upload":
                                    st.success("🟢 **실제 데이터**\n\nevents 테이블이 사용자 업로드 기반")
                                elif ev_source == "synthetic":
                                    st.warning("🟡 **합성 데이터**\n\nevents 테이블이 가짜로 생성됨")
                            with badge_cols[1]:
                                if ev_mapping:
                                    src = ev_mapping.get("mapping_source", "auto")
                                    cov = ev_mapping.get("coverage_rate", 0)
                                    label = "수동 매핑" if src == "manual" else "자동 매핑"
                                    st.info(f"🔁 **{label}**\n\n커버리지 {cov:.0%}")
                                else:
                                    st.info("🔁 **매핑 없음**\n\nevent_type 컬럼 부재")
                            with badge_cols[2]:
                                if id_type == "string_factorized":
                                    st.info(f"🔑 **문자열 ID 변환**\n\n{meta.get('customer_id_unique_count', 0):,}명")
                                else:
                                    st.info("🔑 **수치 ID**\n\n원본 그대로 사용")

                        if pipeline_result.training:
                            completed = pipeline_result.training.stages_completed
                            failed = pipeline_result.training.stages_failed
                            st.caption(f"완료: {len(completed)}개 단계 / 실패: {len(failed)}개 단계")
                            if failed:
                                with st.expander("실패 단계 상세"):
                                    for stage, err in failed.items():
                                        st.text(f"  {stage}: {err[:100]}")

                        st.session_state.pop("mapping_preview", None)
                        clear_dashboard_caches()
                        st.rerun()
                    else:
                        progress_bar.progress(100, text="일부 실패")
                        st.warning(f"⚠️ 파이프라인이 부분적으로 완료되었습니다: {pipeline_result.error or '일부 단계 실패'}")
                        if pipeline_result.training and pipeline_result.training.stages_completed:
                            st.caption(f"완료된 단계: {', '.join(pipeline_result.training.stages_completed)}")
                        clear_dashboard_caches()
                        st.rerun()

                except Exception as exc:
                    progress_bar.progress(100, text="오류 발생")
                    st.error(f"파이프라인 실행 중 오류: {exc}")

    st.session_state.setdefault("dashboard_view", DASHBOARD_VIEW_OPTIONS[0])
    st.session_state["dashboard_view"] = LEGACY_VIEW_REDIRECTS.get(
        st.session_state.get("dashboard_view", DASHBOARD_VIEW_OPTIONS[0]),
        st.session_state.get("dashboard_view", DASHBOARD_VIEW_OPTIONS[0]),
    )
    if st.session_state["dashboard_view"] not in DASHBOARD_VIEW_OPTIONS:
        st.session_state["dashboard_view"] = DASHBOARD_VIEW_OPTIONS[0]

    # 중요: 분석 분야(dashboard_group)를 매 실행마다 현재 세부 화면(dashboard_view)으로
    # 다시 덮어쓰면, 사용자가 다른 대분류를 클릭해도 직전 세부 화면의 그룹
    # 예: "1. 이탈현황" -> "고객 현황"으로 즉시 되돌아간다.
    # 따라서 group은 독립 상태로 유지하고, 선택한 group 안에 현재 view가 없을 때만
    # 해당 group의 첫 세부 화면으로 이동시킨다.
    default_group = VIEW_TO_GROUP.get(st.session_state["dashboard_view"], DASHBOARD_VIEW_GROUPS[0][0])
    st.session_state.setdefault("dashboard_group", default_group)
    if st.session_state["dashboard_group"] not in GROUP_TO_VIEW_OPTIONS:
        st.session_state["dashboard_group"] = default_group

    current_group_options = GROUP_TO_VIEW_OPTIONS.get(st.session_state["dashboard_group"], DASHBOARD_VIEW_OPTIONS)
    if st.session_state["dashboard_view"] not in current_group_options:
        st.session_state["dashboard_view"] = current_group_options[0]

    st.session_state.setdefault("control_threshold", 0.50)
    st.session_state.setdefault("control_budget", 5_000_000)
    st.session_state.setdefault("control_budget_text", str(st.session_state["control_budget"]))
    st.session_state.setdefault("control_top_n", 25)
    st.session_state.setdefault("control_target_cap", 1500)
    st.session_state.setdefault("control_recommendation_per_customer", 3)

group_labels = [group for group, _ in DASHBOARD_VIEW_GROUPS]

_group_icons = {
    "고객 현황": "📊",
    "타겟팅·예산": "🎯",
    "운영·리스크": "⚡",
    "모델 검증·진단": "🧪",
}
_group_label_with_icon = lambda g: f"{_group_icons.get(g, '')} {g}"

# 대분류 라디오 (가로)
selected_group = st.radio(
    "🗂 분석 분야",
    options=group_labels,
    format_func=_group_label_with_icon,
    horizontal=True,
    key="dashboard_group",
)

group_options = list(GROUP_TO_VIEW_OPTIONS.get(selected_group, DASHBOARD_VIEW_OPTIONS))
if st.session_state.get("dashboard_view") not in group_options:
    st.session_state["dashboard_view"] = group_options[0]

view = st.radio(
    f"📌 세부 화면 ({_group_label_with_icon(selected_group)})",
    options=group_options,
    format_func=_view_title_from_option,
    horizontal=True,
    key="dashboard_view",
)

with st.sidebar:
    st.divider()
    st.markdown("#### ⚙️ 분석 컨트롤")

    # 모든 세부 화면에서 같은 widget key를 항상 렌더링한다.
    # 이렇게 해야 1번에서 바꾼 threshold/예산이 3번으로 가도 유지되고,
    # 3번에서 다시 바꾼 값도 4번·5번 등 다른 화면에서 그대로 이어진다.
    threshold = st.slider(
        "이탈 Threshold",
        min_value=0.10,
        max_value=0.90,
        step=0.01,
        key="control_threshold",
        help="이 값 이상인 고객을 이탈 위험군으로 간주합니다. 모든 화면에서 동일하게 유지됩니다.",
    )

    budget_raw = st.text_input(
        "총 마케팅 예산",
        key="control_budget_text",
        help="상한 없이 입력 가능합니다. 쉼표 없이 숫자만 입력해도 됩니다.",
    )

    try:
        budget = parse_unlimited_nonnegative_int(
        budget_raw,
        default=int(st.session_state.get("control_budget", 5_000_000)),
    )
        st.session_state["control_budget"] = budget
    except ValueError:
        st.warning("총 마케팅 예산은 0 이상의 정수로 입력해야 합니다.")
        budget = int(st.session_state.get("control_budget", 5_000_000))
    
    target_cap = st.slider(
        "최대 타겟 고객 수",
        min_value=100,
        max_value=5000,
        step=100,
        key="control_target_cap",
        help="예산이 충분하더라도 이 수를 넘겨 타겟팅하지 않습니다. 모든 화면에서 동일하게 유지됩니다.",
    )

    # top_n은 실시간/설명가능성/리스크 화면에서 주로 쓰지만,
    # 화면 이동 시 값이 사라지지 않도록 항상 렌더링한다.
    top_n = st.slider(
        "차트 기준 표시 고객 수",
        min_value=5,
        max_value=200,
        step=5,
        key="control_top_n",
    )

    if view == "5. 개인화 추천":
        st.caption("최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.")
        recommendation_per_customer = st.slider(
            "고객당 추천 개수",
            min_value=1,
            max_value=5,
            step=1,
            key="control_recommendation_per_customer",
        )
    else:
        recommendation_per_customer = int(st.session_state["control_recommendation_per_customer"])

    preview_selected_customers, _, _ = get_budget_result(
        customers,
        budget=budget,
        threshold=threshold,
        max_customers=target_cap,
    )
    st.caption(
        f"현재 공통 조건: threshold={float(threshold):.2f} / "
        f"예산={int(budget):,}원 / 최종 타겟 고객 수={int(len(preview_selected_customers)):,}명"
    )

with st.sidebar:
    st.divider()
    st.subheader("실행 / 새로고침")
    if notice := st.session_state.pop("dashboard_refresh_notice", None):
        st.success(notice)
    if warning := st.session_state.pop("dashboard_refresh_warning", None):
        st.warning(warning)

    if st.button("데이터/결과 새로고침", use_container_width=True):
        refresh_notice = None
        refresh_warning = None
        if view in REALTIME_REFRESH_VIEWS:
            try:
                tick_payload = advance_realtime_stream(batch_size=250, top_n=max(int(top_n), 50), reset_when_exhausted=True)
                tick_summary = tick_payload.get("summary", {}) if isinstance(tick_payload, dict) else {}
                refresh_notice = (
                    f"실시간 스트림을 {int(tick_summary.get('last_tick_advanced', 0) or 0):,}건 전진했습니다. "
                    f"누적 처리 이벤트 수: {int(tick_summary.get('processed_events', 0) or 0):,}건"
                )
            except Exception as exc:
                refresh_warning = f"실시간 tick 호출에는 실패했지만 화면 캐시는 새로고침했습니다: {exc}"
        clear_dashboard_caches()
        if refresh_notice:
            st.session_state["dashboard_refresh_notice"] = refresh_notice
        if refresh_warning:
            st.session_state["dashboard_refresh_warning"] = refresh_warning
        st.rerun()

    st.caption("실시간 화면에서는 새로고침 시 스트림을 조금씩 더 재생해 수치가 변하도록 했습니다. 나머지 화면은 캐시를 비우고 다시 계산합니다.")

    st.divider()
    st.subheader("LLM 설정")
    st.caption("권장: API 키는 코드에 쓰지 말고 환경변수 OPENAI_API_KEY 또는 Streamlit secrets로 관리하세요.")

    llm_enabled = st.toggle("LLM 요약/질문 기능 사용", value=True)
    llm_api_key = st.text_input(
        "OpenAI API Key (선택)",
        type="password",
        help="비워두면 OPENAI_API_KEY 환경변수를 사용합니다.",
    )
    st.caption("모델이 목록에 없으면 '직접 입력'을 선택해서 모델명을 넣어주세요.")
    _llm_presets = [
        ("GPT-4.1 mini (default)", DEFAULT_MODEL_NAME),
        ("GPT-4.1", "gpt-4.1"),
        ("GPT-4o mini", "gpt-4o-mini"),
        ("GPT-4o", "gpt-4o"),
        ("o4-mini (reasoning)", "o4-mini"),
        ("o3-mini (reasoning)", "o3-mini"),
        ("직접 입력", "__custom__"),
    ]
    _llm_preset_labels = [label for label, _ in _llm_presets]
    _llm_preset_models = {label: model for label, model in _llm_presets}
    _default_label = next((label for label, model in _llm_presets if model == DEFAULT_MODEL_NAME), _llm_presets[0][0])
    llm_model_choice = st.selectbox("LLM 모델 선택", options=_llm_preset_labels, index=_llm_preset_labels.index(_default_label))
    _chosen_model = _llm_preset_models.get(llm_model_choice, DEFAULT_MODEL_NAME)
    if _chosen_model == "__custom__":
        llm_model = st.text_input("LLM 모델명 (직접 입력)", value=DEFAULT_MODEL_NAME)
    else:
        llm_model = _chosen_model

    env_key_configured = bool(os.getenv("OPENAI_API_KEY"))
    if env_key_configured and not llm_api_key:
        st.caption("현재 OPENAI_API_KEY 환경변수를 사용하도록 설정되어 있습니다.")

live_payload = _load_user_live_tables(
    top_n=int(top_n),
    target_cap=int(target_cap),
)

if _is_user_live_mode():
    _render_user_live_status(live_payload)

if _is_user_live_mode() and not live_payload.get("scores", pd.DataFrame()).empty:
    customers = _rename_live_score_columns(live_payload["scores"])

churn_summary, risk_customers = get_churn_status(customers, threshold)
cohort_curve = get_cohort_curve(cohort_df)
top_customers = get_top_high_value_customers(customers, top_n=None)

if _is_user_live_mode() and not live_payload.get("actions", pd.DataFrame()).empty:
    selected_customers, optimize_summary, segment_allocation = _build_live_optimize_payload(
        live_payload["actions"],
        budget=budget,
        threshold=threshold,
        max_customers=target_cap,
        scores_df=live_payload.get("scores", pd.DataFrame()),
    )
else:
    selected_customers, optimize_summary, segment_allocation = get_budget_result(
        customers,
        budget=budget,
        threshold=threshold,
        max_customers=target_cap,
    )

# 외부 CSV/user 결과에서는 일부 정렬·표시 컬럼이 없을 수 있으므로
# 모든 downstream 화면이 같은 스키마를 보도록 즉시 보정한다.
selected_customers = _ensure_retention_target_schema(selected_customers)

baseline_selected_customers, baseline_optimize_summary, baseline_segment_allocation = pd.DataFrame(), {}, pd.DataFrame()

retention_targets = get_retention_targets(customers, threshold)

if view == "5. 개인화 추천":
    if _is_user_live_mode():
        recommendation_summary, personalized_recommendations = _build_dynamic_user_recommendations(
            selected_customers,
            optimize_summary,
            per_customer=recommendation_per_customer,
            budget=budget,
            threshold=threshold,
            max_customers=max(int(target_cap), 1),
        )
        recommendation_error = recommendation_summary.get("error") if isinstance(recommendation_summary, dict) else None
    else:
        try:
            recommendation_limit = max(int(len(selected_customers)), int(target_cap), 1)
            recommendation_summary, personalized_recommendations = fetch_personalized_recommendations(
                limit=recommendation_limit,
                per_customer=recommendation_per_customer,
                budget=budget,
                threshold=threshold,
                max_customers=max(recommendation_limit, int(target_cap)),
                rebuild=True,
            )
        except Exception as exc:
            recommendation_summary, personalized_recommendations = {}, pd.DataFrame()
            recommendation_error = str(exc)
        else:
            recommendation_error = None
else:
    recommendation_summary, personalized_recommendations = {}, pd.DataFrame()
    recommendation_error = None

if view == "6. 실시간 운영 모니터":
    if _is_user_live_mode():
        realtime_scores = _live_scores_to_realtime_df(
            live_payload.get("scores", pd.DataFrame()),
            live_payload.get("actions", pd.DataFrame()),
        )
        score_summary = live_payload.get("score_summary", {}) or {}
        action_summary = live_payload.get("action_summary", {}) or {}
        health_summary = live_payload.get("health", {}) or {}
        realtime_summary = {
            "tracked_customers": int(score_summary.get("scored_customers") or len(realtime_scores)),
            "high_risk_customers": int(score_summary.get("high_risk_customers") or 0),
            "critical_risk_customers": int((realtime_scores.get("realtime_churn_score", pd.Series(dtype=float)) >= 0.85).sum()) if not realtime_scores.empty else 0,
            "triggered_reoptimizations": int(action_summary.get("live_actions") or 0),
            "action_queue_size": int(action_summary.get("queued_actions") or 0),
            "queued_actions_total": int(action_summary.get("queued_actions") or 0),
            "processed_events": int(health_summary.get("processed_event_count", health_summary.get("event_count", 0)) or 0),
            "closed_loop_budget_spent": float(optimize_summary.get("spent", 0.0) or 0.0),
            "daily_channel_allocated": int(action_summary.get("queued_actions") or 0),
            "daily_channel_capacity": max(int(target_cap), 1),
            "high_priority_queue_size": int(action_summary.get("queued_actions") or 0),
        }
        realtime_error = live_payload.get("score_summary", {}).get("error") if isinstance(live_payload.get("score_summary"), dict) else None
    else:
        try:
            realtime_summary, realtime_scores = fetch_realtime_scores(limit=max(int(top_n), 500))
        except Exception as exc:
            realtime_summary, realtime_scores = {}, pd.DataFrame()
            realtime_error = str(exc)
        else:
            realtime_error = None
else:
    realtime_summary, realtime_scores = {}, pd.DataFrame()
    realtime_error = None

if view == "9. 이탈 시점 예측 (Survival Analysis)":
    if st.session_state.get("data_mode", "simulator") == "user":
        _mode_result_dir = Path(_resolve_result_dir_for_mode("user"))
        _bundle = load_insight_data()
        survival_metrics = {}
        _metrics_path = _mode_result_dir / "survival_metrics.json"
        if _metrics_path.exists():
            try:
                survival_metrics = json.loads(_metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                survival_metrics = {}
        survival_predictions = _bundle.survival_predictions.copy().head(int(top_n))
        _coef_path = _mode_result_dir / "survival_top_coefficients.csv"
        survival_coefficients = pd.read_csv(_coef_path) if _coef_path.exists() else pd.DataFrame()
        survival_image_paths = {
            "risk_stratification": str(_mode_result_dir / "survival_risk_stratification.png")
            if (_mode_result_dir / "survival_risk_stratification.png").exists() else None
        }
        survival_error = None
    else:
        try:
            survival_metrics, survival_predictions, survival_coefficients, survival_image_paths = fetch_survival_summary(limit=top_n)
        except Exception as exc:
            survival_metrics, survival_predictions, survival_coefficients, survival_image_paths = {}, pd.DataFrame(), pd.DataFrame(), {}
            survival_error = str(exc)
        else:
            survival_error = None
else:
    survival_metrics, survival_predictions, survival_coefficients, survival_image_paths = {}, pd.DataFrame(), pd.DataFrame(), {}
    survival_error = None

recommendation_context_df = personalized_recommendations.copy()
survival_context_df = survival_predictions.copy()
realtime_context_df = realtime_scores.copy()

insight_bundle = None
global_feature_table = pd.DataFrame()
operational_overview: dict[str, Any] = {}
experiment_overview: dict[str, Any] = {}
realtime_monitor_overview: dict[str, Any] = {}
coupon_risk_overview: dict[str, Any] = {}
data_diagnostics: dict[str, Any] = {}
customer_explanations = pd.DataFrame()

if view in INSIGHT_HEAVY_VIEWS and not (view == "6. 실시간 운영 모니터" and _is_user_live_mode()):
    insight_bundle = load_insight_data()
    if recommendation_context_df.empty:
        recommendation_context_df = insight_bundle.personalized_recommendations.copy()
    if survival_context_df.empty:
        survival_context_df = insight_bundle.survival_predictions.copy()
    if realtime_context_df.empty:
        realtime_context_df = insight_bundle.realtime_scores.copy()

    if view in {"11. 설명가능성 / 고객별 개입 이유"}:
        operational_overview = build_operational_overview(
            customers=customers,
            selected_customers=selected_customers,
            optimize_summary=optimize_summary,
            recommendation_summary=recommendation_summary,
            realtime_summary=realtime_summary,
            survival_metrics=survival_metrics,
            insight_bundle=insight_bundle,
        )

    if view == "10. 증분 성과 / A-B 실험":
        experiment_overview = build_experiment_overview(insight_bundle)

    if view == "6. 실시간 운영 모니터":
        realtime_monitor_overview = build_realtime_monitor_overview(insight_bundle, fallback_scores=realtime_context_df)

    if view == "12. 데이터 진단 / 시뮬레이터 충실도":
        data_diagnostics = build_data_diagnostics(insight_bundle)

    if view == "7. 할인·쿠폰 운영 리스크":
        coupon_risk_overview = build_coupon_risk_overview(insight_bundle)

    if view == "11. 설명가능성 / 고객별 개입 이유":
        global_feature_table = build_global_feature_table(insight_bundle)
        explanation_limit = max(int(len(selected_customers)) if not selected_customers.empty else int(len(insight_bundle.optimization_selected_customers)), int(top_n), 1)
        customer_explanations = build_customer_explanations(
            customers=customers,
            selected_customers=selected_customers if not selected_customers.empty else insight_bundle.optimization_selected_customers,
            recommendation_df=recommendation_context_df,
            survival_predictions=survival_context_df,
            realtime_scores=realtime_context_df,
            top_n=explanation_limit,
        )

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 고객 수", f"{churn_summary['total_customers']:,}")
c2.metric("이탈 위험 고객 수", f"{churn_summary['at_risk_customers']:,}")
c3.metric("위험 고객 비율", pct(churn_summary["risk_rate"]))
c4.metric("평균 이탈 확률", pct(churn_summary["avg_churn_prob"]))

st.divider()

llm_view_title = view
llm_payload: Dict = {}
llm_api_key_value = llm_api_key.strip() if llm_api_key else None

if view == "1. 이탈현황":
    _churn_has_data = (
        isinstance(customers, pd.DataFrame)
        and not customers.empty
        and all(col in customers.columns for col in ["customer_id", "churn_probability"])
    )
    if _simulator_mode_unavailable(
        "이탈현황",
        _churn_has_data,
        "고객 요약 또는 churn score 산출물이 없습니다.",
        "시뮬레이터 데모에서는 python src/main.py --mode simulate --force --randomize → features → train 순서로 실행한 뒤 새로고침하세요.",
    ):
        st.stop()
    st.subheader("이탈현황")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        hist_fig = px.histogram(
            customers,
            x="churn_probability",
            nbins=30,
            title="고객별 이탈 확률 분포",
        )
        hist_fig.update_traces(
            marker_line_color="rgba(255,255,255,0.95)",
            marker_line_width=1.2,
            opacity=0.9,
        )

        hist_fig.update_layout(
            bargap=0.02,
        )

        hist_fig.add_vline(
            x=threshold,
            line_dash="dash",
            annotation_text=f"Threshold={threshold:.2f}",
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        persona_risk = (
            risk_customers.groupby("persona", as_index=False)
            .agg(at_risk_count=("customer_id", "count"))
            .sort_values("at_risk_count", ascending=False)
        )

        bar_fig = px.bar(
            persona_risk,
            x="persona",
            y="at_risk_count",
            title="페르소나별 이탈 위험 고객 수",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("### 이탈 위험 고객 목록")
    display_df = risk_customers[
        ["customer_id", "persona", "churn_probability", "clv", "uplift_score", "uplift_segment"]
    ].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
    display_df["clv"] = display_df["clv"].map(money)
    display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
    _render_dataframe_with_count(display_df, label="이탈 위험 고객 목록")

    llm_payload = {
        "threshold": threshold,
        "kpis": churn_summary,
        "all_customer_numeric_summary": numeric_summary(
            customers, ["churn_probability", "uplift_score", "clv", "expected_roi"]
        ),
        "persona_risk_counts": persona_risk.to_dict(orient="records"),
        "top_risk_customers": dataframe_snapshot(
            risk_customers,
            columns=[
                "customer_id",
                "persona",
                "churn_probability",
                "clv",
                "uplift_score",
                "uplift_segment",
            ],
            max_rows=min(top_n, 12),
        ),
    }

elif view == "2. 코호트 리텐션 곡선":
    _cohort_has_data = isinstance(cohort_df, pd.DataFrame) and not cohort_df.empty
    if _simulator_mode_unavailable(
        "코호트 리텐션 곡선",
        _cohort_has_data,
        "코호트 리텐션 입력 데이터가 없습니다.",
        "시뮬레이터 데모에서는 simulate 결과를 생성한 뒤 코호트 관련 산출물을 준비하고 새로고침하세요.",
    ):
        st.stop()
    st.subheader("코호트 리텐션 분석")

    activity_options = get_available_activity_definitions(cohort_df)
    retention_mode_options = get_available_retention_modes(cohort_df)

    c1, c2 = st.columns(2)
    selected_activity_definition = c1.selectbox(
        "리텐션 활동 정의",
        options=activity_options,
        index=activity_options.index("core_engagement") if "core_engagement" in activity_options else 0,
        format_func=get_activity_definition_label,
        key="cohort_activity_definition",
    )
    selected_retention_mode = c2.selectbox(
        "리텐션 측정 방식",
        options=retention_mode_options,
        index=retention_mode_options.index("rolling") if "rolling" in retention_mode_options else 0,
        format_func=get_retention_mode_label,
        key="cohort_retention_mode",
    )

    cohort_curve = get_cohort_curve(
        cohort_df,
        activity_definition=selected_activity_definition,
        retention_mode=selected_retention_mode,
    )
    cohort_summary = get_cohort_summary(
        cohort_df,
        activity_definition=selected_activity_definition,
        retention_mode=selected_retention_mode,
    )
    display_table = get_cohort_display_table(
        cohort_df,
        activity_definition=selected_activity_definition,
        retention_mode=selected_retention_mode,
    )
    heatmap_df = get_cohort_pivot(
        cohort_df,
        activity_definition=selected_activity_definition,
        retention_mode=selected_retention_mode,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("코호트 수", f"{cohort_summary['cohort_count']:,}")
    avg_size = cohort_summary["avg_cohort_size"]
    m2.metric("평균 코호트 크기", "-" if pd.isna(avg_size) else f"{avg_size:,.0f}")
    month1_ret = cohort_summary["month1_avg_retention"]
    m3.metric("평균 1개월차 리텐션", "-" if pd.isna(month1_ret) else f"{month1_ret:.2%}")
    comparable_ret = cohort_summary["comparable_avg_retention"]
    comparable_period = cohort_summary["comparable_period"]
    comparable_label = "공통 비교 리텐션"
    if comparable_period is not None:
        comparable_label = f"공통 비교({comparable_period}개월차)"
    m4.metric(comparable_label, "-" if pd.isna(comparable_ret) else f"{comparable_ret:.2%}")

    st.caption(
        f"현재 기준: {cohort_summary['selected_activity_label']} / {cohort_summary['selected_retention_mode_label']}. "
        "period 0은 코호트 정의상 100%로 고정하고, 아직 관측할 수 없는 미래 period는 0이 아니라 공란으로 둡니다."
    )

    if selected_retention_mode == "point":
        st.info(
            "해당 월 재방문율(point)은 재활성화 고객 때문에 month 2가 month 1보다 높아질 수 있습니다. "
            "최근/오래된 코호트를 섞어 해석하지 않도록 아래 공통 비교 지표를 함께 보세요."
        )
    else:
        st.info(
            "롤링 리텐션(rolling)은 해당 월 또는 그 이후에 다시 살아난 고객까지 포함하므로 곡선이 단조 감소합니다. "
            "코호트 붕괴 속도를 비교하기에 더 안정적입니다."
        )

    if cohort_summary.get("non_monotonic_cohort_count", 0) > 0:
        st.caption(
            f"참고: 현재 point 기준에서는 {cohort_summary['non_monotonic_cohort_count']}개 코호트에서 "
            "후행 월 리텐션이 앞선 월보다 높게 나타났습니다."
        )

    if cohort_curve.empty:
        st.warning("표시할 코호트 데이터가 없습니다.")
        comparable_df = cohort_curve.copy()
        last_period_df = cohort_curve.copy()
    else:
        line_fig = px.line(
            cohort_curve,
            x="period",
            y="retention_rate",
            color="cohort_month",
            markers=True,
            title=(
                f"가입 코호트별 리텐션 곡선 · "
                f"{get_activity_definition_label(selected_activity_definition)} / {get_retention_mode_label(selected_retention_mode)}"
            ),
        )
        line_fig.update_layout(xaxis_title="경과 기간(개월)", yaxis_title="Retention Rate")
        st.plotly_chart(line_fig, use_container_width=True)

        if not heatmap_df.empty:
            heatmap_fig = px.imshow(
                heatmap_df,
                text_auto=".0%",
                aspect="auto",
                labels={"x": "경과 기간(개월)", "y": "코호트", "color": "Retention"},
                title="코호트 리텐션 히트맵",
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown("### 코호트 리텐션 테이블")
        _render_dataframe_with_count(display_table, label="코호트 리텐션 테이블")

        comparable_df = cohort_curve.copy()
        if comparable_period is not None:
            comparable_df = cohort_curve[cohort_curve["period"] == comparable_period].copy()

        if not comparable_df.empty:
            st.markdown("### 공통 기간 비교")
            comparable_display = comparable_df[
                ["cohort_month", "period", "cohort_size", "retained_customers", "retention_rate"]
            ].copy()
            comparable_display["retention_rate"] = comparable_display["retention_rate"].map(lambda x: f"{x:.2%}")
            _render_dataframe_with_count(
                comparable_display.sort_values("retention_rate", ascending=False),
                label="공통 기간 비교 테이블",
            )

        last_period_df = (
            cohort_curve.sort_values(["cohort_month", "period"])
            .groupby("cohort_month", as_index=False)
            .tail(1)
            .sort_values("retention_rate", ascending=False)
            .reset_index(drop=True)
        )

    llm_payload = {
        "cohort_summary": cohort_summary,
        "selected_activity_definition": selected_activity_definition,
        "selected_retention_mode": selected_retention_mode,
        "retention_curve_summary": numeric_summary(cohort_curve, ["retention_rate"]),
        "cohort_retention_records": cohort_curve.round(4).to_dict(orient="records"),
        "comparable_retention": comparable_df.round(4).to_dict(orient="records"),
        "last_observed_retention": last_period_df.round(4).to_dict(orient="records"),
    }

elif view == "3. Uplift·CLV 세그먼트 분석":
    if _user_mode_unavailable("Uplift Score + CLV 상위 고객 분석", "외부 자사 데이터에는 Treatment/Control 배정 정보가 없어 Uplift Score 계산이 불가합니다."):
        st.stop()
    _uplift_has_data = (
        isinstance(top_customers, pd.DataFrame)
        and not top_customers.empty
        and all(col in top_customers.columns for col in ["customer_id", "uplift_score", "clv"])
    ) or (
        isinstance(customers, pd.DataFrame)
        and not customers.empty
        and all(col in customers.columns for col in ["customer_id", "uplift_score", "clv"])
    )
    if _simulator_mode_unavailable(
        "Uplift·CLV 세그먼트 분석",
        _uplift_has_data,
        "Uplift/CLV 세그먼트 산출물이 없습니다.",
        "시뮬레이터 데모에서는 python src/main.py --mode uplift → clv → segment 순서의 산출물을 먼저 생성하세요.",
    ):
        st.stop()
    st.subheader("Uplift·CLV 세그먼트 분석")


    segment_dist = (
        customers.groupby("uplift_segment", as_index=False)
        .agg(
            customer_count=("customer_id", "nunique"),
            avg_uplift=("uplift_score", "mean"),
            avg_clv=("clv", "mean"),
            avg_expected_profit=("expected_incremental_profit", "mean"),
        )
        .sort_values("customer_count", ascending=False)
    ) if "uplift_segment" in customers.columns else pd.DataFrame()

    if not segment_dist.empty:
        seg_fig = px.bar(
            segment_dist,
            x="uplift_segment",
            y="customer_count",
            text="customer_count",
            hover_data=["avg_uplift", "avg_clv", "avg_expected_profit"],
            title="Uplift 세그먼트별 고객 수",
        )
        st.plotly_chart(seg_fig, use_container_width=True)

        segment_display = segment_dist.copy()
        for col in ["avg_clv", "avg_expected_profit"]:
            if col in segment_display.columns:
                segment_display[col] = segment_display[col].map(money)
        if "avg_uplift" in segment_display.columns:
            segment_display["avg_uplift"] = segment_display["avg_uplift"].map(lambda x: f"{float(x):.3f}")
        _render_dataframe_with_count(segment_display, label="Uplift 세그먼트 요약", prefer_static=True)

    plot_df = top_customers.head(min(len(top_customers), 500)).copy()
    plot_df["customer_label"] = plot_df["customer_id"].astype(str)
    plot_df["bubble_size"] = plot_df["value_score"].clip(lower=0.01)

    scatter_fig = px.scatter(
        plot_df,
        x="uplift_score",
        y="clv",
        size="bubble_size",
        color="uplift_segment",
        hover_data=[
            "customer_id",
            "persona",
            "expected_incremental_profit",
            "value_score",
        ],
        title="상위 고객의 Uplift-CLV 분포",
        labels={"bubble_size": "value_score"},
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.caption(
        "버블 크기는 expected_incremental_profit 대신 value_score(CLV × uplift_score)를 사용합니다. 차트는 성능을 위해 상위 500명만, 아래 테이블은 전체 정렬 결과를 보여줍니다."
    )

    display_columns = [
        "customer_id",
        "persona",
        "uplift_score",
        "clv",
        "value_score",
        "expected_incremental_profit",
        "uplift_segment",
    ]
    display_df = top_customers[[col for col in display_columns if col in top_customers.columns]].copy()
    if "uplift_score" in display_df.columns:
        display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
    if "clv" in display_df.columns:
        display_df["clv"] = display_df["clv"].map(money)
    if "value_score" in display_df.columns:
        display_df["value_score"] = display_df["value_score"].map(money)
    if "expected_incremental_profit" in display_df.columns:
        display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
    _render_dataframe_with_count(display_df, label="상위 고객 테이블")

    llm_payload = {
        "top_n": int(len(top_customers)),
        "segment_distribution": segment_dist.to_dict(orient="records") if not segment_dist.empty else series_distribution(plot_df, "uplift_segment"),
        "numeric_summary": numeric_summary(
            plot_df,
            ["uplift_score", "clv", "expected_incremental_profit"],
        ),
        "top_customers": dataframe_snapshot(
            plot_df,
            columns=[
                "customer_id",
                "persona",
                "uplift_score",
                "clv",
                "expected_incremental_profit",
                "uplift_segment",
            ],
            max_rows=15,
        ),
    }

elif view == "4. 예산 최적화 및 리텐션 타겟":
    if _user_mode_unavailable("예산 최적화 및 리텐션 타겟", "예산 최적화와 최종 타겟 선정은 Uplift 기반 증분 이익 추정과 Treatment/Control 정보에 의존합니다."):
        st.stop()
    _opt_has_data = (
        (isinstance(selected_customers, pd.DataFrame) and not selected_customers.empty)
        or (isinstance(segment_allocation, pd.DataFrame) and not segment_allocation.empty)
        or _nonempty_mapping(optimize_summary)
    )
    if _simulator_mode_unavailable(
        "예산 최적화 및 리텐션 타겟",
        _opt_has_data,
        "예산 최적화 결과 또는 리텐션 타겟 산출물이 없습니다.",
        "시뮬레이터 데모에서는 python src/main.py --mode optimize 및 --mode recommend 를 실행한 뒤 새로고침하세요.",
    ):
        st.stop()
    st.subheader("예산 최적화 및 리텐션 타겟")
    st.caption("기존의 예산 배분, 예상 ROI, 리텐션 대상 고객 목록을 하나로 병합했습니다. 같은 selected_customers/optimize_summary 결과를 반복 표시하지 않도록 탭으로만 구분합니다.")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("총 예산", money(optimize_summary.get("budget", budget)))
    m2.metric("집행 예산", money(optimize_summary.get("spent", 0)))
    m3.metric("잔여 예산", money(optimize_summary.get("remaining", 0)))
    m4.metric("타겟 고객 수", f"{int(optimize_summary.get('num_targeted', len(selected_customers))):,}")
    m5.metric("예상 증분 이익", money(optimize_summary.get("expected_incremental_profit", 0)))
    m6.metric("예상 ROI", pct(float(optimize_summary.get("overall_roi", 0.0))))

    selected_customers = _ensure_retention_target_schema(selected_customers)
    optimized_targets = selected_customers.sort_values(
        ["priority_score", "selection_score", "expected_incremental_profit", "customer_id"],
        ascending=[False, False, False, True],
    ).copy() if not selected_customers.empty else pd.DataFrame()

    tab_budget, tab_roi, tab_targets = st.tabs(["예산 배분", "ROI 분포", "선정 고객"])

    with tab_budget:
        candidate_by_segment = pd.DataFrame(
            {
                "uplift_segment": list(optimize_summary.get("candidate_segment_counts", {}).keys()),
                "candidate_customer_count": list(optimize_summary.get("candidate_segment_counts", {}).values()),
            }
        )
        if not candidate_by_segment.empty:
            cand_fig = px.bar(
                candidate_by_segment,
                x="uplift_segment",
                y="candidate_customer_count",
                text="candidate_customer_count",
                title="세그먼트별 예산 배분 후보 고객 수",
            )
            st.plotly_chart(cand_fig, use_container_width=True)

        if segment_allocation.empty or int(optimize_summary.get("num_targeted", 0)) == 0:
            st.warning("현재 조건에서 예산 배분 대상 고객이 없습니다.")
        else:
            chart_df = segment_allocation.copy()
            label_threshold = float(chart_df["allocated_budget"].max()) * 0.08 if not chart_df.empty else 0.0
            chart_df["customer_count_label"] = np.where(
                (chart_df["customer_count"] >= 5) | (chart_df["allocated_budget"] >= label_threshold),
                chart_df["customer_count"].astype(int).astype(str),
                "",
            )
            if "intervention_intensity" in chart_df.columns and chart_df["intervention_intensity"].nunique() > 1:
                bar_fig = px.bar(
                    chart_df,
                    x="uplift_segment",
                    y="allocated_budget",
                    color="intervention_intensity",
                    barmode="group",
                    text="customer_count_label",
                    hover_data=["customer_count", "expected_profit"],
                    title="세그먼트·개입 강도별 예산 배분",
                )
                bar_fig.update_layout(legend_title_text="개입 강도")
            else:
                bar_fig = px.bar(
                    chart_df,
                    x="uplift_segment",
                    y="allocated_budget",
                    text="customer_count_label",
                    hover_data=["customer_count", "expected_profit"],
                    title="세그먼트별 예산 배분",
                )
            bar_fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(bar_fig, use_container_width=True)

            display_df = segment_allocation.copy()
            if "allocated_budget" in display_df.columns:
                display_df["allocated_budget"] = display_df["allocated_budget"].map(money)
            if "expected_profit" in display_df.columns:
                display_df["expected_profit"] = display_df["expected_profit"].map(money)
            _render_dataframe_with_count(display_df, label="세그먼트별 예산 배분 테이블")

    with tab_roi:
        if selected_customers.empty:
            st.warning("현재 조건에서 ROI 계산 대상이 없습니다.")
        else:
            roi_fig = px.histogram(
                selected_customers,
                x="expected_roi",
                nbins=25,
                title="선정 고객의 예상 ROI 분포",
            )
            roi_fig.update_traces(marker_line_color="rgba(255,255,255,0.95)", marker_line_width=1.2, opacity=0.9)
            roi_fig.update_layout(bargap=0.02)
            st.plotly_chart(roi_fig, use_container_width=True)

            roi_summary = selected_customers[[col for col in ["expected_roi", "coupon_cost", "expected_incremental_profit"] if col in selected_customers.columns]].describe().T.reset_index()
            if not roi_summary.empty:
                _render_dataframe_with_count(roi_summary, label="ROI·비용·기대이익 요약", prefer_static=True)

    with tab_targets:
        if optimized_targets.empty:
            st.warning("현재 조건에서 리텐션 타겟 고객이 없습니다.")
        else:
            priority_chart_df = optimized_targets.head(min(15, len(optimized_targets))).copy()
            priority_fig = px.bar(
                priority_chart_df,
                x="customer_id",
                y="priority_score",
                color="intervention_intensity" if "intervention_intensity" in priority_chart_df.columns else None,
                hover_data=[col for col in ["churn_probability", "uplift_score", "clv", "expected_incremental_profit", "expected_roi"] if col in priority_chart_df.columns],
                title="우선순위 상위 리텐션 대상 고객",
            )
            st.plotly_chart(priority_fig, use_container_width=True)

            display_columns = [
                "customer_id",
                "persona",
                "uplift_segment",
                "churn_probability",
                "uplift_score",
                "clv",
                "intervention_intensity",
                "recommended_action",
                "coupon_cost",
                "expected_incremental_profit",
                "expected_roi",
                "priority_score",
                "recommended_intervention_window",
            ]
            display_df = optimized_targets[[col for col in display_columns if col in optimized_targets.columns]].copy()
            if "churn_probability" in display_df.columns:
                display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{float(x):.3f}")
            if "uplift_score" in display_df.columns:
                display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{float(x):.3f}")
            if "clv" in display_df.columns:
                display_df["clv"] = display_df["clv"].map(money)
            if "coupon_cost" in display_df.columns:
                display_df["coupon_cost"] = display_df["coupon_cost"].map(money)
            if "expected_incremental_profit" in display_df.columns:
                display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
            if "expected_roi" in display_df.columns:
                display_df["expected_roi"] = display_df["expected_roi"].map(lambda x: f"{float(x):.2%}")
            if "priority_score" in display_df.columns:
                display_df["priority_score"] = display_df["priority_score"].map(lambda x: f"{float(x):.3f}")
            _render_dataframe_with_count(
                display_df,
                label="최종 리텐션 타겟 고객 테이블",
                height=min(1100, 180 + 32 * len(display_df)),
            )

    llm_payload = {
        "threshold": threshold,
        "budget": budget,
        "optimize_summary": optimize_summary,
        "segment_allocation": segment_allocation.round(4).to_dict(orient="records") if not segment_allocation.empty else [],
        "target_count": int(len(optimized_targets)),
        "persona_distribution": series_distribution(optimized_targets, "persona") if not optimized_targets.empty else {},
        "segment_distribution": series_distribution(optimized_targets, "uplift_segment") if not optimized_targets.empty else {},
        "target_numeric_summary": numeric_summary(
            optimized_targets,
            ["priority_score", "selection_score", "churn_probability", "uplift_score", "clv", "coupon_cost", "expected_incremental_profit", "expected_roi"],
        ),
    }

elif view == "8. 학습 결과 아티팩트":
    st.subheader("학습 결과 아티팩트")
    st.caption("이 화면은 백엔드 API가 보관 중인 최신 학습 산출물을 읽기 전용으로 표시합니다. 대시보드에서 학습 파라미터를 조정하거나 재학습을 직접 실행하지 않습니다.")

    try:
        training_payload = load_training_artifacts_api()
    except Exception as exc:
        training_payload = {"_load_error": str(exc)}

    churn_metrics = training_payload.get("churn_metrics", {})
    threshold_analysis = training_payload.get("threshold_analysis", {})
    top_feature_importance_df = _artifact_frame(training_payload.get("top_feature_importance"))
    customer_features_df = _artifact_frame(training_payload.get("customer_features"), max_columns=16)
    image_paths = training_payload.get("image_paths", {})
    model_paths = training_payload.get("model_paths", {})
    training_parameters = training_payload.get("training_parameters", {}) or churn_metrics.get("training_parameters", {})

    _training_error = str(training_payload.get("_load_error", "") or "")
    _training_has_data = bool(
        churn_metrics
        or threshold_analysis
        or not top_feature_importance_df.empty
        or not customer_features_df.empty
        or any(_path_exists(path) for path in (image_paths or {}).values())
        or any(_path_exists(path) for path in (model_paths or {}).values())
    )
    if not _training_has_data:
        _simulator_missing_result_box(
            "학습 결과 아티팩트",
            _training_error or "churn_metrics, feature importance, feature store, 학습 이미지/모델 파일을 찾지 못했습니다.",
            "시뮬레이터 데모에서는 python src/main.py --mode simulate --force --randomize → features → train 순서로 실행한 뒤 새로고침하세요.",
        )
    else:
        if not churn_metrics:
            st.warning("학습 결과를 아직 불러오지 못했습니다.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best model", str(churn_metrics.get("best_model_name", "-")))
            m2.metric("Test AUC", f"{float(churn_metrics.get('test_auc_roc', 0.0)):.4f}")
            m3.metric("Selected threshold", f"{float(churn_metrics.get('selected_threshold', 0.0)):.4f}")
            m4.metric("Positive rate", f"{float(churn_metrics.get('positive_rate', 0.0)):.2%}")

            st.markdown("### 학습 메타데이터")
            meta_df = pd.DataFrame(
                [
                    {"key": "train_rows", "value": churn_metrics.get("train_rows")},
                    {"key": "test_rows", "value": churn_metrics.get("test_rows")},
                    {"key": "numeric_feature_count", "value": churn_metrics.get("numeric_feature_count")},
                    {"key": "categorical_feature_count", "value": churn_metrics.get("categorical_feature_count")},
                    {"key": "lightgbm_available", "value": churn_metrics.get("lightgbm_available")},
                    {"key": "model_path", "value": model_paths.get("churn_model")},
                    {"key": "requested_models", "value": training_parameters.get("candidate_models") or training_parameters.get("requested_models")},
                    {"key": "test_size", "value": training_parameters.get("test_size")},
                    {"key": "random_state", "value": training_parameters.get("random_state")},
                    {"key": "shap_sample_size", "value": training_parameters.get("shap_sample_size")},
                ]
            )
            _render_artifact_table(meta_df, label="학습 메타데이터")

        if not top_feature_importance_df.empty:
            st.markdown("### Top feature importance")
            _render_artifact_table(top_feature_importance_df, label="Top feature importance")

        if threshold_analysis and threshold_analysis.get("selected"):
            st.markdown("### 선택된 threshold 요약")
            selected_df = _sanitize_artifact_dataframe(pd.DataFrame([threshold_analysis["selected"]]))
            _render_artifact_table(selected_df, label="선택 threshold 요약")

        if training_parameters:
            st.markdown("### 학습 파라미터 (서버 반영값)")
            training_parameter_df = _sanitize_artifact_dataframe(pd.DataFrame([training_parameters]))
            _render_artifact_table(training_parameter_df, label="학습 파라미터")

        st.markdown("### 학습 시각화")
        image_cols = st.columns(2)
        image_items = [
            ("ROC Curve", image_paths.get("churn_auc_roc")),
            ("Precision-Recall Tradeoff", image_paths.get("churn_precision_recall_tradeoff")),
            ("SHAP Summary", image_paths.get("churn_shap_summary")),
            ("SHAP Local", image_paths.get("churn_shap_local")),
        ]
        for idx, (title, img_path) in enumerate(image_items):
            with image_cols[idx % 2]:
                if img_path and _path_exists(img_path):
                    st.image(img_path, caption=title, use_container_width=True)
                else:
                    st.info(f"{title} 파일이 없습니다.")

        if not customer_features_df.empty:
            st.markdown("### Feature store 미리보기")
            _render_artifact_table(customer_features_df.head(20), use_dataframe=True, height=420, label="Feature store 미리보기")

    llm_payload = {
        "churn_metrics": churn_metrics,
        "training_parameters": training_parameters,
        "threshold_analysis_selected": threshold_analysis.get("selected", {}) if threshold_analysis else {},
        "top_feature_importance": top_feature_importance_df.to_dict(orient="records") if not top_feature_importance_df.empty else [],
        "feature_store_preview": dataframe_snapshot(
            customer_features_df,
            columns=list(customer_features_df.columns[:12]),
            max_rows=10,
        ) if not customer_features_df.empty else [],
    }

elif view == "5. 개인화 추천":
    _recommend_has_data = isinstance(personalized_recommendations, pd.DataFrame) and not personalized_recommendations.empty
    if _simulator_mode_unavailable(
        "개인화 추천",
        _recommend_has_data,
        recommendation_error or "개인화 추천 산출물이 없습니다.",
        "시뮬레이터 데모에서는 python src/main.py --mode recommend 를 실행한 뒤 새로고침하세요.",
    ):
        st.stop()
    st.subheader("최종 타겟 고객 대상 개인화 추천")
    st.caption("예산/임계값으로 선별된 최종 리텐션 타겟 고객에게만 추천을 생성합니다. 추천 점수는 구매 이력 + 최근 관심 + 세그먼트 인기 + 전역 인기를 혼합해 계산합니다.")

    if recommendation_error:
        st.error(f"추천 API 호출 실패: {recommendation_error}")
    elif personalized_recommendations.empty:
        st.warning("표시할 추천 결과가 없습니다. 현재 예산/임계값 조건에서 최종 타겟 고객이 없을 수 있습니다.")
    else:
        if isinstance(recommendation_summary, dict) and recommendation_summary.get("warning"):
            st.warning(str(recommendation_summary.get("warning")))
        budget_context = recommendation_summary.get('budget_context', {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("추천 행 수", f"{recommendation_summary.get('rows', len(personalized_recommendations)):,}")
        m2.metric("커버 고객 수", f"{recommendation_summary.get('customers_covered', personalized_recommendations['customer_id'].nunique()):,}")
        m3.metric("고객당 추천 수", str(recommendation_summary.get('per_customer', recommendation_per_customer)))
        m4.metric("최종 타겟 고객 수", f"{budget_context.get('num_targeted', recommendation_summary.get('customers_covered', 0)):,}")

        category_counts = (
            personalized_recommendations.groupby('recommended_category', as_index=False)
            .agg(recommend_count=('customer_id', 'count'))
            .sort_values('recommend_count', ascending=False)
        )
        fig = px.bar(
            category_counts,
            x='recommended_category',
            y='recommend_count',
            title='추천 카테고리 분포',
        )
        st.plotly_chart(fig, use_container_width=True)

        display_df = personalized_recommendations.copy()
        if 'churn_probability' in display_df.columns:
            display_df['churn_probability'] = display_df['churn_probability'].map(lambda x: f"{x:.3f}")
        if 'uplift_score' in display_df.columns:
            display_df['uplift_score'] = display_df['uplift_score'].map(lambda x: f"{x:.3f}")
        if 'clv' in display_df.columns:
            display_df['clv'] = display_df['clv'].map(money)
        if 'expected_incremental_profit' in display_df.columns:
            display_df['expected_incremental_profit'] = display_df['expected_incremental_profit'].map(money)
        if 'coupon_cost' in display_df.columns:
            display_df['coupon_cost'] = display_df['coupon_cost'].map(money)
        if 'expected_roi' in display_df.columns:
            display_df['expected_roi'] = display_df['expected_roi'].map(lambda x: f"{x:.3f}")
        if 'recommendation_priority' in display_df.columns:
            display_df['recommendation_priority'] = display_df['recommendation_priority'].map(lambda x: f"{x:.3f}")
        if 'target_priority_score' in display_df.columns:
            display_df['target_priority_score'] = display_df['target_priority_score'].map(lambda x: f"{x:.3f}")
        if 'recommendation_score' in display_df.columns:
            display_df['recommendation_score'] = display_df['recommendation_score'].map(lambda x: f"{x:.3f}")
        _render_dataframe_with_count(display_df, label="개인화 추천 테이블")

    llm_payload = {
        'recommendation_summary': recommendation_summary,
        'category_distribution': (
            personalized_recommendations['recommended_category'].value_counts().to_dict()
            if not personalized_recommendations.empty else {}
        ),
        'recommendation_preview': dataframe_snapshot(
            personalized_recommendations,
            columns=[
                'customer_id',
                'persona',
                'recommended_category',
                'recommendation_rank',
                'recommendation_score',
                'reason_tags',
            ],
            max_rows=20,
        ) if not personalized_recommendations.empty else [],
    }

elif view == "6. 실시간 운영 모니터":
    # user mode에서는 PostgreSQL live DB 화면만 렌더링하고,
    # 기존 Redis Streams 기반 simulator 실시간 블록으로 내려가지 않는다.
    # 그렇지 않으면 user mode에서도 realtime/scores API를 호출해 Redis 안내/오류가 같이 표시된다.
    if _is_user_live_mode():
        st.subheader("실시간 운영 모니터")
        st.caption("자사 데이터 모드: PostgreSQL live DB 기준 운영 모니터입니다.")

        from dashboard.services.api_client import (
            fetch_demo_status as _page_fetch_demo_status,
            start_demo_stream as _page_start_demo,
            stop_demo_stream as _page_stop_demo,
            reset_demo_stream as _page_reset_demo,
        )
        try:
            _page_demo = _page_fetch_demo_status()
        except Exception:
            _page_demo = {}
        _page_demo_running = _page_demo.get("running", False)

        st.caption("시연을 시작하면 설정된 간격마다 가상 고객 이벤트(방문, 구매 등)가 자동 생성되고, 이탈 점수 재산정 및 액션 큐가 갱신됩니다.")
        _demo_bar = st.container()
        with _demo_bar:
            if _page_demo_running:
                _ev = _page_demo.get("total_events_sent", 0)
                _new = _page_demo.get("new_customers_created", 0)
                _exist = _page_demo.get("existing_customers_updated", 0)
                st.success(f"시연 실행 중  |  이벤트 {_ev}건  |  신규 {_new}명  |  기존 {_exist}명")
                _dc1, _dc2, _dc3 = st.columns(3)
                with _dc1:
                    if st.button("시연 중지", use_container_width=True, key="pg_demo_stop"):
                        _page_stop_demo()
                        clear_dashboard_caches()
                        st.rerun()
                with _dc2:
                    if st.button("시연 초기화", use_container_width=True, type="secondary", key="pg_demo_reset_running"):
                        _page_reset_demo()
                        clear_dashboard_caches()
                        st.rerun()
                with _dc3:
                    st.caption("10초마다 자동 새로고침")
            else:
                _dc1, _dc2, _dc3, _dc4 = st.columns([1.5, 1.5, 1, 1])
                with _dc1:
                    st.caption("N초마다 이벤트 1건 생성")
                    _pg_interval = st.number_input("간격(초)", min_value=0.5, max_value=30.0, value=2.0, step=0.5, key="pg_demo_interval")
                with _dc2:
                    st.caption("새 고객 vs 기존 고객 비율")
                    _pg_ratio = st.number_input("신규 비율", min_value=0.0, max_value=1.0, value=0.3, step=0.1, key="pg_demo_ratio")
                with _dc3:
                    if st.button("시연 시작", use_container_width=True, type="primary", key="pg_demo_start"):
                        _page_start_demo(interval_seconds=_pg_interval, new_customer_ratio=_pg_ratio)
                        clear_dashboard_caches()
                        st.rerun()
                with _dc4:
                    if st.button("시연 초기화", use_container_width=True, type="secondary", key="pg_demo_reset_idle"):
                        _page_reset_demo()
                        clear_dashboard_caches()
                        st.rerun()

            if _page_demo.get("latest_results"):
                if _page_demo_running:
                    _prev = st.session_state.get("_demo_last_log", [])
                    _seen = {(r["customer_id"], r["event_type"], r.get("churn_score")) for r in _prev}
                    _merged = list(_prev)
                    for _r in _page_demo["latest_results"]:
                        _key = (_r["customer_id"], _r["event_type"], _r.get("churn_score"))
                        if _key not in _seen:
                            _merged.append(_r)
                            _seen.add(_key)
                    st.session_state["_demo_last_log"] = _merged
                _log_data = st.session_state.get("_demo_last_log", _page_demo.get("latest_results", []))
                if _log_data:
                    _log_label = f"이벤트 로그 ({len(_log_data)}건)" if _page_demo_running else f"이벤트 로그 ({len(_log_data)}건, 중지됨)"
                    with st.expander(_log_label, expanded=True):
                        _lines = []
                        for _r in reversed(_log_data):
                            _label = "NEW" if _r.get("is_new") else "UPD"
                            _score_str = f"score={_r['churn_score']:.2f}" if _r.get("churn_score") is not None else ""
                            _action_str = "-> action queued" if _r.get("action_queued") else ""
                            _lines.append(f"[{_label}] #{_r['customer_id']}  {_r['event_type']}  {_score_str}  {_action_str}")
                        st.dataframe(pd.DataFrame({"log": _lines}), height=300, use_container_width=True, hide_index=True)
            elif st.session_state.get("_demo_last_log"):
                _log_data = st.session_state["_demo_last_log"]
                with st.expander(f"이벤트 로그 ({len(_log_data)}건, 중지됨)", expanded=False):
                    _lines = []
                    for _r in reversed(_log_data):
                        _label = "NEW" if _r.get("is_new") else "UPD"
                        _score_str = f"score={_r['churn_score']:.2f}" if _r.get("churn_score") is not None else ""
                        _action_str = "-> action queued" if _r.get("action_queued") else ""
                        _lines.append(f"[{_label}] #{_r['customer_id']}  {_r['event_type']}  {_score_str}  {_action_str}")
                    st.dataframe(pd.DataFrame({"log": _lines}), height=300, use_container_width=True, hide_index=True)

        st.divider()

        health = live_payload.get("health", {}) or {}
        score_summary = live_payload.get("score_summary", {}) or {}
        action_summary = live_payload.get("action_summary", {}) or {}
        rec_summary = live_payload.get("recommendation_summary", {}) or {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("이벤트 수", f"{int(health.get('event_count') or 0):,}")
        c2.metric("실시간 고객 상태", f"{int(health.get('feature_state_count') or 0):,}")
        c3.metric("점수 고객 수", f"{int(score_summary.get('scored_customers') or 0):,}")
        c4.metric("Queued 액션", f"{int(action_summary.get('queued_actions') or 0):,}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("평균 이탈 점수", pct(float(score_summary.get("avg_churn_score") or 0.0)))
        c6.metric("고위험 고객", f"{int(score_summary.get('high_risk_customers') or 0):,}")
        c7.metric("Live 추천", f"{int(rec_summary.get('live_recommendations') or 0):,}")
        c8.metric("최신 점수 갱신", str(score_summary.get("latest_scored_at") or "-"))

        scores_df = live_payload.get("scores", pd.DataFrame()).copy()
        actions_df = live_payload.get("actions", pd.DataFrame()).copy()

        if not scores_df.empty:
            chart_df = scores_df.head(int(top_n)).copy()
            if "customer_id" in chart_df.columns:
                chart_df["customer_id"] = chart_df["customer_id"].astype(str)

            y_col = "churn_score" if "churn_score" in chart_df.columns else "churn_probability"

            fig = px.bar(
                chart_df,
                x="customer_id" if "customer_id" in chart_df.columns else chart_df.index,
                y=y_col,
                hover_data=[
                    col for col in [
                        "clv",
                        "uplift_score",
                        "expected_roi",
                        "risk_segment",
                        "scored_at",
                    ]
                    if col in chart_df.columns
                ],
                title="Live 이탈 점수 Top 고객",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("표시할 live score 데이터가 없습니다.")

        if not actions_df.empty:
            display_cols = [
                col for col in [
                    "customer_id",
                    "recommended_action",
                    "intervention_intensity",
                    "expected_profit",
                    "expected_incremental_profit",
                    "expected_roi",
                    "priority_score",
                    "action_status",
                    "source_type",
                    "trigger_reason",
                    "queued_at",
                    "updated_at",
                ]
                if col in actions_df.columns
            ]

            _render_dataframe_with_count(
                actions_df[display_cols],
                label="Live Action Queue",
                height=520,
            )
        else:
            st.info("현재 queued action이 없습니다. action_threshold를 낮춰 테스트하거나 새 이벤트를 입력하세요.")

        llm_payload = {
            "mode": "user_live",
            "health": health,
            "score_summary": score_summary,
            "action_summary": action_summary,
            "recommendation_summary": rec_summary,
            "score_preview": dataframe_snapshot(
                scores_df,
                columns=[
                    "customer_id",
                    "churn_score",
                    "clv",
                    "uplift_score",
                    "expected_roi",
                    "risk_segment",
                    "scored_at",
                ],
                max_rows=20,
            ) if not scores_df.empty else [],
            "action_preview": dataframe_snapshot(
                actions_df,
                columns=[
                    "customer_id",
                    "recommended_action",
                    "priority_score",
                    "action_status",
                    "source_type",
                    "updated_at",
                ],
                max_rows=20,
            ) if not actions_df.empty else [],
        }

        if _page_demo_running:
            import time as _demo_time
            _placeholder = st.empty()
            _placeholder.caption("다음 자동 새로고침까지 10초...")
            _demo_time.sleep(10)
            clear_dashboard_caches()
            st.rerun()

        st.stop()

    _realtime_has_data = (
        (isinstance(realtime_scores, pd.DataFrame) and not realtime_scores.empty)
        or _nonempty_mapping(realtime_summary)
        or (isinstance(realtime_monitor_overview, dict) and any(isinstance(v, pd.DataFrame) and not v.empty for v in realtime_monitor_overview.values()))
    )
    if _simulator_mode_unavailable(
        "실시간 운영 모니터",
        _realtime_has_data,
        realtime_error or "실시간 스코어 스냅샷 또는 액션 큐 산출물이 없습니다.",
        "시뮬레이터 데모에서는 python src/main.py --mode realtime-bootstrap 및 --mode realtime-replay 를 실행한 뒤 새로고침하세요.",
    ):
        st.stop()
    st.subheader("실시간 운영 모니터")
    st.caption("Redis Streams로 적재된 이벤트를 조금씩 재생하며 고객별 실시간 위험 점수와 액션 큐 상태를 함께 갱신합니다.")

    if realtime_error:
        st.error(f"실시간 스코어 API 호출 실패: {realtime_error}")
        st.info("먼저 Redis를 실행한 뒤 realtime-bootstrap / realtime-produce / realtime-consume(또는 realtime-replay) 명령을 수행하세요.")
    elif realtime_scores.empty:
        st.warning("실시간 스코어 스냅샷이 없습니다. 스트림 소비 결과가 아직 생성되지 않았을 수 있습니다.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("추적 고객 수", f"{int(realtime_summary.get('tracked_customers', 0)):,}")
        m2.metric("고위험 고객 수", f"{int(realtime_summary.get('high_risk_customers', 0)):,}")
        m3.metric("재최적화 트리거 수", f"{int(realtime_summary.get('triggered_reoptimizations', 0)):,}")
        m4.metric("액션 큐 적재 수", f"{int(realtime_summary.get('action_queue_size', 0)):,}")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("임계 위험 고객 수", f"{int(realtime_summary.get('critical_risk_customers', 0)):,}")
        q2.metric("처리 이벤트 수", f"{int(realtime_summary.get('processed_events', 0)):,}")
        q3.metric("폐쇄루프 예산 사용", money(int(realtime_summary.get('closed_loop_budget_spent', 0))))
        q4.metric("채널 할당 수", f"{int(realtime_summary.get('daily_channel_allocated', 0)):,} / {int(realtime_summary.get('daily_channel_capacity', 0)):,}")

        chart_df = realtime_scores.head(min(len(realtime_scores), 20)).copy()
        chart_df['customer_id'] = chart_df['customer_id'].astype(str)
        fig = px.bar(
            chart_df,
            x='customer_id',
            y='realtime_churn_score',
            color='action_queue_status' if 'action_queue_status' in chart_df.columns else None,
            hover_data=['base_churn_probability', 'score_delta', 'last_event_type', 'persona', 'latest_trigger_reason', 'queued_recommended_action'],
            title='실시간 이탈 위험 상위 고객',
        )
        st.plotly_chart(fig, use_container_width=True)

        queued_df = realtime_scores[realtime_scores.get('action_queue_status', pd.Series(index=realtime_scores.index, dtype=object)).astype(str) == 'queued'].copy() if 'action_queue_status' in realtime_scores.columns else pd.DataFrame()
        if not queued_df.empty:
            queue_display = queued_df[[
                col for col in [
                    'customer_id',
                    'persona',
                    'uplift_segment',
                    'realtime_churn_score',
                    'queued_intervention_intensity',
                    'queued_recommended_action',
                    'queued_coupon_cost',
                    'queued_expected_profit',
                    'queued_expected_roi',
                    'latest_trigger_reason',
                    'reoptimization_count',
                ] if col in queued_df.columns
            ]].copy()
            if 'realtime_churn_score' in queue_display.columns:
                queue_display['realtime_churn_score'] = queue_display['realtime_churn_score'].map(lambda x: f"{float(x):.3f}")
            if 'queued_coupon_cost' in queue_display.columns:
                queue_display['queued_coupon_cost'] = queue_display['queued_coupon_cost'].map(money)
            if 'queued_expected_profit' in queue_display.columns:
                queue_display['queued_expected_profit'] = queue_display['queued_expected_profit'].map(money)
            if 'queued_expected_roi' in queue_display.columns:
                queue_display['queued_expected_roi'] = queue_display['queued_expected_roi'].map(lambda x: f"{float(x):.2%}")
            _render_dataframe_with_count(queue_display, label="실시간 부분 재최적화 액션 큐", height=min(520, 180 + 32 * len(queue_display)))

        display_df = realtime_scores.copy()
        for col in ['base_churn_probability', 'realtime_churn_score', 'score_delta', 'behavioral_risk', 'inactivity_signal', 'queued_expected_roi']:
            if col in display_df.columns:
                formatter = (lambda x: f"{float(x):.2%}") if col == 'queued_expected_roi' else (lambda x: f"{float(x):.3f}")
                display_df[col] = display_df[col].map(formatter)
        for money_col in ['clv', 'coupon_cost', 'queued_coupon_cost', 'queued_expected_profit']:
            if money_col in display_df.columns:
                display_df[money_col] = display_df[money_col].map(money)
        if 'expected_roi' in display_df.columns:
            display_df['expected_roi'] = display_df['expected_roi'].map(lambda x: f"{float(x):.3f}")
        _render_dataframe_with_count(display_df, label="실시간 이탈 위험 테이블")

    realtime_summary_display = realtime_monitor_overview.get("summary", realtime_summary) if realtime_monitor_overview else realtime_summary
    st.markdown("### 운영 모니터")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("처리 이벤트 수", f"{int(realtime_summary_display.get('processed_events', 0) or 0):,}")
    q2.metric("재최적화 횟수", f"{int(realtime_summary_display.get('triggered_reoptimizations', 0) or 0):,}")
    q3.metric("큐 적재 수", f"{int(realtime_summary_display.get('queued_actions_total', realtime_summary_display.get('action_queue_size', 0)) or 0):,}")
    cap = int(realtime_summary_display.get('daily_channel_capacity', 0) or 0)
    alloc = int(realtime_summary_display.get('daily_channel_allocated', 0) or 0)
    utilization = alloc / cap if cap > 0 else 0.0
    q4.metric("채널 용량 사용률", pct(utilization))
    q5.metric("고우선순위 큐", f"{int(realtime_summary_display.get('high_priority_queue_size', 0) or 0):,}")

    if realtime_monitor_overview:
        tab1, tab2, tab3 = st.tabs(["큐 상태", "트리거 이유", "행동 신호"])
        with tab1:
            status_df = realtime_monitor_overview.get("status_df", pd.DataFrame())
            queue_df = realtime_monitor_overview.get("queue_df", pd.DataFrame())
            if not status_df.empty:
                fig = px.pie(status_df, names="status", values="count", title="액션 큐 상태 구성")
                st.plotly_chart(fig, use_container_width=True)
            if not queue_df.empty:
                display_df = queue_df.copy()
                for col in ["queued_coupon_cost", "queued_expected_profit"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: money(float(x)) if pd.notna(x) else "")
                for col in ["queued_expected_roi", "realtime_churn_score"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
                _render_dataframe_with_count(display_df, label="실시간 액션 큐 상세", height=min(1200, 220 + 28 * len(display_df)))
        with tab2:
            trigger_df = realtime_monitor_overview.get("trigger_df", pd.DataFrame())
            if not trigger_df.empty:
                fig = px.bar(trigger_df.head(15), x="trigger_reason", y="count", title="주요 트리거 이유", text="count")
                st.plotly_chart(fig, use_container_width=True)
                _render_dataframe_with_count(trigger_df, label="트리거 이유 빈도", prefer_static=True)
        with tab3:
            signal_df = realtime_monitor_overview.get("signal_df", pd.DataFrame())
            if not signal_df.empty:
                fig = px.bar(signal_df, x="signal", y="mean_value", title="행동 신호 평균값")
                st.plotly_chart(fig, use_container_width=True)
                _render_dataframe_with_count(signal_df, label="행동 신호 평균", prefer_static=True)

    llm_payload = {
        'realtime_summary': realtime_summary_display,
        'realtime_preview': dataframe_snapshot(
            realtime_scores,
            columns=[
                'customer_id',
                'persona',
                'realtime_churn_score',
                'score_delta',
                'action_queue_status',
                'queued_recommended_action',
                'latest_trigger_reason',
            ],
            max_rows=20,
        ) if not realtime_scores.empty else [],
        'queue_preview': dataframe_snapshot(realtime_monitor_overview.get("queue_df", pd.DataFrame()), max_rows=20) if realtime_monitor_overview and not realtime_monitor_overview.get("queue_df", pd.DataFrame()).empty else [],
    }

elif view == "9. 이탈 시점 예측 (Survival Analysis)":
    st.subheader("이탈 시점 예측 (Survival Analysis)")
    st.caption('Cox Proportional Hazards 기반으로 landmark 시점 이후 얼마 안에 churn risk 상태로 진입할지를 추정합니다. 분류 모델과 달리 "언제" 위험이 커지는지를 함께 봅니다.')

    if survival_error or not survival_metrics:
        _simulator_missing_result_box(
            "이탈 시점 예측 (Survival Analysis)",
            survival_error or "survival_metrics.json, survival_predictions.csv 또는 survival 모델 산출물을 찾지 못했습니다.",
            "시뮬레이터 데모에서는 python src/main.py --mode survival 실행 후 대시보드를 새로고침하세요.",
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("모델", str(survival_metrics.get('model_name', '-')))
        m2.metric("Test C-index", f"{float(survival_metrics.get('test_concordance_index', 0.0)):.4f}")
        m3.metric("Horizon", f"{int(survival_metrics.get('horizon_days', 0))}일")
        m4.metric("Event rate", f"{float(survival_metrics.get('event_rate', 0.0)):.2%}")

        meta_df = pd.DataFrame([
            {'key': 'landmark_as_of_date', 'value': survival_metrics.get('landmark_as_of_date')},
            {'key': 'train_rows', 'value': survival_metrics.get('train_rows')},
            {'key': 'test_rows', 'value': survival_metrics.get('test_rows')},
            {'key': 'feature_count_before_encoding', 'value': survival_metrics.get('feature_count_before_encoding')},
            {'key': 'feature_count_after_encoding', 'value': survival_metrics.get('feature_count_after_encoding')},
            {'key': 'penalizer', 'value': survival_metrics.get('penalizer')},
        ])
        st.markdown("### Survival 메타데이터")
        _render_artifact_table(meta_df, label="Survival 메타데이터")

        risk_plot = survival_image_paths.get('risk_stratification')
        if risk_plot and _path_exists(risk_plot):
            st.image(risk_plot, caption='예측 위험군별 생존 곡선', use_container_width=True)

        if not survival_predictions.empty:
            chart_df = survival_predictions.head(min(len(survival_predictions), 20)).copy()
            chart_df['customer_id'] = chart_df['customer_id'].astype(str)
            if 'survival_prob_30d' in chart_df.columns:
                fig = px.bar(
                    chart_df,
                    x='customer_id',
                    y='predicted_hazard_ratio',
                    hover_data=['survival_prob_30d', 'predicted_median_time_to_churn_days', 'persona', 'risk_group'],
                    title='단기 churn 위험 상위 고객',
                )
                st.plotly_chart(fig, use_container_width=True)

            display_df = survival_predictions.copy()
            for col in ['predicted_hazard_ratio', 'survival_prob_30d', 'survival_prob_60d', 'survival_prob_90d', 'predicted_median_time_to_churn_days', 'risk_percentile']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}")
            _render_dataframe_with_count(display_df, label="Survival 예측 결과")

        if not survival_coefficients.empty:
            st.markdown("### 주요 hazard coefficient")
            coef_df = survival_coefficients.copy()
            for col in ['coef', 'exp(coef)', 'p', 'abs_coef']:
                if col in coef_df.columns:
                    coef_df[col] = coef_df[col].map(lambda x: f"{float(x):.4f}")
            _render_dataframe_with_count(coef_df, label="주요 hazard coefficient")

    llm_payload = {
        'survival_metrics': survival_metrics,
        'survival_prediction_preview': dataframe_snapshot(
            survival_predictions,
            columns=[
                'customer_id',
                'predicted_hazard_ratio',
                'survival_prob_30d',
                'predicted_median_time_to_churn_days',
                'risk_group',
            ],
            max_rows=20,
        ) if not survival_predictions.empty else [],
        'survival_coefficients': survival_coefficients.head(15).to_dict(orient='records') if not survival_coefficients.empty else [],
    }

elif view == "10. 증분 성과 / A-B 실험":
    if _user_mode_unavailable("증분 성과 / A-B 실험 분석", "A/B 테스트 분석은 Treatment/Control 그룹 분리 데이터가 필수이며, 외부 데이터에는 해당 정보가 없습니다."):
        st.stop()
    st.subheader("증분 성과 / A-B 실험")
    st.caption("정확도보다 더 중요한 운영 지표인 증분 리텐션, 추가 유지 고객 수, 비용 대비 유지 성과, dose-response 결과를 함께 봅니다.")
    # ── Power Analysis 기반 표본 충분성 경고 (개선안 1) ──
    _ab_test_meta = experiment_overview.get("ab_test", {}) or {}
    _power_meta = _ab_test_meta.get("power_analysis", {}) or {}
    _achieved_power = _power_meta.get("achieved_power_with_current_sample", _power_meta.get("achieved_power"))
    _required_n_per_group = _power_meta.get("required_sample_size_per_group", _power_meta.get("required_n_per_group"))
    _current_min_n = _power_meta.get("current_min_group_size", _power_meta.get("min_group_size"))
    if _achieved_power is not None and float(_achieved_power) < 0.80:
        _ratio_text = ""
        if _required_n_per_group and _current_min_n:
            _ratio = float(_current_min_n) / float(_required_n_per_group) * 100
            _ratio_text = f" (필요 표본의 {_ratio:.1f}%)"
        st.error(
            f"⚠️ **검출력 부족 — 결과를 효과 유무의 근거로 사용할 수 없습니다.**\n\n"
            f"현재 표본은 효과 검출에 필요한 수의 일부에 불과합니다{_ratio_text}. "
            f"Achieved power **{float(_achieved_power)*100:.1f}%** (목표 80%). "
            f"아래 수치(증분 리텐션, ROI 등)는 통계적 노이즈일 가능성이 매우 높으며, "
            f"**'효과가 없다'가 아니라 '효과를 측정할 수 없었다'로 해석해야 합니다.**"
        )
    exp_metrics = experiment_overview.get("metrics", {})
    _dose_df_for_check = experiment_overview.get("dose_df", pd.DataFrame())
    _ab_has_data = bool(
        experiment_overview.get("ab_test")
        or not _dose_df_for_check.empty
        or experiment_overview.get("persuadables")
        or any(value not in (None, "", 0, 0.0) for value in exp_metrics.values())
    )
    if not _ab_has_data:
        _simulator_missing_result_box(
            "증분 성과 / A-B 실험",
            "A/B 테스트, dose-response, persuadables 산출물을 찾지 못했습니다.",
            "시뮬레이터 데모에서는 python src/main.py --mode abtest 실행 후 대시보드를 새로고침하세요.",
        )
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("증분 리텐션", pct(float(exp_metrics.get('incremental_retention', 0.0))))
        m2.metric("추가 유지 고객 수", f"{int(round(float(exp_metrics.get('incremental_retained_customers', 0.0)))):,}명")
        m3.metric("쿠폰 집행 총액", money(float(exp_metrics.get('coupon_spend_total', 0.0))))
        cpic_val = exp_metrics.get('incremental_cpic', np.nan)
        _incremental_n = float(exp_metrics.get('incremental_retained_customers', 0.0))
        if pd.notna(cpic_val):
            m4.metric("CPIC", money(float(cpic_val)))
        elif _incremental_n <= 0:
            m4.metric("CPIC", "측정 불가", help="추가 유지 고객 수가 0 이하라 분모가 정의되지 않습니다. 효과 검출 실패 — 표본 확대 후 재측정 필요.")
        else:
            m4.metric("CPIC", "-")
        m5.metric("Z-test p-value", f"{float(exp_metrics.get('p_value', np.nan)):.6f}" if pd.notna(exp_metrics.get('p_value', np.nan)) else "-")

        tab1, tab2, tab3 = st.tabs(["A/B 해석", "개입 강도 효과", "Persuadables 프로필"])

        with tab1:
            ab_test = experiment_overview.get("ab_test", {})
            if ab_test:
                # ── p-value 의미 해석 박스 (개선안 2) ──
                _p_val = exp_metrics.get('p_value', np.nan)
                if pd.notna(_p_val):
                    _p_float = float(_p_val)
                    if _p_float >= 0.05:
                        st.info(
                            f"📊 **p = {_p_float:.4f} 의 의미**\n\n"
                            f"이 수치는 'Treatment와 Control 사이에 차이가 없다'는 가설이 매우 그럴듯하다는 뜻입니다. "
                            f"즉 관측된 증분 리텐션은 **캠페인 실패의 증거가 아니라, 효과를 측정할 수 없었다는 증거**입니다. "
                            f"통계적으로 유의한 결론을 도출하려면 표본 확대 또는 효과 크기 증가가 필요합니다."
                        )
                    else:
                        st.success(
                            f"📊 **p = {_p_float:.4f}** — 두 그룹 간 차이가 통계적으로 유의합니다 (α=0.05 기준)."
                        )
                report_md = ab_test.get("report_markdown", "")
                if report_md:
                    st.markdown(report_md)
            else:
                st.warning("A/B 테스트 산출물을 찾지 못했습니다.")

        with tab2:
            dose_df = experiment_overview.get("dose_df", pd.DataFrame())
            if not dose_df.empty:
                chart_df = dose_df.copy()
                fig = px.bar(
                    chart_df,
                    x="arm",
                    y="retention_rate",
                    hover_data=["samples", "avg_coupon_cost", "effect_prior", "cost_multiplier"],
                    title="개입 강도별 retention rate",
                )
                st.plotly_chart(fig, use_container_width=True)
                display_df = dose_df.copy()
                for col in ["retention_rate", "effect_prior"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}")
                for col in ["avg_coupon_cost", "avg_revenue_post_horizon"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(money)
                _render_dataframe_with_count(display_df, label="dose-response arm 요약")
            else:
                st.warning("dose-response 요약을 찾지 못했습니다.")

        # ── What-if 시나리오 카드 (개선안 3) ──
        st.markdown("---")
        st.markdown("### 💡 What-if: 충분한 표본/효과 크기 시 예상 성과")
        st.caption("현재 표본의 검출력 한계를 보완하기 위해, 효과 크기 가정별 운영 시나리오를 계산합니다. 실제 운영 데이터 누적 후 본 시스템이 동일 분석을 자동 수행합니다.")

        _sample_sizes = _ab_test_meta.get("sample_sizes", {}) or {}
        _business = _ab_test_meta.get("business_metrics", {}) or {}
        _treat_n = float(_sample_sizes.get("treatment", 0)) or float(_current_min_n or 0)
        _coupon_total = float(_business.get("treatment_coupon_cost_total", 0.0)) or float(exp_metrics.get('coupon_spend_total', 0.0))
        # 1인당 매출 추정: 증분 매출 총액 / Treatment 표본 수. 음수면 절댓값으로 추정한 평균 매출 사용.
        _inc_revenue_per_treated = abs(float(_business.get("incremental_revenue_per_treated_customer", 0.0)))
        # 추정 평균 매출 = 1%p 증분당 1명당 매출 환산. 데이터 없으면 100,000원 기본값.
        _avg_revenue_per_retained = _inc_revenue_per_treated * 100 if _inc_revenue_per_treated > 0 else 100000
        _scenarios = [
            ("보수적 (+1%p)", 0.01),
            ("중간 (+2%p)", 0.02),
            ("낙관적 (+5%p)", 0.05),
        ]
        _whatif_rows = []
        for _label, _lift in _scenarios:
            _additional_retained = _treat_n * _lift
            _additional_revenue = _additional_retained * _avg_revenue_per_retained
            _net_profit = _additional_revenue - _coupon_total
            _roi = (_net_profit / _coupon_total) if _coupon_total > 0 else 0.0
            _cpic = (_coupon_total / _additional_retained) if _additional_retained > 0 else 0.0
            _whatif_rows.append({
                "시나리오": _label,
                "증분 리텐션": f"+{_lift*100:.1f}%p",
                "추가 유지 고객": f"{_additional_retained:,.0f}명",
                "추가 매출": money(_additional_revenue),
                "쿠폰비 반영 ROI": f"{_roi*100:+.1f}%",
                "CPIC": money(_cpic) if _additional_retained > 0 else "-",
            })
        _whatif_df = pd.DataFrame(_whatif_rows)
        _render_dataframe_with_count(_whatif_df, label="효과 크기 가정별 시뮬레이션", prefer_static=True)

        st.caption(
            "※ 본 표는 동일 표본·쿠폰비 조건에서 효과 크기만 가정해 산출한 추정치입니다. "
            "현재 시뮬레이터 표본으로는 실제 효과 크기를 신뢰성 있게 검출할 수 없으므로, "
            "운영 데이터가 누적되면 본 시스템이 동일 방식으로 실효 ROI를 자동 산출하도록 설계되어 있습니다."
        )
        
        with tab3:
            persuadables = experiment_overview.get("persuadables", {})
            st.metric("Persuadables 비중", pct(float(persuadables.get('persuadables_share', 0.0))))
            rules = persuadables.get("derived_targeting_rules", [])
            if rules:
                st.markdown("### 도출된 타겟팅 규칙")
                for rule in rules:
                    st.markdown(f"- {rule}")
            numeric_deltas = experiment_overview.get("numeric_deltas", pd.DataFrame())
            if not numeric_deltas.empty:
                _render_dataframe_with_count(numeric_deltas, label="Persuadables 수치 프로필 차이")

    llm_payload = {
        "experiment_metrics": exp_metrics,
        "dose_response": experiment_overview.get("dose_df", pd.DataFrame()).to_dict(orient="records") if not experiment_overview.get("dose_df", pd.DataFrame()).empty else [],
        "persuadables": experiment_overview.get("persuadables", {}),
    }

elif view == "11. 설명가능성 / 고객별 개입 이유":
    st.subheader("설명가능성 / 고객별 개입 이유")
    st.caption("왜 이 고객이 위험군인지, 왜 개입 후보로 뽑혔는지, 무엇을 조심해야 하는지를 운영 언어로 풀어 보여줍니다.")

    _explain_has_data = bool(
        not global_feature_table.empty
        or not customer_explanations.empty
        or not operational_overview.get("persona_df", pd.DataFrame()).empty
    )
    if not _explain_has_data:
        _simulator_missing_result_box(
            "설명가능성 / 고객별 개입 이유",
            "전역 feature importance, 고객별 설명 테이블, 운영 요약 산출물을 찾지 못했습니다.",
            "시뮬레이터 데모에서는 train/explain/recommend 관련 명령을 먼저 실행한 뒤 새로고침하세요.",
        )
    else:
        tab1, tab2 = st.tabs(["전역 설명", "고객별 설명"])

        with tab1:
            if not global_feature_table.empty:
                chart_df = global_feature_table.head(10).copy()
                fig = px.bar(chart_df.iloc[::-1], x="importance", y="feature_display", orientation="h", title="전역 중요 변수 Top 10")
                st.plotly_chart(fig, use_container_width=True)
                display_df = global_feature_table[["feature_display", "importance", "importance_share"]].copy()
                display_df.columns = ["feature", "importance", "importance_share"]
                display_df["importance"] = display_df["importance"].map(lambda x: f"{float(x):.4f}")
                display_df["importance_share"] = display_df["importance_share"].map(lambda x: f"{float(x):.2%}")
                _render_dataframe_with_count(display_df, label="전역 중요 변수")
            else:
                st.warning("전역 중요 변수 파일을 찾지 못했습니다.")

            if not operational_overview.get("persona_df", pd.DataFrame()).empty:
                persona_reason_df = operational_overview["persona_df"].copy()
                if "avg_churn_probability" in persona_reason_df.columns:
                    persona_reason_df["avg_churn_probability"] = persona_reason_df["avg_churn_probability"].map(lambda x: f"{float(x):.3f}")
                if "avg_uplift_score" in persona_reason_df.columns:
                    persona_reason_df["avg_uplift_score"] = persona_reason_df["avg_uplift_score"].map(lambda x: f"{float(x):.3f}")
                if "avg_clv" in persona_reason_df.columns:
                    persona_reason_df["avg_clv"] = persona_reason_df["avg_clv"].map(money)
                _render_dataframe_with_count(persona_reason_df, label="페르소나별 위험·가치 프로필")

        with tab2:
            if not customer_explanations.empty:
                display_df = customer_explanations.copy()
                for col in ["churn_probability", "realtime_churn_score", "uplift_score", "expected_roi", "survival_prob_30d"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
                for col in ["clv", "expected_incremental_profit"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: money(float(x)) if pd.notna(x) else "")
                _render_dataframe_with_count(display_df, label="고객별 선택 이유 / 주의사항", height=min(760, 220 + 34 * len(display_df)))
            else:
                st.warning("설명가능성 테이블을 만들 데이터가 부족합니다.")

    llm_payload = {
        "global_feature_table": global_feature_table.head(15).to_dict(orient="records") if not global_feature_table.empty else [],
        "customer_explanations": customer_explanations.head(20).to_dict(orient="records") if not customer_explanations.empty else [],
    }

elif view == "12. 데이터 진단 / 시뮬레이터 충실도":
    st.subheader("데이터 진단 / 시뮬레이터 충실도")
    st.caption("시뮬레이터가 만든 원천 데이터와 파생 산출물이 운영형 분석에 쓰기 적절한지, 기본적인 정합성과 분포를 함께 점검합니다.")

    checks_df = data_diagnostics.get("checks_df", pd.DataFrame())
    volumes_df = data_diagnostics.get("volumes_df", pd.DataFrame())
    event_mix_df = data_diagnostics.get("event_mix_df", pd.DataFrame())
    distribution_df = data_diagnostics.get("distribution_df", pd.DataFrame())

    _diagnostics_has_data = bool(
        not checks_df.empty
        or not volumes_df.empty
        or not event_mix_df.empty
        or not distribution_df.empty
    )
    if not _diagnostics_has_data:
        _simulator_missing_result_box(
            "데이터 진단 / 시뮬레이터 충실도",
            "시뮬레이터 원천 데이터/산출 데이터 볼륨, 행동 분포, 고객 분포 진단 결과를 찾지 못했습니다.",
            "시뮬레이터 데모에서는 simulate, features, fidelity 관련 명령을 먼저 실행한 뒤 새로고침하세요.",
        )
    else:
        if not checks_df.empty:
            status_counts = checks_df["status"].value_counts().to_dict()
            st.info(f"양호 {status_counts.get('양호', 0)}개 / 주의 {status_counts.get('주의', 0)}개 점검 항목")
            _render_dataframe_with_count(checks_df, label="정합성 점검 결과", prefer_static=True)

        tab1, tab2, tab3 = st.tabs(["데이터 볼륨", "행동 분포", "고객 분포"])

        with tab1:
            _render_dataframe_with_count(volumes_df, label="원천/산출 데이터 볼륨", prefer_static=True)

        with tab2:
            if not event_mix_df.empty:
                fig = px.bar(event_mix_df, x="event_type", y="count", title="이벤트 타입 분포", text="count")
                st.plotly_chart(fig, use_container_width=True)
                display_df = event_mix_df.copy()
                if "share" in display_df.columns:
                    display_df["share"] = display_df["share"].map(lambda x: f"{float(x):.2%}")
                _render_dataframe_with_count(display_df, label="이벤트 타입 분포", prefer_static=True)
            else:
                st.warning("이벤트 분포를 계산할 데이터가 없습니다.")

        with tab3:
            if not distribution_df.empty:
                selected_dimension = st.selectbox("분포 차원 선택", options=sorted(distribution_df["dimension"].unique()), key="diagnostic_dimension")
                subset = distribution_df[distribution_df["dimension"] == selected_dimension].copy()
                fig = px.bar(subset, x="value", y="count", title=f"{selected_dimension} 분포", text="count")
                st.plotly_chart(fig, use_container_width=True)
                subset["share"] = subset["share"].map(lambda x: f"{float(x):.2%}")
                _render_dataframe_with_count(subset, label=f"{selected_dimension} 분포", prefer_static=True)
            else:
                st.warning("고객 분포를 계산할 데이터가 없습니다.")

    llm_payload = {
        "checks": checks_df.to_dict(orient="records") if not checks_df.empty else [],
        "volumes": volumes_df.to_dict(orient="records") if not volumes_df.empty else [],
        "event_mix": event_mix_df.head(20).to_dict(orient="records") if not event_mix_df.empty else [],
        "distribution": distribution_df.head(30).to_dict(orient="records") if not distribution_df.empty else [],
    }

elif view == "7. 할인·쿠폰 운영 리스크":
    _coupon_has_data = False
    if isinstance(coupon_risk_overview, dict):
        _coupon_has_data = bool(
            _nonempty_mapping(coupon_risk_overview.get("metrics", {}))
            or not coupon_risk_overview.get("flags_df", pd.DataFrame()).empty
            or not coupon_risk_overview.get("segment_df", pd.DataFrame()).empty
            or not coupon_risk_overview.get("recommendation_mix", pd.DataFrame()).empty
            or not coupon_risk_overview.get("intensity_mix", pd.DataFrame()).empty
        )
    if _simulator_mode_unavailable(
        "할인·쿠폰 운영 리스크",
        _coupon_has_data,
        "쿠폰 노출/리딤/믹스 리스크 산출물이 없습니다.",
        "시뮬레이터 데모에서는 recommend, abtest 또는 관련 운영 분석 산출물을 먼저 생성한 뒤 새로고침하세요.",
    ):
        st.stop()
    st.subheader("할인·쿠폰 운영 리스크")
    st.caption("쿠폰 노출 누적, 리딤 효율, 강도별 효과, 추천/개입 믹스를 같이 보면서 할인 남발의 부작용 가능성을 점검합니다.")

    risk_metrics = coupon_risk_overview.get("metrics", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("노출 고객 수", f"{int(risk_metrics.get('exposed_customers', 0)):,}명")
    m2.metric("고노출 고객 수", f"{int(risk_metrics.get('high_exposure_customers', 0)):,}명")
    m3.metric("전체 노출 수", f"{int(risk_metrics.get('total_exposures', 0)):,}회")
    m4.metric("오픈율", pct(float(risk_metrics.get('open_rate', 0.0))) if pd.notna(risk_metrics.get('open_rate', np.nan)) else "-")
    m5.metric("리딤률", pct(float(risk_metrics.get('redeem_rate', 0.0))) if pd.notna(risk_metrics.get('redeem_rate', np.nan)) else "-")

    flags_df = coupon_risk_overview.get("flags_df", pd.DataFrame())
    if not flags_df.empty:
        _render_dataframe_with_count(flags_df, label="쿠폰 운영 리스크 플래그", prefer_static=True)

    tab1, tab2, tab3 = st.tabs(["페르소나별 노출", "추천/강도 믹스", "운영 해석"])

    with tab1:
        segment_df = coupon_risk_overview.get("segment_df", pd.DataFrame())
        if not segment_df.empty:
            fig = px.bar(segment_df.head(12), x="persona", y="avg_coupon_exposure", hover_data=[col for col in ["avg_churn_probability", "avg_expected_roi"] if col in segment_df.columns], title="페르소나별 평균 쿠폰 노출")
            st.plotly_chart(fig, use_container_width=True)
            display_df = segment_df.copy()
            for col in ["avg_churn_probability", "avg_expected_roi"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}")
            _render_dataframe_with_count(display_df, label="페르소나별 쿠폰 노출/성과")
        else:
            st.warning("쿠폰 노출 집계를 계산할 데이터가 없습니다.")

    with tab2:
        left, right = st.columns(2)
        recommendation_mix = coupon_risk_overview.get("recommendation_mix", pd.DataFrame())
        intensity_mix = coupon_risk_overview.get("intensity_mix", pd.DataFrame())
        with left:
            if not recommendation_mix.empty:
                fig = px.pie(recommendation_mix, names="recommended_category", values="count", title="추천 카테고리 믹스")
                st.plotly_chart(fig, use_container_width=True)
        with right:
            if not intensity_mix.empty:
                fig = px.bar(intensity_mix, x="intervention_intensity", y="count", title="선정된 개입 강도 믹스", text="count")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        high_prior = insight_bundle.dose_response_summary.get("effect_priors", {}).get("high") if insight_bundle.dose_response_summary else None
        st.markdown("### 운영 해석")
        if high_prior is not None:
            st.markdown(
                "- 고강도 개입의 prior effect가 음수이면 혜택을 세게 줄수록 오히려 성과가 악화될 수 있습니다.\n"
                f"- 현재 high 강도 prior effect: **{float(high_prior):.3f}**"
            )
        else:
            st.markdown("- high 강도 prior effect를 찾지 못했습니다.")
        st.markdown(
            "- 노출 고객 수와 리딤률을 함께 봐야 합니다. 노출은 많은데 리딤이 낮으면 학습효과/피로 누적 가능성이 큽니다.\n"
            "- price_sensitive 성향이 강한 고객군은 단기 반응은 좋을 수 있지만, 장기적으로는 마진 희석과 할인 의존이 커질 수 있습니다.\n"
            "- support 이슈형 고객은 쿠폰보다 서비스 회복 메시지나 CS 해결이 더 나을 수 있습니다."
        )

    llm_payload = {
        "coupon_risk_metrics": risk_metrics,
        "risk_flags": flags_df.to_dict(orient="records") if not flags_df.empty else [],
        "segment_df": coupon_risk_overview.get("segment_df", pd.DataFrame()).head(15).to_dict(orient="records") if not coupon_risk_overview.get("segment_df", pd.DataFrame()).empty else [],
        "intensity_mix": coupon_risk_overview.get("intensity_mix", pd.DataFrame()).to_dict(orient="records") if not coupon_risk_overview.get("intensity_mix", pd.DataFrame()).empty else [],
    }

current_view_key = view.split(".")[0]
current_model_name = llm_model.strip() or DEFAULT_MODEL_NAME

if llm_enabled:
    render_llm_summary(
        view_key=current_view_key,
        view_title=llm_view_title,
        payload=llm_payload,
        api_key=llm_api_key_value,
        model_name=current_model_name,
    )

with st.sidebar:
    render_sidebar_chatbot_launcher(
        view_key=current_view_key,
        view_title=llm_view_title,
        llm_enabled=llm_enabled,
        api_key=llm_api_key_value,
        payload=llm_payload,
        model_name=current_model_name,
    )

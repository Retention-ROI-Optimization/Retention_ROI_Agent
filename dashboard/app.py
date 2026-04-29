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
from dashboard.services.decision_engine_service import (
    aggregate_enhanced_segment_allocation,
    get_baseline_budget_result,
    get_decision_engine_factor_table,
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
    ("1", "이탈현황"),
    ("2", "코호트 리텐션 곡선"),
    ("3", "Uplift + CLV 상위 고객"),
    ("4", "예산 배분 결과"),
    ("5", "예상 최적화 ROI"),
    ("6", "리텐션 대상 고객 목록"),
    ("7", "학습 결과 아티팩트"),
    ("8", "Uplift/최적화 결과 (실시간)"),
    ("9", "개인화 추천"),
    ("10", "실시간 위험 스코어링 / 운영 모니터"),
    ("11", "이탈 시점 예측 (Survival Analysis)"),
    ("12", "의사결정 엔진 비교"),
    ("13", "운영 한눈에 보기"),
    ("14", "증분 성과 / A-B 실험"),
    ("15", "설명가능성 / 고객별 개입 이유"),
    ("16", "데이터 진단 / 시뮬레이터 충실도"),
    ("17", "할인·쿠폰 운영 리스크"),
)

DASHBOARD_VIEW_OPTIONS: tuple[str, ...] = tuple(f"{n}. {t}" for n, t in DASHBOARD_VIEW_ITEMS)
VIEW_OPTION_BY_NUM: dict[str, str] = {num: f"{num}. {title}" for num, title in DASHBOARD_VIEW_ITEMS}
DASHBOARD_VIEW_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("핵심 분석", ("1", "2", "3", "4", "5", "6", "7")),
    ("실시간·예측", ("8", "9", "10", "11", "12")),
    ("운영 인사이트", ("13", "14", "15", "16", "17")),
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
REALTIME_REFRESH_VIEWS: set[str] = {
    "8. Uplift/최적화 결과 (실시간)",
    "10. 실시간 위험 스코어링 / 운영 모니터",
    "13. 운영 한눈에 보기",
}
INSIGHT_HEAVY_VIEWS: set[str] = {
    "10. 실시간 위험 스코어링 / 운영 모니터",
    "13. 운영 한눈에 보기",
    "14. 증분 성과 / A-B 실험",
    "15. 설명가능성 / 고객별 개입 이유",
    "16. 데이터 진단 / 시뮬레이터 충실도",
    "17. 할인·쿠폰 운영 리스크",
}


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
        }

        .oai-table-wrapper thead th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: #f8fafc;
            color: #0f172a;
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
    return _file_version_token([
        "data/raw/customer_summary.csv",
        "data/raw/cohort_retention.csv",
    ])


def _result_data_token() -> str:
    return _file_version_token([
        "results/churn_top10_feature_importance.json",
        "results/optimization_selected_customers.csv",
        "results/personalized_recommendations.csv",
        "results/realtime_scores_snapshot.csv",
        "results/realtime_scores_summary.json",
        "results/realtime_action_queue_snapshot.csv",
        "results/realtime_action_queue_summary.json",
        "results/survival_predictions.csv",
        "results/uplift_segmentation.csv",
        "results/ab_test_results.json",
        "results/dose_response_summary.json",
        "results/customer_segment_summary.json",
        "results/persuadables_analysis.json",
        "results/optimization_summary.json",
        "results/personalized_recommendation_summary.json",
        "results/clv_validation_metrics.json",
        "results/feature_engineering_summary.json",
        "results/churn_metrics.json",
    ])


@st.cache_data(show_spinner=False)
def _load_app_bundle_cached(_token: str):
    return load_dashboard_bundle(include_optional=False)


@st.cache_data(show_spinner=False)
def _load_insight_bundle_cached(_raw_token: str, _result_token: str):
    return load_dashboard_insight_bundle()


def load_app_data():
    return _load_app_bundle_cached(_raw_data_token())


def load_insight_data():
    return _load_insight_bundle_cached(_raw_data_token(), _result_data_token())


def clear_dashboard_caches() -> None:
    _load_app_bundle_cached.clear()
    _load_insight_bundle_cached.clear()


def load_training_artifacts_api():
    return fetch_training_artifacts()


def load_saved_results_artifacts_api(
    budget: int,
    threshold: float,
    max_customers: int | None,
    rebuild: bool = False,
):
    return fetch_saved_results_artifacts(
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
        rebuild=rebuild,
    )


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


def render_sidebar_chatbot_launcher(
    view_key: str,
    view_title: str,
    llm_enabled: bool,
    api_key: Optional[str],
):
    st.divider()
    st.subheader("AI 챗봇")

    chatbot_image_path = resolve_chatbot_image()
    if chatbot_image_path:
        st.image(chatbot_image_path, use_container_width=True)
        st.caption("현재 화면의 표·그래프를 바탕으로 대화를 이어갈 수 있습니다.")
    else:
        st.markdown(
            f"""
            <div class="sidebar-chatbot-card">
                <div class="sidebar-chatbot-emoji">🤖</div>
                <div class="sidebar-chatbot-title">AI 분석 챗봇</div>
                <div class="sidebar-chatbot-desc">
                    현재 보고 있는 표·그래프를 함께 보면서
                    질문을 이어갈 수 있습니다.
                </div>
                <div class="chatbot-view-chip">{view_title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ready, status_message = get_llm_status(api_key)

    if st.button(
        "🤖 AI 챗봇 열기",
        key=f"open_chatbot_{view_key}",
        use_container_width=True,
        disabled=(not llm_enabled) or (not ready),
    ):
        st.session_state["llm_chat_open"] = True
        st.session_state["llm_chat_view_key"] = view_key
        st.rerun()

    if not llm_enabled:
        st.caption("LLM 기능이 꺼져 있어 챗봇을 열 수 없습니다.")
    elif not ready:
        pass
    else:
        st.caption("현재 화면 문맥을 유지한 채 질문할 수 있습니다.")


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
    render_status_pill(f"실제 시뮬레이터 산출물 사용 중: {bundle.source_dir}", "success")

with st.sidebar:
    st.header("제어 패널")

    st.session_state.setdefault("dashboard_view", DASHBOARD_VIEW_OPTIONS[0])
    st.session_state.setdefault("dashboard_group", VIEW_TO_GROUP.get(st.session_state["dashboard_view"], DASHBOARD_VIEW_GROUPS[0][0]))
    st.session_state.setdefault("control_threshold", 0.50)
    st.session_state.setdefault("control_budget", 5_000_000)
    st.session_state.setdefault("control_top_n", 25)
    st.session_state.setdefault("control_target_cap", 1500)
    st.session_state.setdefault("control_recommendation_per_customer", 3)

    default_view = st.session_state.get("dashboard_view", DASHBOARD_VIEW_OPTIONS[0])
    if default_view not in DASHBOARD_VIEW_OPTIONS:
        default_view = DASHBOARD_VIEW_OPTIONS[0]
        st.session_state["dashboard_view"] = default_view
    default_group = VIEW_TO_GROUP.get(default_view, DASHBOARD_VIEW_GROUPS[0][0])
    if st.session_state.get("dashboard_group") not in [g for g, _ in DASHBOARD_VIEW_GROUPS]:
        st.session_state["dashboard_group"] = default_group

    group_labels = [group for group, _ in DASHBOARD_VIEW_GROUPS]
    selected_group = st.selectbox("대분류", options=group_labels, key="dashboard_group")

    group_options = list(GROUP_TO_VIEW_OPTIONS.get(selected_group, DASHBOARD_VIEW_OPTIONS))
    if st.session_state.get("dashboard_view") not in group_options:
        st.session_state["dashboard_view"] = group_options[0]

    view = st.radio(
        "세부 화면",
        options=group_options,
        format_func=_view_title_from_option,
        key="dashboard_view",
        label_visibility="visible",
    )

    threshold = float(st.session_state["control_threshold"])
    budget = int(st.session_state["control_budget"])
    top_n = int(st.session_state["control_top_n"])
    target_cap = int(st.session_state["control_target_cap"])
    recommendation_per_customer = int(st.session_state["control_recommendation_per_customer"])

    if view in {"1. 이탈현황", "4. 예산 배분 결과", "5. 예상 최적화 ROI", "6. 리텐션 대상 고객 목록", "8. Uplift/최적화 결과 (실시간)", "9. 개인화 추천", "12. 의사결정 엔진 비교", "13. 운영 한눈에 보기"}:
        threshold = st.slider(
            "이탈 Threshold",
            min_value=0.10,
            max_value=0.90,
            step=0.01,
            key="control_threshold",
            help="이 값 이상인 고객을 이탈 위험군으로 간주합니다.",
        )

    if view in {"10. 실시간 위험 스코어링 / 운영 모니터", "11. 이탈 시점 예측 (Survival Analysis)", "15. 설명가능성 / 고객별 개입 이유", "17. 할인·쿠폰 운영 리스크"}:
        top_n = st.slider(
            "차트 기준 표시 고객 수",
            min_value=5,
            max_value=200,
            step=5,
            key="control_top_n",
        )

    if view == "9. 개인화 추천":
        st.caption("최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.")
        recommendation_per_customer = st.slider(
            "고객당 추천 개수",
            min_value=1,
            max_value=5,
            step=1,
            key="control_recommendation_per_customer",
        )

    if view in {"4. 예산 배분 결과", "5. 예상 최적화 ROI", "6. 리텐션 대상 고객 목록", "8. Uplift/최적화 결과 (실시간)", "9. 개인화 추천", "12. 의사결정 엔진 비교", "13. 운영 한눈에 보기"}:
        budget = int(st.number_input("총 마케팅 예산", step=100000, key="control_budget"))
        target_cap = st.slider(
            "최대 타겟 고객 수",
            min_value=100,
            max_value=5000,
            step=100,
            key="control_target_cap",
            help="예산이 충분하더라도 이 수를 넘겨 타겟팅하지 않습니다.",
        )

    if view == "9. 개인화 추천":
        preview_selected_customers, _, _ = get_budget_result(
            customers,
            budget=budget,
            threshold=threshold,
            max_customers=target_cap,
        )
        st.caption(f"현재 조건의 최종 타겟 고객 수: {int(len(preview_selected_customers)):,}명")

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

    render_sidebar_chatbot_launcher(
        view_key=view.split(".")[0],
        view_title=view,
        llm_enabled=llm_enabled,
        api_key=llm_api_key.strip() if llm_api_key else None,
    )


churn_summary, risk_customers = get_churn_status(customers, threshold)
cohort_curve = get_cohort_curve(cohort_df)
top_customers = get_top_high_value_customers(customers, top_n=None)
selected_customers, optimize_summary, segment_allocation = get_budget_result(
    customers,
    budget=budget,
    threshold=threshold,
    max_customers=target_cap,
)

if view == "12. 의사결정 엔진 비교":
    baseline_selected_customers, baseline_optimize_summary, baseline_segment_allocation = get_baseline_budget_result(
        customers,
        budget=budget,
        threshold=threshold,
        max_customers=target_cap,
    )
else:
    baseline_selected_customers, baseline_optimize_summary, baseline_segment_allocation = pd.DataFrame(), {}, pd.DataFrame()

retention_targets = get_retention_targets(customers, threshold)

if view == "9. 개인화 추천":
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

if view == "10. 실시간 위험 스코어링 / 운영 모니터":
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

if view == "11. 이탈 시점 예측 (Survival Analysis)":
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

if view in INSIGHT_HEAVY_VIEWS:
    insight_bundle = load_insight_data()
    if recommendation_context_df.empty:
        recommendation_context_df = insight_bundle.personalized_recommendations.copy()
    if survival_context_df.empty:
        survival_context_df = insight_bundle.survival_predictions.copy()
    if realtime_context_df.empty:
        realtime_context_df = insight_bundle.realtime_scores.copy()

    if view in {"13. 운영 한눈에 보기", "15. 설명가능성 / 고객별 개입 이유"}:
        operational_overview = build_operational_overview(
            customers=customers,
            selected_customers=selected_customers,
            optimize_summary=optimize_summary,
            recommendation_summary=recommendation_summary,
            realtime_summary=realtime_summary,
            survival_metrics=survival_metrics,
            insight_bundle=insight_bundle,
        )

    if view == "14. 증분 성과 / A-B 실험":
        experiment_overview = build_experiment_overview(insight_bundle)

    if view == "10. 실시간 위험 스코어링 / 운영 모니터":
        realtime_monitor_overview = build_realtime_monitor_overview(insight_bundle, fallback_scores=realtime_context_df)

    if view == "16. 데이터 진단 / 시뮬레이터 충실도":
        data_diagnostics = build_data_diagnostics(insight_bundle)

    if view == "17. 할인·쿠폰 운영 리스크":
        coupon_risk_overview = build_coupon_risk_overview(insight_bundle)

    if view == "15. 설명가능성 / 고객별 개입 이유":
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

elif view == "3. Uplift + CLV 상위 고객":
    st.subheader("Uplift Score + CLV 상위 고가치 고객 목록")

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
            "churn_probability",
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

    display_df = top_customers[
        [
            "customer_id",
            "persona",
            "churn_probability",
            "uplift_score",
            "clv",
            "value_score",
            "expected_incremental_profit",
            "uplift_segment",
        ]
    ].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
    display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
    display_df["clv"] = display_df["clv"].map(money)
    display_df["value_score"] = display_df["value_score"].map(money)
    display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
    _render_dataframe_with_count(display_df, label="상위 고객 테이블")

    llm_payload = {
        "top_n": int(len(top_customers)),
        "segment_distribution": series_distribution(plot_df, "uplift_segment"),
        "numeric_summary": numeric_summary(
            plot_df,
            ["uplift_score", "clv", "churn_probability", "expected_incremental_profit"],
        ),
        "top_customers": dataframe_snapshot(
            plot_df,
            columns=[
                "customer_id",
                "persona",
                "churn_probability",
                "uplift_score",
                "clv",
                "expected_incremental_profit",
                "uplift_segment",
            ],
            max_rows=15,
        ),
    }

elif view == "4. 예산 배분 결과":
    st.subheader("예산 배분 결과")
    st.caption("이 화면은 저장된 optimize 결과 파일이 아니라 현재 입력값으로 다시 계산한 결과입니다.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 예산", money(optimize_summary["budget"]))
    m2.metric("집행 예산", money(optimize_summary["spent"]))
    m3.metric("잔여 예산", money(optimize_summary["remaining"]))
    m4.metric("타겟 고객 수", f"{optimize_summary['num_targeted']:,}")

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

    if segment_allocation.empty or optimize_summary["num_targeted"] == 0:
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
            bar_fig.update_traces(textposition="outside", cliponaxis=False)
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
        display_df["allocated_budget"] = display_df["allocated_budget"].map(money)
        display_df["expected_profit"] = display_df["expected_profit"].map(money)
        _render_dataframe_with_count(display_df, label="세그먼트별 예산 배분 테이블")

    llm_payload = {
        "budget_summary": optimize_summary,
        "segment_allocation": segment_allocation.round(4).to_dict(orient="records"),
        "selected_customer_numeric_summary": numeric_summary(
            selected_customers, ["coupon_cost", "expected_incremental_profit", "expected_roi"]
        ),
    }

elif view == "5. 예상 최적화 ROI":
    st.subheader("예상 최적화 ROI")
    st.caption("이 화면도 현재 입력값 기준의 실시간 재계산 결과입니다.")

    m1, m2, m3 = st.columns(3)
    m1.metric("예상 증분 이익", money(optimize_summary["expected_incremental_profit"]))
    m2.metric("예상 ROI", pct(optimize_summary["overall_roi"]))
    m3.metric("선정 고객 수", f"{optimize_summary['num_targeted']:,}")

    top_roi = selected_customers.copy()
    if selected_customers.empty:
        st.warning("현재 조건에서 ROI 계산 대상이 없습니다.")
    else:
        roi_fig = px.histogram(
            selected_customers,
            x="expected_roi",
            nbins=25,
            title="선정 고객의 예상 ROI 분포",
        )
        roi_fig.update_traces(
            marker_line_color="rgba(255,255,255,0.95)",
            marker_line_width=1.2,
            opacity=0.9,
        )

        roi_fig.update_layout(
            bargap=0.02,
        )

        st.plotly_chart(roi_fig, use_container_width=True)

        top_roi = selected_customers.sort_values(["expected_roi", "expected_incremental_profit", "customer_id"], ascending=[False, False, True]).copy()
        display_df = top_roi[
            [
                "customer_id",
                "persona",
                "uplift_segment",
                "intervention_intensity",
                "recommended_action",
                "uplift_score",
                "clv",
                "coupon_cost",
                "expected_incremental_profit",
                "expected_roi",
            ]
        ].copy()
        display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
        display_df["clv"] = display_df["clv"].map(money)
        display_df["coupon_cost"] = display_df["coupon_cost"].map(money)
        display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
        display_df["expected_roi"] = display_df["expected_roi"].map(lambda x: f"{x:.2%}")
        _render_dataframe_with_count(display_df, label="최적화로 선정된 전체 고객 테이블", height=min(900, 180 + 32 * len(display_df)))

    llm_payload = {
        "optimize_summary": optimize_summary,
        "roi_numeric_summary": numeric_summary(
            selected_customers,
            ["expected_roi", "coupon_cost", "expected_incremental_profit"],
        ),
    }

elif view == "6. 리텐션 대상 고객 목록":
    st.subheader("리텐션 대상 고객 목록")
    st.caption("현재 budget / threshold / 최대 타겟 고객 수 조건에서 실제로 마케팅 대상으로 선정된 전체 고객을 보여줍니다.")

    optimized_targets = selected_customers.sort_values(
        ["priority_score", "selection_score", "expected_incremental_profit", "customer_id"],
        ascending=[False, False, False, True],
    ).copy()

    if optimized_targets.empty:
        st.warning("현재 조건에서 리텐션 타겟 고객이 없습니다.")
    else:
        priority_chart_df = optimized_targets.head(min(15, len(optimized_targets))).copy()
        priority_fig = px.bar(
            priority_chart_df,
            x="customer_id",
            y="priority_score",
            color="intervention_intensity" if "intervention_intensity" in priority_chart_df.columns else None,
            hover_data=["churn_probability", "uplift_score", "clv", "expected_incremental_profit", "expected_roi"],
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
            display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
        if "uplift_score" in display_df.columns:
            display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
        if "clv" in display_df.columns:
            display_df["clv"] = display_df["clv"].map(money)
        if "coupon_cost" in display_df.columns:
            display_df["coupon_cost"] = display_df["coupon_cost"].map(money)
        if "expected_incremental_profit" in display_df.columns:
            display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(money)
        if "expected_roi" in display_df.columns:
            display_df["expected_roi"] = display_df["expected_roi"].map(lambda x: f"{x:.2%}")
        if "priority_score" in display_df.columns:
            display_df["priority_score"] = display_df["priority_score"].map(lambda x: f"{x:.3f}")
        _render_dataframe_with_count(
            display_df,
            label="최적화 후 실제 선정된 전체 마케팅 고객 테이블",
            height=min(1100, 180 + 32 * len(display_df)),
        )

    llm_payload = {
        "threshold": threshold,
        "budget": budget,
        "target_count": int(len(optimized_targets)),
        "persona_distribution": series_distribution(optimized_targets, "persona"),
        "segment_distribution": series_distribution(optimized_targets, "uplift_segment"),
        "numeric_summary": numeric_summary(
            optimized_targets, ["priority_score", "selection_score", "churn_probability", "uplift_score", "clv", "expected_incremental_profit", "expected_roi"]
        ),
    }

elif view == "7. 학습 결과 아티팩트":
    st.subheader("학습 결과 아티팩트")
    st.caption("이 화면은 백엔드 API가 보관 중인 최신 학습 산출물을 읽기 전용으로 표시합니다. 대시보드에서 학습 파라미터를 조정하거나 재학습을 직접 실행하지 않습니다.")

    try:
        training_payload = load_training_artifacts_api()
    except Exception as exc:
        st.error(f"학습 결과 API 호출 실패: {exc}")
        training_payload = {}

    churn_metrics = training_payload.get("churn_metrics", {})
    threshold_analysis = training_payload.get("threshold_analysis", {})
    top_feature_importance_df = _artifact_frame(training_payload.get("top_feature_importance"))
    customer_features_df = _artifact_frame(training_payload.get("customer_features"), max_columns=16)
    image_paths = training_payload.get("image_paths", {})
    model_paths = training_payload.get("model_paths", {})
    training_parameters = training_payload.get("training_parameters", {}) or churn_metrics.get("training_parameters", {})

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
            if img_path:
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

elif view == "8. Uplift/최적화 결과 (실시간)":
    st.subheader("Uplift/최적화 결과 (실시간)")
    st.caption("Uplift 결과는 최신 raw 데이터를 기준으로 필요 시 다시 만들고, 최적화 결과는 현재 budget/threshold/max-customers 조건으로 즉시 다시 계산합니다.")
    rebuild_saved_results = st.button("현재 조건으로 Uplift/최적화 다시 계산", key="rebuild_saved_results")

    try:
        saved_payload = load_saved_results_artifacts_api(
            int(budget),
            float(threshold),
            int(target_cap) if target_cap else None,
            rebuild=rebuild_saved_results,
        )
    except Exception as exc:
        st.error(f"저장 결과 API 호출 실패: {exc}")
        saved_payload = {}

    uplift_summary = saved_payload.get("uplift_summary", {})
    uplift_segmentation_df = _artifact_frame(saved_payload.get("uplift_segmentation"))
    optimization_summary = saved_payload.get("optimization_summary", {})
    optimization_segment_budget_df = _artifact_frame(saved_payload.get("optimization_segment_budget"))
    optimization_selected_customers_df = _artifact_frame(saved_payload.get("optimization_selected_customers"))
    saved_parameters = saved_payload.get("parameters", {})

    if saved_parameters:
        st.caption(
            f"현재 반영 조건 · budget={money(saved_parameters.get('budget', 0))}, "
            f"threshold={float(saved_parameters.get('threshold', 0.0)):.2f}, "
            f"max_customers={int(saved_parameters.get('max_customers') or 0):,}"
        )

    uplift_tab, optimize_tab = st.tabs(["Uplift 결과", "Optimize 결과"])

    with uplift_tab:
        if not uplift_summary and uplift_segmentation_df.empty:
            st.warning("저장된 uplift 결과를 찾지 못했습니다.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Uplift rows", int(uplift_summary.get("rows", len(uplift_segmentation_df))))
            segment_counts = uplift_summary.get("segment_counts", {})
            m2.metric("세그먼트 종류 수", len(segment_counts))

            if segment_counts:
                seg_df = pd.DataFrame(
                    {
                        "uplift_segment": list(segment_counts.keys()),
                        "customer_count": list(segment_counts.values()),
                    }
                )
                fig = px.bar(seg_df, x="uplift_segment", y="customer_count", text="customer_count")
                st.plotly_chart(fig, use_container_width=True)

            if not uplift_segmentation_df.empty:
                _render_dataframe_with_count(
                    uplift_segmentation_df,
                    label="Uplift 세그먼트 미리보기",
                )

    with optimize_tab:
        if not optimization_summary and optimization_segment_budget_df.empty:
            st.warning("저장된 optimize 결과를 찾지 못했습니다.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("현재 예산", money(optimization_summary.get("budget", 0)))
            m2.metric("현재 집행 예산", money(optimization_summary.get("spent", 0)))
            m3.metric("현재 잔여 예산", money(optimization_summary.get("remaining", 0)))
            m4.metric("현재 타겟 고객 수", f"{int(optimization_summary.get('num_targeted', 0)):,}")

            intensity_counts = optimization_summary.get("selected_intensity_counts", {}) if optimization_summary else {}
            if intensity_counts:
                st.markdown("### 선택된 개입 강도 구성")
                intensity_df = pd.DataFrame({
                    "intervention_intensity": list(intensity_counts.keys()),
                    "customer_count": list(intensity_counts.values()),
                })
                intensity_fig = px.bar(
                    intensity_df,
                    x="intervention_intensity",
                    y="customer_count",
                    text="customer_count",
                    title="선택된 고객의 개입 강도 분포",
                )
                st.plotly_chart(intensity_fig, use_container_width=True)

            if not optimization_segment_budget_df.empty:
                display_df = optimization_segment_budget_df.copy()
                if "allocated_budget" in display_df.columns:
                    display_df["allocated_budget"] = display_df["allocated_budget"].map(money)
                if "expected_profit" in display_df.columns:
                    display_df["expected_profit"] = display_df["expected_profit"].map(money)
                result_heading = "### 세그먼트·강도별 실시간 결과" if "intervention_intensity" in display_df.columns else "### 세그먼트별 실시간 결과"
                result_label = "세그먼트·강도별 실시간 결과" if "intervention_intensity" in display_df.columns else "세그먼트별 실시간 결과"
                st.markdown(result_heading)
                _render_dataframe_with_count(display_df, label=result_label)

            if not optimization_selected_customers_df.empty:
                st.markdown("### 현재 조건에서 선정된 고객")
                _render_dataframe_with_count(
                    optimization_selected_customers_df,
                    label="현재 조건에서 선정된 고객",
                )

    llm_payload = {
        "uplift_summary": uplift_summary,
        "optimization_summary": optimization_summary,
        "optimization_segment_budget": optimization_segment_budget_df.to_dict(orient="records") if not optimization_segment_budget_df.empty else [],
        "optimization_selected_preview": dataframe_snapshot(
            optimization_selected_customers_df,
            columns=list(optimization_selected_customers_df.columns[:12]),
            max_rows=12,
        ) if not optimization_selected_customers_df.empty else [],
    }

elif view == "9. 개인화 추천":
    st.subheader("최종 타겟 고객 대상 개인화 추천")
    st.caption("예산/임계값으로 선별된 최종 리텐션 타겟 고객에게만 추천을 생성합니다. 추천 점수는 구매 이력 + 최근 관심 + 세그먼트 인기 + 전역 인기를 혼합해 계산합니다.")

    if recommendation_error:
        st.error(f"추천 API 호출 실패: {recommendation_error}")
    elif personalized_recommendations.empty:
        st.warning("표시할 추천 결과가 없습니다. 현재 예산/임계값 조건에서 최종 타겟 고객이 없을 수 있습니다.")
    else:
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

elif view == "10. 실시간 위험 스코어링 / 운영 모니터":
    st.subheader("실시간 위험 스코어링 / 운영 모니터")
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

elif view == "11. 이탈 시점 예측 (Survival Analysis)":
    st.subheader("이탈 시점 예측 (Survival Analysis)")
    st.caption('Cox Proportional Hazards 기반으로 landmark 시점 이후 얼마 안에 churn risk 상태로 진입할지를 추정합니다. 분류 모델과 달리 "언제" 위험이 커지는지를 함께 봅니다.')

    if survival_error:
        st.error(f"Survival API 호출 실패: {survival_error}")
    elif not survival_metrics:
        st.warning("Survival 분석 결과를 아직 불러오지 못했습니다.")
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
        if risk_plot:
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

elif view == "12. 의사결정 엔진 비교":
    st.subheader("의사결정 엔진 비교")
    st.caption("기존 예산 최적화(이탈·업리프트·ROI 중심)와 현재 의사결정 엔진(이탈 시점 + intervention window + 개입 강도)을 같은 예산 조건에서 비교합니다.")

    enhanced_segment_summary = aggregate_enhanced_segment_allocation(segment_allocation)
    factor_table = get_decision_engine_factor_table()

    baseline_profit = float(baseline_optimize_summary.get("expected_incremental_profit", 0.0))
    enhanced_profit = float(optimize_summary.get("expected_incremental_profit", 0.0))
    baseline_roi = float(baseline_optimize_summary.get("overall_roi", 0.0))
    enhanced_roi = float(optimize_summary.get("overall_roi", 0.0))
    baseline_targeted = int(baseline_optimize_summary.get("num_targeted", 0))
    enhanced_targeted = int(optimize_summary.get("num_targeted", 0))
    enhanced_window = float(optimize_summary.get("avg_intervention_window_days", 0.0) or 0.0)
    enhanced_urgency = float(optimize_summary.get("avg_timing_urgency_score", 0.0) or 0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "예상 증분 이익 변화",
        money(enhanced_profit),
        delta=money(enhanced_profit - baseline_profit),
    )
    m2.metric(
        "ROI 변화",
        pct(enhanced_roi),
        delta=f"{(enhanced_roi - baseline_roi) * 100:.2f}%p",
    )
    m3.metric(
        "선정 고객 수 변화",
        f"{enhanced_targeted:,}",
        delta=f"{enhanced_targeted - baseline_targeted:+d}명",
    )
    m4.metric(
        "평균 개입 윈도우",
        f"{enhanced_window:.1f}일",
        delta=f"긴급도 {enhanced_urgency:.3f}",
    )

    st.info(
        "현재 엔진은 '누가 위험한가'뿐 아니라 '얼마나 빨리 떠날 것 같은가'와 '어느 강도의 개입이 더 맞는가'까지 함께 고려합니다. "
        "그래서 같은 예산이어도 단순 고ROI 고객 모음이 아니라, 더 시급한 고객에게 적절한 강도로 예산이 재배분됩니다."
    )

    comparison_rows = pd.DataFrame([
        {
            "비교 항목": "선정 고객 수",
            "기존 엔진": baseline_targeted,
            "현재 엔진": enhanced_targeted,
            "변화": enhanced_targeted - baseline_targeted,
        },
        {
            "비교 항목": "집행 예산",
            "기존 엔진": baseline_optimize_summary.get("spent", 0),
            "현재 엔진": optimize_summary.get("spent", 0),
            "변화": int(optimize_summary.get("spent", 0)) - int(baseline_optimize_summary.get("spent", 0)),
        },
        {
            "비교 항목": "예상 증분 이익",
            "기존 엔진": baseline_profit,
            "현재 엔진": enhanced_profit,
            "변화": enhanced_profit - baseline_profit,
        },
        {
            "비교 항목": "예상 ROI",
            "기존 엔진": baseline_roi,
            "현재 엔진": enhanced_roi,
            "변화": enhanced_roi - baseline_roi,
        },
    ])

    display_compare = comparison_rows.copy()
    currency_rows = {"집행 예산", "예상 증분 이익"}
    ratio_rows = {"예상 ROI"}
    for idx, row in display_compare.iterrows():
        metric_name = row["비교 항목"]
        if metric_name in currency_rows:
            display_compare.loc[idx, "기존 엔진"] = money(row["기존 엔진"])
            display_compare.loc[idx, "현재 엔진"] = money(row["현재 엔진"])
            display_compare.loc[idx, "변화"] = money(row["변화"])
        elif metric_name in ratio_rows:
            display_compare.loc[idx, "기존 엔진"] = pct(row["기존 엔진"])
            display_compare.loc[idx, "현재 엔진"] = pct(row["현재 엔진"])
            display_compare.loc[idx, "변화"] = f"{row['변화'] * 100:.2f}%p"
        else:
            display_compare.loc[idx, "기존 엔진"] = f"{int(row['기존 엔진']):,}"
            display_compare.loc[idx, "현재 엔진"] = f"{int(row['현재 엔진']):,}"
            display_compare.loc[idx, "변화"] = f"{int(row['변화']):+d}"

    st.markdown("### 엔진이 고려하는 요소")
    _render_dataframe_with_count(factor_table, label="의사결정 요소 비교")

    st.markdown("### 결과가 어떻게 달라졌는가")
    _render_dataframe_with_count(display_compare, label="기존 엔진 vs 현재 엔진")

    compare_chart_df = pd.DataFrame([
        {"engine": "기존 엔진", "metric": "예상 증분 이익", "value": baseline_profit},
        {"engine": "현재 엔진", "metric": "예상 증분 이익", "value": enhanced_profit},
        {"engine": "기존 엔진", "metric": "집행 예산", "value": float(baseline_optimize_summary.get("spent", 0))},
        {"engine": "현재 엔진", "metric": "집행 예산", "value": float(optimize_summary.get("spent", 0))},
    ])
    compare_fig = px.bar(
        compare_chart_df,
        x="metric",
        y="value",
        color="engine",
        barmode="group",
        title="핵심 수치 비교",
    )
    st.plotly_chart(compare_fig, use_container_width=True)

    baseline_segment_chart = baseline_segment_allocation.copy()
    baseline_segment_chart["engine"] = "기존 엔진"
    enhanced_segment_chart = enhanced_segment_summary.copy()
    enhanced_segment_chart["engine"] = "현재 엔진"
    segment_compare_df = pd.concat([baseline_segment_chart, enhanced_segment_chart], ignore_index=True)

    if not segment_compare_df.empty:
        segment_fig = px.bar(
            segment_compare_df,
            x="uplift_segment",
            y="allocated_budget",
            color="engine",
            barmode="group",
            hover_data=["customer_count", "expected_profit"],
            title="세그먼트별 예산 재배분 비교",
        )
        st.plotly_chart(segment_fig, use_container_width=True)

    left_col, right_col = st.columns(2)

    with left_col:
        if not selected_customers.empty and "timing_priority_bucket" in selected_customers.columns:
            timing_df = (
                selected_customers.groupby("timing_priority_bucket", as_index=False)
                .agg(customer_count=("customer_id", "nunique"))
                .sort_values("customer_count", ascending=False)
            )
            timing_fig = px.bar(
                timing_df,
                x="timing_priority_bucket",
                y="customer_count",
                text="customer_count",
                title="현재 엔진이 잡아낸 개입 시점 분포",
            )
            st.plotly_chart(timing_fig, use_container_width=True)

    with right_col:
        intensity_counts = optimize_summary.get("selected_intensity_counts", {}) if optimize_summary else {}
        if intensity_counts:
            intensity_df = pd.DataFrame({
                "intervention_intensity": list(intensity_counts.keys()),
                "customer_count": list(intensity_counts.values()),
            })
            intensity_fig = px.bar(
                intensity_df,
                x="intervention_intensity",
                y="customer_count",
                text="customer_count",
                title="현재 엔진의 개입 강도 선택 결과",
            )
            st.plotly_chart(intensity_fig, use_container_width=True)

    if not selected_customers.empty:
        st.markdown("### 현재 엔진이 실제로 선택한 고객 예시")
        preview_df = selected_customers.copy()
        preview_columns = [
            "customer_id",
            "persona",
            "uplift_segment",
            "recommended_intervention_window",
            "intervention_intensity",
            "churn_probability",
            "uplift_score",
            "coupon_cost",
            "expected_incremental_profit",
            "expected_roi",
        ]
        preview_columns = [column for column in preview_columns if column in preview_df.columns]
        preview_df = preview_df[preview_columns].head(20).copy()
        if "churn_probability" in preview_df.columns:
            preview_df["churn_probability"] = preview_df["churn_probability"].map(lambda x: f"{float(x):.3f}")
        if "uplift_score" in preview_df.columns:
            preview_df["uplift_score"] = preview_df["uplift_score"].map(lambda x: f"{float(x):.3f}")
        if "coupon_cost" in preview_df.columns:
            preview_df["coupon_cost"] = preview_df["coupon_cost"].map(money)
        if "expected_incremental_profit" in preview_df.columns:
            preview_df["expected_incremental_profit"] = preview_df["expected_incremental_profit"].map(money)
        if "expected_roi" in preview_df.columns:
            preview_df["expected_roi"] = preview_df["expected_roi"].map(lambda x: f"{float(x):.2%}")
        _render_dataframe_with_count(preview_df, label="현재 엔진 선택 고객 예시")

    llm_payload = {
        "decision_engine_factors": factor_table.to_dict(orient="records"),
        "baseline_summary": baseline_optimize_summary,
        "enhanced_summary": optimize_summary,
        "enhanced_segment_summary": enhanced_segment_summary.to_dict(orient="records") if not enhanced_segment_summary.empty else [],
        "selected_customer_preview": dataframe_snapshot(
            selected_customers,
            columns=[
                "customer_id",
                "uplift_segment",
                "recommended_intervention_window",
                "intervention_intensity",
                "expected_incremental_profit",
                "expected_roi",
            ],
            max_rows=15,
        ) if not selected_customers.empty else [],
    }

elif view == "13. 운영 한눈에 보기":
    st.subheader("운영 한눈에 보기")
    st.caption("현 시점의 위험 고객 규모, 예산 집행, 추천 생성, 실시간 액션 큐, 세그먼트 구성을 한 화면에서 묶어 보여줍니다.")

    overview_cards = operational_overview.get("summary_cards", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("현재 리텐션 타겟", f"{int(overview_cards.get('selected_count', 0)):,}명")
    m2.metric("예상 증분 이익", money(float(overview_cards.get('expected_profit', 0.0))))
    m3.metric("예상 ROI", pct(float(overview_cards.get('overall_roi', 0.0))))
    m4.metric("추천 생성 건수", f"{int(overview_cards.get('recommended_rows', 0)):,}건")
    m5.metric("실시간 큐 적재", f"{int(overview_cards.get('queued_actions', 0)):,}건")

    st.info(
        "사용자가 보통 궁금해하는 질문을 기준으로 묶었습니다: 지금 누구를 잡고 있는가, 왜 그 고객들인가, "
        "이벤트는 어느 단계에서 많이 쌓이고 있는가, 실시간 큐에 얼마나 액션이 대기 중인가, 어떤 세그먼트가 가장 중요한가."
    )

    tab1, tab2, tab3 = st.tabs(["운영 파이프라인", "행동/세그먼트", "지금 봐야 할 포인트"])

    with tab1:
        pipeline_rows = pd.DataFrame([
            {"stage": "전체 고객", "count": int(len(customers)), "note": "현재 customer_summary 기준 전체 모수"},
            {"stage": "위험 고객(현재 threshold)", "count": int(churn_summary.get('at_risk_customers', 0)), "note": "threshold 이상인 고객"},
            {"stage": "최종 타겟 고객", "count": int(len(selected_customers)), "note": "예산·ROI·타이밍을 고려해 선별"},
            {"stage": "개인화 추천 생성", "count": int(recommendation_summary.get('rows', 0) or 0), "note": "최종 타겟 대상 추천 산출"},
            {"stage": "실시간 액션 큐", "count": int(realtime_monitor_overview.get('summary', {}).get('queued_actions_total', 0) or 0), "note": "즉시/후속 조치 대기"},
        ])
        fig = px.funnel(pipeline_rows, x="count", y="stage", title="운영 파이프라인 요약")
        st.plotly_chart(fig, use_container_width=True)
        _render_dataframe_with_count(pipeline_rows, label="운영 파이프라인 단계")

    with tab2:
        funnel_df = operational_overview.get("funnel_df", pd.DataFrame())
        persona_df = operational_overview.get("persona_df", pd.DataFrame())
        segment_df = operational_overview.get("segment_df", pd.DataFrame())

        left, right = st.columns(2)
        with left:
            if not funnel_df.empty:
                fig = px.bar(funnel_df, x="stage", y="events", title="이벤트 단계별 발생량", text="events")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("이벤트 믹스를 계산할 데이터가 없습니다.")
        with right:
            if not persona_df.empty:
                plot_df = persona_df.head(10).copy()
                fig = px.bar(
                    plot_df,
                    x="persona",
                    y="avg_churn_probability",
                    hover_data=[col for col in ["customers", "avg_uplift_score", "avg_clv"] if col in plot_df.columns],
                    title="페르소나별 평균 이탈 위험",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("페르소나 집계를 계산할 수 없습니다.")

        if not segment_df.empty:
            display_segment_df = segment_df.head(10).copy()
            for col in ["avg_clv", "avg_priority_score"]:
                if col in display_segment_df.columns:
                    display_segment_df[col] = display_segment_df[col].map(money)
            for col in ["avg_churn_probability", "avg_uplift"]:
                if col in display_segment_df.columns:
                    display_segment_df[col] = display_segment_df[col].map(lambda x: f"{float(x):.3f}")
            _render_dataframe_with_count(display_segment_df, label="핵심 고객 세그먼트")

    with tab3:
        cA, cB, cC = st.columns(3)
        cA.metric("평균 개입 윈도우", f"{float(overview_cards.get('avg_window', 0.0)):.1f}일")
        cB.metric("Survival C-index", f"{float(overview_cards.get('survival_c_index', 0.0)):.4f}")
        cC.metric("누적 주문 건수", f"{int(overview_cards.get('order_count', 0)):,}건")
        st.markdown("### 지금 바로 확인할 것")
        st.markdown(
            "- 현재 threshold와 예산에서 실제로 선별된 고객 수가 얼마나 되는지\n"
            "- 세그먼트별로 churn만 높은지, uplift와 CLV까지 같이 높은지\n"
            "- 실시간 액션 큐가 채널 용량보다 빠르게 늘고 있지 않은지\n"
            "- 추천 건수는 충분한데 실제 선택 고객이 적다면 예산/ROI 제약이 과한지"
        )

    llm_payload = {
        "overview_cards": overview_cards,
        "pipeline": pipeline_rows.to_dict(orient="records"),
        "persona_summary": operational_overview.get("persona_df", pd.DataFrame()).head(10).to_dict(orient="records") if not operational_overview.get("persona_df", pd.DataFrame()).empty else [],
        "segment_summary": operational_overview.get("segment_df", pd.DataFrame()).head(10).to_dict(orient="records") if not operational_overview.get("segment_df", pd.DataFrame()).empty else [],
    }

elif view == "14. 증분 성과 / A-B 실험":
    st.subheader("증분 성과 / A-B 실험")
    st.caption("정확도보다 더 중요한 운영 지표인 증분 리텐션, 추가 유지 고객 수, 비용 대비 유지 성과, dose-response 결과를 함께 봅니다.")

    exp_metrics = experiment_overview.get("metrics", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("증분 리텐션", pct(float(exp_metrics.get('incremental_retention', 0.0))))
    m2.metric("추가 유지 고객 수", f"{int(round(float(exp_metrics.get('incremental_retained_customers', 0.0)))):,}명")
    m3.metric("쿠폰 집행 총액", money(float(exp_metrics.get('coupon_spend_total', 0.0))))
    cpic_val = exp_metrics.get('incremental_cpic', np.nan)
    m4.metric("CPIC", money(float(cpic_val)) if pd.notna(cpic_val) else "-")
    m5.metric("Z-test p-value", f"{float(exp_metrics.get('p_value', np.nan)):.6f}" if pd.notna(exp_metrics.get('p_value', np.nan)) else "-")

    tab1, tab2, tab3 = st.tabs(["A/B 해석", "개입 강도 효과", "Persuadables 프로필"])

    with tab1:
        ab_test = experiment_overview.get("ab_test", {})
        if ab_test:
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

elif view == "15. 설명가능성 / 고객별 개입 이유":
    st.subheader("설명가능성 / 고객별 개입 이유")
    st.caption("왜 이 고객이 위험군인지, 왜 개입 후보로 뽑혔는지, 무엇을 조심해야 하는지를 운영 언어로 풀어 보여줍니다.")

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

elif view == "16. 데이터 진단 / 시뮬레이터 충실도":
    st.subheader("데이터 진단 / 시뮬레이터 충실도")
    st.caption("시뮬레이터가 만든 원천 데이터와 파생 산출물이 운영형 분석에 쓰기 적절한지, 기본적인 정합성과 분포를 함께 점검합니다.")

    checks_df = data_diagnostics.get("checks_df", pd.DataFrame())
    volumes_df = data_diagnostics.get("volumes_df", pd.DataFrame())
    event_mix_df = data_diagnostics.get("event_mix_df", pd.DataFrame())
    distribution_df = data_diagnostics.get("distribution_df", pd.DataFrame())

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

elif view == "17. 할인·쿠폰 운영 리스크":
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

if llm_enabled:
    current_view_key = view.split(".")[0]
    current_model_name = llm_model.strip() or DEFAULT_MODEL_NAME

    render_llm_summary(
        view_key=current_view_key,
        view_title=llm_view_title,
        payload=llm_payload,
        api_key=llm_api_key_value,
        model_name=current_model_name,
    )

    if (
        st.session_state.get("llm_chat_open", False)
        and st.session_state.get("llm_chat_view_key") == current_view_key
    ):
        open_chatbot_dialog(
            view_key=current_view_key,
            view_title=llm_view_title,
            payload=llm_payload,
            api_key=llm_api_key_value,
            model_name=current_model_name,
        )
        inject_draggable_chat_dialog()

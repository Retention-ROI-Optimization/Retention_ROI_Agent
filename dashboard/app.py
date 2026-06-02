import hashlib
import html
import json
import math
import os

# Limit only the number of rows rendered in heavy HTML preview tables.
# This does not limit the full dataset used by the pipeline.
TABLE_DISPLAY_ROW_LIMIT = int(os.getenv("TABLE_DISPLAY_ROW_LIMIT", "500"))
CHURN_TIMING_DISPLAY_ROW_LIMIT = int(os.getenv("CHURN_TIMING_DISPLAY_ROW_LIMIT", "500"))
import re
from functools import lru_cache
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
    fetch_demo_status,
    seed_user_live_from_artifacts,
)
from dashboard.services.churn_service import get_churn_status
from dashboard.services.counterfactual_service import build_counterfactual_retention_lab
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
from dashboard.services.optimize_service import build_budget_sensitivity_map, get_budget_result
from dashboard.services.uplift_service import (
    get_retention_targets,
    get_top_high_value_customers,
)
from dashboard.utils.formatters import money, pct
from dashboard.ui_budget_formula import budget_formula_html
from dashboard.ui_labels import (
    drop_duplicate_metric_columns,
    localize_plotly_figure,
    translate_column as friendly_translate_column,
    translate_text as friendly_translate_text,
    translate_value as friendly_translate_value,
)
from dashboard.ui_llm_language import llm_language_instruction, llm_language_name


DASHBOARD_VIEW_ITEMS: tuple[tuple[str, str], ...] = (
    # 내부 키는 기존 렌더링 분기와 호환되도록 일부 원래 번호를 유지한다.
    # 화면에는 CORE_VIEW_DISPLAY_LABELS를 통해 1~5로 재정렬된 번호만 보여준다.
    ("1", "이탈현황"),
    ("9", "이탈 시점 예측"),
    ("4", "예산 최적화 및 리텐션 타겟"),
    ("13", "고객별 대응 전략 비교"),
    ("5", "개인화 추천"),
    ("6", "실시간 운영 모니터"),
    ("14", "주간 액션 성과 리뷰"),
)
DASHBOARD_VIEW_OPTIONS: tuple[str, ...] = tuple(f"{n}. {t}" for n, t in DASHBOARD_VIEW_ITEMS)
VIEW_OPTION_BY_NUM: dict[str, str] = {num: f"{num}. {title}" for num, title in DASHBOARD_VIEW_ITEMS}

DASHBOARD_VIEW_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("핵심 화면", ("1", "9", "4", "13", "5", "6", "14")),
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

CORE_VIEW_DISPLAY_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "1. 이탈현황": "① 이탈 현황",
        "9. 이탈 시점 예측": "② 이탈 시점 예측",
        "4. 예산 최적화 및 리텐션 타겟": "③ 예산 배분·타겟 고객",
        "13. 고객별 대응 전략 비교": "④ 고객별 대응 전략 비교",
        "5. 개인화 추천": "⑤ 개인화 추천",
        "6. 실시간 운영 모니터": "⑥ 실시간 운영 모니터",
        "14. 주간 액션 성과 리뷰": "⑦ 주간 액션 성과 리뷰",
    },
    "en": {
        "1. 이탈현황": "① Churn Status",
        "9. 이탈 시점 예측": "② Churn Timing",
        "4. 예산 최적화 및 리텐션 타겟": "③ Budget Allocation & Targets",
        "13. 고객별 대응 전략 비교": "④ Counterfactual Retention Lab",
        "5. 개인화 추천": "⑤ Personalized Recommendations",
        "6. 실시간 운영 모니터": "⑥ Real-time Operations",
        "14. 주간 액션 성과 리뷰": "⑦ Weekly Action Review",
    },
    "ja": {
        "1. 이탈현황": "① 離脱状況",
        "9. 이탈 시점 예측": "② 離脱時点予測",
        "4. 예산 최적화 및 리텐션 타겟": "③ 予算配分・対象顧客",
        "13. 고객별 대응 전략 비교": "④ 反事実リテンション実験室",
        "5. 개인화 추천": "⑤ パーソナライズ推薦",
        "6. 실시간 운영 모니터": "⑥ リアルタイム運用",
        "14. 주간 액션 성과 리뷰": "⑦ 週次アクション成果レビュー",
    },
}

LANGUAGE_OPTIONS: dict[str, str] = {
    "한국어": "ko",
    "English": "en",
    "日本語": "ja",
}
LANGUAGE_LABEL_BY_CODE: dict[str, str] = {v: k for k, v in LANGUAGE_OPTIONS.items()}

DOMAIN_MODE_OPTIONS: dict[str, dict[str, str]] = {
    "ecommerce": {
        "ko": "이커머스 모드",
        "en": "E-commerce Mode",
        "ja": "ECモード",
    },
    "finance": {
        "ko": "금융 모드",
        "en": "Finance Mode",
        "ja": "金融モード",
    },
}
DOMAIN_DIRS: dict[str, dict[str, str]] = {
    "ecommerce": {"data": "data/raw_ecommerce", "results": "results_ecommerce", "models": "models_ecommerce", "features": "data/feature_store_ecommerce"},
    "finance": {"data": "data/raw_finance", "results": "results_finance", "models": "models_finance", "features": "data/feature_store_finance"},
    "user": {"data": "data/raw_user", "results": "results_user", "models": "models_user", "features": "data/feature_store_user"},
    "simulator": {"data": "data/raw_simulator", "results": "results_simulator", "models": "models_simulator", "features": "data/feature_store_simulator"},
}
BUSINESS_UPLOAD_MODES: set[str] = {"ecommerce", "finance", "user"}

FINANCE_COLUMN_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "customer_id": "금융 고객 ID", "financial_customer_id": "금융 고객 ID",
        "persona": "금융 고객 유형", "recommended_category": "추천 금융상품/서비스",
        "item_category": "금융상품/서비스", "financial_product": "금융상품/서비스",
        "order_id": "거래 ID", "transaction_id": "거래 ID",
        "order_time": "거래 시각", "transaction_time": "거래 시각",
        "gross_amount": "거래 금액", "transaction_amount": "거래 금액",
        "discount_amount": "혜택 금액", "benefit_amount": "혜택 금액",
        "net_amount": "순거래 금액", "net_transaction_amount": "순거래 금액",
        "coupon_used": "혜택 사용 여부", "retention_benefit_used": "혜택 사용 여부",
        "coupon_cost": "금융 혜택/개입 비용", "queued_coupon_cost": "큐 금융 혜택 비용",
        "coupon_exposure_count": "혜택 제안 횟수", "coupon_redeem_count": "혜택 수락 횟수",
        "coupon_fatigue_score": "혜택 피로도", "coupon_affinity": "금융 혜택 반응도",
        "discount_dependency_score": "금리·수수료 혜택 의존도",
        "discount_pressure_score": "혜택 압박도", "discount_effect_penalty": "혜택 효과 페널티",
        "price_sensitivity": "금리·수수료 민감도", "purchase_last_30": "최근 30일 금융거래",
        "purchase_prev_30": "직전 30일 금융거래", "purchase_change_rate": "금융거래 변화율",
        "monetary": "금융 거래/잔고 금액", "frequency": "거래 빈도", "recency_days": "마지막 금융거래 경과일",
        "financial_event_type": "금융 이벤트 유형", "event_type": "금융 이벤트 유형",
        "account_balance_current": "현재 계좌잔고", "avg_balance": "평균잔고",
        "loan_balance": "대출잔액", "loan_amount": "대출금액", "credit_limit": "신용한도",
        "card_spend_total": "카드 이용금액", "aum": "운용자산", "credit_score": "신용점수",
        "credit_risk_score": "신용위험 점수", "delinquency_days": "연체일수",
        "missed_payment_count": "미납/연체 횟수", "tenure_months": "거래기간(개월)",
        "product_count": "보유 금융상품 수", "risk_grade": "리스크 등급",
        "account_status": "계좌/거래 상태", "intervention_cost": "금융 개입 비용",
        "benefit_offer_count": "혜택 제안 횟수", "benefit_accept_count": "혜택 수락 횟수",
        "financial_benefit_affinity": "금융 혜택 반응도", "rate_fee_sensitivity": "금리·수수료 민감도",
        "service_contact_propensity": "상담/민원 가능성",
    },
    "en": {
        "customer_id": "Financial Customer ID", "financial_customer_id": "Financial Customer ID",
        "persona": "Financial Customer Type", "recommended_category": "Recommended Financial Product/Service",
        "item_category": "Financial Product/Service", "financial_product": "Financial Product/Service",
        "order_id": "Transaction ID", "transaction_id": "Transaction ID",
        "order_time": "Transaction Time", "transaction_time": "Transaction Time",
        "gross_amount": "Transaction Amount", "transaction_amount": "Transaction Amount",
        "discount_amount": "Benefit Amount", "benefit_amount": "Benefit Amount",
        "net_amount": "Net Transaction Amount", "net_transaction_amount": "Net Transaction Amount",
        "coupon_used": "Benefit Used", "retention_benefit_used": "Benefit Used",
        "coupon_cost": "Financial Benefit/Intervention Cost", "queued_coupon_cost": "Queued Financial Benefit Cost",
        "coupon_exposure_count": "Benefit Offers", "coupon_redeem_count": "Accepted Benefits",
        "coupon_fatigue_score": "Benefit Fatigue", "coupon_affinity": "Financial Benefit Affinity",
        "discount_dependency_score": "Rate/Fee Benefit Dependency",
        "discount_pressure_score": "Benefit Pressure", "discount_effect_penalty": "Benefit Effect Penalty",
        "price_sensitivity": "Rate/Fee Sensitivity", "purchase_last_30": "Financial Transactions Last 30d",
        "purchase_prev_30": "Financial Transactions Previous 30d", "purchase_change_rate": "Financial Transaction Change Rate",
        "monetary": "Financial Value/Balance", "frequency": "Transaction Frequency", "recency_days": "Days Since Last Financial Activity",
        "financial_event_type": "Financial Event Type", "event_type": "Financial Event Type",
        "account_balance_current": "Current Account Balance", "avg_balance": "Average Balance",
        "loan_balance": "Loan Balance", "loan_amount": "Loan Amount", "credit_limit": "Credit Limit",
        "card_spend_total": "Card Spend", "aum": "Assets Under Management", "credit_score": "Credit Score",
        "credit_risk_score": "Credit Risk Score", "delinquency_days": "Days Past Due",
        "missed_payment_count": "Missed Payment Count", "tenure_months": "Relationship Tenure (Months)",
        "product_count": "Financial Products Held", "risk_grade": "Risk Grade",
        "account_status": "Account/Relationship Status", "intervention_cost": "Financial Intervention Cost",
        "benefit_offer_count": "Benefit Offers", "benefit_accept_count": "Accepted Benefits",
        "financial_benefit_affinity": "Financial Benefit Affinity", "rate_fee_sensitivity": "Rate/Fee Sensitivity",
        "service_contact_propensity": "Service Contact Propensity",
    },
    "ja": {},
}

FINANCE_VALUE_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "purchase": "금융거래", "구매": "금융거래", "총 구매": "총 금융거래",
        "order": "거래", "주문": "거래", "주문 내역": "거래 내역", "주문 기록 없음": "거래 기록 없음",
        "add_to_cart": "신청시작/관심상품", "cart": "신청/관심", "장바구니": "신청/관심", "장바구니 담기": "신청 시작",
        "page_view": "계좌·상품조회", "상품조회": "금융상품조회", "search": "금융상품 탐색",
        "coupon_offer": "금융 혜택 제안", "discount_offer": "금리·수수료 우대 제안",
        "personalized_coupon": "맞춤 금융 혜택", "coupon": "금융 혜택", "쿠폰": "금융 혜택",
        "coupon_used": "혜택 사용", "쿠폰 사용": "혜택 사용",
        "own_purchase_history": "고객 본인의 과거 금융거래 이력",
        "recent_browse_signal": "최근 금융상품 조회 신호",
        "category_affinity": "금융상품 관심도", "price_affinity": "금리·수수료 반응 가능성",
        "purchase_gap_increase": "금융거래 간격이 길어짐",
        "recent_activity_drop": "최근 금융 활동이 줄어듦",
        "coupon_cost": "금융 혜택/개입 비용", "queued_coupon_cost": "큐 금융 혜택 비용",
        "coupon_affinity": "금융 혜택 반응도", "coupon_exposure_count": "혜택 제안 횟수",
        "coupon_redeem_count": "혜택 수락 횟수", "coupon_fatigue_score": "혜택 피로도",
        "discount_dependency_score": "금리·수수료 혜택 의존도",
        "discount_pressure_score": "혜택 압박도", "discount_effect_penalty": "혜택 효과 페널티",
        "price_sensitivity": "금리·수수료 민감도",
        "purchase_last_30": "최근 30일 금융거래", "purchase_prev_30": "직전 30일 금융거래",
        "purchase_change_rate": "금융거래 변화율", "avg_coupon_exposure": "평균 혜택 제안 횟수",
        "recommended_category": "추천 금융상품/서비스", "item_category": "금융상품/서비스",
        "fashion": "카드/소비", "beauty": "예·적금", "grocery": "입출금계좌", "sports": "대출", "health": "보험/연금",
    },
    "en": {
        "purchase": "Financial transaction", "Purchase": "Financial transaction",
        "order": "Transaction", "Order": "Transaction", "add_to_cart": "Application/interest start",
        "cart": "Application/interest", "page_view": "Account/product view", "search": "Financial product search",
        "coupon_offer": "Financial benefit offer", "discount_offer": "Rate/fee benefit offer",
        "personalized_coupon": "Personalized financial benefit", "coupon": "Financial benefit",
        "own_purchase_history": "Own financial transaction history",
        "recent_browse_signal": "Recent financial product view signal",
        "category_affinity": "Financial product affinity", "price_affinity": "Rate/fee sensitivity",
        "purchase_gap_increase": "Longer financial transaction gap",
        "recent_activity_drop": "Recent financial activity drop",
        "coupon_cost": "Financial benefit/intervention cost", "queued_coupon_cost": "Queued financial benefit cost",
        "coupon_affinity": "Financial benefit affinity", "coupon_exposure_count": "Benefit offers",
        "coupon_redeem_count": "Accepted benefits", "coupon_fatigue_score": "Benefit fatigue",
        "discount_dependency_score": "Rate/fee benefit dependency",
        "discount_pressure_score": "Benefit pressure", "discount_effect_penalty": "Benefit effect penalty",
        "price_sensitivity": "Rate/fee sensitivity",
        "purchase_last_30": "Financial transactions last 30d", "purchase_prev_30": "Financial transactions previous 30d",
        "purchase_change_rate": "Financial transaction change rate", "avg_coupon_exposure": "Average benefit offers",
        "recommended_category": "Recommended financial product/service", "item_category": "Financial product/service",
    },
    "ja": {},
}

FINANCE_RUNTIME_REPLACEMENTS: dict[str, dict[str, str]] = {
    "ko": {
        "방문·검색·장바구니·구매·쿠폰·카테고리 선호 기반": "접속·상품탐색·신청시작·금융거래·혜택 반응·금융상품 선호 기반",
        "고객 구매 이력": "고객 금융거래 이력",
        "최근 관심": "최근 금융상품 관심",
        "세그먼트 인기": "유사 금융고객군 선호",
        "전역 인기를": "전체 금융고객 선호를",
        "방문, 구매 등": "접속, 금융거래 등",
        "마지막 활동(이벤트/주문)": "마지막 금융 활동(이벤트/거래)",
        "업종별 방문·구매 주기": "금융 채널 접속·거래 주기",
        "쿠폰 집행 총액": "금융 혜택 집행 총액",
        "쿠폰비 반영 ROI": "금융 혜택 비용 반영 ROI",
        "할인·쿠폰 운영 리스크": "금융 혜택 운영 리스크",
        "쿠폰 노출/리딤/믹스 리스크": "혜택 제안/수락/믹스 리스크",
        "쿠폰 노출 누적": "혜택 제안 누적",
        "할인 남발": "금리·수수료 혜택 남발",
    },
    "en": {}, "ja": {},
}

UI_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "고객 이탈 예측·개입 최적화·ROI 분석 플랫폼": "Customer Churn, Intervention Optimization & ROI Platform",
        "누가 이탈할 가능성이 높은지뿐 아니라, 언제 개입해야 하는지, 누구에게 예산을 우선 배분할지, 어떤 액션을 추천할지까지 연결해 보여주는 운영형 리텐션 분석 플랫폼입니다.": "An operational retention platform that connects churn risk, intervention timing, budget priority, and recommended actions.",
        "핵심 화면": "Core Views",
        "분석 화면": "Analysis View",
        "분석 모드 선택": "Choose Analysis Mode",
        "어떤 산업 데이터로 분석할지 선택하세요.": "Choose the industry domain for your dataset.",
        "금융 모드": "Finance Mode",
        "이커머스 모드": "E-commerce Mode",
        "언어": "Language",
        "현재 분석 모드": "Current Mode",
        "사용 데이터셋": "Dataset",
        "미선택": "Not selected",
        "제어 패널": "Control Panel",
        "분석 컨트롤": "Analysis Controls",
        "데이터/결과 새로고침": "Refresh current view",
        "실행 / 새로고침": "Run / Refresh",
        "이탈현황": "Churn Status",
        "이탈 현황": "Churn Status",
        "예산 최적화 및 리텐션 타겟": "Budget Allocation & Retention Targets",
        "최종 타겟 고객 대상 개인화 추천": "Personalized Recommendations for Final Targets",
        "실시간 운영 모니터": "Real-time Operations Monitor",
        "이탈 위험 고객 목록": "At-risk Customer List",
        "세그먼트별 예산 배분 후보 고객 수": "Candidate Customers by Segment",
        "세그먼트별 예산 배분 테이블": "Segment Budget Allocation Table",
        "최종 리텐션 타겟 고객 테이블": "Final Retention Target Customers",
        "고객별 선택 이유 / 주의사항": "Customer-level Reasons / Cautions",
        "개인화 추천 테이블": "Personalized Recommendation Table",
        "실시간 이탈 위험 테이블": "Real-time Churn Risk Table",
        "실시간 액션 큐 상세": "Real-time Action Queue Details",
        "Live Action Queue": "Live Action Queue",
        "용어 설명": "Terminology",
    },
    "ja": {
        "고객 이탈 예측·개입 최적화·ROI 분석 플랫폼": "顧客離脱予測・介入最適化・ROI分析プラットフォーム",
        "누가 이탈할 가능성이 높은지뿐 아니라, 언제 개입해야 하는지, 누구에게 예산을 우선 배분할지, 어떤 액션을 추천할지까지 연결해 보여주는 운영형 리텐션 분석 플랫폼입니다.": "離脱リスク、介入タイミング、予算優先度、推奨アクションを一つにつなぐ運用型リテンション分析基盤です。",
        "핵심 화면": "主要画面",
        "분석 화면": "分析画面",
        "분석 모드 선택": "分析モード選択",
        "어떤 산업 데이터로 분석할지 선택하세요.": "分析するデータの業界ドメインを選択してください。",
        "금융 모드": "金融モード",
        "이커머스 모드": "ECモード",
        "언어": "言語",
        "현재 분석 모드": "現在のモード",
        "사용 데이터셋": "使用データセット",
        "미선택": "未選択",
        "제어 패널": "コントロールパネル",
        "분석 컨트롤": "分析コントロール",
        "데이터/결과 새로고침": "現在画面を更新",
        "실행 / 새로고침": "実行 / 更新",
        "이탈현황": "離脱状況",
        "이탈 현황": "離脱状況",
        "예산 최적화 및 리텐션 타겟": "予算配分・リテンション対象",
        "최종 타겟 고객 대상 개인화 추천": "最終対象顧客への推薦",
        "실시간 운영 모니터": "リアルタイム運用モニター",
        "이탈 위험 고객 목록": "離脱リスク顧客一覧",
        "세그먼트별 예산 배분 후보 고객 수": "セグメント別候補顧客数",
        "세그먼트별 예산 배분 테이블": "セグメント別予算配分表",
        "최종 리텐션 타겟 고객 테이블": "最終リテンション対象顧客",
        "고객별 선택 이유 / 주의사항": "顧客別選定理由・注意事項",
        "개인화 추천 테이블": "パーソナライズ推薦表",
        "실시간 이탈 위험 테이블": "リアルタイム離脱リスク表",
        "실시간 액션 큐 상세": "リアルタイムアクションキュー詳細",
        "Live Action Queue": "Live Action Queue",
        "용어 설명": "用語説明",
    },
}

COLUMN_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "customer_id": "고객 ID", "persona": "고객 유형", "churn_probability": "이탈 확률", "churn_score": "이탈 점수", "realtime_churn_score": "실시간 이탈 점수", "base_churn_probability": "기준 이탈 확률", "score_delta": "점수 변화",
        "clv": "고객 생애가치(CLV)", "uplift_score": "개입 효과 점수", "uplift_segment": "개입 반응 세그먼트", "risk_segment": "위험 등급", "expected_roi": "예상 ROI", "expected_incremental_profit": "예상 증분이익", "expected_profit": "예상 이익", "coupon_cost": "쿠폰/개입 비용",
        "allocated_budget": "배정 예산", "customer_count": "선정 고객 수", "candidate_customer_count": "후보 고객 수", "intervention_intensity": "개입 강도", "recommended_action": "추천 액션", "priority_score": "우선순위 점수", "selection_score": "선정 점수", "recommended_intervention_window": "추천 개입 시점",
        "recommended_category": "추천 카테고리", "recommendation_rank": "추천 순위", "recommendation_score": "추천 점수", "recommendation_priority": "추천 우선순위", "target_priority_score": "타겟 우선순위", "reason_tags": "추천 이유", "action_status": "액션 상태", "source_type": "발생 경로", "trigger_reason": "트리거 이유",
        "queued_at": "큐 적재 시각", "updated_at": "갱신 시각", "scored_at": "점수 산출 시각", "latest_trigger_reason": "최근 트리거 이유", "queued_recommended_action": "큐 추천 액션", "queued_intervention_intensity": "큐 개입 강도", "queued_coupon_cost": "큐 쿠폰 비용", "queued_expected_profit": "큐 예상 이익", "queued_expected_roi": "큐 예상 ROI", "reoptimization_count": "재최적화 횟수",
        "feature": "변수", "feature_display": "변수명", "importance": "중요도", "importance_share": "중요도 비중", "reason_summary": "선정 이유", "caution": "주의사항", "next_best_action": "다음 추천 액션", "survival_prob_30d": "30일 생존확률", "action_queue_status": "액션 큐 상태", "expected_churn_period": "예상 이탈 시점", "expected_churn_date": "예상 이탈 날짜", "expected_loss_30d": "예상 손실액", "churn_within_30d_probability": "30일 내 이탈 가능성",
    },
    "en": {
        "customer_id": "Customer ID", "persona": "Customer Type", "churn_probability": "Churn Probability", "churn_score": "Churn Score", "realtime_churn_score": "Real-time Churn Score", "base_churn_probability": "Base Churn Probability", "score_delta": "Score Delta",
        "clv": "Customer Lifetime Value (CLV)", "uplift_score": "Uplift Score", "uplift_segment": "Uplift Segment", "risk_segment": "Risk Segment", "expected_roi": "Expected ROI", "expected_incremental_profit": "Expected Incremental Profit", "expected_profit": "Expected Profit", "coupon_cost": "Coupon/Intervention Cost",
        "allocated_budget": "Allocated Budget", "customer_count": "Selected Customers", "candidate_customer_count": "Candidate Customers", "intervention_intensity": "Intervention Intensity", "recommended_action": "Recommended Action", "priority_score": "Priority Score", "selection_score": "Selection Score", "recommended_intervention_window": "Recommended Timing",
        "recommended_category": "Recommended Category", "recommendation_rank": "Rank", "recommendation_score": "Recommendation Score", "recommendation_priority": "Recommendation Priority", "target_priority_score": "Target Priority", "reason_tags": "Reason Tags", "action_status": "Action Status", "source_type": "Source Type", "trigger_reason": "Trigger Reason",
        "queued_at": "Queued At", "updated_at": "Updated At", "scored_at": "Scored At", "latest_trigger_reason": "Latest Trigger Reason", "queued_recommended_action": "Queued Action", "queued_intervention_intensity": "Queued Intensity", "queued_coupon_cost": "Queued Coupon Cost", "queued_expected_profit": "Queued Expected Profit", "queued_expected_roi": "Queued Expected ROI", "reoptimization_count": "Re-optimization Count",
        "feature": "Feature", "feature_display": "Feature", "importance": "Importance", "importance_share": "Importance Share", "reason_summary": "Reason Summary", "caution": "Caution", "next_best_action": "Next Best Action", "survival_prob_30d": "30-day Survival Probability", "action_queue_status": "Action Queue Status", "expected_churn_period": "Expected Churn Timing", "expected_churn_date": "Expected Churn Date", "expected_loss_30d": "Expected Loss", "churn_within_30d_probability": "Churn Chance within 30 Days",
    },
    "ja": {
        "customer_id": "顧客ID", "persona": "顧客タイプ", "churn_probability": "離脱確率", "churn_score": "離脱スコア", "realtime_churn_score": "リアルタイム離脱スコア", "base_churn_probability": "基準離脱確率", "score_delta": "スコア変化",
        "clv": "顧客生涯価値(CLV)", "uplift_score": "介入効果スコア", "uplift_segment": "介入反応セグメント", "risk_segment": "リスク区分", "expected_roi": "予想ROI", "expected_incremental_profit": "予想増分利益", "expected_profit": "予想利益", "coupon_cost": "クーポン/介入費用",
        "allocated_budget": "配分予算", "customer_count": "選定顧客数", "candidate_customer_count": "候補顧客数", "intervention_intensity": "介入強度", "recommended_action": "推奨アクション", "priority_score": "優先度スコア", "selection_score": "選定スコア", "recommended_intervention_window": "推奨介入時点",
        "recommended_category": "推薦カテゴリ", "recommendation_rank": "推薦順位", "recommendation_score": "推薦スコア", "recommendation_priority": "推薦優先度", "target_priority_score": "対象優先度", "reason_tags": "推薦理由", "action_status": "アクション状態", "source_type": "発生経路", "trigger_reason": "トリガー理由",
        "queued_at": "キュー登録時刻", "updated_at": "更新時刻", "scored_at": "スコア算出時刻", "latest_trigger_reason": "最新トリガー理由", "queued_recommended_action": "キュー推奨アクション", "queued_intervention_intensity": "キュー介入強度", "queued_coupon_cost": "キュー費用", "queued_expected_profit": "キュー予想利益", "queued_expected_roi": "キュー予想ROI", "reoptimization_count": "再最適化回数",
        "feature": "変数", "feature_display": "変数名", "importance": "重要度", "importance_share": "重要度比率", "reason_summary": "選定理由", "caution": "注意事項", "next_best_action": "次の推奨アクション", "survival_prob_30d": "30日生存確率", "action_queue_status": "アクションキュー状態", "expected_churn_period": "予想離脱時点", "expected_churn_date": "予想離脱日", "expected_loss_30d": "予想損失額", "churn_within_30d_probability": "30日以内の離脱可能性",
    },
}

TERM_CAPTIONS: dict[str, dict[str, str]] = {
    "ko": {
        "CLV": "CLV는 고객이 앞으로 가져올 것으로 추정되는 생애가치입니다.",
        "Uplift": "Uplift는 개입했을 때 이탈 방지·구매 증가가 얼마나 추가로 발생할지 나타내는 점수입니다.",
        "ROI": "ROI는 투입 비용 대비 기대 이익의 비율입니다. 100%는 비용만큼의 이익, 0% 이하는 손실 가능성을 의미합니다.",
        "Priority": "우선순위 점수는 이탈 위험, 개입 효과, 고객 가치, 비용을 합쳐 타겟 순서를 정한 값입니다.",
    },
    "en": {
        "CLV": "CLV is the estimated lifetime value a customer may generate in the future.",
        "Uplift": "Uplift estimates the incremental retention or purchase effect caused by an intervention.",
        "ROI": "ROI is expected profit relative to intervention cost; 100% means profit equals the cost.",
        "Priority": "Priority score combines churn risk, uplift, customer value, and cost to rank targets.",
    },
    "ja": {
        "CLV": "CLVは顧客が将来もたらすと推定される生涯価値です。",
        "Uplift": "Upliftは介入によって追加で得られる離脱防止・購買増加効果の推定値です。",
        "ROI": "ROIは介入費用に対する期待利益の比率です。100%は費用と同額の利益を意味します。",
        "Priority": "優先度スコアは離脱リスク、介入効果、顧客価値、費用を組み合わせた順位付け指標です。",
    },
}

UI_TEXT["en"].update({
    "전체 고객 수": "Total Customers", "이탈 위험 고객 수": "At-risk Customers", "위험 고객 비율": "Risk Rate", "평균 이탈 확률": "Avg. Churn Probability",
    "이탈 임계값": "Churn Threshold", "총 마케팅 예산": "Total Marketing Budget", "최대 타겟 고객 수": "Max Target Customers", "차트 기준 표시 고객 수": "Rows/Customers to Display", "고객당 추천 개수": "Recommendations per Customer",
    "총 예산": "Total Budget", "집행 예산": "Spent Budget", "잔여 예산": "Remaining Budget", "타겟 고객 수": "Target Customers", "예상 증분 이익": "Expected Incremental Profit",
    "표시 추천 행 수": "Displayed Recommendation Rows", "추천 대상 고객 수": "Recommended Customers", "평균 추천 수/고객": "Avg. Recommendations / Customer", "현재 최종 타겟 고객 수": "Current Final Target Customers", "추천 카테고리 분포": "Recommendation Category Distribution",
    "이벤트 수": "Events", "상태 보유 고객 수": "Customers with Live State", "이탈점수 산출 고객 수": "Scored Customers", "Queued 액션": "Queued Actions", "평균 이탈 점수": "Avg. Churn Score", "현재 기준 이탈 위험 고객 수": "At-risk Customers by Current Threshold", "실시간 추천 후보 수": "Live Recommendation Candidates", "최신 점수 갱신": "Latest Score Update",
    "추적 고객 수": "Tracked Customers", "재최적화 트리거 수": "Re-optimization Triggers", "액션 큐 적재 수": "Action Queue Size", "임계 위험 고객 수": "Critical-risk Customers", "처리 이벤트 수": "Processed Events", "폐쇄루프 예산 사용": "Closed-loop Budget Used", "채널 할당 수": "Channel Allocations", "운영 모니터": "Operations Monitor", "재최적화 횟수": "Re-optimizations", "큐 적재 수": "Queued Actions", "채널 용량 사용률": "Channel Capacity Utilization", "고우선순위 큐": "High-priority Queue",
})
UI_TEXT["ja"].update({
    "전체 고객 수": "全顧客数", "이탈 위험 고객 수": "離脱リスク顧客数", "위험 고객 비율": "リスク顧客比率", "평균 이탈 확률": "平均離脱確率",
    "이탈 임계값": "離脱リスク基準", "총 마케팅 예산": "総マーケティング予算", "최대 타겟 고객 수": "最大対象顧客数", "차트 기준 표시 고객 수": "表示件数", "고객당 추천 개수": "顧客あたり推薦数",
    "총 예산": "総予算", "집행 예산": "使用予算", "잔여 예산": "残予算", "타겟 고객 수": "対象顧客数", "예상 증분 이익": "予想増分利益",
    "표시 추천 행 수": "表示推薦行数", "추천 대상 고객 수": "推薦対象顧客数", "평균 추천 수/고객": "平均推薦数/顧客", "현재 최종 타겟 고객 수": "現在の最終対象顧客数", "추천 카테고리 분포": "推薦カテゴリ分布",
    "이벤트 수": "イベント数", "상태 보유 고객 수": "Live状態保有顧客数", "이탈점수 산출 고객 수": "離脱スコア算出顧客数", "Queued 액션": "Queuedアクション", "평균 이탈 점수": "平均離脱スコア", "현재 기준 이탈 위험 고객 수": "現在基準の離脱リスク顧客数", "실시간 추천 후보 수": "リアルタイム推薦候補数", "최신 점수 갱신": "最新スコア更新",
    "추적 고객 수": "追跡顧客数", "재최적화 트리거 수": "再最適化トリガー数", "액션 큐 적재 수": "アクションキュー数", "임계 위험 고객 수": "重大リスク顧客数", "처리 이벤트 수": "処理イベント数", "폐쇄루프 예산 사용": "閉ループ予算使用", "채널 할당 수": "チャネル割当数", "운영 모니터": "運用モニター", "재최적화 횟수": "再最適化回数", "큐 적재 수": "キュー数", "채널 용량 사용률": "チャネル容量使用率", "고우선순위 큐": "高優先度キュー",
})

UI_TEXT["en"].update({
    "LLM 결과 요약": "LLM Result Summary",
    "LLM결과요약": "LLM Result Summary",
    "현재 화면의 지표·표·그래프에서 추린 요약 컨텍스트만 바탕으로 응답합니다.": "The response is based only on the summary context extracted from the current screen's metrics, tables, and charts.",
    "AI가 현재 화면의 결과를 요약하는 중입니다...": "AI is summarizing the current screen...",
    "AI 요약 생성 중 오류가 발생했습니다": "An error occurred while generating the AI summary",
    "추가 질문은 사이드바의 AI 챗봇 버튼을 눌러 이어서 대화할 수 있습니다.": "For follow-up questions, open the AI chatbot in the sidebar.",
    "AI 챗봇": "AI Chatbot",
    "챗봇 닫기": "Close Chatbot",
    "챗봇 열기": "Open Chatbot",
    "LLM 기능이 꺼져 있어 챗봇을 열 수 없습니다.": "The chatbot cannot be opened because the LLM feature is disabled.",
    "현재 화면": "Current View",
    "화면의 표·그래프를 보면서 질문할 수 있습니다.": "You can ask questions while viewing the tables and charts on this screen.",
    "고정된 챗봇 컨텍스트": "Pinned Chatbot Context",
    "화면을 이동해도 챗봇은 처음 열었던 화면의 데이터로 유지됩니다.": "Even when you move between views, the chatbot keeps the data from the view where it was first opened.",
    "현재 화면으로 컨텍스트 갱신": "Refresh Context to Current View",
    "대화 지우기": "Clear Chat",
    "컨텍스트": "Context",
    "현재 화면에 대해 질문하세요...": "Ask about the current view...",
    "현재 화면에 대해 질문하세요.": "Ask about the current view.",
    "AI 답변 생성 중": "Generating AI answer",
    "AI 답변 생성 중 오류가 발생했습니다": "An error occurred while generating the AI answer",
    "표시할 데이터가 없습니다.": "No data to display.",
    "검색": "Search",
    "고객 ID 검색": "Search Customer ID",
    "분포 차원 선택": "Choose Distribution Dimension",
    "LLM 설정": "LLM Settings",
    "권장: API 키는 코드에 쓰지 말고 환경변수 OPENAI_API_KEY 또는 Streamlit secrets로 관리하세요.": "Recommended: manage API keys through the OPENAI_API_KEY environment variable or Streamlit secrets instead of writing them in code.",
    "LLM 요약/질문 기능 사용": "Enable LLM summaries/questions",
    "OpenAI API Key (선택)": "OpenAI API Key (Optional)",
    "비워두면 OPENAI_API_KEY 환경변수를 사용합니다.": "Leave empty to use the OPENAI_API_KEY environment variable.",
    "모델이 목록에 없으면 '직접 입력'을 선택해서 모델명을 넣어주세요.": "If the model is not listed, choose 'Manual Input' and enter the model name.",
    "LLM 모델 선택": "Choose LLM Model",
    "LLM 모델명 (직접 입력)": "LLM Model Name (Manual Input)",
    "현재 OPENAI_API_KEY 환경변수를 사용하도록 설정되어 있습니다.": "The app is currently configured to use the OPENAI_API_KEY environment variable.",
    "자사 데이터 Live DB 연결됨": "Live DB connected",
    "자사 데이터 Live DB 상태 확인 실패": "Live DB health check failed",
    "최신 이벤트": "Latest Event",
    "Live DB 상태": "Live DB Status",
    "저장 추천후보": "Saved Recommendation Candidates",
    "현재 데이터셋과 Live DB가 일치하지 않아 CSV/결과 파일 기준으로 표시합니다.": "The Live DB does not match the current dataset, so the dashboard is using the CSV/result files.",
    "모드/데이터셋 변경": "Change Mode/Dataset",
    "기존 결과로 대시보드 보기": "Open Dashboard with Existing Results",
    "학습 완료. 대시보드로 이동합니다.": "Training completed. Opening the dashboard.",
    "PostgreSQL user-live DB 초기 적재 완료": "PostgreSQL user-live DB seeding completed",
    "PostgreSQL user-live DB 자동 적재 실패": "PostgreSQL user-live DB automatic seeding failed",
    "주간 액션 성과 리뷰": "Weekly Action Performance Review",
    "이 화면은 실제 집행 결과가 아닌, 추천 데이터 기반의 시뮬레이션 리뷰입니다. 실행률과 성과 노이즈 슬라이더로 가상 시나리오를 조정할 수 있습니다.": "This is a simulation review based on recommendation data, not actual execution results. Adjust the execution rate and performance noise sliders to explore scenarios.",
    "전체 실행률": "Overall Execution Rate",
    "고쿠폰 실행률": "High Coupon Execution Rate",
    "성과 노이즈": "Performance Noise",
    "시뮬레이션 시드": "Simulation Seed",
    "총 추천 건수": "Total Recommendations",
    "총 집행 건수": "Total Executed",
    "총 집행 예산": "Total Budget Spent",
    "기대 이익 합계": "Expected Profit Sum",
    "실제 이익 합계": "Actual Profit Sum",
    "손실 액션 수": "Loss Actions",
    "기대 vs 실제 ROI": "Expected vs Actual ROI",
    "추천 카테고리별 기대 vs 실제 ROI": "Expected vs Actual ROI by Recommended Category",
    "세그먼트별 손익": "Segment P&L",
    "세그먼트별 손익 히트맵": "Segment P&L Heatmap",
    "손실 Top N": "Top N Losses",
    "손실 액션 Top 20": "Top 20 Loss Actions",
    "판정 분포": "Outcome Distribution",
    "액션 판정 분포": "Action Outcome Distribution",
    "전체 액션 상세": "Full Action Detail",
    "주간 액션 성과 리뷰 테이블": "Weekly Action Performance Review Table",
    "다음 주 정책 조정 제안": "Next Week Policy Adjustment Suggestions",
    "아래 제안은 이번 주 시뮬레이션 성과를 기반으로 자동 생성된 운영 힌트입니다.": "The suggestions below are auto-generated operational hints based on this week's simulated performance.",
    "적정 판단": "Good Decision",
    "기대 미달": "Underperformed",
    "과잉 투자": "Over-Invested",
    "타겟 오류": "Wrong Target",
    "실행 누락": "Missed Opportunity",
    "기대 이익": "Expected Profit",
    "실제 이익": "Actual Profit",
    "손실 액션 상세": "Loss Action Details",
    "개인화 추천 또는 최적화 선정 고객 산출물이 없습니다.": "No personalized recommendation or optimization result data found.",
    "실행 여부": "Executed",
    "판정": "Outcome",
    "건수": "Count",
    "과투자 추정 금액": "Estimated Over-Investment",
    "기대 미달 고객 수": "Underperformed Customer Count",
    "쿠폰 강도별 실제 전환율": "Conversion Rate by Coupon Intensity",
    "기대 미달 원인 분포": "Underperformance Cause Distribution",
    "지난주 리텐션 액션 결과": "Last Week Retention Action Results",
    "기대 대비": "vs Expected",
    "원": "",
    "건": " actions",
    "예상과 다른 반응을 보인 고객": "Customers with Unexpected Outcomes",
    "클릭하면 해당 고객의 이벤트 로그, 주문 내역, 쿠폰 이력을 확인할 수 있습니다.": "Click to view event logs, order history, and coupon history for each customer.",
    "전체": "All",
    "판정 필터": "Outcome Filter",
    "해당 판정의 고객이 없습니다.": "No customers with this outcome.",
    "추천 카테고리": "Recommended Category",
    "쿠폰 사용": "Coupon Used",
    "전환": "Converted",
    "개입 강도": "Intervention Intensity",
    "이벤트 로그": "Event Log",
    "주문 내역": "Order History",
    "쿠폰 이력": "Coupon History",
    "최근": "Recent",
    "건만 표시": " shown",
    "이벤트 기록 없음": "No event records",
    "총 구매": "Total Purchases",
    "회": " times",
    "주문 기록 없음": "No order records",
    "총 쿠폰 지급": "Total Coupons Issued",
    "쿠폰 이력 없음": "No coupon history",
    "상위": "Top",
    "적절한 비용으로 기대 이상의 성과": "Good ROI and profit with reasonable cost",
    "이익은 있지만 ROI가 기대보다 낮음": "Profitable but ROI below expectations",
    "쿠폰 비용 대비 성과 부족": "Poor performance relative to coupon cost",
    "잘못된 대상에 액션 집행": "Action executed on wrong target",
    "미실행으로 기회 손실 발생": "Opportunity lost due to non-execution",
    "시뮬레이션 설정": "Simulation Settings",
    "CRM 담당자가 추천 액션 중 실제 실행하는 비율": "Proportion of recommended actions actually executed by CRM operator",
    "고비용 쿠폰 추천의 실행 비율 (보통 더 낮음)": "Execution rate for high-cost coupon recommendations (usually lower)",
    "실제 성과가 예상에서 벗어나는 정도": "How much actual performance deviates from predictions",
    "실행": "Executed",
    "미실행": "Not Executed",
    "손실 액션이 없습니다!": "No loss actions!",
    "실행된 액션이 없습니다.": "No executed actions.",
    "쿠폰": "Coupon",
    "손익": "P&L",
    "실행 여부": "Executed",
})
UI_TEXT["ja"].update({
    "LLM 결과 요약": "LLM結果サマリー",
    "LLM결과요약": "LLM結果サマリー",
    "현재 화면의 지표·표·그래프에서 추린 요약 컨텍스트만 바탕으로 응답합니다.": "現在画面の指標・表・グラフから抽出した要約コンテキストだけに基づいて応答します。",
    "AI가 현재 화면의 결과를 요약하는 중입니다...": "AIが現在画面の結果を要約しています...",
    "AI 요약 생성 중 오류가 발생했습니다": "AI要約の生成中にエラーが発生しました",
    "추가 질문은 사이드바의 AI 챗봇 버튼을 눌러 이어서 대화할 수 있습니다.": "追加質問はサイドバーのAIチャットボットから続けられます。",
    "AI 챗봇": "AIチャットボット",
    "챗봇 닫기": "チャットボットを閉じる",
    "챗봇 열기": "チャットボットを開く",
    "LLM 기능이 꺼져 있어 챗봇을 열 수 없습니다.": "LLM機能がオフのためチャットボットを開けません。",
    "현재 화면": "現在の画面",
    "화면의 표·그래프를 보면서 질문할 수 있습니다.": "画面の表・グラフを見ながら質問できます。",
    "고정된 챗봇 컨텍스트": "固定されたチャットボットコンテキスト",
    "화면을 이동해도 챗봇은 처음 열었던 화면의 데이터로 유지됩니다.": "画面を移動しても、チャットボットは最初に開いた画面のデータを維持します。",
    "현재 화면으로 컨텍스트 갱신": "現在画面でコンテキストを更新",
    "실시간 화면에서는 새로고침 시 최신 DB/캐시 상태를 다시 읽습니다. 나머지 화면도 캐시를 비우고 다시 계산합니다.": "リアルタイム画面では更新時に最新のDB/キャッシュ状態を再読み込みします。他の画面もキャッシュを削除して再計算します。",
    "LLM 요약은 API 키가 준비된 경우에만 메인 화면에 표시됩니다.": "LLM要約はAPIキーが準備されている場合のみメイン画面に表示されます。",
    "대화 지우기": "会話を削除",
    "컨텍스트": "コンテキスト",
    "현재 화면에 대해 질문하세요...": "現在画面について質問してください...",
    "현재 화면에 대해 질문하세요.": "現在画面について質問してください。",
    "AI 답변 생성 중": "AI回答を生成中",
    "AI 답변 생성 중 오류가 발생했습니다": "AI回答の生成中にエラーが発生しました",
    "표시할 데이터가 없습니다.": "表示するデータがありません。",
    "검색": "検索",
    "고객 ID 검색": "顧客ID検索",
    "분포 차원 선택": "分布次元を選択",
    "LLM 설정": "LLM設定",
    "권장: API 키는 코드에 쓰지 말고 환경변수 OPENAI_API_KEY 또는 Streamlit secrets로 관리하세요.": "推奨: APIキーはコードに書かず、OPENAI_API_KEY環境変数またはStreamlit secretsで管理してください。",
    "LLM 요약/질문 기능 사용": "LLM要約/質問機能を使用",
    "OpenAI API Key (선택)": "OpenAI API Key（任意）",
    "비워두면 OPENAI_API_KEY 환경변수를 사용합니다.": "空欄の場合はOPENAI_API_KEY環境変数を使用します。",
    "모델이 목록에 없으면 '직접 입력'을 선택해서 모델명을 넣어주세요.": "モデルが一覧にない場合は「直接入力」を選択してモデル名を入力してください。",
    "LLM 모델 선택": "LLMモデル選択",
    "LLM 모델명 (직접 입력)": "LLMモデル名（直接入力）",
    "현재 OPENAI_API_KEY 환경변수를 사용하도록 설정되어 있습니다.": "現在OPENAI_API_KEY環境変数を使用する設定です。",
    "자사 데이터 Live DB 연결됨": "Live DB接続済み",
    "자사 데이터 Live DB 상태 확인 실패": "Live DB状態確認失敗",
    "최신 이벤트": "最新イベント",
    "Live DB 상태": "Live DB状態",
    "저장 추천후보": "保存推薦候補",
    "현재 데이터셋과 Live DB가 일치하지 않아 CSV/결과 파일 기준으로 표시합니다.": "現在のデータセットとLive DBが一致しないため、CSV/結果ファイル基準で表示します。",
    "모드/데이터셋 변경": "モード/データセット変更",
    "기존 결과로 대시보드 보기": "既存結果でダッシュボードを開く",
    "학습 완료. 대시보드로 이동합니다.": "学習完了。ダッシュボードへ移動します。",
    "PostgreSQL user-live DB 초기 적재 완료": "PostgreSQL user-live DB初期投入完了",
    "PostgreSQL user-live DB 자동 적재 실패": "PostgreSQL user-live DB自動投入失敗",
    "주간 액션 성과 리뷰": "週次アクション成果レビュー",
    "이 화면은 실제 집행 결과가 아닌, 추천 데이터 기반의 시뮬레이션 리뷰입니다. 실행률과 성과 노이즈 슬라이더로 가상 시나리오를 조정할 수 있습니다.": "この画面は実際の実行結果ではなく、推薦データ基盤のシミュレーションレビューです。実行率と成果ノイズスライダーで仮想シナリオを調整できます。",
    "전체 실행률": "全体実行率",
    "고쿠폰 실행률": "高クーポン実行率",
    "성과 노이즈": "成果ノイズ",
    "시뮬레이션 시드": "シミュレーションシード",
    "총 추천 건수": "総推薦件数",
    "총 집행 건수": "総実行件数",
    "총 집행 예산": "総実行予算",
    "기대 이익 합계": "期待利益合計",
    "실제 이익 합계": "実際利益合計",
    "손실 액션 수": "損失アクション数",
    "기대 vs 실제 ROI": "期待 vs 実際 ROI",
    "추천 카테고리별 기대 vs 실제 ROI": "推薦カテゴリー別 期待 vs 実際 ROI",
    "세그먼트별 손익": "セグメント別損益",
    "세그먼트별 손익 히트맵": "セグメント別損益ヒートマップ",
    "손실 Top N": "損失 Top N",
    "손실 액션 Top 20": "損失アクション Top 20",
    "판정 분포": "判定分布",
    "액션 판정 분포": "アクション判定分布",
    "전체 액션 상세": "全アクション詳細",
    "주간 액션 성과 리뷰 테이블": "週次アクション成果レビューテーブル",
    "다음 주 정책 조정 제안": "来週の方針調整提案",
    "아래 제안은 이번 주 시뮬레이션 성과를 기반으로 자동 생성된 운영 힌트입니다.": "以下の提案は今週のシミュレーション成果に基づき自動生成された運用ヒントです。",
    "적정 판단": "適切な判断",
    "기대 미달": "期待未達",
    "과잉 투자": "過剰投資",
    "타겟 오류": "対象誤り",
    "실행 누락": "実行漏れ",
    "기대 이익": "期待利益",
    "실제 이익": "実際利益",
    "손실 액션 상세": "損失アクション詳細",
    "개인화 추천 또는 최적화 선정 고객 산출물이 없습니다.": "パーソナライズ推薦または最適化選定顧客の産出物がありません。",
    "실행 여부": "実行有無",
    "판정": "判定",
    "건수": "件数",
    "과투자 추정 금액": "過剰投資推定額",
    "기대 미달 고객 수": "期待未達顧客数",
    "쿠폰 강도별 실제 전환율": "クーポン強度別実際転換率",
    "기대 미달 원인 분포": "期待未達原因分布",
    "지난주 리텐션 액션 결과": "先週のリテンションアクション結果",
    "기대 대비": "期待比",
    "원": "ウォン",
    "건": "件",
    "예상과 다른 반응을 보인 고객": "予想と異なる反応を示した顧客",
    "클릭하면 해당 고객의 이벤트 로그, 주문 내역, 쿠폰 이력을 확인할 수 있습니다.": "クリックすると顧客のイベントログ、注文履歴、クーポン履歴を確認できます。",
    "전체": "全体",
    "판정 필터": "判定フィルター",
    "해당 판정의 고객이 없습니다.": "該当判定の顧客がいません。",
    "추천 카테고리": "推薦カテゴリー",
    "쿠폰 사용": "クーポン使用",
    "전환": "転換",
    "개입 강도": "介入強度",
    "이벤트 로그": "イベントログ",
    "주문 내역": "注文履歴",
    "쿠폰 이력": "クーポン履歴",
    "최근": "最近",
    "건만 표시": "件のみ表示",
    "이벤트 기록 없음": "イベント記録なし",
    "총 구매": "総購入",
    "회": "回",
    "주문 기록 없음": "注文記録なし",
    "총 쿠폰 지급": "総クーポン支給",
    "쿠폰 이력 없음": "クーポン履歴なし",
    "상위": "上位",
    "적절한 비용으로 기대 이상의 성과": "適切なコストで期待以上の成果",
    "이익은 있지만 ROI가 기대보다 낮음": "利益はあるがROIが期待より低い",
    "쿠폰 비용 대비 성과 부족": "クーポンコスト比成果不足",
    "잘못된 대상에 액션 집행": "誤った対象にアクション実行",
    "미실행으로 기회 손실 발생": "未実行による機会損失",
    "시뮬레이션 설정": "シミュレーション設定",
    "CRM 담당자가 추천 액션 중 실제 실행하는 비율": "CRM担当者が推薦アクションのうち実際に実行する割合",
    "고비용 쿠폰 추천의 실행 비율 (보통 더 낮음)": "高コストクーポン推薦の実行率（通常より低い）",
    "실제 성과가 예상에서 벗어나는 정도": "実際の成果が予測から外れる程度",
    "실행": "実行",
    "미실행": "未実行",
    "손실 액션이 없습니다!": "損失アクションがありません！",
    "실행된 액션이 없습니다.": "実行されたアクションがありません。",
    "쿠폰": "クーポン",
    "손익": "損益",
    "실행 여부": "実行有無",
})


# ============================================================
# [UX/i18n PATCH] 쉬운 표현, 값 라벨, 핵심 뷰 안내문
# ============================================================
EXTRA_UI_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "뷰 안내": "View guide",
        "이 화면을 보는 이유": "Why this view matters",
        "확인할 정보": "What to check",
        "활용 목적": "How to use it",
        "이탈 위험이 높은 고객을 먼저 확인해 리텐션 대응의 출발점을 잡습니다.": "Start by identifying customers with high churn risk.",
        "전체 위험 규모와 고객별 위험도를 함께 보며 대응 우선순위를 정합니다.": "Check the overall risk size and each customer's risk level to prioritize actions.",
        "예산 화면과 추천 화면으로 넘어가기 전에 어떤 고객군이 문제인지 빠르게 파악하는 목적입니다.": "Use this as the starting point before budget allocation and personalized recommendations.",
        "한정된 예산을 어떤 고객·세그먼트에 먼저 쓸지 결정하는 화면입니다.": "Decide which customers and segments deserve budget first.",
        "예상 이익, 비용, 고객 반응 가능성을 함께 보며 최종 타겟을 검토합니다.": "Review final targets using expected profit, cost, and response likelihood together.",
        "운영자는 이 화면을 바탕으로 캠페인 집행 대상과 예산 배분 근거를 설명할 수 있습니다.": "Use this view to explain campaign targets and the rationale behind budget allocation.",
        "최종 타겟 고객에게 어떤 상품·혜택·액션을 제안할지 확인하는 화면입니다.": "See which product, benefit, or action should be suggested to each final target.",
        "추천 점수와 추천 이유를 통해 고객별 다음 행동을 바로 실행 가능한 형태로 확인합니다.": "Use recommendation scores and reasons to turn model output into concrete next actions.",
        "단순 예측을 넘어 실제 CRM·마케팅 액션으로 연결하는 목적입니다.": "This view turns prediction into CRM and marketing execution.",
        "실시간 이벤트가 들어올 때 고객 위험도와 액션 큐가 어떻게 바뀌는지 확인합니다.": "Monitor how customer risk and the action queue change as live events arrive.",
        "새 이벤트, 고위험 고객, 큐 적재 상태를 함께 보며 운영 이상 여부를 점검합니다.": "Check live events, high-risk customers, and queue status together to spot operational issues.",
        "시연이나 실제 운영에서 시스템이 데이터 변화에 반응하는지 검증하는 목적입니다.": "Use this view to verify that the system reacts correctly during demos or real operations.",
        "현재 화면은 업로드된 CSV 산출물을 기준으로 표시합니다. 원본 CSV에 Treatment/Control이 없으면 전처리 단계의 자동 배정 및 쉬운 추정값이 사용됩니다.": "This view uses outputs generated from the uploaded CSV. If the original CSV has no Treatment/Control column, the preprocessing step creates a simple estimated comparison group.",
        "예산 배분 후보, 최종 선정 고객, 고객별 선택 이유만 남긴 핵심 운영 화면입니다.": "This core operations view keeps only candidate segments, final targets, and customer-level reasons.",
        "세그먼트별 후보 고객 수를 계산할 데이터가 없습니다.": "There is not enough data to calculate candidate customers by segment.",
        "현재 조건에서 예산 배분 대상 고객이 없습니다.": "No customers match the current budget-allocation conditions.",
        "현재 조건에서 리텐션 타겟 고객이 없습니다.": "No retention target customers match the current conditions.",
        "고객별 설명 테이블을 만들 데이터가 부족합니다. 학습 파이프라인의 explainability 단계가 생성한 산출물을 확인하세요.": "There is not enough data to build customer-level explanations. Check the explanation output from the training pipeline.",
        "현재 예산·이탈 임계값으로 선별된 최종 타겟 고객에게만 새 추천을 생성합니다. 추천 점수는 고객 구매 이력, 최근 관심, 세그먼트 인기, 전역 인기를 혼합해 계산합니다.": "New recommendations are generated only for final targets selected by the current budget and churn-risk threshold. Scores combine purchase history, recent interests, segment popularity, and overall popularity.",
        "현재 조건에서 생성된 추천이 없습니다. 최종 타겟 고객 수가 0명이면 예산을 늘리거나 이탈 임계값을 낮춰야 합니다. 저장된 과거 후보를 현재 추천처럼 표시하지 않습니다.": "No recommendations were generated under the current conditions. If final targets are zero, raise the budget or lower the churn-risk threshold. Saved past candidates are not shown as current recommendations.",
        "이벤트 스트림을 재생하며 고객별 실시간 위험 점수와 액션 큐 상태를 함께 갱신합니다.": "Replay live events and update each customer's risk score and action-queue status together.",
        "실시간 스코어 API 호출 실패": "Real-time score API call failed",
        "먼저 Redis를 실행한 뒤 realtime-bootstrap / realtime-produce / realtime-consume(또는 realtime-replay) 명령을 수행하세요.": "Start Redis first, then run realtime-bootstrap / realtime-produce / realtime-consume or realtime-replay.",
        "실시간 스코어 스냅샷이 없습니다. 스트림 소비 결과가 아직 생성되지 않았을 수 있습니다.": "No real-time score snapshot is available yet. Stream consumption may not have produced results.",
        "큐 상태": "Queue status",
        "트리거 이유": "Trigger reason",
        "행동 신호": "Behavior signal",
        "액션 큐 상태 구성": "Action queue status mix",
        "주요 트리거 이유": "Main trigger reasons",
        "트리거 이유 빈도": "Trigger reason frequency",
        "행동 신호 평균값": "Average behavior signal values",
        "행동 신호 평균": "Behavior signal average",
        "실시간 부분 재최적화 액션 큐": "Real-time re-optimized action queue",
        "Live 이탈 점수 Top 고객": "Top live churn-risk customers",
        "표시할 live score 데이터가 없습니다.": "No live score data to display.",
        "현재 queued action이 없습니다. action_threshold를 낮춰 테스트하거나 새 이벤트를 입력하세요.": "No queued actions now. Lower the action threshold for testing or add new events.",
        "시연을 시작하면 설정된 간격마다 가상 고객 이벤트(방문, 구매 등)가 자동 생성되고, 이탈 점수 재산정 및 액션 큐가 갱신됩니다.": "When the demo starts, virtual customer events are generated at the chosen interval, then churn scores and the action queue are updated.",
        "시연 실행 중": "Demo running",
        "시연 중지": "Stop demo",
        "시연 초기화": "Reset demo",
        "시연 시작": "Start demo",
        "10초마다 자동 새로고침": "Auto-refresh every 10 seconds",
        "N초마다 이벤트 1건 생성": "Generate one event every N seconds",
        "간격(초)": "Interval (seconds)",
        "새 고객 vs 기존 고객 비율": "New vs existing customer ratio",
        "신규 비율": "New-customer ratio",
        "이벤트 로그": "Event log",
        "중지됨": "stopped",
        "다음 자동 새로고침까지 10초...": "Next auto-refresh in 10 seconds...",
        "현재 화면은": "Current view uses",
        "기준 PostgreSQL live DB 운영 모니터입니다.": "PostgreSQL live DB operations monitor.",
        "고객별 이탈 확률 분포": "Customer churn-risk distribution",
        "실시간 이탈 위험 상위 고객": "Top real-time churn-risk customers",
        "세그먼트·개입 강도별 예산 배분": "Budget allocation by customer group and intervention level",
        "세그먼트별 예산 배분": "Budget allocation by customer group",
        "개입 강도": "Intervention level",
        "추천 기준": "Recommendation basis",
        "예산": "Budget",
        "이탈 임계값": "Churn-risk threshold",
        "최대 타겟": "Max targets",
        "명": "customers",
        "행": "rows",
        "건": "items",
        "전체": "total",
        "중": "of",
        "일치": "matched",
        "학습 단계에서는 예산과 이탈 임계값을 조절하지 않습니다. 학습이 끝난 뒤 대시보드의 분석 컨트롤에서 운영 조건을 바꿔 비교하세요.": "Budget and churn-risk threshold are not adjusted during training. After training, change operating conditions from the dashboard analysis controls.",
        "학습 설정": "Training settings",
        "이탈 고객 정의": "Churn definition",
        "학습 예산": "Training budget",
        "학습 이탈 임계값": "Training churn threshold",
        "이탈 기준·학습": "Churn definition & training",
        "Step 5. 이탈 기준·학습": "Step 5. Churn definition & training",
        "이탈 기준: N일 이상 비활성": "Churn definition: inactive for N+ days",
        "총 개입 예산": "Total intervention budget",
        "업로드 샘플": "Uploaded sample",
        "시작 중...": "Starting...",
        "학습 실패": "Training failed",
        "CSV 검증": "CSV validation",
        "전처리": "Preprocessing",
        "피처 생성": "Feature generation",
        "이탈 모델 학습": "Churn model training",
        "Uplift/CLV 계산": "Response/profit estimation",
        "예산 최적화": "Budget optimization",
        "추천/설명 생성": "Recommendation/explanation generation",
        "OpenAI API 키가 설정되지 않았습니다. 사이드바에 키를 입력하거나 OPENAI_API_KEY 환경변수를 설정하세요.": "OpenAI API key is not configured. Enter a key in the sidebar or set the OPENAI_API_KEY environment variable.",
        "안녕하세요. 현재 보고 있는 화면 기준으로 답해드릴게요.": "Hi. I will answer based on the dashboard view you are currently seeing.",
        "왜 이 지표가 높/낮은지": "Why a metric is high or low",
        "어떤 고객/세그먼트가 핵심인지": "Which customers or customer groups matter most",
        "예산·threshold에서 뭘 바꾸면 좋을지": "What to change in budget or churn-risk threshold",
        "AI가 답변하는 중입니다...": "AI is answering...",
        "AI 분석 챗봇": "AI analysis chatbot",
        "드래그해서 이동": "Drag to move",
        "닫기": "Close",
        "실시간 화면에서는 새로고침 시 최신 DB/캐시 상태를 다시 읽습니다. 나머지 화면도 캐시를 비우고 다시 계산합니다.": "On the real-time view, refresh reloads the latest DB/cache state. Other views also clear cache and recalculate.",
    "LLM 요약은 API 키가 준비된 경우에만 메인 화면에 표시됩니다.": "The LLM summary is shown on the main screen only when an API key is ready.",
    "대화 지우기": "Clear chat",
    },
    "ja": {
        "뷰 안내": "画面ガイド",
        "이 화면을 보는 이유": "この画面を見る理由",
        "확인할 정보": "確認する情報",
        "활용 목적": "活用目的",
        "이탈 위험이 높은 고객을 먼저 확인해 리텐션 대응의 출발점을 잡습니다.": "まず離脱リスクの高い顧客を確認し、リテンション対応の出発点を決めます。",
        "전체 위험 규모와 고객별 위험도를 함께 보며 대응 우선순위를 정합니다.": "全体のリスク規模と顧客別リスクを見ながら、対応優先順位を決めます。",
        "예산 화면과 추천 화면으로 넘어가기 전에 어떤 고객군이 문제인지 빠르게 파악하는 목적입니다.": "予算配分や推薦画面に進む前に、どの顧客群が問題かを素早く把握するための画面です。",
        "한정된 예산을 어떤 고객·세그먼트에 먼저 쓸지 결정하는 화면입니다.": "限られた予算をどの顧客・顧客群に優先投入するかを決める画面です。",
        "예상 이익, 비용, 고객 반응 가능성을 함께 보며 최종 타겟을 검토합니다.": "予想利益、費用、顧客の反応見込みを合わせて最終対象を確認します。",
        "운영자는 이 화면을 바탕으로 캠페인 집행 대상과 예산 배분 근거를 설명할 수 있습니다.": "運用担当者はこの画面をもとに、キャンペーン対象と予算配分の根拠を説明できます。",
        "최종 타겟 고객에게 어떤 상품·혜택·액션을 제안할지 확인하는 화면입니다.": "最終対象顧客にどの商品・特典・アクションを提案するかを確認する画面です。",
        "추천 점수와 추천 이유를 통해 고객별 다음 행동을 바로 실행 가능한 형태로 확인합니다.": "推薦スコアと理由から、顧客別の次アクションを実行しやすい形で確認します。",
        "단순 예측을 넘어 실제 CRM·마케팅 액션으로 연결하는 목적입니다.": "単なる予測を実際のCRM・マーケティング施策につなげるための画面です。",
        "실시간 이벤트가 들어올 때 고객 위험도와 액션 큐가 어떻게 바뀌는지 확인합니다.": "リアルタイムイベントにより顧客リスクとアクションキューがどう変わるかを確認します。",
        "새 이벤트, 고위험 고객, 큐 적재 상태를 함께 보며 운영 이상 여부를 점검합니다.": "新規イベント、高リスク顧客、キュー状態を一緒に見て運用上の異常を点検します。",
        "시연이나 실제 운영에서 시스템이 데이터 변화에 반응하는지 검증하는 목적입니다.": "デモや実運用で、システムがデータ変化に反応しているかを検証するための画面です。",
        "현재 화면은 업로드된 CSV 산출물을 기준으로 표시합니다. 원본 CSV에 Treatment/Control이 없으면 전처리 단계의 자동 배정 및 쉬운 추정값이 사용됩니다.": "この画面はアップロードCSVから生成された結果を基準に表示します。元CSVにTreatment/Controlがない場合は、前処理で作成した簡易推定値を使用します。",
        "예산 배분 후보, 최종 선정 고객, 고객별 선택 이유만 남긴 핵심 운영 화면입니다.": "候補顧客群、最終対象顧客、顧客別の選定理由だけを残した主要運用画面です。",
        "세그먼트별 후보 고객 수를 계산할 데이터가 없습니다.": "顧客群別の候補顧客数を計算できるデータがありません。",
        "현재 조건에서 예산 배분 대상 고객이 없습니다.": "現在条件で予算配分対象となる顧客はいません。",
        "현재 조건에서 리텐션 타겟 고객이 없습니다.": "現在条件でリテンション対象となる顧客はいません。",
        "고객별 설명 테이블을 만들 데이터가 부족합니다. 학습 파이프라인의 explainability 단계가 생성한 산출물을 확인하세요.": "顧客別説明テーブルを作成するデータが不足しています。学習パイプラインの説明結果を確認してください。",
        "현재 예산·이탈 임계값으로 선별된 최종 타겟 고객에게만 새 추천을 생성합니다. 추천 점수는 고객 구매 이력, 최근 관심, 세그먼트 인기, 전역 인기를 혼합해 계산합니다.": "現在の予算・離脱リスク基準で選ばれた最終対象顧客にだけ新しい推薦を生成します。推薦スコアは購買履歴、最近の関心、顧客群の人気、全体人気を組み合わせて計算します。",
        "현재 조건에서 생성된 추천이 없습니다. 최종 타겟 고객 수가 0명이면 예산을 늘리거나 이탈 임계값을 낮춰야 합니다. 저장된 과거 후보를 현재 추천처럼 표시하지 않습니다.": "現在条件で生成された推薦はありません。最終対象顧客が0人の場合は、予算を増やすか離脱リスク基準を下げてください。保存済みの過去候補は現在推薦として表示しません。",
        "이벤트 스트림을 재생하며 고객별 실시간 위험 점수와 액션 큐 상태를 함께 갱신합니다.": "イベントストリームを再生し、顧客別のリアルタイムリスクスコアとアクションキュー状態を更新します。",
        "실시간 스코어 API 호출 실패": "リアルタイムスコアAPI呼び出し失敗",
        "먼저 Redis를 실행한 뒤 realtime-bootstrap / realtime-produce / realtime-consume(또는 realtime-replay) 명령을 수행하세요.": "まずRedisを起動し、realtime-bootstrap / realtime-produce / realtime-consume または realtime-replay を実行してください。",
        "실시간 스코어 스냅샷이 없습니다. 스트림 소비 결과가 아직 생성되지 않았을 수 있습니다.": "リアルタイムスコアスナップショットがありません。ストリーム処理結果がまだ生成されていない可能性があります。",
        "큐 상태": "キュー状態",
        "트리거 이유": "トリガー理由",
        "행동 신호": "行動シグナル",
        "액션 큐 상태 구성": "アクションキュー状態構成",
        "주요 트리거 이유": "主なトリガー理由",
        "트리거 이유 빈도": "トリガー理由頻度",
        "행동 신호 평균값": "行動シグナル平均値",
        "행동 신호 평균": "行動シグナル平均",
        "실시간 부분 재최적화 액션 큐": "リアルタイム部分再最適化アクションキュー",
        "Live 이탈 점수 Top 고객": "Live離脱リスク上位顧客",
        "표시할 live score 데이터가 없습니다.": "表示するlive scoreデータがありません。",
        "현재 queued action이 없습니다. action_threshold를 낮춰 테스트하거나 새 이벤트를 입력하세요.": "現在queued actionはありません。テストではaction_thresholdを下げるか新しいイベントを入力してください。",
        "시연을 시작하면 설정된 간격마다 가상 고객 이벤트(방문, 구매 등)가 자동 생성되고, 이탈 점수 재산정 및 액션 큐가 갱신됩니다.": "デモを開始すると、設定間隔ごとに仮想顧客イベント（訪問・購入など）が自動生成され、離脱スコアとアクションキューが更新されます。",
        "시연 실행 중": "デモ実行中",
        "시연 중지": "デモ停止",
        "시연 초기화": "デモ初期化",
        "시연 시작": "デモ開始",
        "10초마다 자동 새로고침": "10秒ごとに自動更新",
        "N초마다 이벤트 1건 생성": "N秒ごとにイベント1件を生成",
        "간격(초)": "間隔（秒）",
        "새 고객 vs 기존 고객 비율": "新規顧客と既存顧客の比率",
        "신규 비율": "新規比率",
        "이벤트 로그": "イベントログ",
        "중지됨": "停止中",
        "다음 자동 새로고침까지 10초...": "次の自動更新まで10秒...",
        "현재 화면은": "現在画面は",
        "기준 PostgreSQL live DB 운영 모니터입니다.": "基準のPostgreSQL live DB運用モニターです。",
        "고객별 이탈 확률 분포": "顧客別離脱リスク分布",
        "실시간 이탈 위험 상위 고객": "リアルタイム離脱リスク上位顧客",
        "세그먼트·개입 강도별 예산 배분": "顧客群・介入レベル別予算配分",
        "세그먼트별 예산 배분": "顧客群別予算配分",
        "개입 강도": "介入レベル",
        "추천 기준": "推薦基準",
        "예산": "予算",
        "이탈 임계값": "離脱リスク基準",
        "최대 타겟": "最大対象",
        "명": "人",
        "행": "行",
        "건": "件",
        "전체": "全体",
        "중": "中",
        "일치": "一致",
        "학습 단계에서는 예산과 이탈 임계값을 조절하지 않습니다. 학습이 끝난 뒤 대시보드의 분석 컨트롤에서 운영 조건을 바꿔 비교하세요.": "学習段階では予算と離脱リスク基準を調整しません。学習後にダッシュボードの分析コントロールで運用条件を変えて比較してください。",
        "학습 설정": "学習設定",
        "이탈 고객 정의": "離脱顧客定義",
        "학습 예산": "学習予算",
        "학습 이탈 임계값": "学習離脱リスク基準",
        "이탈 기준·학습": "離脱基準・学習",
        "Step 5. 이탈 기준·학습": "Step 5. 離脱基準・学習",
        "이탈 기준: N일 이상 비활성": "離脱基準: N日以上非アクティブ",
        "총 개입 예산": "総介入予算",
        "업로드 샘플": "アップロードサンプル",
        "시작 중...": "開始中...",
        "학습 실패": "学習失敗",
        "CSV 검증": "CSV検証",
        "전처리": "前処理",
        "피처 생성": "特徴量生成",
        "이탈 모델 학습": "離脱モデル学習",
        "Uplift/CLV 계산": "反応・利益推定",
        "예산 최적화": "予算最適化",
        "추천/설명 생성": "推薦・説明生成",
        "OpenAI API 키가 설정되지 않았습니다. 사이드바에 키를 입력하거나 OPENAI_API_KEY 환경변수를 설정하세요.": "OpenAI APIキーが設定されていません。サイドバーにキーを入力するか、OPENAI_API_KEY環境変数を設定してください。",
        "안녕하세요. 현재 보고 있는 화면 기준으로 답해드릴게요.": "こんにちは。現在表示している画面を基準に回答します。",
        "왜 이 지표가 높/낮은지": "なぜこの指標が高い/低いのか",
        "어떤 고객/세그먼트가 핵심인지": "どの顧客・顧客群が重要か",
        "예산·threshold에서 뭘 바꾸면 좋을지": "予算や離脱リスク基準で何を変えるべきか",
        "AI가 답변하는 중입니다...": "AIが回答中です...",
        "AI 분석 챗봇": "AI分析チャットボット",
        "드래그해서 이동": "ドラッグして移動",
        "닫기": "閉じる",
        "대화 지우기": "会話を削除",
    },
}
for _lang, _mapping in EXTRA_UI_TEXT.items():
    UI_TEXT.setdefault(_lang, {}).update(_mapping)
UI_TEXT.setdefault("en", {}).update({"학습 대상": "Training target", "파일": "File", "신규": "New", "기존": "Existing", "학습 시작": "Start training", "NEW": "New", "UPD": "Updated"})
UI_TEXT.setdefault("ja", {}).update({"학습 대상": "学習対象", "파일": "ファイル", "신규": "新規", "기존": "既存", "학습 시작": "学習開始", "NEW": "新規", "UPD": "更新"})

UI_TEXT.setdefault("en", {}).update({
    "이탈 시점 예측": "Churn Timing Prediction",
    "고객별 이탈 시점과 예상 손실": "Customer Churn Timing and Expected Loss",
    "고객별로 언제쯤 이탈할 가능성이 큰지와 그때 잃을 수 있는 금액만 표로 보여줍니다.": "Shows only when each customer is likely to churn and the potential loss in a table.",
    "이탈 시점 예측 결과가 없습니다.": "No churn timing prediction results are available.",
    "survival_predictions.csv가 없거나 survival 분석이 아직 실행되지 않았습니다.": "survival_predictions.csv is missing or survival analysis has not been run yet.",
    "시뮬레이터 데모에서는 python src/main.py --mode survival 실행 후 대시보드를 새로고침하세요.": "For the simulator demo, run python src/main.py --mode survival and refresh the dashboard.",
    "예상 손실액은 고객 생애가치(CLV)에 30일 내 이탈 가능성을 곱해 계산합니다. CLV가 없으면 최근 구매금액을 보수적 대체값으로 사용합니다.": "Expected loss is calculated as customer lifetime value (CLV) multiplied by the 30-day churn chance. If CLV is unavailable, recent spend is used as a conservative fallback.",
    "고객이 언제 이탈할 가능성이 큰지 미리 확인하는 화면입니다.": "Use this view to see when each customer is likely to churn.",
    "예상 이탈 시점과 예상 손실액만 남겨 긴급 대응이 필요한 고객을 빠르게 찾습니다.": "It keeps only expected timing and expected loss so urgent customers are easy to find.",
    "예산 배분 전에 먼저 연락해야 할 고객의 시간 우선순위를 정하는 목적입니다.": "Use it to set time-based contact priority before budget allocation.",
    "약": "about",
    "일 이내": "days",
    "알 수 없음": "Unknown",
    "표시 고객 수": "Displayed customers",
    "30일 내 이탈 가능성 기준": "30-day churn chance threshold",
    "이 기준 이상인 고객만 테이블에 표시됩니다. 0%로 두면 모든 고객을 표시합니다.": "Only customers at or above this threshold appear in the table. Set it to 0% to show every customer.",
    "이 표는 선택한 30일 내 이탈 가능성 이상인 고객을 모두 보여줍니다.": "This table shows all customers whose 30-day churn chance is at or above the selected threshold.",
    "현재 기준 이상 고객": "Customers above current threshold",
    "명": " customers",
    "이탈 기준 설정 안내": "Churn definition guide",
    "이 슬라이더는 고객을 언제부터 이탈로 볼지 정하는 기준입니다. 예를 들어 30일로 두면 마지막 활동 후 30일 이상 지난 고객을 이탈 사례로 학습합니다.": "This slider defines when a customer should be treated as churned. For example, if it is set to 30 days, customers with no activity for 30 days after their last activity are learned as churn cases.",
    "이 기준은 이탈 모델 학습, 생존분석, 이탈 시점 예측의 기준이 됩니다. 업종별 방문·구매 주기에 맞게 조절하세요.": "This setting becomes the basis for churn model training, survival analysis, and churn timing prediction. Adjust it to match the visit or purchase cycle of your business.",
})
UI_TEXT.setdefault("ja", {}).update({
    "이탈 시점 예측": "離脱時点予測",
    "고객별 이탈 시점과 예상 손실": "顧客別の離脱時点と予想損失",
    "고객별로 언제쯤 이탈할 가능성이 큰지와 그때 잃을 수 있는 금액만 표로 보여줍니다.": "顧客ごとにいつ離脱しそうか、その時に失う可能性のある金額だけを表で表示します。",
    "이탈 시점 예측 결과가 없습니다.": "離脱時点予測結果がありません。",
    "survival_predictions.csv가 없거나 survival 분석이 아직 실행되지 않았습니다.": "survival_predictions.csvがないか、survival分析がまだ実行されていません。",
    "시뮬레이터 데모에서는 python src/main.py --mode survival 실행 후 대시보드를 새로고침하세요.": "シミュレーターデモでは python src/main.py --mode survival を実行してからダッシュボードを更新してください。",
    "예상 손실액은 고객 생애가치(CLV)에 30일 내 이탈 가능성을 곱해 계산합니다. CLV가 없으면 최근 구매금액을 보수적 대체값으로 사용합니다.": "予想損失額は顧客生涯価値(CLV)に30日以内の離脱可能性を掛けて計算します。CLVがない場合は最近の購入金額を保守的な代替値として使います。",
    "고객이 언제 이탈할 가능성이 큰지 미리 확인하는 화면입니다.": "顧客がいつ離脱しそうかを事前に確認する画面です。",
    "예상 이탈 시점과 예상 손실액만 남겨 긴급 대응이 필요한 고객을 빠르게 찾습니다.": "予想離脱時点と予想損失額だけを残し、緊急対応が必要な顧客を素早く見つけます。",
    "예산 배분 전에 먼저 연락해야 할 고객의 시간 우선순위를 정하는 목적입니다.": "予算配分の前に、先に連絡すべき顧客の時間優先度を決めるための画面です。",
    "약": "約",
    "일 이내": "日以内",
    "알 수 없음": "不明",
    "표시 고객 수": "表示顧客数",
    "30일 내 이탈 가능성 기준": "30日以内の離脱可能性基準",
    "이 기준 이상인 고객만 테이블에 표시됩니다. 0%로 두면 모든 고객을 표시합니다.": "この基準以上の顧客だけを表に表示します。0%にするとすべての顧客を表示します。",
    "이 표는 선택한 30일 내 이탈 가능성 이상인 고객을 모두 보여줍니다.": "この表は、選択した30日以内の離脱可能性以上の顧客をすべて表示します。",
    "현재 기준 이상 고객": "現在の基準以上の顧客",
    "명": "人",
    "이탈 기준 설정 안내": "離脱基準の設定案内",
    "이 슬라이더는 고객을 언제부터 이탈로 볼지 정하는 기준입니다. 예를 들어 30일로 두면 마지막 활동 후 30일 이상 지난 고객을 이탈 사례로 학습합니다.": "このスライダーは、顧客をいつから離脱とみなすかを決める基準です。たとえば30日に設定すると、最後の活動から30日以上活動がない顧客を離脱事例として学習します。",
    "이 기준은 이탈 모델 학습, 생존분석, 이탈 시점 예측의 기준이 됩니다. 업종별 방문·구매 주기에 맞게 조절하세요.": "この基準は、離脱モデル学習、生存分析、離脱時点予測の基準になります。業種ごとの訪問・購入サイクルに合わせて調整してください。",
})


UI_TEXT.setdefault("en", {}).update({
    "0%로 두어도 전체 행을 한 번에 렌더링하지 않고, 운영 우선순위가 높은 고객부터 제한된 수만 빠르게 표시합니다.": "Even at 0%, the view does not render every row at once; it quickly shows a limited preview starting with the highest-priority customers.",
    "현재 표시는 운영 우선순위 상위 고객만 보여줍니다.": "Current display shows only the top customers by operational priority.",
    "표시 고객 수 제한": "Displayed row limit",
    "이 표는 선택한 기준 이상 고객 중 운영 우선순위가 높은 고객부터 빠르게 보여줍니다.": "This table quickly shows eligible customers above the selected threshold, starting from the highest operational priority.",
})
UI_TEXT.setdefault("ja", {}).update({
    "0%로 두어도 전체 행을 한 번에 렌더링하지 않고, 운영 우선순위가 높은 고객부터 제한된 수만 빠르게 표시합니다.": "0%に設定しても全行を一度に描画せず、運用優先度の高い顧客から限定件数だけ高速表示します。",
    "현재 표시는 운영 우선순위 상위 고객만 보여줍니다.": "現在の表示は運用優先度上位の顧客のみです。",
    "표시 고객 수 제한": "表示顧客数の上限",
    "이 표는 선택한 기준 이상 고객 중 운영 우선순위가 높은 고객부터 빠르게 보여줍니다.": "この表は、選択基準以上の顧客のうち運用優先度が高い顧客から高速表示します。",
})


# ============================================================
# [PATCH] Remaining visible i18n fragments and real-time no-chart labels
# ============================================================
UI_TEXT.setdefault("en", {}).update({
    "5번 화면은 저장 후보를 그대로 쓰지 않고 현재 예산·임계값 타겟 기준으로 새 추천을 만듭니다.": "View 5 generates new recommendations from the current budget and threshold targets instead of reusing saved candidates.",
    "저장 후보를 그대로 쓰지 않고 현재 예산·임계값 타겟 기준으로 새 추천을 만듭니다.": "New recommendations are generated from the current budget and threshold targets instead of saved candidates.",
    "이 값 이상인 고객을 이탈 위험군으로 간주합니다. 모든 화면에서 동일하게 유지됩니다.": "Customers at or above this value are treated as churn-risk customers. The value is shared across all views.",
    "상한 없이 입력 가능합니다. 쉼표 없이 숫자만 입력해도 됩니다.": "There is no upper limit. You may enter numbers without commas.",
    "상한 없이 입력 가능합니다. 1 이상의 정수만 입력하세요.": "There is no upper limit. Enter an integer of 1 or higher.",
    "총 마케팅 예산은 0 이상의 정수로 입력해야 합니다.": "Total marketing budget must be a non-negative integer.",
    "최대 타겟 고객 수는 1 이상의 정수여야 합니다.": "Max target customers must be an integer of 1 or higher.",
    "최대 타겟 고객 수는 1 이상의 정수로 입력해야 합니다.": "Max target customers must be entered as an integer of 1 or higher.",
    "최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.": "Recommendations are generated only for final retention targets after budget and threshold filters.",
    "현재 공통 조건": "Current shared conditions",
    "최종 타겟 고객 수": "Final target customers",
    "실시간 그래프는 시연 집중도를 높이기 위해 숨겼습니다. 아래 표에서 최신 고객 위험도와 액션 큐를 확인하세요.": "Real-time charts are hidden to keep the demo focused. Check the latest customer risk and action queue in the tables below.",
    "실시간 스코어 상위 고객": "Top real-time score customers",
    "실시간 운영 모니터 그래프는 제거하고 표 중심으로 표시합니다.": "Real-time operations charts have been removed and the view is table-first.",
    "분석 컨트롤 값은 언어 전환 시에도 유지됩니다.": "Analysis control values are preserved when the language changes.",
    "화면 전환 최적화가 적용되어 Live DB 조회와 무거운 산출물 로딩을 필요한 화면에서만 수행합니다.": "View-switch optimization is enabled: Live DB calls and heavy artifact loads run only where needed.",
    "고객 위험도 목록": "Customer risk list",
    "액션 큐 목록": "Action queue list",
    "상태 구성 목록": "Status mix list",
    "트리거 이유 목록": "Trigger reason list",
    "행동 신호 목록": "Behavior signal list",
    "예금·대출·카드·거래·잔고·연체·상담 이력 기반 이탈/해지 위험과 캠페인 우선순위를 분석합니다.": "Analyze churn/cancellation risk and campaign priority from deposits, loans, cards, transactions, balances, delinquency, and service history.",
    "방문·검색·장바구니·구매·쿠폰·카테고리 선호 기반 이탈 위험과 개인화 추천을 분석합니다.": "Analyze churn risk and personalized recommendations from visits, searches, carts, purchases, coupons, and category preferences.",
    "에 이전 학습 결과가 있습니다.": " has existing training results.",
    "CSV 구조를 분석하고 자동 매핑하는 중입니다...": "Analyzing the CSV structure and auto-mapping columns...",
    "업로드 완료": "Upload complete",
    "업로드 파일을 찾지 못했습니다.": "The uploaded file was not found.",
    "업로드 파일을 찾지 못했습니다. 이전 단계로 돌아가세요.": "The uploaded file was not found. Go back to the previous step.",
    "분석할 CSV/TSV 파일을 업로드하면 다음 단계로 이동할 수 있습니다.": "Upload a CSV/TSV file to continue to the next step.",
    "시스템 역할": "System role",
    "업로드 컬럼": "Uploaded column",
    "원본 값": "Original value",
    "빈도": "Frequency",
    "내부 표준 값": "Internal standard value",
    "자동 매핑 커버리지": "Auto-mapping coverage",
    "event_type/timestamp 조합이 부족합니다. 스냅샷 데이터로 진행하면 일부 실시간·행동 시계열 분석은 제한됩니다.": "The event_type/timestamp combination is insufficient. If you proceed with snapshot data, some real-time and behavior time-series analyses will be limited.",
    "스냅샷 데이터로 진행": "Proceed with snapshot data",
    "이탈 기준: N일 이상 비활성": "Churn definition: inactive for N+ days",
    "완료": "Complete",
    "부분 완료": "Partially complete",
    "일부 단계 실패": "Some steps failed",
    "산출물을 확인하세요.": "Please check the generated outputs.",
    "이전 단계로": "Previous step",
    "다음": "Next",
    "이전": "Previous",
})
UI_TEXT.setdefault("ja", {}).update({
    "5번 화면은 저장 후보를 그대로 쓰지 않고 현재 예산·임계값 타겟 기준으로 새 추천을 만듭니다.": "5番画面は保存候補をそのまま使わず、現在の予算・閾値で選ばれた対象を基準に新しい推薦を作成します。",
    "저장 후보를 그대로 쓰지 않고 현재 예산·임계값 타겟 기준으로 새 추천을 만듭니다.": "保存候補をそのまま使わず、現在の予算・閾値で選ばれた対象を基準に新しい推薦を作成します。",
    "이 값 이상인 고객을 이탈 위험군으로 간주합니다. 모든 화면에서 동일하게 유지됩니다.": "この値以上の顧客を離脱リスク顧客とみなします。すべての画面で同じ値を維持します。",
    "상한 없이 입력 가능합니다. 쉼표 없이 숫자만 입력해도 됩니다.": "上限なく入力できます。カンマなしの数字だけでも入力できます。",
    "상한 없이 입력 가능합니다. 1 이상의 정수만 입력하세요.": "上限なく入力できます。1以上の整数を入力してください。",
    "총 마케팅 예산은 0 이상의 정수로 입력해야 합니다.": "総マーケティング予算は0以上の整数で入力してください。",
    "최대 타겟 고객 수는 1 이상의 정수여야 합니다.": "最大対象顧客数は1以上の整数である必要があります。",
    "최대 타겟 고객 수는 1 이상의 정수로 입력해야 합니다.": "最大対象顧客数は1以上の整数で入力してください。",
    "최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.": "予算・閾値適用後の最終リテンション対象にだけ推薦を生成します。",
    "현재 공통 조건": "現在の共通条件",
    "최종 타겟 고객 수": "最終対象顧客数",
    "실시간 그래프는 시연 집중도를 높이기 위해 숨겼습니다. 아래 표에서 최신 고객 위험도와 액션 큐를 확인하세요.": "デモの集中度を高めるため、リアルタイムグラフは非表示にしました。下の表で最新の顧客リスクとアクションキューを確認してください。",
    "실시간 스코어 상위 고객": "リアルタイムスコア上位顧客",
    "실시간 운영 모니터 그래프는 제거하고 표 중심으로 표시합니다.": "リアルタイム運用モニターのグラフは削除し、表中心で表示します。",
    "분석 컨트롤 값은 언어 전환 시에도 유지됩니다.": "分析コントロールの値は言語変更時にも維持されます。",
    "화면 전환 최적화가 적용되어 Live DB 조회와 무거운 산출물 로딩을 필요한 화면에서만 수행합니다.": "画面切り替え最適化により、Live DB照会と重い出力読み込みは必要な画面でのみ実行します。",
    "고객 위험도 목록": "顧客リスク一覧",
    "액션 큐 목록": "アクションキュー一覧",
    "상태 구성 목록": "状態構成一覧",
    "트리거 이유 목록": "トリガー理由一覧",
    "행동 신호 목록": "行動シグナル一覧",
    "예금·대출·카드·거래·잔고·연체·상담 이력 기반 이탈/해지 위험과 캠페인 우선순위를 분석합니다.": "預金・融資・カード・取引・残高・延滞・相談履歴を基に離脱/解約リスクとキャンペーン優先順位を分析します。",
    "방문·검색·장바구니·구매·쿠폰·카테고리 선호 기반 이탈 위험과 개인화 추천을 분석합니다.": "訪問・検索・カート・購入・クーポン・カテゴリ嗜好を基に離脱リスクとパーソナライズ推薦を分析します。",
    "에 이전 학습 결과가 있습니다.": "に以前の学習結果があります。",
    "CSV 구조를 분석하고 자동 매핑하는 중입니다...": "CSV構造を分析し、自動マッピングしています...",
    "업로드 완료": "アップロード完了",
    "업로드 파일을 찾지 못했습니다.": "アップロードファイルが見つかりません。",
    "업로드 파일을 찾지 못했습니다. 이전 단계로 돌아가세요.": "アップロードファイルが見つかりません。前の段階に戻ってください。",
    "분석할 CSV/TSV 파일을 업로드하면 다음 단계로 이동할 수 있습니다.": "分析するCSV/TSVファイルをアップロードすると次の段階へ進めます。",
    "시스템 역할": "システム役割",
    "업로드 컬럼": "アップロード列",
    "원본 값": "元の値",
    "빈도": "頻度",
    "내부 표준 값": "内部標準値",
    "자동 매핑 커버리지": "自動マッピングカバレッジ",
    "event_type/timestamp 조합이 부족합니다. 스냅샷 데이터로 진행하면 일부 실시간·행동 시계열 분석은 제한됩니다.": "event_type/timestampの組み合わせが不足しています。スナップショットデータで進むと一部のリアルタイム・行動時系列分析は制限されます。",
    "스냅샷 데이터로 진행": "スナップショットデータで進む",
    "이탈 기준: N일 이상 비활성": "離脱基準: N日以上非アクティブ",
    "완료": "完了",
    "부분 완료": "一部完了",
    "일부 단계 실패": "一部段階失敗",
    "산출물을 확인하세요.": "出力を確認してください。",
    "이전 단계로": "前の段階へ",
    "다음": "次へ",
    "이전": "前へ",
})
# ============================================================
# [/PATCH]
# ============================================================

EXTRA_COLUMN_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "selection_reason": "선정 이유", "reason_summary": "선정 이유", "watchout": "주의사항", "caution": "주의사항", "next_best_action": "다음 추천 액션",
        "uplift_segment": "고객 반응 유형", "risk_group": "위험 그룹", "risk_segment": "위험 등급", "customer_count_label": "고객 수", "recommend_count": "추천 수",
        "status": "상태", "signal": "행동 신호", "mean_value": "평균값", "count": "수", "log": "이벤트 로그",
    },
    "en": {
        "selection_reason": "Reason Selected", "reason_summary": "Reason Selected", "watchout": "Caution", "caution": "Caution", "next_best_action": "Next Action",
        "uplift_segment": "Response Type", "risk_group": "Risk Group", "risk_segment": "Risk Level", "customer_count_label": "Customer Count", "recommend_count": "Recommendations",
        "status": "Status", "signal": "Behavior Signal", "mean_value": "Average Value", "count": "Count", "log": "Event Log",
    },
    "ja": {
        "selection_reason": "選定理由", "reason_summary": "選定理由", "watchout": "注意事項", "caution": "注意事項", "next_best_action": "次の推奨アクション",
        "uplift_segment": "顧客反応タイプ", "risk_group": "リスクグループ", "risk_segment": "リスク等級", "customer_count_label": "顧客数", "recommend_count": "推薦数",
        "status": "状態", "signal": "行動シグナル", "mean_value": "平均値", "count": "数", "log": "イベントログ",
    },
}
for _lang, _mapping in EXTRA_COLUMN_LABELS.items():
    COLUMN_LABELS.setdefault(_lang, {}).update(_mapping)

COUNTERFACTUAL_UI_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "고객별 대응 전략 비교": "Counterfactual Retention Lab",
        "무개입 대비 평균 개선": "Avg. improvement vs no action",
        "양수 개선 고객": "Positive-improvement customers",
        "A/B 검증 권장": "A/B validation recommended",
        "최종 추천 분포": "Final recommendation distribution",
        "고객별 반사실 손익 비교": "Customer-level counterfactual profit comparison",
        "고객별 시나리오 상세": "Customer scenario details",
        "무개입": "No action",
        "5,000원 혜택": "5,000 KRW benefit",
        "상담 전화": "Consultation call",
        "푸시/이메일": "Push/email",
        "7일 대기": "Wait 7 days",
        "반사실 실험실은 실제 집행 결과가 아니라 기존 churn·uplift·CLV·survival 신호를 조합한 의사결정 시뮬레이션입니다. 실제 증분 ROI는 holdout/A-B 검증으로 확인해야 합니다.": "The lab is a decision simulation from churn, uplift, CLV, and survival signals, not realized campaign results. Validate true incremental ROI with holdout/A-B tests.",
    },
    "ja": {
        "고객별 대응 전략 비교": "反事実リテンション実験室",
        "무개입 대비 평균 개선": "無介入比の平均改善",
        "양수 개선 고객": "改善が正の顧客",
        "A/B 검증 권장": "A/B検証推奨",
        "최종 추천 분포": "最終推薦分布",
        "고객별 반사실 손익 비교": "顧客別反事実損益比較",
        "고객별 시나리오 상세": "顧客別シナリオ詳細",
        "무개입": "無介入",
        "5,000원 혜택": "5,000ウォン特典",
        "상담 전화": "相談電話",
        "푸시/이메일": "プッシュ/メール",
        "7일 대기": "7日待機",
        "반사실 실험실은 실제 집행 결과가 아니라 기존 churn·uplift·CLV·survival 신호를 조합한 의사결정 시뮬레이션입니다. 실제 증분 ROI는 holdout/A-B 검증으로 확인해야 합니다.": "反事実実験室は実際の施策結果ではなく、churn・uplift・CLV・survival信号を組み合わせた意思決定シミュレーションです。真の増分ROIはholdout/A-B検証で確認する必要があります。",
    },
}
COUNTERFACTUAL_COLUMN_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "expected_no_action_net_profit": "무개입 예상 순이익",
        "expected_net_profit_coupon_5000": "쿠폰 개입 예상 순이익",
        "expected_net_profit_consult_call": "상담 개입 예상 순이익",
        "expected_net_profit_push_email": "푸시/이메일 예상 순이익",
        "expected_net_profit_wait_7d": "7일 대기 예상 순이익",
        "best_expected_net_profit": "최선 시나리오 예상 순이익",
        "incremental_vs_no_action": "무개입 대비 개선액",
        "final_recommendation": "최종 추천",
        "recommendation_reason": "추천 근거",
        "confidence": "신뢰도",
        "confidence_score": "신뢰도 점수",
        "ab_test_recommended": "A/B 검증 권장",
        "expected_churn_period": "예상 이탈 시점",
    },
    "en": {
        "expected_no_action_net_profit": "No-action expected net profit",
        "expected_net_profit_coupon_5000": "Coupon expected net profit",
        "expected_net_profit_consult_call": "Call expected net profit",
        "expected_net_profit_push_email": "Push/email expected net profit",
        "expected_net_profit_wait_7d": "Wait-7d expected net profit",
        "best_expected_net_profit": "Best-scenario expected net profit",
        "incremental_vs_no_action": "Improvement vs no action",
        "final_recommendation": "Final recommendation",
        "recommendation_reason": "Recommendation rationale",
        "confidence": "Confidence",
        "confidence_score": "Confidence score",
        "ab_test_recommended": "A/B validation recommended",
        "expected_churn_period": "Expected churn timing",
    },
    "ja": {
        "expected_no_action_net_profit": "無介入の予想純利益",
        "expected_net_profit_coupon_5000": "クーポン介入の予想純利益",
        "expected_net_profit_consult_call": "相談介入の予想純利益",
        "expected_net_profit_push_email": "プッシュ/メールの予想純利益",
        "expected_net_profit_wait_7d": "7日待機の予想純利益",
        "best_expected_net_profit": "最善シナリオの予想純利益",
        "incremental_vs_no_action": "無介入比の改善額",
        "final_recommendation": "最終推薦",
        "recommendation_reason": "推薦根拠",
        "confidence": "信頼度",
        "confidence_score": "信頼度スコア",
        "ab_test_recommended": "A/B検証推奨",
        "expected_churn_period": "予想離脱時点",
    },
}
for _lang, _mapping in COUNTERFACTUAL_UI_TEXT.items():
    UI_TEXT.setdefault(_lang, {}).update(_mapping)
for _lang, _mapping in COUNTERFACTUAL_COLUMN_LABELS.items():
    COLUMN_LABELS.setdefault(_lang, {}).update(_mapping)

VIEW_INTRO_LINES: dict[str, list[str]] = {
    "1": [
        "이탈 위험이 높은 고객을 먼저 확인해 리텐션 대응의 출발점을 잡습니다.",
        "전체 위험 규모와 고객별 위험도를 함께 보며 대응 우선순위를 정합니다.",
        "예산 화면과 추천 화면으로 넘어가기 전에 어떤 고객군이 문제인지 빠르게 파악하는 목적입니다.",
    ],
    "9": [
        "고객이 언제 이탈할 가능성이 큰지 미리 확인하는 화면입니다.",
        "예상 이탈 시점과 예상 손실액만 남겨 긴급 대응이 필요한 고객을 빠르게 찾습니다.",
        "예산 배분 전에 먼저 연락해야 할 고객의 시간 우선순위를 정하는 목적입니다.",
    ],
    "4": [
        "한정된 예산을 어떤 고객·세그먼트에 먼저 쓸지 결정하는 화면입니다.",
        "예상 이익, 비용, 고객 반응 가능성을 함께 보며 최종 타겟을 검토합니다.",
        "운영자는 이 화면을 바탕으로 캠페인 집행 대상과 예산 배분 근거를 설명할 수 있습니다.",
    ],
    "5": [
        "최종 타겟 고객에게 어떤 상품·혜택·액션을 제안할지 확인하는 화면입니다.",
        "추천 점수와 추천 이유를 통해 고객별 다음 행동을 바로 실행 가능한 형태로 확인합니다.",
        "단순 예측을 넘어 실제 CRM·마케팅 액션으로 연결하는 목적입니다.",
    ],
    "6": [
        "실시간 이벤트가 들어올 때 고객 위험도와 액션 큐가 어떻게 바뀌는지 확인합니다.",
        "새 이벤트, 고위험 고객, 큐 적재 상태를 함께 보며 운영 이상 여부를 점검합니다.",
        "시연이나 실제 운영에서 시스템이 데이터 변화에 반응하는지 검증하는 목적입니다.",
    ],
    "13": [
        "같은 고객에게 아무것도 하지 않을 때와 여러 개입을 했을 때의 기대 손익을 직접 비교합니다.",
        "무개입, 쿠폰, 상담, 푸시/이메일, 7일 대기 전략의 예상 순이익과 신뢰도를 함께 봅니다.",
        "운영자는 추천 액션을 맹목적으로 따르지 않고, 비용·효과·대기 옵션을 비교해 실험 대상으로 보낼지 결정할 수 있습니다.",
    ],
    "14": [
        "지난주 실행한 리텐션 액션의 기대 대비 실제 성과를 빠르게 점검합니다.",
        "액션별 ROI, 이익/손실, 세그먼트별 손익을 함께 보며 어떤 판단이 맞았고 틀렸는지 확인합니다.",
        "다음 주 예산·타겟·전략 조정 근거를 만들어 캠페인 운영을 개선하는 목적입니다.",
    ],
}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "sure_things": "이미 반응 가능성이 높은 고객", "sleeping_dogs": "건드리면 이탈 위험이 커질 수 있는 고객", "lost_causes": "개입 효과가 낮은 고객", "persuadables": "개입하면 반응할 가능성이 높은 고객",
        "vip_loyal": "충성 VIP 고객", "regular_loyal": "충성 일반 고객", "vip_at_risk": "이탈 위험 VIP 고객", "regular_at_risk": "이탈 위험 일반 고객", "new_customer": "신규 고객", "dormant": "휴면 고객",
        "high_uplift": "개입 반응 높음", "very_high_uplift": "개입 반응 매우 높음", "medium_uplift": "개입 반응 보통", "low_uplift": "개입 반응 낮음", "negative_uplift": "개입 비추천", "unknown_segment": "분류 정보 없음", "live": "실시간 고객", "live_user": "실시간 고객",
        "high": "높음", "medium": "보통", "low": "낮음", "critical": "매우 높음", "queued": "큐에 적재됨", "not_queued": "미적재", "pending": "대기 중", "sent": "발송 완료", "completed": "완료", "failed": "실패",
        "generic_retention_offer": "기본 리텐션 혜택", "coupon_offer": "쿠폰 혜택", "discount_offer": "할인 혜택", "service_recovery": "서비스 회복 안내", "loyalty_reward": "충성 고객 보상", "personalized_coupon": "개인 맞춤 쿠폰", "retention_action": "리텐션 액션",
        "page_view": "페이지 방문", "purchase": "구매", "cart": "장바구니", "add_to_cart": "장바구니 담기", "search": "검색", "login": "로그인", "NEW": "신규", "UPD": "기존 갱신", "High risk": "높은 위험", "Medium risk": "중간 위험", "Low risk": "낮은 위험",
    },
    "en": {
        "sure_things": "Already likely to respond", "sleeping_dogs": "Avoid unnecessary intervention", "lost_causes": "Low expected response", "persuadables": "Likely to respond if contacted",
        "vip_loyal": "Loyal VIP customer", "regular_loyal": "Loyal regular customer", "vip_at_risk": "At-risk VIP customer", "regular_at_risk": "At-risk regular customer", "new_customer": "New customer", "dormant": "Inactive customer",
        "high_uplift": "High response potential", "very_high_uplift": "Very high response potential", "medium_uplift": "Medium response potential", "low_uplift": "Low response potential", "negative_uplift": "Intervention not recommended", "unknown_segment": "No group info", "live": "Live customer", "live_user": "Live customer",
        "high": "High", "medium": "Medium", "low": "Low", "critical": "Critical", "queued": "Queued", "not_queued": "Not queued", "pending": "Pending", "sent": "Sent", "completed": "Completed", "failed": "Failed",
        "generic_retention_offer": "Basic retention offer", "coupon_offer": "Coupon offer", "discount_offer": "Discount offer", "service_recovery": "Service recovery message", "loyalty_reward": "Loyalty reward", "personalized_coupon": "Personalized coupon", "retention_action": "Retention action",
        "page_view": "Page visit", "purchase": "Purchase", "cart": "Cart", "add_to_cart": "Add to cart", "search": "Search", "login": "Login", "NEW": "New", "UPD": "Updated existing", "High risk": "High risk", "Medium risk": "Medium risk", "Low risk": "Low risk",
    },
    "ja": {
        "sure_things": "すでに反応しやすい顧客", "sleeping_dogs": "過度な介入を避ける顧客", "lost_causes": "反応見込みが低い顧客", "persuadables": "介入すると反応しやすい顧客",
        "vip_loyal": "ロイヤルVIP顧客", "regular_loyal": "ロイヤル一般顧客", "vip_at_risk": "離脱リスクVIP顧客", "regular_at_risk": "離脱リスク一般顧客", "new_customer": "新規顧客", "dormant": "休眠顧客",
        "high_uplift": "反応見込み高", "very_high_uplift": "反応見込み非常に高", "medium_uplift": "反応見込み中", "low_uplift": "反応見込み低", "negative_uplift": "介入非推奨", "unknown_segment": "分類情報なし", "live": "リアルタイム顧客", "live_user": "リアルタイム顧客",
        "high": "高", "medium": "中", "low": "低", "critical": "重大", "queued": "キュー登録済み", "not_queued": "未登録", "pending": "待機中", "sent": "送信済み", "completed": "完了", "failed": "失敗",
        "generic_retention_offer": "基本リテンション特典", "coupon_offer": "クーポン特典", "discount_offer": "割引特典", "service_recovery": "サービス回復メッセージ", "loyalty_reward": "ロイヤル顧客特典", "personalized_coupon": "個別クーポン", "retention_action": "リテンション施策",
        "page_view": "ページ訪問", "purchase": "購入", "cart": "カート", "add_to_cart": "カート追加", "search": "検索", "login": "ログイン", "NEW": "新規", "UPD": "既存更新", "High risk": "高リスク", "Medium risk": "中リスク", "Low risk": "低リスク",
    },
}


# Additional plain-language labels for customer types and generated segment names.
_VALUE_LABEL_SUPPLEMENTS = {
    "ko": {
        "new_signup": "가입 초기 고객",
        "churn_progressing": "이탈 조짐 고객",
        "explorer": "탐색 고객",
        "price_sensitive": "가격 민감 고객",
        "High Value-Lost Causes": "고가치·개입 효과 낮은 고객",
        "High Value-Persuadables": "고가치·개입 반응 가능 고객",
        "High Value-Sure Things": "고가치·이미 반응 가능 고객",
        "New Customers": "신규 고객군",
    },
    "en": {
        "new_signup": "Newly signed-up customer",
        "churn_progressing": "Showing churn signs",
        "explorer": "Exploring customer",
        "price_sensitive": "Price-sensitive customer",
        "High Value-Lost Causes": "High-value, low response",
        "High Value-Persuadables": "High-value, likely persuaded",
        "High Value-Sure Things": "High-value, already responsive",
        "New Customers": "New customer group",
    },
    "ja": {
        "new_signup": "登録直後の顧客",
        "churn_progressing": "離脱兆候のある顧客",
        "explorer": "探索中の顧客",
        "price_sensitive": "価格重視顧客",
        "High Value-Lost Causes": "高価値・反応見込み低",
        "High Value-Persuadables": "高価値・反応見込みあり",
        "High Value-Sure Things": "高価値・すでに反応しやすい",
        "New Customers": "新規顧客群",
    },
}
for _lang, _mapping in _VALUE_LABEL_SUPPLEMENTS.items():
    VALUE_LABELS.setdefault(_lang, {}).update(_mapping)

PHRASE_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "개입 반응 가능성이 큼": "high response potential", "고객 가치가 높음": "high customer value", "예상 ROI가 양호함": "good expected ROI", "단기 이탈 가속 주의": "watch for short-term churn acceleration", "가격·서비스·타이밍 리스크를 함께 점검": "check price, service, and timing risks together",
    },
    "ja": {
        "개입 반응 가능성이 큼": "介入反応の可能性が高い", "고객 가치가 높음": "顧客価値が高い", "예상 ROI가 양호함": "予想ROIが良好", "단기 이탈 가속 주의": "短期離脱の加速に注意", "가격·서비스·타이밍 리스크를 함께 점검": "価格・サービス・タイミングリスクを一緒に確認",
    },
}



# ============================================================
# [FULL I18N PATCH] Runtime UI translation coverage
# - Adds translations for remaining dashboard/wizard/control messages.
# - Runtime wrappers below also translate unwrapped Streamlit/Plotly labels.
# ============================================================
FULL_UI_TEXT_PATCH: dict[str, dict[str, str]] = {
    "en": {
        "실제 data/raw 산출물을 찾지 못해 mock data로 실행 중입니다.": "Running with mock data because no real data/raw outputs were found.",
        "시뮬레이터 데모 산출물이 아직 없습니다. docker compose up만 실행하면 일부 모델 검증/생존분석/실험 산출물은 생성되지 않습니다.": "Simulator demo outputs are not available yet. Running only docker compose up does not create some model validation, survival-analysis, or experiment outputs.",
        "python src/main.py --mode train, survival, abtest, fidelity 등 필요한 시뮬레이터 산출 명령을 먼저 실행하세요.": "Run the required simulator output commands first, such as python src/main.py --mode train, survival, abtest, or fidelity.",
        "이 값 이상인 고객을 이탈 위험군으로 간주합니다. 모든 화면에서 동일하게 유지됩니다.": "Customers at or above this value are treated as churn-risk customers. This value stays the same across all views.",
        "상한 없이 입력 가능합니다. 쉼표 없이 숫자만 입력해도 됩니다.": "No upper limit. You may enter numbers without commas.",
        "상한 없이 입력 가능합니다. 1 이상의 정수만 입력하세요.": "No upper limit. Enter an integer of 1 or higher.",
        "현재 공통 조건": "Current common conditions",
        "threshold": "threshold",
        "예산": "budget",
        "최종 타겟 고객 수": "final target customers",
        "원": "KRW",
        "실시간 화면에서는 새로고침 시 최신 DB/캐시 상태를 다시 읽습니다. 나머지 화면도 캐시를 비우고 다시 계산합니다.": "On the real-time view, refresh reloads the latest DB/cache state. Other views also clear cache and recalculate.",
        "최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.": "Recommendations are generated only for final retention targets after applying the budget and threshold.",
        "고객당 추천 개수": "Recommendations per customer",
        "코호트 리텐션 분석": "Cohort Retention Analysis",
        "현재 기준": "Current basis",
        "period 0은 코호트 정의상 100%로 고정하고, 아직 관측할 수 없는 미래 period는 0이 아니라 공란으로 둡니다.": "Period 0 is fixed at 100% by cohort definition; future periods that cannot yet be observed are left blank, not zero.",
        "해당 월 재방문율(point)은 재활성화 고객 때문에 month 2가 month 1보다 높아질 수 있습니다. 최근/오래된 코호트를 섞어 해석하지 않도록 아래 공통 비교 지표를 함께 보세요.": "Monthly return rate (point) can be higher in month 2 than month 1 because of reactivated customers. Use the common comparison metrics below to avoid mixing recent and old cohorts.",
        "롤링 리텐션(rolling)은 해당 월 또는 그 이후에 다시 살아난 고객까지 포함하므로 곡선이 단조 감소합니다. 코호트 붕괴 속도를 비교하기에 더 안정적입니다.": "Rolling retention includes customers who return in that month or later, so the curve decreases monotonically. It is more stable for comparing cohort decay speed.",
        "참고: 현재 point 기준에서는": "Note: under the current point basis,",
        "개 코호트에서 후행 월 리텐션이 앞선 월보다 높게 나타났습니다.": "cohorts show later-month retention higher than earlier-month retention.",
        "표시할 코호트 데이터가 없습니다.": "No cohort data to display.",
        "가입 코호트별 리텐션 곡선": "Retention curve by signup cohort",
        "코호트 리텐션 히트맵": "Cohort retention heatmap",
        "경과 기간(개월)": "Elapsed period (months)",
        "코호트": "Cohort",
        "코호트 리텐션 테이블": "Cohort retention table",
        "공통 기간 비교": "Common-period comparison",
        "공통 기간 비교 테이블": "Common-period comparison table",
        "Uplift·CLV 세그먼트 분석": "Uplift and CLV Segment Analysis",
        "Uplift 세그먼트별 고객 수": "Customers by uplift segment",
        "Uplift 세그먼트 요약": "Uplift segment summary",
        "상위 고객의 Uplift-CLV 분포": "Uplift-CLV distribution of top customers",
        "버블 크기는 expected_incremental_profit 대신 value_score(CLV × uplift_score)를 사용합니다. 차트는 성능을 위해 상위 500명만, 아래 테이블은 전체 정렬 결과를 보여줍니다.": "Bubble size uses value_score (CLV × uplift_score) instead of expected_incremental_profit. For performance, the chart shows only the top 500 customers, while the table shows the full sorted result.",
        "상위 고객 테이블": "Top customer table",
        "학습 결과 아티팩트": "Training Result Artifacts",
        "이 화면은 백엔드 API가 보관 중인 최신 학습 산출물을 읽기 전용으로 표시합니다. 대시보드에서 학습 파라미터를 조정하거나 재학습을 직접 실행하지 않습니다.": "This view displays the latest training outputs stored by the backend API in read-only mode. Training parameters are not changed and retraining is not run directly from the dashboard.",
        "학습 결과를 아직 불러오지 못했습니다.": "Training results could not be loaded yet.",
        "학습 메타데이터": "Training metadata",
        "선택된 threshold 요약": "Selected threshold summary",
        "선택 threshold 요약": "Selected threshold summary",
        "학습 파라미터 (서버 반영값)": "Training parameters (server-applied values)",
        "학습 파라미터": "Training parameters",
        "학습 시각화": "Training visualization",
        "Feature store 미리보기": "Feature store preview",
        "파일이 없습니다.": "file is missing.",
        "추천 API 호출 실패": "Recommendation API call failed",
        "기준 PostgreSQL live DB 운영 모니터입니다.": "PostgreSQL live DB operations monitor.",
        "시연 실행 중": "Demo running",
        "시연 중지": "Stop demo",
        "시연 초기화": "Reset demo",
        "시연 시작": "Start demo",
        "10초마다 자동 새로고침": "Auto-refresh every 10 seconds",
        "N초마다 이벤트 1건 생성": "Generate one event every N seconds",
        "간격(초)": "Interval (seconds)",
        "새 고객 vs 기존 고객 비율": "New vs existing customer ratio",
        "신규 비율": "New-customer ratio",
        "이벤트 로그": "Event log",
        "다음 자동 새로고침까지 10초...": "Next auto-refresh in 10 seconds...",
        "Live 이탈 점수 Top 고객": "Top live churn-risk customers",
        "실시간 부분 재최적화 액션 큐": "Real-time partially re-optimized action queue",
        "이탈 시점 예측 (Survival Analysis)": "Churn Timing Prediction (Survival Analysis)",
        "Cox Proportional Hazards 기반으로 landmark 시점 이후 얼마 안에 churn risk 상태로 진입할지를 추정합니다. 분류 모델과 달리 \"언제\" 위험이 커지는지를 함께 봅니다.": "Based on Cox Proportional Hazards, this estimates how soon customers enter a churn-risk state after the landmark point. Unlike a classification model, it also shows when risk increases.",
        "survival_metrics.json, survival_predictions.csv 또는 survival 모델 산출물을 찾지 못했습니다.": "survival_metrics.json, survival_predictions.csv, or survival model outputs were not found.",
        "시뮬레이터 데모에서는 python src/main.py --mode survival 실행 후 대시보드를 새로고침하세요.": "For the simulator demo, run python src/main.py --mode survival and refresh the dashboard.",
        "모델": "Model",
        "일": "days",
        "Survival 메타데이터": "Survival metadata",
        "예측 위험군별 생존 곡선": "Survival curves by predicted risk group",
        "단기 churn 위험 상위 고객": "Top customers by short-term churn risk",
        "Survival 예측 결과": "Survival prediction results",
        "주요 hazard coefficient": "Key hazard coefficients",
        "증분 성과 / A-B 실험": "Incremental Performance / A-B Experiment",
        "정확도보다 더 중요한 운영 지표인 증분 리텐션, 추가 유지 고객 수, 비용 대비 유지 성과, dose-response 결과를 함께 봅니다.": "This view shows operational metrics that matter more than accuracy: incremental retention, additional retained customers, retention performance per cost, and dose-response results.",
        "검출력 부족 — 결과를 효과 유무의 근거로 사용할 수 없습니다.": "Insufficient statistical power — do not use this result as evidence of whether the effect exists.",
        "현재 표본은 효과 검출에 필요한 수의 일부에 불과합니다": "The current sample is only a fraction of the size needed to detect the effect",
        "아래 수치(증분 리텐션, ROI 등)는 통계적 노이즈일 가능성이 매우 높으며": "The numbers below, such as incremental retention and ROI, are highly likely to be statistical noise",
        "효과가 없다": "there is no effect",
        "효과를 측정할 수 없었다": "the effect could not be measured",
        "증분 리텐션": "Incremental retention",
        "추가 유지 고객 수": "Additional retained customers",
        "쿠폰 집행 총액": "Total coupon spend",
        "측정 불가": "Not measurable",
        "추가 유지 고객 수가 0 이하라 분모가 정의되지 않습니다. 효과 검출 실패 — 표본 확대 후 재측정 필요.": "The denominator is undefined because additional retained customers are zero or below. Effect detection failed — increase the sample size and measure again.",
        "A/B 해석": "A/B interpretation",
        "개입 강도 효과": "Intervention intensity effect",
        "Persuadables 프로필": "Persuadables profile",
        "두 그룹 간 차이가 통계적으로 유의합니다": "the difference between the two groups is statistically significant",
        "기준": "basis",
        "A/B 테스트 산출물을 찾지 못했습니다.": "A/B test outputs were not found.",
        "개입 강도별 retention rate": "Retention rate by intervention intensity",
        "dose-response arm 요약": "Dose-response arm summary",
        "dose-response 요약을 찾지 못했습니다.": "Dose-response summary was not found.",
        "What-if: 충분한 표본/효과 크기 시 예상 성과": "What-if: Expected performance with enough sample size/effect size",
        "현재 표본의 검출력 한계를 보완하기 위해, 효과 크기 가정별 운영 시나리오를 계산합니다. 실제 운영 데이터 누적 후 본 시스템이 동일 분석을 자동 수행합니다.": "To supplement the power limitation of the current sample, this calculates operating scenarios by assumed effect size. After real operating data accumulates, the system runs the same analysis automatically.",
        "보수적": "Conservative",
        "중간": "Medium",
        "낙관적": "Optimistic",
        "시나리오": "Scenario",
        "추가 유지 고객": "Additional retained customers",
        "추가 매출": "Additional revenue",
        "쿠폰비 반영 ROI": "ROI after coupon cost",
        "효과 크기 가정별 시뮬레이션": "Simulation by assumed effect size",
        "본 표는 동일 표본·쿠폰비 조건에서 효과 크기만 가정해 산출한 추정치입니다.": "This table is an estimate calculated by assuming only the effect size under the same sample and coupon-cost conditions.",
        "운영 데이터가 누적되면 본 시스템이 동일 방식으로 실효 ROI를 자동 산출하도록 설계되어 있습니다.": "The system is designed to automatically calculate realized ROI in the same way once operating data accumulates.",
        "Persuadables 비중": "Persuadables share",
        "도출된 타겟팅 규칙": "Derived targeting rules",
        "Persuadables 수치 프로필 차이": "Numeric profile differences of persuadables",
        "설명가능성 / 고객별 개입 이유": "Explainability / Customer-level Intervention Reasons",
        "왜 이 고객이 위험군인지, 왜 개입 후보로 뽑혔는지, 무엇을 조심해야 하는지를 운영 언어로 풀어 보여줍니다.": "This explains in operational language why each customer is risky, why they were selected for intervention, and what to be careful about.",
        "전역 설명": "Global explanation",
        "고객별 설명": "Customer-level explanation",
        "전역 중요 변수 Top 10": "Top 10 global important features",
        "전역 중요 변수": "Global important features",
        "전역 중요 변수 파일을 찾지 못했습니다.": "The global feature-importance file was not found.",
        "페르소나별 위험·가치 프로필": "Risk/value profile by persona",
        "설명가능성 테이블을 만들 데이터가 부족합니다.": "There is not enough data to build the explainability table.",
        "데이터 진단 / 시뮬레이터 충실도": "Data Diagnostics / Simulator Fidelity",
        "시뮬레이터가 만든 원천 데이터와 파생 산출물이 운영형 분석에 쓰기 적절한지, 기본적인 정합성과 분포를 함께 점검합니다.": "This checks whether the simulator's raw data and derived outputs are suitable for operational analysis by reviewing basic consistency and distributions.",
        "시뮬레이터 원천 데이터/산출 데이터 볼륨, 행동 분포, 고객 분포 진단 결과를 찾지 못했습니다.": "Simulator raw/output data volume, behavior distribution, and customer distribution diagnostics were not found.",
        "양호": "Good",
        "주의": "Warning",
        "점검 항목": "check items",
        "정합성 점검 결과": "Consistency check results",
        "데이터 볼륨": "Data volume",
        "행동 분포": "Behavior distribution",
        "고객 분포": "Customer distribution",
        "원천/산출 데이터 볼륨": "Raw/output data volume",
        "이벤트 타입 분포": "Event type distribution",
        "이벤트 분포를 계산할 데이터가 없습니다.": "There is no data to calculate event distribution.",
        "분포 차원 선택": "Choose distribution dimension",
        "분포": "distribution",
        "고객 분포를 계산할 데이터가 없습니다.": "There is no data to calculate customer distribution.",
        "할인·쿠폰 운영 리스크": "Discount/Coupon Operations Risk",
        "쿠폰 노출/리딤/믹스 리스크 산출물이 없습니다.": "Coupon exposure/redemption/mix risk outputs are not available.",
        "쿠폰 노출 누적, 리딤 효율, 강도별 효과, 추천/개입 믹스를 같이 보면서 할인 남발의 부작용 가능성을 점검합니다.": "Review cumulative coupon exposure, redemption efficiency, effects by intensity, and recommendation/intervention mix to check for side effects from excessive discounting.",
        "노출 고객 수": "Exposed customers",
        "고노출 고객 수": "Highly exposed customers",
        "전체 노출 수": "Total exposures",
        "오픈율": "Open rate",
        "리딤률": "Redemption rate",
        "쿠폰 운영 리스크 플래그": "Coupon operations risk flags",
        "페르소나별 노출": "Exposure by persona",
        "추천/강도 믹스": "Recommendation/intensity mix",
        "운영 해석": "Operational interpretation",
        "페르소나별 평균 쿠폰 노출": "Average coupon exposure by persona",
        "페르소나별 쿠폰 노출/성과": "Coupon exposure/performance by persona",
        "쿠폰 노출 집계를 계산할 데이터가 없습니다.": "There is no data to aggregate coupon exposure.",
        "추천 카테고리 믹스": "Recommended category mix",
        "선정된 개입 강도 믹스": "Selected intervention intensity mix",
        "고강도 개입의 prior effect가 음수이면 혜택을 세게 줄수록 오히려 성과가 악화될 수 있습니다.": "If the prior effect of high-intensity intervention is negative, stronger benefits may actually worsen performance.",
        "현재 high 강도 prior effect": "Current high-intensity prior effect",
        "high 강도 prior effect를 찾지 못했습니다.": "High-intensity prior effect was not found.",
        "노출 고객 수와 리딤률을 함께 봐야 합니다.": "Review exposed customers and redemption rate together.",
        "노출은 많은데 리딤이 낮으면 학습효과/피로 누적 가능성이 큽니다.": "High exposure with low redemption may indicate learning effects or accumulated fatigue.",
        "price_sensitive 성향이 강한 고객군은 단기 반응은 좋을 수 있지만, 장기적으로는 마진 희석과 할인 의존이 커질 수 있습니다.": "Price-sensitive customers may respond in the short term, but over time they can dilute margin and become dependent on discounts.",
        "support 이슈형 고객은 쿠폰보다 서비스 회복 메시지나 CS 해결이 더 나을 수 있습니다.": "For customers with support issues, service recovery messages or CS resolution may work better than coupons.",
        "금융/이커머스 원천 CSV를 업로드하세요. 고객 스냅샷, 거래, 이벤트 로그 형태를 모두 허용합니다.": "Upload a finance/e-commerce source CSV. Customer snapshots, transactions, and event logs are all supported.",
        "금융 데이터 권장 컬럼": "Recommended finance columns",
        "이커머스 데이터 권장 컬럼": "Recommended e-commerce columns",
        "CSV/TSV 파일": "CSV/TSV file",
        "CSV 구조를 분석하고 자동 매핑하는 중입니다...": "Analyzing the CSV structure and auto-mapping columns...",
        "업로드 완료": "Upload completed",
        "분석할 CSV/TSV 파일을 업로드하면 다음 단계로 이동할 수 있습니다.": "Upload a CSV/TSV file to move to the next step.",
        "컬럼 매핑 검토": "Review column mapping",
        "업로드 파일을 찾지 못했습니다. 이전 단계로 돌아가세요.": "The uploaded file was not found. Go back to the previous step.",
        "업로드 파일을 찾지 못했습니다.": "The uploaded file was not found.",
        "시스템 역할": "System role",
        "업로드 컬럼": "Uploaded column",
        "설명": "Description",
        "고객을 식별하는 ID": "ID that identifies the customer",
        "이벤트·거래 발생 시각": "Event/transaction timestamp",
        "방문/구매/거래/상담 등 행동 유형": "Behavior type such as visit, purchase, transaction, or consultation",
        "주문금액·거래금액·잔고 등 금액성 컬럼": "Amount-related column such as order amount, transaction amount, or balance",
        "분석 피처로 사용할 수 있는 컬럼": "Column usable as an analysis feature",
        "매핑 안 함": "Do not map",
        "이벤트·거래 타입 매핑": "Event/transaction type mapping",
        "원본 값": "Original value",
        "빈도": "Frequency",
        "내부 표준 값": "Internal standard value",
        "자동 매핑 커버리지": "Auto-mapping coverage",
        "event_type/timestamp 조합이 부족합니다. 스냅샷 데이터로 진행하면 일부 실시간·행동 시계열 분석은 제한됩니다.": "The event_type/timestamp combination is insufficient. If you proceed with snapshot data, some real-time and behavioral time-series analyses will be limited.",
        "스냅샷 데이터로 진행": "Proceed with snapshot data",
        "다음": "Next",
        "이전": "Previous",
        "이전 단계로": "Back to previous step",
        "오류": "Error",
        "학습 실패": "Training failed",
        "일부 단계 실패": "Some steps failed",
        "산출물을 확인하세요.": "Check the outputs.",
        "파이프라인 실행 중 오류": "Pipeline execution error",
        "완료": "Completed",
        "실패": "Failed",
        "완료된 단계": "Completed steps",
        "실패 단계 상세": "Failed step details",
        "검증 통과": "Validation passed",
        "관련성": "relevance",
        "컬럼 매핑": "Column mapping",
        "왼쪽은 **시스템 스키마 칼럼**, 오른쪽은 **자사 CSV 컬럼** 입니다. 오른쪽 셀을 더블클릭하면 매핑 컬럼을 변경할 수 있습니다.": "The left side is the system schema column and the right side is your CSV column. Double-click the right cell to change the mapped column.",
        "시스템 스키마": "System schema",
        "자사 CSV 컬럼": "Your CSV column",
        "시스템 스키마 (고정)": "System schema (fixed)",
        "자사 CSV 컬럼 ▼": "Your CSV column ▼",
        "시스템에서 사용하는 표준 역할명 — 변경 불가": "Standard role name used by the system — cannot be changed",
        "자동 감지된 결과 — 잘못 매핑되었으면 ▼ 클릭해서 변경": "Auto-detected result — click ▼ to change if it is wrong",
        "event_type 값 매핑": "event_type value mapping",
        "당신의 CSV에 있는 event_type 값입니다.": "event_type values found in your CSV.",
        "해당 값이 데이터에 등장한 횟수": "Number of times this value appears in the data",
        "이 원본 값을 어떤 표준 이벤트로 분류할지 선택하세요.": "Choose which standard event this original value should map to.",
        "event_type 또는 timestamp 컬럼이 감지되지 않았습니다.": "event_type or timestamp column was not detected.",
        "합성 이벤트 데이터": "synthetic event data",
        "신뢰할 수 없습니다": "cannot be trusted",
        "그래도 합성 이벤트로 진행 (제한된 분석만 신뢰 가능)": "Proceed with synthetic events anyway (only limited analyses are reliable)",
        "체크하면 시스템이 가짜 이벤트를 생성해서 학습합니다. 결과 해석에 주의하세요.": "If checked, the system generates synthetic events for training. Interpret the results carefully.",
        "이탈 고객 정의": "Churn customer definition",
        "마지막 활동(이벤트/주문) 이후 며칠 동안 활동이 없으면 \"이탈\"로 분류할지 정합니다. 업종에 따라 적절한 값이 다릅니다.": "Set how many inactive days after the last activity/event/order should classify a customer as churned. The right value differs by industry.",
        "서비스 성격별 권장 기준": "Recommended 기준 by service type",
        "데일리 앱": "daily apps",
        "일반 커머스, 라이프스타일": "general commerce and lifestyle",
        "정기 구독 서비스": "subscription services",
        "접속 기록이 없으면": "if there is no access record",
        "이탈로 간주합니다": "the customer is treated as churned",
        "현재 설정": "Current setting",
        "마지막 활동": "last activity",
        "일 후 이탈": "days later as churned",
        "event_type/timestamp 컬럼이 없어 진행 불가. 위에서 합성 진행에 동의하면 활성화됩니다.": "Cannot proceed because event_type/timestamp columns are missing. It will be enabled if you agree to synthetic processing above.",
        "학습 완료. 대시보드로 이동합니다.": "Training completed. Moving to the dashboard.",
        "전처리, 모델 학습, user-live DB 초기 적재가 완료되었습니다! 이제 터미널에서 curl 이벤트를 주입하면 실시간 운영 모니터에 반영됩니다.": "Preprocessing, model training, and initial user-live DB seeding are complete. Now curl events injected from the terminal will be reflected in the real-time operations monitor.",
        "전처리 및 모델 학습이 완료되었습니다! 대시보드가 자동으로 새로고침됩니다.": "Preprocessing and model training are complete. The dashboard will refresh automatically.",
        "PostgreSQL user-live DB 자동 적재는 실패했습니다. 시연 전 RETENTION_USER_DB_URL, PostgreSQL 실행 상태, API 로그를 확인하세요. 필요하면 터미널에서 seed-from-user-artifacts를 수동 호출하면 됩니다.": "Automatic PostgreSQL user-live DB seeding failed. Before the demo, check RETENTION_USER_DB_URL, PostgreSQL status, and API logs. If needed, call seed-from-user-artifacts manually from the terminal.",
        "seed 오류": "seed error",
        "실제 데이터": "real data",
        "합성 데이터": "synthetic data",
        "문자열 ID 변환": "string ID conversion",
        "수치 ID": "numeric ID",
        "원본 그대로 사용": "used as-is",
        "매핑 양호": "mapping looks good",
        "검토 권장": "review recommended",
        "수정 필요": "needs correction",
        "자동 매핑 실패한": "auto-mapping failed for",
        "개 값": "values",
        "필요시 직접 수정해 주세요": "please adjust manually if needed",
        "매핑 후 분포 (예상)": "Expected distribution after mapping",
        "업로드 데이터의 평균 활동/구매 주기를 기준으로": "Based on the average activity/purchase cycle in the uploaded data",
        "일을 추천합니다": "days is recommended",
        "학습 대상": "Training target",
        "파일": "File",
        "신규": "New",
        "기존": "Existing",
        "NEW": "New",
        "UPD": "Updated",
        "행": "rows",
        "열": "columns",
        "개": "items",
        "회": "times",
        "명": "customers",
        "건": "items",
    },
    "ja": {
        "실제 data/raw 산출물을 찾지 못해 mock data로 실행 중입니다.": "実際のdata/raw出力が見つからないため、mock dataで実行中です。",
        "시뮬레이터 데모 산출물이 아직 없습니다. docker compose up만 실행하면 일부 모델 검증/생존분석/실험 산출물은 생성되지 않습니다.": "シミュレーターデモの出力がまだありません。docker compose upだけでは一部のモデル検証・生存分析・実験出力は作成されません。",
        "python src/main.py --mode train, survival, abtest, fidelity 등 필요한 시뮬레이터 산출 명령을 먼저 실행하세요.": "python src/main.py --mode train、survival、abtest、fidelityなど、必要なシミュレーター出力コマンドを先に実行してください。",
        "이 값 이상인 고객을 이탈 위험군으로 간주합니다. 모든 화면에서 동일하게 유지됩니다.": "この値以上の顧客を離脱リスク顧客とみなします。この値は全画面で同じまま維持されます。",
        "상한 없이 입력 가능합니다. 쉼표 없이 숫자만 입력해도 됩니다.": "上限なしで入力できます。カンマなしの数字だけでも入力できます。",
        "상한 없이 입력 가능합니다. 1 이상의 정수만 입력하세요.": "上限なしで入力できます。1以上の整数を入力してください。",
        "현재 공통 조건": "現在の共通条件",
        "threshold": "しきい値",
        "예산": "予算",
        "최종 타겟 고객 수": "最終対象顧客数",
        "원": "ウォン",
        "실시간 화면에서는 새로고침 시 최신 DB/캐시 상태를 다시 읽습니다. 나머지 화면도 캐시를 비우고 다시 계산합니다.": "リアルタイム画面では更新時に最新のDB/キャッシュ状態を再読み込みします。他の画面もキャッシュをクリアして再計算します。",
        "최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.": "予算としきい値を適用した最終リテンション対象顧客にのみ推薦を生成します。",
        "고객당 추천 개수": "顧客あたり推薦数",
        "코호트 리텐션 분석": "コホートリテンション分析",
        "현재 기준": "現在基準",
        "period 0은 코호트 정의상 100%로 고정하고, 아직 관측할 수 없는 미래 period는 0이 아니라 공란으로 둡니다.": "period 0はコホート定義上100%に固定し、まだ観測できない未来periodは0ではなく空欄にします。",
        "해당 월 재방문율(point)은 재활성화 고객 때문에 month 2가 month 1보다 높아질 수 있습니다. 최근/오래된 코호트를 섞어 해석하지 않도록 아래 공통 비교 지표를 함께 보세요.": "該当月の再訪率(point)は、再活性化顧客によりmonth 2がmonth 1より高くなる場合があります。新旧コホートを混同しないよう、下の共通比較指標も確認してください。",
        "롤링 리텐션(rolling)은 해당 월 또는 그 이후에 다시 살아난 고객까지 포함하므로 곡선이 단조 감소합니다. 코호트 붕괴 속도를 비교하기에 더 안정적입니다.": "ローリングリテンション(rolling)は、その月以降に戻った顧客も含むため曲線が単調減少します。コホートの崩壊速度比較により安定的です。",
        "참고: 현재 point 기준에서는": "参考: 現在のpoint基準では",
        "개 코호트에서 후행 월 리텐션이 앞선 월보다 높게 나타났습니다.": "個のコホートで後続月リテンションが前月より高く表示されました。",
        "표시할 코호트 데이터가 없습니다.": "表示するコホートデータがありません。",
        "가입 코호트별 리텐션 곡선": "加入コホート別リテンション曲線",
        "코호트 리텐션 히트맵": "コホートリテンションヒートマップ",
        "경과 기간(개월)": "経過期間（月）",
        "코호트": "コホート",
        "코호트 리텐션 테이블": "コホートリテンション表",
        "공통 기간 비교": "共通期間比較",
        "공통 기간 비교 테이블": "共通期間比較表",
        "Uplift·CLV 세그먼트 분석": "Uplift・CLVセグメント分析",
        "Uplift 세그먼트별 고객 수": "Upliftセグメント別顧客数",
        "Uplift 세그먼트 요약": "Upliftセグメント要約",
        "상위 고객의 Uplift-CLV 분포": "上位顧客のUplift-CLV分布",
        "버블 크기는 expected_incremental_profit 대신 value_score(CLV × uplift_score)를 사용합니다. 차트는 성능을 위해 상위 500명만, 아래 테이블은 전체 정렬 결과를 보여줍니다.": "バブルサイズはexpected_incremental_profitの代わりにvalue_score（CLV × uplift_score）を使用します。性能のためチャートは上位500人のみ、下の表は全体の並び替え結果を表示します。",
        "상위 고객 테이블": "上位顧客テーブル",
        "학습 결과 아티팩트": "学習結果アーティファクト",
        "이 화면은 백엔드 API가 보관 중인 최신 학습 산출물을 읽기 전용으로 표시합니다. 대시보드에서 학습 파라미터를 조정하거나 재학습을 직접 실행하지 않습니다.": "この画面はバックエンドAPIが保管している最新学習出力を読み取り専用で表示します。ダッシュボードで学習パラメータを調整したり再学習を直接実行したりしません。",
        "학습 결과를 아직 불러오지 못했습니다.": "学習結果をまだ読み込めません。",
        "학습 메타데이터": "学習メタデータ",
        "선택된 threshold 요약": "選択しきい値要約",
        "선택 threshold 요약": "選択しきい値要約",
        "학습 파라미터 (서버 반영값)": "学習パラメータ（サーバー反映値）",
        "학습 파라미터": "学習パラメータ",
        "학습 시각화": "学習可視化",
        "Feature store 미리보기": "Feature storeプレビュー",
        "파일이 없습니다.": "ファイルがありません。",
        "추천 API 호출 실패": "推薦API呼び出し失敗",
        "기준 PostgreSQL live DB 운영 모니터입니다.": "基準のPostgreSQL live DB運用モニターです。",
        "시연 실행 중": "デモ実行中",
        "시연 중지": "デモ停止",
        "시연 초기화": "デモ初期化",
        "시연 시작": "デモ開始",
        "10초마다 자동 새로고침": "10秒ごとに自動更新",
        "N초마다 이벤트 1건 생성": "N秒ごとにイベントを1件生成",
        "간격(초)": "間隔（秒）",
        "새 고객 vs 기존 고객 비율": "新規顧客と既存顧客の比率",
        "신규 비율": "新規比率",
        "이벤트 로그": "イベントログ",
        "다음 자동 새로고침까지 10초...": "次の自動更新まで10秒...",
        "Live 이탈 점수 Top 고객": "Live離脱リスク上位顧客",
        "실시간 부분 재최적화 액션 큐": "リアルタイム部分再最適化アクションキュー",
        "이탈 시점 예측 (Survival Analysis)": "離脱時点予測（Survival Analysis）",
        "Cox Proportional Hazards 기반으로 landmark 시점 이후 얼마 안에 churn risk 상태로 진입할지를 추정합니다. 분류 모델과 달리 \"언제\" 위험이 커지는지를 함께 봅니다.": "Cox Proportional Hazardsに基づき、landmark時点後どれくらいでchurn risk状態に入るかを推定します。分類モデルと異なり「いつ」リスクが高まるかも確認します。",
        "survival_metrics.json, survival_predictions.csv 또는 survival 모델 산출물을 찾지 못했습니다.": "survival_metrics.json、survival_predictions.csv、またはsurvivalモデル出力が見つかりません。",
        "시뮬레이터 데모에서는 python src/main.py --mode survival 실행 후 대시보드를 새로고침하세요.": "シミュレーターデモでは python src/main.py --mode survival を実行後、ダッシュボードを更新してください。",
        "모델": "モデル",
        "일": "日",
        "Survival 메타데이터": "Survivalメタデータ",
        "예측 위험군별 생존 곡선": "予測リスク群別生存曲線",
        "단기 churn 위험 상위 고객": "短期churnリスク上位顧客",
        "Survival 예측 결과": "Survival予測結果",
        "주요 hazard coefficient": "主要hazard coefficient",
        "증분 성과 / A-B 실험": "増分成果 / A-B実験",
        "정확도보다 더 중요한 운영 지표인 증분 리텐션, 추가 유지 고객 수, 비용 대비 유지 성과, dose-response 결과를 함께 봅니다.": "精度より重要な運用指標である増分リテンション、追加維持顧客数、費用対効果、dose-response結果を一緒に確認します。",
        "검출력 부족 — 결과를 효과 유무의 근거로 사용할 수 없습니다.": "検出力不足 — 結果を効果有無の根拠として使えません。",
        "현재 표본은 효과 검출에 필요한 수의 일부에 불과합니다": "現在の標本は効果検出に必要な数の一部に過ぎません",
        "아래 수치(증분 리텐션, ROI 등)는 통계적 노이즈일 가능성이 매우 높으며": "下の数値（増分リテンション、ROIなど）は統計的ノイズである可能性が非常に高く",
        "효과가 없다": "効果がない",
        "효과를 측정할 수 없었다": "効果を測定できなかった",
        "증분 리텐션": "増分リテンション",
        "추가 유지 고객 수": "追加維持顧客数",
        "쿠폰 집행 총액": "クーポン実行総額",
        "측정 불가": "測定不可",
        "추가 유지 고객 수가 0 이하라 분모가 정의되지 않습니다. 효과 검출 실패 — 표본 확대 후 재측정 필요.": "追加維持顧客数が0以下のため分母が定義できません。効果検出失敗 — 標本拡大後に再測定が必要です。",
        "A/B 해석": "A/B解釈",
        "개입 강도 효과": "介入強度効果",
        "Persuadables 프로필": "Persuadablesプロフィール",
        "두 그룹 간 차이가 통계적으로 유의합니다": "2群間の差は統計的に有意です",
        "기준": "基準",
        "A/B 테스트 산출물을 찾지 못했습니다.": "A/Bテスト出力が見つかりません。",
        "개입 강도별 retention rate": "介入強度別retention rate",
        "dose-response arm 요약": "dose-response arm要約",
        "dose-response 요약을 찾지 못했습니다.": "dose-response要約が見つかりません。",
        "What-if: 충분한 표본/효과 크기 시 예상 성과": "What-if: 十分な標本/効果サイズ時の予想成果",
        "현재 표본의 검출력 한계를 보완하기 위해, 효과 크기 가정별 운영 시나리오를 계산합니다. 실제 운영 데이터 누적 후 본 시스템이 동일 분석을 자동 수행합니다.": "現在標本の検出力限界を補完するため、効果サイズ仮定別の運用シナリオを計算します。実運用データ蓄積後、本システムが同じ分析を自動実行します。",
        "보수적": "保守的",
        "중간": "中間",
        "낙관적": "楽観的",
        "시나리오": "シナリオ",
        "추가 유지 고객": "追加維持顧客",
        "추가 매출": "追加売上",
        "쿠폰비 반영 ROI": "クーポン費反映ROI",
        "효과 크기 가정별 시뮬레이션": "効果サイズ仮定別シミュレーション",
        "본 표는 동일 표본·쿠폰비 조건에서 효과 크기만 가정해 산출한 추정치입니다.": "本表は同一標本・クーポン費条件で効果サイズだけを仮定して算出した推定値です。",
        "운영 데이터가 누적되면 본 시스템이 동일 방식으로 실효 ROI를 자동 산출하도록 설계되어 있습니다.": "運用データが蓄積されると、本システムが同じ方式で実効ROIを自動算出するよう設計されています。",
        "Persuadables 비중": "Persuadables比率",
        "도출된 타겟팅 규칙": "導出されたターゲティング規則",
        "Persuadables 수치 프로필 차이": "Persuadables数値プロフィール差",
        "설명가능성 / 고객별 개입 이유": "説明可能性 / 顧客別介入理由",
        "왜 이 고객이 위험군인지, 왜 개입 후보로 뽑혔는지, 무엇을 조심해야 하는지를 운영 언어로 풀어 보여줍니다.": "なぜこの顧客がリスク群なのか、なぜ介入候補に選ばれたのか、何に注意すべきかを運用言語で説明します。",
        "전역 설명": "全体説明",
        "고객별 설명": "顧客別説明",
        "전역 중요 변수 Top 10": "全体重要変数Top 10",
        "전역 중요 변수": "全体重要変数",
        "전역 중요 변수 파일을 찾지 못했습니다.": "全体重要変数ファイルが見つかりません。",
        "페르소나별 위험·가치 프로필": "ペルソナ別リスク・価値プロフィール",
        "설명가능성 테이블을 만들 데이터가 부족합니다.": "説明可能性テーブルを作成するデータが不足しています。",
        "데이터 진단 / 시뮬레이터 충실도": "データ診断 / シミュレーター忠実度",
        "시뮬레이터가 만든 원천 데이터와 파생 산출물이 운영형 분석에 쓰기 적절한지, 기본적인 정합성과 분포를 함께 점검합니다.": "シミュレーターが作成した原始データと派生出力が運用型分析に適切か、基本的な整合性と分布を確認します。",
        "시뮬레이터 원천 데이터/산출 데이터 볼륨, 행동 분포, 고객 분포 진단 결과를 찾지 못했습니다.": "シミュレーター原始/出力データ量、行動分布、顧客分布診断結果が見つかりません。",
        "양호": "良好",
        "주의": "注意",
        "점검 항목": "点検項目",
        "정합성 점검 결과": "整合性点検結果",
        "데이터 볼륨": "データ量",
        "행동 분포": "行動分布",
        "고객 분포": "顧客分布",
        "원천/산출 데이터 볼륨": "原始/出力データ量",
        "이벤트 타입 분포": "イベントタイプ分布",
        "이벤트 분포를 계산할 데이터가 없습니다.": "イベント分布を計算するデータがありません。",
        "분포 차원 선택": "分布次元を選択",
        "분포": "分布",
        "고객 분포를 계산할 데이터가 없습니다.": "顧客分布を計算するデータがありません。",
        "할인·쿠폰 운영 리스크": "割引・クーポン運用リスク",
        "쿠폰 노출/리딤/믹스 리스크 산출물이 없습니다.": "クーポン露出/リディーム/ミックスリスク出力がありません。",
        "쿠폰 노출 누적, 리딤 효율, 강도별 효과, 추천/개입 믹스를 같이 보면서 할인 남발의 부작용 가능성을 점검합니다.": "クーポン露出累積、リディーム効率、強度別効果、推薦/介入ミックスを確認し、割引乱発の副作用可能性を点検します。",
        "노출 고객 수": "露出顧客数",
        "고노출 고객 수": "高露出顧客数",
        "전체 노출 수": "総露出数",
        "오픈율": "開封率",
        "리딤률": "リディーム率",
        "쿠폰 운영 리스크 플래그": "クーポン運用リスクフラグ",
        "페르소나별 노출": "ペルソナ別露出",
        "추천/강도 믹스": "推薦/強度ミックス",
        "운영 해석": "運用解釈",
        "페르소나별 평균 쿠폰 노출": "ペルソナ別平均クーポン露出",
        "페르소나별 쿠폰 노출/성과": "ペルソナ別クーポン露出/成果",
        "쿠폰 노출 집계를 계산할 데이터가 없습니다.": "クーポン露出集計を計算するデータがありません。",
        "추천 카테고리 믹스": "推薦カテゴリミックス",
        "선정된 개입 강도 믹스": "選定された介入強度ミックス",
        "고강도 개입의 prior effect가 음수이면 혜택을 세게 줄수록 오히려 성과가 악화될 수 있습니다.": "高強度介入のprior effectが負の場合、特典を強めるほど成果が悪化する可能性があります。",
        "현재 high 강도 prior effect": "現在のhigh強度prior effect",
        "high 강도 prior effect를 찾지 못했습니다.": "high強度prior effectが見つかりません。",
        "노출 고객 수와 리딤률을 함께 봐야 합니다.": "露出顧客数とリディーム率を一緒に見る必要があります。",
        "노출은 많은데 리딤이 낮으면 학습효과/피로 누적 가능성이 큽니다.": "露出が多いのにリディームが低い場合、学習効果/疲労蓄積の可能性が大きいです。",
        "price_sensitive 성향이 강한 고객군은 단기 반응은 좋을 수 있지만, 장기적으로는 마진 희석과 할인 의존이 커질 수 있습니다.": "price_sensitive傾向が強い顧客群は短期反応は良い可能性がありますが、長期的にはマージン希薄化と割引依存が大きくなる可能性があります。",
        "support 이슈형 고객은 쿠폰보다 서비스 회복 메시지나 CS 해결이 더 나을 수 있습니다.": "support問題型顧客にはクーポンよりサービス回復メッセージやCS解決が有効な場合があります。",
        "금융/이커머스 원천 CSV를 업로드하세요. 고객 스냅샷, 거래, 이벤트 로그 형태를 모두 허용합니다.": "金融/ECの元CSVをアップロードしてください。顧客スナップショット、取引、イベントログ形式をすべて許可します。",
        "금융 데이터 권장 컬럼": "金融データ推奨カラム",
        "이커머스 데이터 권장 컬럼": "ECデータ推奨カラム",
        "CSV/TSV 파일": "CSV/TSVファイル",
        "CSV 구조를 분석하고 자동 매핑하는 중입니다...": "CSV構造を分析し、自動マッピング中です...",
        "업로드 완료": "アップロード完了",
        "분석할 CSV/TSV 파일을 업로드하면 다음 단계로 이동할 수 있습니다.": "分析するCSV/TSVファイルをアップロードすると次の段階に進めます。",
        "컬럼 매핑 검토": "カラムマッピング確認",
        "업로드 파일을 찾지 못했습니다. 이전 단계로 돌아가세요.": "アップロードファイルが見つかりません。前の段階に戻ってください。",
        "업로드 파일을 찾지 못했습니다.": "アップロードファイルが見つかりません。",
        "시스템 역할": "システム役割",
        "업로드 컬럼": "アップロードカラム",
        "설명": "説明",
        "고객을 식별하는 ID": "顧客を識別するID",
        "이벤트·거래 발생 시각": "イベント・取引発生時刻",
        "방문/구매/거래/상담 등 행동 유형": "訪問/購入/取引/相談などの行動タイプ",
        "주문금액·거래금액·잔고 등 금액성 컬럼": "注文金額・取引金額・残高など金額系カラム",
        "분석 피처로 사용할 수 있는 컬럼": "分析特徴量として使用できるカラム",
        "매핑 안 함": "マッピングしない",
        "이벤트·거래 타입 매핑": "イベント・取引タイプマッピング",
        "원본 값": "元の値",
        "빈도": "頻度",
        "내부 표준 값": "内部標準値",
        "자동 매핑 커버리지": "自動マッピングカバレッジ",
        "event_type/timestamp 조합이 부족합니다. 스냅샷 데이터로 진행하면 일부 실시간·행동 시계열 분석은 제한됩니다.": "event_type/timestampの組み合わせが不足しています。スナップショットデータで進むと一部のリアルタイム・行動時系列分析は制限されます。",
        "스냅샷 데이터로 진행": "スナップショットデータで進む",
        "다음": "次へ",
        "이전": "前へ",
        "이전 단계로": "前の段階へ",
        "오류": "エラー",
        "학습 실패": "学習失敗",
        "일부 단계 실패": "一部段階失敗",
        "산출물을 확인하세요.": "出力を確認してください。",
        "파이프라인 실행 중 오류": "パイプライン実行中エラー",
        "완료": "完了",
        "실패": "失敗",
        "완료된 단계": "完了した段階",
        "실패 단계 상세": "失敗段階詳細",
        "검증 통과": "検証通過",
        "관련성": "関連性",
        "컬럼 매핑": "カラムマッピング",
        "왼쪽은 **시스템 스키마 칼럼**, 오른쪽은 **자사 CSV 컬럼** 입니다. 오른쪽 셀을 더블클릭하면 매핑 컬럼을 변경할 수 있습니다.": "左は**システムスキーマカラム**、右は**自社CSVカラム**です。右セルをダブルクリックするとマッピングカラムを変更できます。",
        "시스템 스키마": "システムスキーマ",
        "자사 CSV 컬럼": "自社CSVカラム",
        "시스템 스키마 (고정)": "システムスキーマ（固定）",
        "자사 CSV 컬럼 ▼": "自社CSVカラム ▼",
        "시스템에서 사용하는 표준 역할명 — 변경 불가": "システムで使用する標準役割名 — 変更不可",
        "자동 감지된 결과 — 잘못 매핑되었으면 ▼ 클릭해서 변경": "自動検出結果 — 誤っている場合は▼をクリックして変更",
        "event_type 값 매핑": "event_type値マッピング",
        "당신의 CSV에 있는 event_type 값입니다.": "あなたのCSVにあるevent_type値です。",
        "해당 값이 데이터에 등장한 횟수": "その値がデータに登場した回数",
        "이 원본 값을 어떤 표준 이벤트로 분류할지 선택하세요.": "この元の値をどの標準イベントに分類するか選択してください。",
        "event_type 또는 timestamp 컬럼이 감지되지 않았습니다.": "event_typeまたはtimestampカラムが検出されませんでした。",
        "합성 이벤트 데이터": "合成イベントデータ",
        "신뢰할 수 없습니다": "信頼できません",
        "그래도 합성 이벤트로 진행 (제한된 분석만 신뢰 가능)": "それでも合成イベントで進む（限定的な分析のみ信頼可能）",
        "체크하면 시스템이 가짜 이벤트를 생성해서 학습합니다. 결과 해석에 주의하세요.": "チェックするとシステムが偽イベントを生成して学習します。結果解釈に注意してください。",
        "이탈 고객 정의": "離脱顧客定義",
        "마지막 활동(이벤트/주문) 이후 며칠 동안 활동이 없으면 \"이탈\"로 분류할지 정합니다. 업종에 따라 적절한 값이 다릅니다.": "最後の活動（イベント/注文）後、何日間活動がなければ「離脱」と分類するかを決めます。業種により適切な値は異なります。",
        "서비스 성격별 권장 기준": "サービス性格別推奨基準",
        "데일리 앱": "デイリーアプリ",
        "일반 커머스, 라이프스타일": "一般コマース、ライフスタイル",
        "정기 구독 서비스": "定期購読サービス",
        "접속 기록이 없으면": "接続記録がなければ",
        "이탈로 간주합니다": "離脱とみなします",
        "현재 설정": "現在設定",
        "마지막 활동": "最後の活動",
        "일 후 이탈": "日後に離脱",
        "event_type/timestamp 컬럼이 없어 진행 불가. 위에서 합성 진행에 동의하면 활성화됩니다.": "event_type/timestampカラムがないため進行不可。上で合成進行に同意すると有効化されます。",
        "학습 완료. 대시보드로 이동합니다.": "学習完了。ダッシュボードへ移動します。",
        "전처리, 모델 학습, user-live DB 초기 적재가 완료되었습니다! 이제 터미널에서 curl 이벤트를 주입하면 실시간 운영 모니터에 반영됩니다.": "前処理、モデル学習、user-live DB初期投入が完了しました。これで端末からcurlイベントを注入するとリアルタイム運用モニターに反映されます。",
        "전처리 및 모델 학습이 완료되었습니다! 대시보드가 자동으로 새로고침됩니다.": "前処理とモデル学習が完了しました。ダッシュボードが自動で更新されます。",
        "PostgreSQL user-live DB 자동 적재는 실패했습니다. 시연 전 RETENTION_USER_DB_URL, PostgreSQL 실행 상태, API 로그를 확인하세요. 필요하면 터미널에서 seed-from-user-artifacts를 수동 호출하면 됩니다.": "PostgreSQL user-live DB自動投入に失敗しました。デモ前にRETENTION_USER_DB_URL、PostgreSQL実行状態、APIログを確認してください。必要なら端末でseed-from-user-artifactsを手動呼び出ししてください。",
        "seed 오류": "seedエラー",
        "실제 데이터": "実データ",
        "합성 데이터": "合成データ",
        "문자열 ID 변환": "文字列ID変換",
        "수치 ID": "数値ID",
        "원본 그대로 사용": "元のまま使用",
        "매핑 양호": "マッピング良好",
        "검토 권장": "確認推奨",
        "수정 필요": "修正必要",
        "자동 매핑 실패한": "自動マッピングに失敗した",
        "개 값": "個の値",
        "필요시 직접 수정해 주세요": "必要に応じて直接修正してください",
        "매핑 후 분포 (예상)": "マッピング後分布（予想）",
        "업로드 데이터의 평균 활동/구매 주기를 기준으로": "アップロードデータの平均活動/購入周期を基準に",
        "일을 추천합니다": "日を推奨します",
        "학습 대상": "学習対象",
        "파일": "ファイル",
        "신규": "新規",
        "기존": "既存",
        "NEW": "新規",
        "UPD": "更新",
        "행": "行",
        "열": "列",
        "개": "個",
        "회": "回",
        "명": "人",
        "건": "件",
    },
}
for _lang, _mapping in FULL_UI_TEXT_PATCH.items():
    UI_TEXT.setdefault(_lang, {}).update(_mapping)


# ============================================================
# [PATCH] Table-cell i18n expansion
# Many values in the dashboard are generated by the training/explanation
# pipeline, not by Streamlit widgets. They arrive as table cell values such as
# "이탈 위험이 높음" or "price_sensitive", so widget-level T(...) wrapping alone
# cannot translate them. Keep these mappings close to the UI layer so display
# language changes do not mutate source artifacts.
# ============================================================
_EXTRA_VALUE_LABELS_PATCH: dict[str, dict[str, str]] = {
    "en": {
        "로열VIP고객": "Loyal VIP customer",
        "로열 VIP 고객": "Loyal VIP customer",
        "로열일반고객": "Loyal regular customer",
        "로열 일반 고객": "Loyal regular customer",
        "충성VIP고객": "Loyal VIP customer",
        "충성 VIP 고객": "Loyal VIP customer",
        "충성일반고객": "Loyal regular customer",
        "충성 일반 고객": "Loyal regular customer",
        "이탈위험VIP고객": "At-risk VIP customer",
        "이탈 위험 VIP 고객": "At-risk VIP customer",
        "이탈위험일반고객": "At-risk regular customer",
        "이탈 위험 일반 고객": "At-risk regular customer",
        "price_sensitive": "Price-sensitive customer",
        "churn_progressing": "Churn-progressing customer",
        "churn_risk": "Churn-risk customer",
        "at_risk": "At-risk customer",
        "loyal_vip": "Loyal VIP customer",
        "loyal_regular": "Loyal regular customer",
        "vip_customer": "VIP customer",
        "regular_customer": "Regular customer",
        "medium_high": "Medium-high",
        "very_low": "Very low",
        "very_high": "Very high",
    },
    "ja": {
        "로열VIP고객": "ロイヤルVIP顧客",
        "로열 VIP 고객": "ロイヤルVIP顧客",
        "로열일반고객": "ロイヤル一般顧客",
        "로열 일반 고객": "ロイヤル一般顧客",
        "충성VIP고객": "ロイヤルVIP顧客",
        "충성 VIP 고객": "ロイヤルVIP顧客",
        "충성일반고객": "ロイヤル一般顧客",
        "충성 일반 고객": "ロイヤル一般顧客",
        "이탈위험VIP고객": "離脱リスクVIP顧客",
        "이탈 위험 VIP 고객": "離脱リスクVIP顧客",
        "이탈위험일반고객": "離脱リスク一般顧客",
        "이탈 위험 일반 고객": "離脱リスク一般顧客",
        "price_sensitive": "価格敏感顧客",
        "churn_progressing": "離脱進行顧客",
        "churn_risk": "離脱リスク顧客",
        "at_risk": "離脱リスク顧客",
        "loyal_vip": "ロイヤルVIP顧客",
        "loyal_regular": "ロイヤル一般顧客",
        "vip_customer": "VIP顧客",
        "regular_customer": "一般顧客",
        "medium_high": "中高",
        "very_low": "非常に低い",
        "very_high": "非常に高い",
    },
}
for _lang, _mapping in _EXTRA_VALUE_LABELS_PATCH.items():
    VALUE_LABELS.setdefault(_lang, {}).update(_mapping)

_EXTRA_PHRASE_LABELS_PATCH: dict[str, dict[str, str]] = {
    "en": {
        "이탈 위험이 높음": "high churn risk",
        "이탈 위험 높음": "high churn risk",
        "이탈 위험이 큼": "high churn risk",
        "개입 반응 가능성이 큼": "high response potential",
        "개입 반응의 가능성이 큼": "high response potential",
        "고객 가치가 높음": "high customer value",
        "고객 가치 높음": "high customer value",
        "예상 ROI가 양호함": "good expected ROI",
        "예상 ROI 양호": "good expected ROI",
        "예상 이익이 큼": "high expected profit",
        "예상 증분이익이 큼": "high expected incremental profit",
        "단기 이탈 가속 주의": "watch for short-term churn acceleration",
        "가격·서비스·타이밍 리스크를 함께 점검": "check price, service, and timing risks together",
        "가격/서비스/타이밍 리스크를 함께 점검": "check price, service, and timing risks together",
        "쿠폰 비용 대비 수익성 확인": "check profitability against coupon cost",
        "과도한 할인 의존 주의": "avoid over-reliance on discounts",
        "최근 활동 감소": "recent activity decreased",
        "구매 간격 증가": "purchase interval increased",
        "재방문 감소": "revisit frequency decreased",
        "장바구니 이탈 증가": "cart abandonment increased",
        "개인화 쿠폰 제안": "offer a personalized coupon",
        "리텐션 혜택 제안": "offer a retention benefit",
        "서비스 회복 안내": "send a service recovery message",
        "로열티 보상 제안": "offer a loyalty reward",
        "우선 개입 권장": "priority intervention recommended",
        "관찰 필요": "monitor closely",
        "발송 보류": "hold delivery",
    },
    "ja": {
        "이탈 위험이 높음": "離脱リスクが高い",
        "이탈 위험 높음": "離脱リスクが高い",
        "이탈 위험이 큼": "離脱リスクが高い",
        "개입 반응 가능성이 큼": "介入反応の可能性が高い",
        "개입 반응의 가능성이 큼": "介入反応の可能性が高い",
        "고객 가치가 높음": "顧客価値が高い",
        "고객 가치 높음": "顧客価値が高い",
        "예상 ROI가 양호함": "予想ROIが良好",
        "예상 ROI 양호": "予想ROIが良好",
        "예상 이익이 큼": "予想利益が大きい",
        "예상 증분이익이 큼": "予想増分利益が大きい",
        "단기 이탈 가속 주의": "短期離脱の加速に注意",
        "가격·서비스·타이밍 리스크를 함께 점검": "価格・サービス・タイミングリスクを一緒に確認",
        "가격/서비스/타이밍 리스크를 함께 점검": "価格・サービス・タイミングリスクを一緒に確認",
        "쿠폰 비용 대비 수익성 확인": "クーポン費用に対する収益性を確認",
        "과도한 할인 의존 주의": "過度な割引依存に注意",
        "최근 활동 감소": "最近の活動が減少",
        "구매 간격 증가": "購入間隔が増加",
        "재방문 감소": "再訪問が減少",
        "장바구니 이탈 증가": "カート離脱が増加",
        "개인화 쿠폰 제안": "個別クーポンを提案",
        "리텐션 혜택 제안": "リテンション特典を提案",
        "서비스 회복 안내": "サービス回復メッセージを送信",
        "로열티 보상 제안": "ロイヤルティ特典を提案",
        "우선 개입 권장": "優先介入を推奨",
        "관찰 필요": "継続観察が必要",
        "발송 보류": "送信保留",
    },
}
for _lang, _mapping in _EXTRA_PHRASE_LABELS_PATCH.items():
    PHRASE_LABELS.setdefault(_lang, {}).update(_mapping)
# ============================================================
# [/PATCH]
# ============================================================


UI_TEXT.setdefault("ko", {}).update({
    "Retention Rate": "리텐션율",
    "Retention": "리텐션",
    "이탈 기준값": "이탈 기준값",
    "검색": "검색",
    "고객 ID 검색": "고객 ID 검색",
    "전체": "전체",
    "건": "건",
    "중": "중",
    "일치": "일치",
})
# ============================================================
# [PATCH] Human-friendly dashboard wording, table formatting and glossary captions
# ============================================================
_HUMAN_COLUMN_LABELS_PATCH: dict[str, dict[str, str]] = {
    "ko": {
        "value_score": "고객 가치 점수",
        "expected_roi_2": "예상 ROI",
        "avg_expected_roi": "평균 예상 ROI",
        "avg_churn_probability": "평균 이탈 확률",
        "avg_coupon_exposure": "평균 쿠폰 노출 횟수",
        "coupon_exposure_count": "쿠폰 노출 횟수",
        "redeem_rate": "혜택 사용률",
        "open_rate": "메시지 확인률",
        "event_type": "이벤트 유형",
        "dimension": "분포 기준",
        "value": "분포 값",
        "share": "비중",
        "period": "경과 기간(개월)",
        "cohort_month": "가입 코호트",
        "cohort_size": "코호트 고객 수",
        "retained_customers": "잔존 고객 수",
        "retention_rate": "리텐션율",
        "customer_count": "고객 수",
        "recommend_count": "추천 건수",
    },
    "en": {
        "value_score": "Customer Value Score",
        "expected_roi_2": "Expected ROI",
        "avg_expected_roi": "Average Expected ROI",
        "avg_churn_probability": "Average Churn Probability",
        "avg_coupon_exposure": "Average Coupon Exposure",
        "coupon_exposure_count": "Coupon Exposures",
        "redeem_rate": "Redeem Rate",
        "open_rate": "Open Rate",
        "event_type": "Event Type",
        "dimension": "Dimension",
        "value": "Value",
        "share": "Share",
        "period": "Elapsed Months",
        "cohort_month": "Signup Cohort",
        "cohort_size": "Cohort Size",
        "retained_customers": "Retained Customers",
        "retention_rate": "Retention Rate",
        "customer_count": "Customers",
        "recommend_count": "Recommendations",
    },
    "ja": {
        "value_score": "顧客価値スコア",
        "expected_roi_2": "予想ROI",
        "avg_expected_roi": "平均予想ROI",
        "avg_churn_probability": "平均離脱確率",
        "avg_coupon_exposure": "平均クーポン露出回数",
        "coupon_exposure_count": "クーポン露出回数",
        "redeem_rate": "特典利用率",
        "open_rate": "メッセージ確認率",
        "event_type": "イベント種別",
        "dimension": "分布基準",
        "value": "分布値",
        "share": "比率",
        "period": "経過期間(月)",
        "cohort_month": "登録コホート",
        "cohort_size": "コホート顧客数",
        "retained_customers": "継続顧客数",
        "retention_rate": "リテンション率",
        "customer_count": "顧客数",
        "recommend_count": "推薦数",
    },
}
for _lang, _mapping in _HUMAN_COLUMN_LABELS_PATCH.items():
    COLUMN_LABELS.setdefault(_lang, {}).update(_mapping)

_HUMAN_VALUE_LABELS_PATCH: dict[str, dict[str, str]] = {
    "ko": {
        "new_signup": "가입 초기 고객", "new sign up": "가입 초기 고객", "new signup": "가입 초기 고객",
        "new_customer": "신규 고객", "new customers": "신규 고객군",
        "churn_progressing": "이탈 조짐 고객", "churn progressing": "이탈 조짐 고객",
        "explorer": "탐색 고객", "price_sensitive": "가격 민감 고객", "price sensitive": "가격 민감 고객",
        "support_issue": "서비스 불편 경험 고객", "support issue": "서비스 불편 경험 고객",
        "regular_customer": "일반 고객", "vip_customer": "VIP 고객", "vip": "VIP 고객",
        "dormant_customer": "휴면 고객", "dormant user": "휴면 고객", "loyal_customer": "충성 고객",
        "at_risk_customer": "이탈 위험 고객", "high_value_customer": "고가치 고객", "low_value_customer": "저가치 고객",
        "mid": "보통", "middle": "보통", "moderate": "보통", "medium": "보통", "low": "낮음", "high": "높음",
        "critical": "매우 높음", "very_high": "매우 높음", "very low": "매우 낮음", "very_low": "매우 낮음",
        "medium_high": "다소 높음", "medium-low": "다소 낮음", "중강도": "보통 수준 개입", "고강도": "높은 수준 개입", "저강도": "낮은 수준 개입",
        "generic_retention_offer": "기본 리텐션 혜택 제안", "generic retention offer": "기본 리텐션 혜택 제안",
        "personalized_retention_offer": "개인 맞춤 리텐션 혜택 제안", "personalized retention offer": "개인 맞춤 리텐션 혜택 제안",
        "light_retention_message": "가벼운 재방문 유도 메시지", "light retention message": "가벼운 재방문 유도 메시지",
        "service_recovery_message": "서비스 불편 회복 안내", "service recovery message": "서비스 불편 회복 안내",
        "coupon_offer": "쿠폰 혜택 제안", "discount_offer": "할인 혜택 제안", "loyalty_reward": "충성 고객 보상 제안",
        "monitor (>60d)": "60일 이후까지 관찰", "monitor(>60d)": "60일 이후까지 관찰", "monitor >60d": "60일 이후까지 관찰", "monitor": "관찰 필요",
        "follow_up_soon": "빠른 후속 연락 필요", "immediate_contact": "즉시 연락 권장",
        "own_purchase_history": "고객 본인의 과거 구매 이력", "recent_browse_signal": "최근 둘러본 상품·카테고리 신호",
        "segment_popularity": "비슷한 고객군에서 인기 있는 항목", "global_popularity": "전체 고객에게 인기 있는 항목",
        "category_affinity": "관심 카테고리와의 관련성", "recent_interest": "최근 관심 행동", "price_affinity": "가격·할인 반응 가능성",
        "high_churn_risk": "이탈 위험이 높음", "high_customer_value": "고객 가치가 높음", "good_expected_roi": "예상 ROI가 양호함",
        "recent_activity_drop": "최근 활동이 줄어듦", "purchase_gap_increase": "구매 간격이 길어짐",
        "queued": "큐에 적재됨", "not_queued": "큐에 없음", "queued action": "큐에 적재된 액션", "action queued": "액션 큐에 적재됨",
        "pending": "대기 중", "sent": "발송 완료", "completed": "완료", "failed": "실패",
    },
    "en": {
        "mid": "Medium", "middle": "Medium", "moderate": "Medium",
        "generic retention offer": "Basic retention offer", "personalized retention offer": "Personalized retention offer",
        "light retention message": "Light retention message", "monitor (>60d)": "Monitor after 60 days",
        "own_purchase_history": "Own purchase history", "recent_browse_signal": "Recent browsing signal",
        "segment_popularity": "Popular with similar customers", "global_popularity": "Popular overall",
    },
    "ja": {
        "mid": "中", "middle": "中", "moderate": "中",
        "generic retention offer": "基本リテンション特典の提案", "personalized retention offer": "個別リテンション特典の提案",
        "light retention message": "軽い再訪問促進メッセージ", "monitor (>60d)": "60日後まで観察",
        "own_purchase_history": "本人の過去購入履歴", "recent_browse_signal": "最近閲覧した商品・カテゴリのシグナル",
        "segment_popularity": "類似顧客群で人気の項目", "global_popularity": "全体で人気の項目",
    },
}
for _lang, _mapping in _HUMAN_VALUE_LABELS_PATCH.items():
    VALUE_LABELS.setdefault(_lang, {}).update(_mapping)

_HUMAN_PHRASE_LABELS_PATCH: dict[str, dict[str, str]] = {
    "ko": {
        "Generic retention offer": "기본 리텐션 혜택 제안", "generic retention offer": "기본 리텐션 혜택 제안",
        "Personalized retention offer": "개인 맞춤 리텐션 혜택 제안", "personalized_retention_offer": "개인 맞춤 리텐션 혜택 제안",
        "Light retention message": "가벼운 재방문 유도 메시지", "Monitor (>60d)": "60일 이후까지 관찰", "Monitor(>60d)": "60일 이후까지 관찰", "Monitor >60d": "60일 이후까지 관찰", "중강도": "보통 수준 개입", "고강도": "높은 수준 개입", "저강도": "낮은 수준 개입",
        "own_purchase_history": "고객 본인의 과거 구매 이력", "recent_browse_signal": "최근 둘러본 상품·카테고리 신호", "segment_popularity": "비슷한 고객군에서 인기 있는 항목", "global_popularity": "전체 고객에게 인기 있는 항목",
        "price_sensitive": "가격 민감 고객", "new_signup": "가입 초기 고객", "churn_progressing": "이탈 조짐 고객", "expected roi 2": "예상 ROI", "Expected ROI 2": "예상 ROI",
        "Retention Rate": "리텐션율", "Retention": "리텐션", "value_score": "고객 가치 점수", "count": "건수",
    },
    "en": {}, "ja": {},
}
for _lang, _mapping in _HUMAN_PHRASE_LABELS_PATCH.items():
    PHRASE_LABELS.setdefault(_lang, {}).update(_mapping)

_HUMAN_TERM_CAPTIONS_PATCH: dict[str, dict[str, str]] = {
    "ko": {
        "CustomerType": "고객 유형은 고객의 최근 행동·가치·이탈 조짐을 사람이 이해하기 쉽게 묶은 분류입니다.",
        "ChurnProbability": "이탈 확률은 고객이 설정한 이탈 기준에 가까워지거나 서비스를 떠날 가능성을 0~100%로 표현한 값입니다.",
        "ChurnTiming": "예상 이탈 시점은 현재 상태가 유지될 때 고객이 이탈 상태에 가까워질 것으로 보는 예상 시기입니다.",
        "ExpectedLoss": "예상 손실액은 해당 고객이 이탈할 경우 잃을 수 있는 매출·고객가치를 원화로 환산한 값입니다.",
        "ExpectedProfit": "예상 이익은 이 고객에게 개입했을 때 추가로 얻을 것으로 기대되는 금액입니다.",
        "ExpectedROI": "예상 ROI는 개입 비용 1원당 기대 이익이 얼마나 되는지 보여주는 효율 지표입니다. 값이 높을수록 비용 대비 효과가 좋습니다.",
        "InterventionIntensity": "개입 강도는 고객에게 제공할 혜택이나 연락의 세기를 낮음·보통·높음처럼 단순화한 값입니다.",
        "RecommendedAction": "추천 액션은 고객에게 지금 제안하면 좋을 혜택·메시지·관찰 조치입니다.",
        "RecommendationReason": "추천 이유는 이 액션이 선택된 근거입니다. 예를 들어 과거 구매 이력, 최근 탐색 행동, 비슷한 고객군의 인기 항목 등이 포함됩니다.",
        "ActionStatus": "액션 상태는 추천 액션이 아직 대기 중인지, 큐에 적재됐는지, 발송됐는지 같은 처리 상태입니다.",
        "CustomerValueScore": "고객 가치 점수는 고객 생애가치와 개입 반응 가능성을 함께 반영해 우선순위를 정하기 위한 보조 점수입니다.",
        "RecommendationScore": "추천 점수는 특정 상품·혜택·메시지가 해당 고객에게 적합하다고 판단한 정도입니다.",
        "Priority": "우선순위 점수는 이탈 위험, 개입 효과, 고객 가치, 비용을 합쳐 먼저 대응할 고객을 정한 값입니다.",
        "CLV": "CLV는 고객이 앞으로 가져올 것으로 추정되는 생애가치입니다.",
        "Uplift": "Uplift는 개입했을 때 이탈 방지·구매 증가가 얼마나 추가로 발생할지 나타내는 점수입니다.",
    },
    "en": {
        "CustomerType": "Customer type is a plain-language group based on recent behavior, value, and churn signs.",
        "ChurnProbability": "Churn probability shows the likelihood of a customer leaving or reaching the configured churn condition.",
        "ChurnTiming": "Expected churn timing estimates when the customer may approach churn if the current pattern continues.",
        "ExpectedLoss": "Expected loss is the revenue or customer value that may be lost if the customer churns.",
        "ExpectedProfit": "Expected profit is the additional profit expected from intervening with this customer.",
        "ExpectedROI": "Expected ROI shows how much profit is expected per unit of intervention cost.",
        "InterventionIntensity": "Intervention intensity simplifies the strength of the benefit or contact into levels such as low, medium, and high.",
        "RecommendedAction": "Recommended action is the benefit, message, or monitoring action suggested for the customer.",
        "RecommendationReason": "Recommendation reason explains why the action was selected, such as purchase history, recent browsing, or segment popularity.",
        "ActionStatus": "Action status shows whether the recommendation is pending, queued, sent, or completed.",
        "CustomerValueScore": "Customer value score is a helper score combining value and expected response.",
        "RecommendationScore": "Recommendation score estimates how suitable an item, benefit, or message is for the customer.",
        "Priority": "Priority score ranks customers by churn risk, intervention effect, value, and cost.",
        "CLV": "CLV is the estimated lifetime value a customer may generate in the future.",
        "Uplift": "Uplift estimates the incremental retention or purchase effect caused by an intervention.",
    },
    "ja": {
        "CustomerType": "顧客タイプは最近の行動、価値、離脱兆候を分かりやすくまとめた分類です。",
        "ChurnProbability": "離脱確率は顧客が設定した離脱条件に近づく、または離脱する可能性を示します。",
        "ChurnTiming": "予想離脱時点は現在の傾向が続く場合に離脱状態へ近づくと見込まれる時期です。",
        "ExpectedLoss": "予想損失額は顧客が離脱した場合に失う可能性のある売上・顧客価値です。",
        "ExpectedProfit": "予想利益はこの顧客に介入した場合に追加で得られると期待される金額です。",
        "ExpectedROI": "予想ROIは介入費用1単位あたりの期待利益を示す効率指標です。",
        "InterventionIntensity": "介入強度は特典や連絡の強さを低・中・高のように単純化した値です。",
        "RecommendedAction": "推奨アクションは顧客に提案する特典、メッセージ、または観察施策です。",
        "RecommendationReason": "推薦理由は、そのアクションが選ばれた根拠です。",
        "ActionStatus": "アクション状態は推薦が待機中、キュー登録済み、送信済みなどかを示します。",
        "CustomerValueScore": "顧客価値スコアは価値と反応見込みを合わせた補助スコアです。",
        "RecommendationScore": "推薦スコアは項目・特典・メッセージの適合度を示します。",
        "Priority": "優先度スコアは離脱リスク、介入効果、顧客価値、費用を組み合わせた順位付け指標です。",
        "CLV": "CLVは顧客が将来もたらすと推定される生涯価値です。",
        "Uplift": "Upliftは介入によって追加で得られる離脱防止・購買増加効果の推定値です。",
    },
}
for _lang, _mapping in _HUMAN_TERM_CAPTIONS_PATCH.items():
    TERM_CAPTIONS.setdefault(_lang, {}).update(_mapping)
# ============================================================
# [/PATCH]
# ============================================================

# Data-facing labels are fixed to Korean for finance/e-commerce tables and charts.
# Keep this patch close to the final label dictionaries so it wins over generic
# multilingual mappings without touching backend schemas or model inputs.
_DATA_KO_COLUMN_PATCH: dict[str, str] = {
    # Common source/detail table columns
    "timestamp": "이벤트 시각",
    "event_time": "이벤트 시각",
    "event_type": "이벤트 유형",
    "last_event_type": "최근 이벤트 유형",
    "item_category": "상품/서비스 카테고리",
    "category": "카테고리",
    "quantity": "수량",
    "session_id": "세션 ID",
    "event_id": "이벤트 ID",
    "order_id": "주문 ID",
    "order_time": "주문 시각",
    "gross_amount": "주문 금액",
    "discount_amount": "할인 금액",
    "net_amount": "실결제 금액",
    "coupon_used": "쿠폰 사용 여부",
    "campaign_id": "캠페인 ID",
    "campaign_type": "캠페인 유형",
    "exposure_time": "노출 시각",
    "channel": "채널",
    "redeemed": "사용 여부",
    "redeem_time": "사용 시각",
    "cost": "비용",
    "discount_rate": "할인율",
    "assigned_at": "배정 시각",
    "treatment_group": "실험군",
    "treatment_flag": "개입 여부",
    "control_group": "대조군",
    "actual_profit": "실제 이익",
    "actual_roi": "실제 ROI",
    "actual_conversion": "실제 전환 여부",
    "coupon_redeemed": "쿠폰 사용 여부",
    "outcome_label": "결과 분류",
    "executed": "실행 여부",
    "intervention_intensity_label": "개입 강도",
    # Feature names that often appear as values in feature importance tables too.
    "recency_days": "마지막 활동 경과일",
    "frequency": "활동 빈도",
    "monetary": "거래 금액",
    "visits_last_7": "최근 7일 방문 수",
    "visits_prev_7": "직전 7일 방문 수",
    "visit_change_rate": "방문 변화율",
    "purchase_last_30": "최근 30일 구매 수",
    "purchase_prev_30": "직전 30일 구매 수",
    "purchase_change_rate": "구매 변화율",
    "inactivity_days": "비활성 일수",
    "coupon_exposure_count": "쿠폰 노출 횟수",
    "coupon_redeem_count": "쿠폰 사용 횟수",
    "coupon_fatigue_score": "쿠폰 피로도",
    "coupon_affinity": "쿠폰 반응도",
    "discount_dependency_score": "할인 의존도",
    "discount_pressure_score": "할인 압박도",
    "discount_effect_penalty": "할인 효과 페널티",
    "price_sensitivity": "가격 민감도",
    "support_contact_propensity": "고객지원 문의 가능성",
    "avg_coupon_exposure": "평균 쿠폰 노출 횟수",
    "avg_churn_probability": "평균 이탈 확률",
    "avg_expected_roi": "평균 예상 ROI",
    "count": "건수",
    "value": "값",
}
COLUMN_LABELS.setdefault("ko", {}).update(_DATA_KO_COLUMN_PATCH)

_DATA_KO_VALUE_PATCH: dict[str, str] = {
    # E-commerce event/action/product labels shown in tables and chart legends.
    "visit": "방문",
    "page_view": "페이지 방문",
    "screen_view": "화면 조회",
    "product_view": "상품 조회",
    "view_item": "상품 조회",
    "search": "검색",
    "add_to_cart": "장바구니 담기",
    "cart": "장바구니",
    "wishlist_add": "찜하기",
    "favorite": "즐겨찾기",
    "purchase": "구매",
    "order": "주문",
    "checkout": "결제",
    "support_contact": "고객지원 문의",
    "other": "기타",
    "ignore": "제외",
    "retention_coupon": "리텐션 쿠폰",
    "personalized_coupon": "개인 맞춤 쿠폰",
    "coupon": "쿠폰",
    "coupon_used": "쿠폰 사용",
    "coupon_redeemed": "쿠폰 사용",
    "no_coupon": "쿠폰 없음",
    "fashion": "패션",
    "beauty": "뷰티",
    "personal_care": "생활/개인관리",
    "grocery": "식품/생활",
    "sports": "스포츠",
    "health": "헬스케어",
    "electronics": "전자제품",
    "home": "홈/리빙",
    "books": "도서",
    "kids": "키즈",
    "pet": "반려동물",
    "own_purchase_history": "본인 구매 이력",
    "recent_browse_signal": "최근 탐색 신호",
    "segment_popularity": "유사 고객군 인기",
    "global_popularity": "전체 인기",
    # Feature values in model artifact tables.
    "recency_days": "마지막 활동 경과일",
    "frequency": "활동 빈도",
    "monetary": "거래 금액",
    "visits_last_7": "최근 7일 방문 수",
    "visits_prev_7": "직전 7일 방문 수",
    "visit_change_rate": "방문 변화율",
    "purchase_last_30": "최근 30일 구매 수",
    "purchase_prev_30": "직전 30일 구매 수",
    "purchase_change_rate": "구매 변화율",
    "inactivity_days": "비활성 일수",
    "coupon_exposure_count": "쿠폰 노출 횟수",
    "coupon_redeem_count": "쿠폰 사용 횟수",
    "coupon_fatigue_score": "쿠폰 피로도",
    "coupon_affinity": "쿠폰 반응도",
    "discount_dependency_score": "할인 의존도",
    "discount_pressure_score": "할인 압박도",
    "discount_effect_penalty": "할인 효과 페널티",
    "price_sensitivity": "가격 민감도",
    "support_contact_propensity": "고객지원 문의 가능성",
    "financial_retention_offer": "금융 리텐션 혜택",
}
VALUE_LABELS.setdefault("ko", {}).update(_DATA_KO_VALUE_PATCH)

_FINANCE_KO_COLUMN_PATCH: dict[str, str] = {
    "timestamp": "금융 이벤트 시각",
    "event_time": "금융 이벤트 시각",
    "event_type": "금융 이벤트 유형",
    "last_event_type": "최근 금융 이벤트 유형",
    "item_category": "금융상품/서비스",
    "category": "금융상품 분류",
    "quantity": "거래 수량/건수",
    "order_id": "거래 ID",
    "order_time": "거래 시각",
    "gross_amount": "거래 금액",
    "discount_amount": "혜택 금액",
    "net_amount": "순거래 금액",
    "coupon_used": "혜택 사용 여부",
    "campaign_id": "금융 캠페인 ID",
    "campaign_type": "금융 캠페인 유형",
    "exposure_time": "혜택 제안 시각",
    "channel": "접촉 채널",
    "redeemed": "혜택 수락 여부",
    "redeem_time": "혜택 수락 시각",
    "cost": "금융 개입 비용",
    "discount_rate": "금리·수수료 우대율",
    "treatment_group": "개입군",
    "treatment_flag": "금융 개입 여부",
    "actual_conversion": "실제 금융거래 전환 여부",
    "coupon_redeemed": "혜택 수락 여부",
    "outcome_label": "성과 판정",
    "executed": "실행 여부",
    "intervention_intensity_label": "개입 강도",
    "visits_last_7": "최근 7일 금융채널 접속 수",
    "visits_prev_7": "직전 7일 금융채널 접속 수",
    "visit_change_rate": "금융채널 접속 변화율",
    "inactivity_days": "금융 비활성 일수",
    "support_contact_propensity": "상담/민원 가능성",
    "avg_coupon_exposure": "평균 혜택 제안 횟수",
}
FINANCE_COLUMN_LABELS.setdefault("ko", {}).update(_FINANCE_KO_COLUMN_PATCH)

_FINANCE_KO_VALUE_PATCH: dict[str, str] = {
    "visit": "금융채널 접속",
    "page_view": "계좌·상품 조회",
    "screen_view": "금융 화면 조회",
    "product_view": "금융상품 조회",
    "view_item": "금융상품 조회",
    "search": "금융상품 탐색",
    "add_to_cart": "신청 시작/관심상품",
    "cart": "신청/관심상품",
    "wishlist_add": "관심상품 저장",
    "favorite": "관심상품 저장",
    "purchase": "금융거래",
    "order": "거래",
    "checkout": "거래 완료",
    "support_contact": "상담/민원",
    "other": "기타 금융활동",
    "ignore": "제외",
    "retention_coupon": "금융 리텐션 혜택",
    "personalized_coupon": "맞춤 금융 혜택",
    "coupon": "금융 혜택",
    "coupon_used": "혜택 사용",
    "coupon_redeemed": "혜택 수락",
    "no_coupon": "혜택 없음",
    "fashion": "카드/소비",
    "beauty": "예·적금",
    "personal_care": "생활금융",
    "grocery": "입출금계좌",
    "sports": "대출",
    "health": "보험/연금",
    "electronics": "디지털금융",
    "home": "주거금융",
    "books": "금융교육/콘텐츠",
    "kids": "가족금융",
    "pet": "펫보험/특화상품",
    "패션": "카드/소비",
    "뷰티": "예·적금",
    "생활/개인관리": "생활금융",
    "식품/생활": "입출금계좌",
    "스포츠": "대출",
    "헬스케어": "보험/연금",
    "전자제품": "디지털금융",
    "홈/리빙": "주거금융",
    "페이지 방문": "계좌·상품 조회",
    "상품 조회": "금융상품 조회",
    "검색": "금융상품 탐색",
    "고객지원 문의": "상담/민원",
    "own_purchase_history": "고객 본인의 과거 금융거래 이력",
    "recent_browse_signal": "최근 금융상품 조회 신호",
    "segment_popularity": "유사 금융고객군 선호",
    "global_popularity": "전체 금융고객 선호",
    "recency_days": "마지막 금융거래 경과일",
    "frequency": "금융거래 빈도",
    "monetary": "금융 거래/잔고 금액",
    "visits_last_7": "최근 7일 금융채널 접속 수",
    "visits_prev_7": "직전 7일 금융채널 접속 수",
    "visit_change_rate": "금융채널 접속 변화율",
    "inactivity_days": "금융 비활성 일수",
    "support_contact_propensity": "상담/민원 가능성",
    "financial_retention_offer": "금융 리텐션 혜택",
}
FINANCE_VALUE_LABELS.setdefault("ko", {}).update(_FINANCE_KO_VALUE_PATCH)

_FINANCE_RUNTIME_REPLACEMENTS_KO_PATCH: dict[str, str] = {
    "추천 카테고리 믹스": "추천 금융상품 믹스",
    "추천 카테고리 분포": "추천 금융상품 분포",
    "추천 카테고리": "추천 금융상품",
    "이벤트 타입 분포": "금융 이벤트 유형 분포",
    "이벤트 유형 분포": "금융 이벤트 유형 분포",
    "페르소나별 평균 쿠폰 노출": "금융 고객 유형별 평균 혜택 제안",
    "쿠폰 운영 리스크 플래그": "금융 혜택 운영 리스크 플래그",
    "페르소나별 쿠폰 노출/성과": "금융 고객 유형별 혜택 제안/성과",
    "총 쿠폰 지급": "총 금융 혜택 지급",
    "쿠폰 이력": "금융 혜택 이력",
    "쿠폰 사용": "혜택 사용",
    "쿠폰 비용": "금융 혜택 비용",
    "주문 내역": "거래 내역",
    "주문 기록 없음": "거래 기록 없음",
    "총 구매": "총 금융거래",
    "구매 변화율": "금융거래 변화율",
    "최근 30일 구매 수": "최근 30일 금융거래 수",
    "직전 30일 구매 수": "직전 30일 금융거래 수",
    "장바구니": "신청/관심상품",
    "할인": "금리·수수료 혜택",
}
FINANCE_RUNTIME_REPLACEMENTS.setdefault("ko", {}).update(_FINANCE_RUNTIME_REPLACEMENTS_KO_PATCH)


# Plain Korean display labels for customer groups, product names, and action text.
# These are presentation labels only; model-facing columns such as coupon_cost or
# action_id remain unchanged for compatibility with the existing pipeline.
_BUSINESS_CUSTOMER_TYPE_KO_PATCH: dict[str, str] = {
    "High Value-Persuadables": "가치가 높고 연락하면 반응할 가능성이 큰 고객",
    "High Value-Sure Things": "가치가 높고 이미 반응 가능성이 큰 고객",
    "High Value-Lost Causes": "가치는 높지만 지금 개입 효과가 낮은 고객",
    "Low Value-Persuadables": "가치는 낮지만 연락하면 반응할 수 있는 고객",
    "Low Value-Sure Things": "가치는 낮고 이미 반응 가능성이 있는 고객",
    "Low Value-Lost Causes": "가치와 개입 효과가 모두 낮은 고객",
    "New Customers": "가입 초기 고객",
    "Persuadables": "연락하면 반응할 가능성이 큰 고객",
    "Sure Things": "이미 반응 가능성이 큰 고객",
    "Lost Causes": "지금 개입 효과가 낮은 고객",
    "Sleeping Dogs": "불필요한 개입을 피해야 하는 고객",
    "persuadables": "연락하면 반응할 가능성이 큰 고객",
    "sure_things": "이미 반응 가능성이 큰 고객",
    "lost_causes": "지금 개입 효과가 낮은 고객",
    "sleeping_dogs": "불필요한 개입을 피해야 하는 고객",
    "vip_loyal": "VIP 충성 고객",
    "regular_loyal": "일반 충성 고객",
    "loyal_regular": "일반 충성 고객",
    "loyal_vip_customer": "VIP 충성 고객",
    "loyal regular customer": "일반 충성 고객",
    "vip_at_risk": "이탈 위험이 큰 VIP 고객",
    "regular_at_risk": "이탈 위험이 큰 일반 고객",
    "new_customer": "신규 고객",
    "new_signup": "가입 초기 고객",
    "churn_progressing": "이탈 조짐이 보이는 고객",
    "price_sensitive": "가격·혜택에 민감한 고객",
    "coupon_sensitive": "혜택에 민감한 고객",
    "explorer": "상품을 둘러보는 탐색 고객",
    "dormant": "휴면 고객",
    "dormant_risk": "활동이 줄어 이탈 위험이 큰 고객",
    "inactive_risk": "활동이 줄어 이탈 위험이 큰 고객",
    "churn_risk": "이탈 위험이 큰 고객",
    "high_churn_risk": "이탈 위험이 매우 큰 고객",
    "medium_churn_risk": "이탈 위험이 보통인 고객",
    "low_churn_risk": "이탈 위험이 낮은 고객",
    "at_risk": "이탈 위험 고객",
    "credit_revolver": "카드 리볼빙 이용 고객",
    "loan_watch": "대출 관찰 대상 고객",
    "대출_watch": "대출 관찰 대상 고객",
    "unknown_segment": "고객 유형 미분류",
    "live": "실시간 고객",
    "live_user": "실시간 고객",
    "high_uplift": "개입하면 반응할 가능성이 높은 고객",
    "very_high_uplift": "개입 반응 가능성이 매우 높은 고객",
    "medium_uplift": "개입 반응 가능성이 보통인 고객",
    "low_uplift": "개입 반응 가능성이 낮은 고객",
    "negative_uplift": "개입을 권장하지 않는 고객",
}
VALUE_LABELS.setdefault("ko", {}).update(_BUSINESS_CUSTOMER_TYPE_KO_PATCH)
PHRASE_LABELS.setdefault("ko", {}).update(_BUSINESS_CUSTOMER_TYPE_KO_PATCH)

_FINANCE_PRODUCT_ACTION_KO_PATCH: dict[str, str] = {
    # Finance products/services that may come from uploaded CSVs or generated recommendations.
    "deposit": "예금",
    "deposits": "예금",
    "savings": "적금",
    "saving": "적금",
    "savings_account": "적금",
    "checking": "입출금계좌",
    "checking_account": "입출금계좌",
    "account": "입출금계좌",
    "credit_card": "신용카드",
    "debit_card": "체크카드",
    "card": "카드",
    "loan": "대출",
    "loans": "대출",
    "mortgage": "주택담보대출",
    "personal_loan": "신용대출",
    "insurance": "보험",
    "pension": "연금",
    "fund": "펀드",
    "funds": "펀드",
    "investment": "투자상품",
    "wealth": "자산관리",
    "wealth_management": "자산관리",
    "asset_management": "자산관리",
    "remittance": "송금",
    "transfer": "이체",
    "digital_banking": "디지털금융",
    "mobile_banking": "모바일뱅킹",
    "retention_action": "고객 유지 상담",
    "generic_retention_offer": "고객 유지 상담 및 금융 혜택 안내",
    "Generic retention offer": "고객 유지 상담 및 금융 혜택 안내",
    "generic retention offer": "고객 유지 상담 및 금융 혜택 안내",
    "personalized_retention_offer": "고객 맞춤 금융 혜택 안내",
    "Personalized retention offer": "고객 맞춤 금융 혜택 안내",
    "high_value_retention_coupon": "고가치 고객 전담 상담 및 우대조건 제안",
    "coupon_offer": "금리·수수료 우대 혜택 안내",
    "Coupon campaign": "금리·수수료 우대 혜택 안내",
    "coupon": "금리·수수료 우대 혜택",
    "coupon_5000": "금리·수수료 우대 혜택",
    "5,000원 쿠폰": "금리·수수료 우대 혜택",
    "5,000원 혜택": "금리·수수료 우대 혜택",
    "discount_offer": "금리·수수료 우대조건 안내",
    "loyalty_reward": "우수 고객 우대조건 안내",
    "service_recovery": "불편사항 해결 상담",
    "service_recovery_message": "불편사항 해결 상담 안내",
    "retention_message": "금융상품 이용 안내 메시지",
    "light_retention_message": "가벼운 금융상품 이용 안내",
    "priority_human_followup": "담당자 우선 상담",
    "low_risk_upsell_offer": "관심 금융상품 추가 안내",
    "monitor_only": "추가 행동 관찰",
    "monitoring": "관찰",
    "benefit": "금융 혜택",
    "crm": "담당자 상담",
    "message": "안내 메시지",
    "upsell": "추가 금융상품 안내",
    # Existing retail categories reused by the common simulator are displayed as finance products.
    "fashion": "카드/소비",
    "beauty": "예·적금",
    "personal_care": "생활금융",
    "grocery": "입출금계좌",
    "sports": "대출",
    "health": "보험/연금",
    "electronics": "디지털금융",
    "home": "주거금융",
    "books": "금융교육/콘텐츠",
    "kids": "가족금융",
    "pet": "펫보험/특화상품",
    # Common finance/customer-segment codes from real banking datasets.
    "credit_revolver": "카드 리볼빙 이용 고객",
    "card_revolver": "카드 리볼빙 이용 고객",
    "revolver": "리볼빙 이용 고객",
    "loan_watch": "대출 관찰 대상 고객",
    "대출_watch": "대출 관찰 대상 고객",
    "loan_watchlist": "대출 관찰 대상 고객",
    "credit_watch": "신용위험 관찰 대상 고객",
    "credit_risk_watch": "신용위험 관찰 대상 고객",
    "dormant_risk": "활동이 줄어 이탈 위험이 큰 고객",
    "inactive_risk": "활동이 줄어 이탈 위험이 큰 고객",
    "churn_risk": "이탈 위험이 큰 고객",
    "high_churn_risk": "이탈 위험이 매우 큰 고객",
    "medium_churn_risk": "이탈 위험이 보통인 고객",
    "low_churn_risk": "이탈 위험이 낮은 고객",
    "at_risk": "이탈 위험 고객",
    "watch": "관찰 대상",
}
FINANCE_VALUE_LABELS.setdefault("ko", {}).update(_FINANCE_PRODUCT_ACTION_KO_PATCH)

_BUSINESS_ACTION_KO_PATCH: dict[str, str] = {
    "VIP concierge + personalized offer": "VIP 고객 전담 상담 및 맞춤 혜택 안내",
    "Loyalty touchpoint": "충성 고객 감사 안내",
    "Deep-dive outreach": "담당자 심층 상담",
    "Coupon campaign": "맞춤 혜택 안내",
    "No Action": "미개입 관찰",
    "Light reminder": "가벼운 재방문 안내",
    "Onboarding sequence": "가입 초기 이용 안내",
    "Immediate (<=14d)": "14일 이내 즉시 연락",
    "Near-term (15-30d)": "15~30일 안에 연락",
    "Planned (31-60d)": "31~60일 안에 계획적으로 연락",
    "Monitor (>60d)": "60일 이후 관찰",
    "Monitor(>60d)": "60일 이후 관찰",
    "Monitor >60d": "60일 이후 관찰",
    "low": "낮은 수준 개입",
    "mid": "보통 수준 개입",
    "medium": "보통 수준 개입",
    "high": "높은 수준 개입",
    "저강도": "낮은 수준 개입",
    "중강도": "보통 수준 개입",
    "고강도": "높은 수준 개입",
}
VALUE_LABELS.setdefault("ko", {}).update(_BUSINESS_ACTION_KO_PATCH)
PHRASE_LABELS.setdefault("ko", {}).update(_BUSINESS_ACTION_KO_PATCH)
FINANCE_VALUE_LABELS.setdefault("ko", {}).update(_BUSINESS_ACTION_KO_PATCH)

_FINANCE_RUNTIME_ACTION_REPLACEMENTS: dict[str, str] = {
    "5,000원 쿠폰 예상 순이익": "금리·수수료 우대 혜택 예상 순이익",
    "5,000원 혜택 예상 순이익": "금리·수수료 우대 혜택 예상 순이익",
    "쿠폰 예상 순이익": "금융 혜택 예상 순이익",
    "쿠폰에 반응할 가능성이 있는 고객에게 5,000원 혜택을 제공하는 전략": "금리·수수료 우대나 수수료 면제처럼 비용이 정해진 금융 혜택을 제안하는 전략",
    "쿠폰보다 서비스 회복 메시지나 CS 해결": "금융 혜택보다 불편사항 해결 상담",
    "쿠폰, 상담, 알림": "금융 혜택, 상담, 알림",
    "쿠폰": "금융 혜택",
}
FINANCE_RUNTIME_REPLACEMENTS.setdefault("ko", {}).update(_FINANCE_RUNTIME_ACTION_REPLACEMENTS)

_ACTION_NAME_FINANCE_KO: dict[str, str] = {
    **_FINANCE_PRODUCT_ACTION_KO_PATCH,
    "기본 리텐션 혜택": "고객 유지 상담 및 금융 혜택 안내",
    "기본 리텐션 혜택 제안": "고객 유지 상담 및 금융 혜택 안내",
    "기본 고객 유지 혜택 안내": "고객 유지 상담 및 금융 혜택 안내",
    "개인 맞춤 리텐션 혜택 제안": "고객 맞춤 금융 혜택 안내",
    "개인화 리텐션 혜택": "고객 맞춤 금융 혜택 안내",
    "개인 맞춤 쿠폰": "고객 맞춤 금융 혜택 안내",
    "쿠폰 혜택": "금리·수수료 우대 혜택 안내",
    "쿠폰 혜택 제안": "금리·수수료 우대 혜택 안내",
    "할인 혜택": "금리·수수료 우대조건 안내",
    "할인 혜택 제안": "금리·수수료 우대조건 안내",
}

_ACTION_NAME_ECOMMERCE_KO: dict[str, str] = {
    "generic_retention_offer": "기본 재방문 혜택 안내",
    "Generic retention offer": "기본 재방문 혜택 안내",
    "generic retention offer": "기본 재방문 혜택 안내",
    "personalized_retention_offer": "고객 맞춤 혜택 안내",
    "Personalized retention offer": "고객 맞춤 혜택 안내",
    "high_value_retention_coupon": "고가치 고객 맞춤 쿠폰 안내",
    "coupon_offer": "쿠폰 혜택 안내",
    "Coupon campaign": "쿠폰 혜택 안내",
    "discount_offer": "할인 혜택 안내",
    "loyalty_reward": "충성 고객 보상 안내",
    "service_recovery": "서비스 불편 회복 안내",
    "service_recovery_message": "서비스 불편 회복 안내",
    "retention_message": "재방문 안내 메시지",
    "light_retention_message": "가벼운 재방문 안내",
    "priority_human_followup": "담당자 우선 연락",
    "low_risk_upsell_offer": "관심 상품 추가 추천",
    "monitor_only": "추가 행동 관찰",
    "retention_action": "고객 유지 액션",
    "기본 고객 유지 혜택 안내": "기본 재방문 혜택 안내",
    "기본 리텐션 혜택": "기본 재방문 혜택 안내",
    "기본 리텐션 혜택 제안": "기본 재방문 혜택 안내",
}

_WINDOW_LABELS_KO: dict[str, str] = {
    "Immediate (<=14d)": "14일 이내 즉시 연락",
    "Near-term (15-30d)": "15~30일 안에 연락",
    "Planned (31-60d)": "31~60일 안에 계획적으로 연락",
    "Monitor (>60d)": "60일 이후 관찰",
    "Monitor(>60d)": "60일 이후 관찰",
    "Monitor >60d": "60일 이후 관찰",
    "14일 이내 즉시 연락": "14일 이내 즉시 연락",
    "15~30일 안에 연락": "15~30일 안에 연락",
    "31~60일 안에 계획적으로 연락": "31~60일 안에 계획적으로 연락",
    "60일 이후 관찰": "60일 이후 관찰",
}

_INTENSITY_DESCRIPTIONS_FINANCE_KO: dict[str, str] = {
    "low": "낮은 수준: 문자·앱 알림처럼 부담이 작은 안내부터 진행합니다.",
    "mid": "보통 수준: 맞춤 금융 혜택 안내와 상담 연결을 함께 진행합니다.",
    "medium": "보통 수준: 맞춤 금융 혜택 안내와 상담 연결을 함께 진행합니다.",
    "high": "높은 수준: 담당자 상담과 우대조건 제안을 우선 진행합니다.",
    "저강도": "낮은 수준: 문자·앱 알림처럼 부담이 작은 안내부터 진행합니다.",
    "중강도": "보통 수준: 맞춤 금융 혜택 안내와 상담 연결을 함께 진행합니다.",
    "고강도": "높은 수준: 담당자 상담과 우대조건 제안을 우선 진행합니다.",
    "낮은 수준 개입": "낮은 수준: 문자·앱 알림처럼 부담이 작은 안내부터 진행합니다.",
    "보통 수준 개입": "보통 수준: 맞춤 금융 혜택 안내와 상담 연결을 함께 진행합니다.",
    "높은 수준 개입": "높은 수준: 담당자 상담과 우대조건 제안을 우선 진행합니다.",
}

_INTENSITY_DESCRIPTIONS_ECOMMERCE_KO: dict[str, str] = {
    "low": "낮은 수준: 푸시·이메일 같은 가벼운 안내부터 진행합니다.",
    "mid": "보통 수준: 맞춤 혜택 안내와 재방문 유도를 함께 진행합니다.",
    "medium": "보통 수준: 맞춤 혜택 안내와 재방문 유도를 함께 진행합니다.",
    "high": "높은 수준: 고가치 고객에게 더 적극적인 혜택과 연락을 진행합니다.",
    "저강도": "낮은 수준: 푸시·이메일 같은 가벼운 안내부터 진행합니다.",
    "중강도": "보통 수준: 맞춤 혜택 안내와 재방문 유도를 함께 진행합니다.",
    "고강도": "높은 수준: 고가치 고객에게 더 적극적인 혜택과 연락을 진행합니다.",
    "낮은 수준 개입": "낮은 수준: 푸시·이메일 같은 가벼운 안내부터 진행합니다.",
    "보통 수준 개입": "보통 수준: 맞춤 혜택 안내와 재방문 유도를 함께 진행합니다.",
    "높은 수준 개입": "높은 수준: 고가치 고객에게 더 적극적인 혜택과 연락을 진행합니다.",
}


def _lookup_plain_korean_label(raw: Any, mapping: dict[str, str]) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text in mapping:
        return mapping[text]
    norm = re.sub(r"[\s_\-:：/\.()\[\]{}]+", "", text).lower()
    for src, dst in mapping.items():
        if norm == re.sub(r"[\s_\-:：/\.()\[\]{}]+", "", str(src)).lower():
            return str(dst)
    return None


_FINANCE_COMPOUND_EXACT_KO: dict[str, str] = {
    "credit_revolver": "카드 리볼빙 이용 고객",
    "card_revolver": "카드 리볼빙 이용 고객",
    "revolver": "리볼빙 이용 고객",
    "loan_watch": "대출 관찰 대상 고객",
    "대출_watch": "대출 관찰 대상 고객",
    "loan_watchlist": "대출 관찰 대상 고객",
    "credit_watch": "신용위험 관찰 대상 고객",
    "credit_risk_watch": "신용위험 관찰 대상 고객",
    "dormant_risk": "활동이 줄어 이탈 위험이 큰 고객",
    "inactive_risk": "활동이 줄어 이탈 위험이 큰 고객",
    "churn_risk": "이탈 위험이 큰 고객",
    "high_churn_risk": "이탈 위험이 매우 큰 고객",
    "medium_churn_risk": "이탈 위험이 보통인 고객",
    "low_churn_risk": "이탈 위험이 낮은 고객",
    "at_risk": "이탈 위험 고객",
}

_FINANCE_COMPOUND_TOKEN_KO: dict[str, str] = {
    "credit": "신용",
    "card": "카드",
    "revolver": "리볼빙 이용",
    "revolving": "리볼빙 이용",
    "loan": "대출",
    "loans": "대출",
    "mortgage": "주택담보대출",
    "debt": "대출",
    "watch": "관찰 대상",
    "watchlist": "관찰 대상",
    "risk": "위험",
    "risky": "위험",
    "dormant": "활동 감소",
    "inactive": "활동 감소",
    "churn": "이탈",
    "high": "높은",
    "medium": "보통",
    "mid": "보통",
    "low": "낮은",
    "vip": "VIP",
    "new": "신규",
    "signup": "가입 초기",
    "segment": "고객군",
    "customer": "고객",
    "customers": "고객",
}


def _norm_lookup_key(value: Any) -> str:
    return re.sub(r"[\s_\-:：/\.()\[\]{}]+", "", str(value or "")).lower()


def _humanize_finance_compound_label(value: Any, column: Any = "") -> str | None:
    """Render mixed finance codes such as credit_revolver or 대출_watch in plain Korean."""
    raw = str(value or "").strip()
    if not raw:
        return None

    exact_label = _lookup_plain_korean_label(raw, _FINANCE_COMPOUND_EXACT_KO)
    if exact_label:
        return exact_label

    # Try exact finance/product/customer dictionaries before composing tokens.
    for mapping in (_BUSINESS_CUSTOMER_TYPE_KO_PATCH, _FINANCE_PRODUCT_ACTION_KO_PATCH):
        exact_label = _lookup_plain_korean_label(raw, mapping)
        if exact_label:
            return exact_label

    # Only compose values that look like backend codes or mixed Korean/English codes.
    if not ("_" in raw or "-" in raw or re.search(r"[A-Za-z]", raw)):
        return None

    parts = [part for part in re.split(r"[\s_\-/]+", raw) if part]
    if not parts:
        return None

    translated: list[str] = []
    for part in parts:
        label = _lookup_plain_korean_label(part, _FINANCE_PRODUCT_ACTION_KO_PATCH)
        if not label:
            label = _lookup_plain_korean_label(part, _BUSINESS_CUSTOMER_TYPE_KO_PATCH)
        if not label:
            label = _FINANCE_COMPOUND_TOKEN_KO.get(part.lower())
        if not label and re.search(r"[가-힣]", part):
            label = part
        if not label:
            return None
        if not translated or translated[-1] != label:
            translated.append(label)

    phrase = " ".join(translated).strip()
    if not phrase or phrase == raw:
        return None

    col_norm = _norm_lookup_key(column)
    code_norm = _norm_lookup_key(raw)
    customer_like = (
        any(token in col_norm for token in ["persona", "segment", "customertype", "고객유형", "customersegment"])
        or any(token in code_norm for token in ["risk", "watch", "revolver", "dormant", "inactive", "churn"])
    )
    if customer_like and not phrase.endswith(("고객", "고객군", "대상")):
        phrase = f"{phrase} 고객"
    return phrase


def _column_display_label_for_wizard(column_name: Any, mode: str | None = None) -> str:
    """Korean label for upload-wizard column names without mutating the mapping key."""
    raw = str(column_name or "").strip()
    if not raw:
        return "미지정"
    finance = (mode or _business_mode()) == "finance"
    base_map = {
        "customer_id": "고객 ID", "client_no": "고객 번호", "client_id": "고객 ID", "user_id": "고객 ID", "member_id": "고객 ID",
        "account_no": "계좌 번호", "account_id": "계좌 ID",
        "timestamp": "발생 시각", "event_time": "이벤트 시각", "transaction_date": "거래일", "transaction_time": "거래 시각", "txn_time": "거래 시각",
        "event_type": "이벤트 유형", "banking_event": "금융 이벤트 유형", "transaction_type": "거래 유형", "txn_type": "거래 유형",
        "amount": "금액", "transaction_amount": "거래금액", "txn_amount": "거래금액", "gross_amount": "거래금액", "net_amount": "순거래금액",
        "category": "상품 유형", "product_type": "상품 유형", "item_category": "상품/서비스 유형", "financial_product": "금융상품",
        "churn_flag": "이탈 여부", "is_churned": "이탈 여부", "churn_label": "이탈 라벨", "account_status": "계좌/거래 상태",
        "persona": "고객 유형", "customer_segment": "고객 유형", "segment": "고객군", "risk_segment": "위험 등급",
        "quantity": "수량", "product_count": "상품 수", "region": "지역", "branch": "지점", "channel": "채널",
        "credit_score": "신용점수", "delinquency_days": "연체일수", "loan_balance": "대출잔액", "account_balance": "계좌잔고",
    }
    finance_map = {
        "category": "금융상품 유형", "product_type": "금융상품 유형", "item_category": "금융상품/서비스", "financial_product": "금융상품/서비스",
        "event_type": "금융 이벤트 유형", "amount": "금융거래 금액", "quantity": "보유/거래 상품 수", "product_count": "보유 금융상품 수",
        "timestamp": "거래/이벤트 시각", "churn_flag": "금융 이탈 여부", "persona": "금융 고객 유형", "customer_segment": "금융 고객 유형",
    }
    mapping = {**base_map, **(finance_map if finance else {})}
    label = _lookup_plain_korean_label(raw, mapping)
    if label:
        return label
    domain = _domain_column_label(raw, "ko") if finance else None
    if domain:
        return domain
    generic = COLUMN_LABELS.get("ko", {}).get(raw)
    if generic:
        return generic
    compound = _humanize_finance_compound_label(raw, "column") if finance else None
    if compound:
        return compound
    # Last resort: make backend-looking names readable in Korean-ish spacing without
    # exposing underscores. This is still better than raw snake_case in the UI.
    return raw.replace("_", " ").replace("-", " ").strip()


def _schema_role_label_for_wizard(role: Any, mode: str | None = None) -> str:
    raw = str(role or "").strip()
    finance = (mode or _business_mode()) == "finance"
    finance_roles = {
        "customer_id": "금융 고객 식별값",
        "timestamp": "거래/이벤트 시각",
        "event_type": "금융 이벤트 유형",
        "amount": "거래금액·잔고 금액",
        "category": "금융상품 유형",
        "churn_flag": "금융 이탈 여부",
        "persona": "금융 고객 유형",
        "quantity": "보유/거래 상품 수",
        "region": "지역",
    }
    ecommerce_roles = {
        "customer_id": "고객 식별값",
        "timestamp": "이벤트 시각",
        "event_type": "이커머스 이벤트 유형",
        "amount": "주문·결제 금액",
        "category": "상품 카테고리",
        "churn_flag": "이탈 여부",
        "persona": "고객 유형",
        "quantity": "상품 수량",
        "region": "지역",
    }
    return (finance_roles if finance else ecommerce_roles).get(raw, _column_display_label_for_wizard(raw, mode))


def _unique_display_lookup(values: list[Any], formatter) -> tuple[dict[str, str], dict[str, str]]:
    raw_to_display: dict[str, str] = {}
    display_to_raw: dict[str, str] = {}
    counts: dict[str, int] = {}
    for value in values:
        raw = str(value)
        base = str(formatter(raw)).strip() or raw
        count = counts.get(base, 0) + 1
        counts[base] = count
        label = base if count == 1 else f"{base} {count}"
        raw_to_display[raw] = label
        display_to_raw[label] = raw
    return raw_to_display, display_to_raw


_INTERNAL_EVENT_LABELS_KO_FINANCE: dict[str, str] = {
    "visit": "접속/방문",
    "page_view": "계좌·상품 조회",
    "search": "금융상품 탐색",
    "add_to_cart": "신청 시작/관심상품",
    "purchase": "금융거래",
    "support_contact": "상담/민원",
    "other": "기타",
    "ignore": "분석 제외",
}

_INTERNAL_EVENT_LABELS_KO_ECOMMERCE: dict[str, str] = {
    "visit": "방문",
    "page_view": "상품 조회",
    "search": "검색",
    "add_to_cart": "장바구니 담기",
    "purchase": "구매",
    "support_contact": "고객지원 문의",
    "other": "기타",
    "ignore": "분석 제외",
}


def _event_display_label_for_wizard(value: Any, mode: str | None = None, *, internal: bool = False) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "미지정"
    finance = (mode or _business_mode()) == "finance"
    if internal:
        return (_INTERNAL_EVENT_LABELS_KO_FINANCE if finance else _INTERNAL_EVENT_LABELS_KO_ECOMMERCE).get(raw, raw.replace("_", " "))
    if finance:
        label = _humanize_finance_compound_label(raw, "event_type")
        if label:
            return label
    translated = _domain_translate_value("event_type", raw) if finance else raw
    translated = _translate_cell_value(translated)
    return str(translated).replace("_", " ").strip()


def _humanize_business_action_text(value: Any) -> str:
    """Turn generated action codes into a sentence that a business user can execute."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    # Only action-like strings should be rewritten as sentences.
    has_action_signal = any(token in raw for token in ["·", "|", "Near-term", "Immediate", "Planned", "Monitor", "retention", "coupon", "Coupon", "혜택", "개입"])
    if not has_action_signal:
        return raw

    mode = "finance" if _is_finance_display_mode() else "ecommerce"
    action_map = _ACTION_NAME_FINANCE_KO if mode == "finance" else _ACTION_NAME_ECOMMERCE_KO
    intensity_map = _INTENSITY_DESCRIPTIONS_FINANCE_KO if mode == "finance" else _INTENSITY_DESCRIPTIONS_ECOMMERCE_KO

    normalized = raw.replace(" | ", " · ").replace(";", " · ")
    parts = [part.strip() for part in re.split(r"\s*·\s*", normalized) if part.strip()]
    if not parts:
        return raw

    action_label = None
    intensity_desc = None
    window_label = None
    other_parts: list[str] = []

    for part in parts:
        mapped_window = _lookup_plain_korean_label(part, _WINDOW_LABELS_KO)
        mapped_intensity = _lookup_plain_korean_label(part, intensity_map)
        mapped_action = _lookup_plain_korean_label(part, action_map)
        if mapped_window:
            window_label = mapped_window
        elif mapped_intensity:
            intensity_desc = mapped_intensity
        elif mapped_action:
            action_label = mapped_action
        else:
            # The part may already be partially translated. Try finance/e-commerce value labels too.
            mapped_value = _lookup_plain_korean_label(part, FINANCE_VALUE_LABELS.get("ko", {})) if mode == "finance" else _lookup_plain_korean_label(part, VALUE_LABELS.get("ko", {}))
            if mapped_value and mapped_value != part:
                if any(keyword in mapped_value for keyword in ["일", "관찰", "연락"]):
                    window_label = mapped_value
                elif "수준" in mapped_value or mapped_value in {"높음", "보통", "낮음"}:
                    intensity_desc = intensity_map.get(mapped_value, mapped_value)
                else:
                    action_label = mapped_value
            else:
                other_parts.append(part)

    if not action_label:
        if len(parts) == 1:
            mapped_single = _lookup_plain_korean_label(parts[0], action_map)
            return mapped_single or raw
        return raw

    if action_label in {"관찰", "추가 행동 관찰", "미개입 관찰"}:
        base_sentence = "추가 비용을 바로 쓰지 않고 고객 행동을 더 관찰합니다."
    else:
        base_sentence = f"{action_label}를 진행합니다."
    if window_label:
        base_sentence = f"{window_label} {base_sentence}"
    if intensity_desc:
        base_sentence = f"{base_sentence} {intensity_desc}"
    if other_parts and len(parts) <= 3:
        # Preserve a short unknown qualifier without exposing code-like separators.
        base_sentence = f"{base_sentence} 참고: {' / '.join(other_parts)}."
    return re.sub(r"\s+", " ", base_sentence).strip()


def _humanize_business_display_value(column: Any, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return ""
    column_norm = re.sub(r"[\s_\-:：/\.()\[\]{}]+", "", str(column or "")).lower()

    if "persona" in column_norm or "segment" in column_norm or "customer_type" in column_norm or "고객유형" in column_norm:
        label = _lookup_plain_korean_label(raw, _BUSINESS_CUSTOMER_TYPE_KO_PATCH)
        if label:
            return label
        if _is_finance_display_mode():
            compound_label = _humanize_finance_compound_label(raw, column)
            if compound_label:
                return compound_label

    if "recommendedaction" in column_norm or "queuedrecommendedaction" in column_norm or "action" in column_norm or "추천액션" in column_norm:
        sentence = _humanize_business_action_text(raw)
        if sentence != raw:
            return sentence

    if _is_finance_display_mode() and ("category" in column_norm or "product" in column_norm or "상품" in column_norm or "recommended" in column_norm):
        label = _lookup_plain_korean_label(raw, _FINANCE_PRODUCT_ACTION_KO_PATCH)
        if label:
            return label
        compound_label = _humanize_finance_compound_label(raw, column)
        if compound_label:
            return compound_label

    if _is_finance_display_mode():
        compound_label = _humanize_finance_compound_label(raw, column)
        if compound_label:
            return compound_label

    # Values that look like generated actions should be sentence-like even when
    # they arrive through generic log/detail columns.
    sentence = _humanize_business_action_text(raw)
    if sentence != raw:
        return sentence
    return raw

LEGACY_VIEW_REDIRECTS: dict[str, str] = {
    "2. 예산 배분·타겟 고객": "4. 예산 최적화 및 리텐션 타겟",
    "3. 개인화 추천": "5. 개인화 추천",
    "4. 실시간 운영 모니터": "6. 실시간 운영 모니터",
    "6. 의사결정 엔진 비교": "4. 예산 최적화 및 리텐션 타겟",
    "3. Uplift + CLV 상위 고객": "4. 예산 최적화 및 리텐션 타겟",
    "4. 예산 배분 결과": "4. 예산 최적화 및 리텐션 타겟",
    "5. 예상 최적화 ROI": "4. 예산 최적화 및 리텐션 타겟",
    "6. 리텐션 대상 고객 목록": "4. 예산 최적화 및 리텐션 타겟",
    "7. 학습 결과 아티팩트": "5. 개인화 추천",
    "8. Uplift/최적화 결과": "4. 예산 최적화 및 리텐션 타겟",
    "8. Uplift/최적화 결과 (실시간)": "4. 예산 최적화 및 리텐션 타겟",
    "9. 개인화 추천": "5. 개인화 추천",
    "10. 실시간 운영 모니터": "6. 실시간 운영 모니터",
    "10. 실시간 위험 스코어링 / 운영 모니터": "6. 실시간 운영 모니터",
    "11. 이탈 시점 예측 (Survival Analysis)": "9. 이탈 시점 예측",
    "12. 의사결정 엔진 비교": "4. 예산 최적화 및 리텐션 타겟",
    "13. 운영 한눈에 보기": "6. 실시간 운영 모니터",
    "14. 증분 성과 / A-B 실험": "4. 예산 최적화 및 리텐션 타겟",
    "15. 설명가능성 / 고객별 개입 이유": "4. 예산 최적화 및 리텐션 타겟",
    "17. 할인·쿠폰 운영 리스크": "4. 예산 최적화 및 리텐션 타겟",
    "7. 실시간 운영 모니터": "6. 실시간 운영 모니터",
    "8. 할인·쿠폰 운영 리스크": "4. 예산 최적화 및 리텐션 타겟",
    "9. 학습 결과 아티팩트": "5. 개인화 추천",
    "10. 이탈 시점 예측 (Survival Analysis)": "9. 이탈 시점 예측",
    "11. 증분 성과 / A-B 실험": "4. 예산 최적화 및 리텐션 타겟",
    "12. 설명가능성 / 고객별 개입 이유": "4. 예산 최적화 및 리텐션 타겟",
    "6. 개인화 추천": "5. 개인화 추천",
    "8. 이탈 시점 예측 (Survival Analysis)": "9. 이탈 시점 예측",
    "9. 의사결정 엔진 비교": "4. 예산 최적화 및 리텐션 타겟",
    "10. 증분 성과 / A-B 실험": "4. 예산 최적화 및 리텐션 타겟",
    "11. 설명가능성 / 고객별 개입 이유": "4. 예산 최적화 및 리텐션 타겟",
    "13. 할인·쿠폰 운영 리스크": "4. 예산 최적화 및 리텐션 타겟",
    "9. 이탈 시점 예측 (Survival Analysis)": "9. 이탈 시점 예측",
}
REALTIME_REFRESH_VIEWS: set[str] = {"6. 실시간 운영 모니터"}
INSIGHT_HEAVY_VIEWS: set[str] = {"4. 예산 최적화 및 리텐션 타겟", "6. 실시간 운영 모니터", "14. 주간 액션 성과 리뷰"}


def _language_code() -> str:
    return st.session_state.get("language_code", "ko") if hasattr(st, "session_state") else "ko"


def _data_label_language_code() -> str:
    """Language for data-facing labels: table headers, event values, product names, and chart axes.

    The dashboard may translate surrounding UI text, but business data labels must
    remain Korean in finance/e-commerce modes so demos and exported screenshots do
    not mix English backend schema names with Korean business terminology.
    """
    try:
        if _business_mode() in {"ecommerce", "finance"}:
            return "ko"
    except Exception:
        pass
    return _language_code()


def _normalize_i18n_key(text: str) -> str:
    """번역 키 비교용 정규화: 공백/언더스코어/대소문자 차이를 흡수한다."""
    return re.sub(r"[\s_\-:：/\.()\[\]{}]+", "", str(text or "")).lower()


def T(text: str) -> str:
    code = _language_code()
    raw = str(text)

    # 1) 정확히 등록된 UI 문구 우선
    direct = UI_TEXT.get(code, {}).get(raw)
    if direct is not None:
        return direct

    # 2) "LLM결과요약" vs "LLM 결과 요약", "고객 id" vs "고객 ID" 같은 표기 차이 보정
    normalized = _normalize_i18n_key(raw)
    for ko_key, translated in UI_TEXT.get(code, {}).items():
        if _normalize_i18n_key(ko_key) == normalized:
            return translated

    # 3) 컬럼 라벨도 일반 텍스트로 들어오는 경우가 있어 역매핑한다.
    column_labels = COLUMN_LABELS.get(code, COLUMN_LABELS.get("ko", {}))
    for canonical, translated in column_labels.items():
        if _normalize_i18n_key(canonical) == normalized:
            return translated
        for labels_by_lang in COLUMN_LABELS.values():
            localized = labels_by_lang.get(canonical)
            if localized and _normalize_i18n_key(localized) == normalized:
                return translated

    friendly = friendly_translate_text(raw, code)
    if friendly != raw:
        return _domain_translate_text(friendly)

    return _domain_translate_text(raw)


def _replace_runtime_token(text: str, src: str, dst: str) -> str:
    if not src:
        return text
    if re.search(r"[A-Za-z]", src):
        return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])", dst, text)
    return text.replace(src, dst)


def _translate_runtime_text(text: Any) -> str:
    """Translate runtime/service/UI messages, including dynamic f-string fragments."""
    raw = str(text or "")
    if not raw:
        return ""

    translated = T(raw)
    if translated != raw:
        return translated

    code = _language_code()
    out = raw

    # Replace the longest known Korean UI fragments first so dynamic f-strings such as
    # "현재 공통 조건: ..." are translated without needing an exact full-string key.
    for mapping in (UI_TEXT.get(code, {}), PHRASE_LABELS.get(code, {}), VALUE_LABELS.get(code, {})):
        for src, dst in sorted(mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
            src = str(src)
            out = _replace_runtime_token(out, src, str(dst))

    api_key_msg = "OpenAI API 키가 설정되지 않았습니다. 사이드바에 키를 입력하거나 OPENAI_API_KEY 환경변수를 설정하세요."
    out = out.replace(api_key_msg, T(api_key_msg))
    out = friendly_translate_value(out, code)
    out = friendly_translate_text(out, code)
    out = _domain_translate_text(out)
    return out


def _translation_destination_set(mapping: dict[str, str]) -> set[str]:
    return {str(v).strip() for v in mapping.values() if str(v).strip()}



def _collapse_repeated_customer_words(value: Any) -> str:
    """Collapse accidental repeated UI suffixes such as '고객 고객 고객'.

    The display translation layer may receive values that were already localized
    by a previous rerun/cache. This helper keeps the text readable without
    changing the original data used by the pipeline.
    """
    text = str(value or "")
    if not text:
        return ""

    # Common Korean/Japanese/English repeated display tokens caused by broad
    # value translation. Keep the loop bounded and conservative.
    for _ in range(4):
        before = text
        text = re.sub(r"(고객)(?:\s+\1)+", r"\1", text)
        text = re.sub(r"(고객군)(?:\s+\1)+", r"\1", text)
        text = re.sub(r"(ユーザー)(?:\s+\1)+", r"\1", text)
        text = re.sub(r"(customer)(?:\s+\1)+", r"\1", text, flags=re.IGNORECASE)
        if text == before:
            break
    return text.strip()


def _translate_cell_value(value: Any) -> Any:
    """Translate a scalar cell value safely and idempotently for display only."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " · ".join(str(_translate_cell_value(v)) for v in value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""

    humanized = _humanize_business_display_value("__value__", stripped)
    if isinstance(humanized, str) and humanized != stripped:
        return _collapse_repeated_customer_words(humanized)

    # In finance mode, convert raw product/event/feature values to finance terms
    # before generic e-commerce labels can translate values such as ``purchase``
    # or ``fashion`` into retail wording.
    if _is_finance_display_mode():
        finance_value = _domain_translate_value("__value__", stripped)
        if isinstance(finance_value, str) and finance_value != stripped:
            return _collapse_repeated_customer_words(finance_value)

    return _translate_cell_value_cached(_data_label_language_code(), stripped)


@lru_cache(maxsize=20000)
def _translate_cell_value_cached(language_code: str, stripped: str) -> str:
    """Cached, idempotent cell-value translation.

    The previous implementation performed broad substring replacement for every
    object cell. In Korean mode, a value that was already translated, such as
    "충성 VIP 고객", could be translated again because the generic key "vip"
    was replaced with "VIP 고객". Repeated reruns could therefore produce
    "충성 VIP 고객 고객 고객". This function first detects already-translated
    destination labels and skips risky short substring keys.
    """
    if stripped == "":
        return ""
    code = language_code or _data_label_language_code()
    value_labels = VALUE_LABELS.get(code, VALUE_LABELS.get("ko", {}))
    phrase_labels = PHRASE_LABELS.get(code, PHRASE_LABELS.get("ko", {}))
    norm = _normalize_i18n_key(stripped)

    # Already localized values must be returned as-is. This makes display
    # translation idempotent even if a dataframe was pre-translated elsewhere.
    for mapping in (value_labels, phrase_labels):
        for dst in _translation_destination_set(mapping):
            if stripped == dst or norm == _normalize_i18n_key(dst):
                return _collapse_repeated_customer_words(stripped)

    for src, dst in value_labels.items():
        src_text = str(src)
        if stripped == src_text or norm == _normalize_i18n_key(src_text):
            return _collapse_repeated_customer_words(str(dst))
    for src, dst in phrase_labels.items():
        src_text = str(src)
        if stripped == src_text or norm == _normalize_i18n_key(src_text):
            return _collapse_repeated_customer_words(str(dst))

    out = stripped.replace(" | ", " · ").replace(";", " · ")
    replacement_items = list(phrase_labels.items()) + list(value_labels.items())

    for src, dst in sorted(replacement_items, key=lambda item: len(str(item[0])), reverse=True):
        src_text = str(src).strip()
        dst_text = str(dst).strip()
        if not src_text or not dst_text:
            continue

        src_norm = _normalize_i18n_key(src_text)
        # Exact matches are handled above. For substring replacement, do not use
        # very short/generic tokens such as "vip", "high", "low". These caused
        # already-friendly labels to grow suffixes like "고객 고객".
        if len(src_norm) <= 4:
            continue
        if dst_text in out:
            continue

        variants = {
            src_text,
            src_text.replace("_", " "),
            src_text.replace("_", "-"),
            src_text.replace("_", " ").title(),
            src_text.replace("_", " ").capitalize(),
        }
        for variant in sorted(variants, key=len, reverse=True):
            if not variant or variant == dst_text:
                continue
            flags = re.IGNORECASE if re.fullmatch(r"[A-Za-z0-9_\-\s()<>/+.]+", variant) else 0
            try:
                out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(variant)}(?![A-Za-z0-9_])", dst_text, out, flags=flags)
            except re.error:
                out = out.replace(variant, dst_text)

    if "," in out:
        parts = [part.strip() for part in out.split(",")]
        translated_parts = []
        for part in parts:
            part_norm = _normalize_i18n_key(part)
            translated = next((str(dst) for src, dst in value_labels.items() if part_norm == _normalize_i18n_key(str(src))), part)
            translated_parts.append(translated)
        out = ", ".join(translated_parts)

    out = out.replace("-> action queued", "→ " + value_labels.get("queued", "queued"))
    out = out.replace("score=", "risk=")
    return out




def _translate_dataframe_values_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = _map_object_series_unique(
                out[col],
                lambda v: _translate_cell_value(v) if not _is_missing_live_value(v) else v,
            )
    return out




def _translate_ui_arg(value: Any) -> Any:
    """Translate labels/messages passed to UI widgets while preserving non-text data."""
    if isinstance(value, str):
        return _translate_runtime_text(value)
    if isinstance(value, list):
        return [_translate_ui_arg(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_translate_ui_arg(v) for v in value)
    if isinstance(value, dict):
        return {k: (_translate_ui_arg(v) if isinstance(v, str) else v) for k, v in value.items()}
    return value




def _strip_duplicate_suffix(column: str) -> str:
    """Remove Streamlit/Pandas duplicate suffixes such as _2 or ' 2'."""
    return re.sub(r"(?:[\s_]+\d+)$", "", str(column or "")).strip()


def _is_money_column(column: str) -> bool:
    norm = _normalize_i18n_key(_strip_duplicate_suffix(column))
    if "roi" in norm or "rate" in norm or "probability" in norm or "score" in norm:
        return False
    money_tokens = ["clv", "customerlifetimevalue", "고객생애가치", "顧客生涯価値", "expectedprofit", "expectedincrementalprofit", "expectedloss", "expectedloss30d", "queuedexpectedprofit", "couponcost", "queuedcouponcost", "allocatedbudget", "budget", "spend", "amount", "revenue", "profit", "loss", "cost", "monetary", "predictedclv12m", "예상이익", "예상증분이익", "예상손실액", "배정예산", "집행예산", "잔여예산", "쿠폰비용", "개입비용", "予想利益", "予想損失", "配分予算", "費用"]
    return any(token.lower() in norm for token in money_tokens)


def _is_probability_column(column: str) -> bool:
    norm = _normalize_i18n_key(_strip_duplicate_suffix(column))
    if "roi" in norm:
        return False
    probability_tokens = ["probability", "prob", "rate", "share", "survivalprob", "churnwithin30dprobability", "이탈확률", "가능성", "비율", "확률", "리텐션율", "생존확률", "離脱確率", "可能性", "比率", "率", "生存確率"]
    return any(token.lower() in norm for token in probability_tokens)


def _is_roi_column(column: str) -> bool:
    return "roi" in _normalize_i18n_key(_strip_duplicate_suffix(column))


def _coerce_float_for_display(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if raw == "" or raw in {"-", "—"}:
            return None
        if raw.endswith("%"):
            try:
                return float(raw[:-1].replace(",", "")) / 100.0
            except ValueError:
                return None
        raw = re.sub(r"[₩원円$€£,\s]", "", raw).replace("배", "")
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _format_money_display(value: Any) -> str:
    numeric = _coerce_float_for_display(value)
    if numeric is None:
        return _translate_cell_value(value)
    return money(float(round(numeric)))


def _format_probability_display(value: Any) -> str:
    if isinstance(value, str) and value.strip().endswith("%"):
        return value.strip()
    numeric = _coerce_float_for_display(value)
    if numeric is None:
        return _translate_cell_value(value)
    percent_value = numeric * 100.0 if abs(numeric) <= 1.0 else numeric
    return f"{percent_value:.1f}%" if abs(percent_value) >= 10 else f"{percent_value:.2f}%"


def _format_roi_display(value: Any) -> str:
    if isinstance(value, str) and (value.strip().endswith("%") or value.strip().endswith("배")):
        return value.strip()
    numeric = _coerce_float_for_display(value)
    if numeric is None:
        return _translate_cell_value(value)
    code = _language_code()
    if code == "en":
        return f"{numeric:.1f}x"
    if code == "ja":
        return f"約{numeric:.1f}倍"
    return f"약 {numeric:.1f}배"


def _format_table_value_by_column(column: str, value: Any) -> Any:
    if _is_missing_live_value(value):
        return ""
    if _is_roi_column(column):
        return _format_roi_display(value)
    if _is_money_column(column):
        return _format_money_display(value)
    if _is_probability_column(column):
        return _format_probability_display(value)
    if isinstance(value, str):
        humanized = _humanize_business_display_value(column, value)
        if isinstance(humanized, str) and humanized != value:
            return _collapse_repeated_customer_words(humanized)
        domain_first = _domain_translate_value(column, value)
        return _translate_cell_value(domain_first)
    return value


def _dedupe_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop backend duplicate display columns before translated headers become '... 2'."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    keep: list[Any] = []
    seen: set[str] = set()
    for col in df.columns:
        raw = str(col)
        base = _strip_duplicate_suffix(raw)
        raw_norm = _normalize_i18n_key(raw)
        base_norm = _normalize_i18n_key(base)
        if raw_norm in {"expectedroi2", "expectedroi02", "예상roi2"}:
            continue
        translated_base = _translate_column_name(base)
        label_norm = _normalize_i18n_key(translated_base)
        if base_norm != raw_norm and label_norm in seen:
            continue
        if label_norm in seen:
            continue
        seen.add(label_norm)
        keep.append(col)
    return df.loc[:, keep].copy()


_CHART_LABEL_PATCH: dict[str, dict[str, str]] = {
    "ko": {
        "retention rate": "리텐션율", "retention": "리텐션", "period": "경과 기간(개월)", "cohort_month": "가입 코호트", "cohort": "코호트", "count": "건수", "value": "값", "customer_count": "고객 수", "candidate_customer_count": "후보 고객 수", "recommend_count": "추천 건수", "uplift_segment": "고객 반응 유형", "intervention_intensity": "개입 강도", "allocated_budget": "배정 예산", "expected_profit": "예상 이익", "expected_roi": "예상 ROI", "churn_probability": "이탈 확률", "clv": "고객 생애가치(CLV)", "uplift_score": "개입 효과 점수", "value_score": "고객 가치 점수", "event_type": "이벤트 유형", "avg_coupon_exposure": "평균 쿠폰 노출 횟수", "recommended_category": "추천 카테고리", "importance": "중요도", "feature_display": "변수명", "persona": "고객 유형", "avg_churn_probability": "평균 이탈 확률", "avg_expected_roi": "평균 예상 ROI",
    },
    "en": {}, "ja": {},
}


def _friendly_chart_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    code = _data_label_language_code()
    translated_col = _translate_column_name(raw)
    if translated_col and translated_col != raw.replace("_", " "):
        return translated_col
    mapping = _CHART_LABEL_PATCH.get(code, _CHART_LABEL_PATCH.get("ko", {}))
    norm = raw.lower().strip()
    if norm in mapping:
        return mapping[norm]
    norm_key = _normalize_i18n_key(raw)
    for src, dst in mapping.items():
        if _normalize_i18n_key(src) == norm_key:
            return dst
    out = _translate_runtime_text(raw)
    for src, dst in sorted(mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(str(src))}(?![A-Za-z0-9_])", str(dst), out, flags=re.IGNORECASE)
    return _translate_cell_value(out)


def _translate_plotly_values(values: Any) -> Any:
    try:
        if values is None:
            return values
        if isinstance(values, np.ndarray):
            if values.dtype.kind in {"U", "S", "O"}:
                return np.array([_translate_cell_value(v) if isinstance(v, str) else v for v in values], dtype=object)
            return values
        if isinstance(values, (list, tuple, pd.Series, pd.Index)):
            translated = [_translate_cell_value(v) if isinstance(v, str) else v for v in list(values)]
            return tuple(translated) if isinstance(values, tuple) else translated
    except Exception:
        return values
    return values


def _localize_plotly_figure(fig: Any) -> Any:
    """Translate Plotly axis titles, legend labels, trace names and categorical ticks."""
    try:
        layout = getattr(fig, "layout", None)
        if layout is not None:
            if getattr(layout, "title", None) is not None and getattr(layout.title, "text", None):
                layout.title.text = _friendly_chart_text(layout.title.text)
            if getattr(layout, "legend", None) is not None and getattr(layout.legend, "title", None) is not None and getattr(layout.legend.title, "text", None):
                layout.legend.title.text = _friendly_chart_text(layout.legend.title.text)
            for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
                raw_axis_title = getattr(axis.title, "text", None) if getattr(axis, "title", None) is not None else None
                if raw_axis_title:
                    if _is_money_column(str(raw_axis_title)):
                        try:
                            axis.tickprefix = "₩"
                            axis.separatethousands = True
                            axis.tickformat = ",.0f"
                        except Exception:
                            pass
                    axis.title.text = _friendly_chart_text(raw_axis_title)
                if getattr(axis, "ticktext", None) is not None:
                    axis.ticktext = _translate_plotly_values(axis.ticktext)
            if getattr(layout, "coloraxis", None) is not None:
                colorbar = getattr(layout.coloraxis, "colorbar", None)
                if colorbar is not None and getattr(colorbar, "title", None) is not None and getattr(colorbar.title, "text", None):
                    colorbar.title.text = _friendly_chart_text(colorbar.title.text)
    except Exception:
        pass
    try:
        for trace in getattr(fig, "data", []) or []:
            if getattr(trace, "name", None):
                trace.name = _translate_cell_value(trace.name)
            for attr in ("x", "y", "labels", "text", "hovertext"):
                if hasattr(trace, attr):
                    try:
                        setattr(trace, attr, _translate_plotly_values(getattr(trace, attr)))
                    except Exception:
                        pass
            if getattr(trace, "hovertemplate", None):
                trace.hovertemplate = _friendly_chart_text(trace.hovertemplate)
    except Exception:
        pass
    return fig

def _install_i18n_runtime_patches() -> None:
    """Translate remaining unwrapped Streamlit and Plotly labels at render time.

    This is intentionally limited to labels/help/title/caption-like fields so dataset
    columns, widget keys, and user-uploaded values are not mutated.
    """
    if getattr(st, "_retention_i18n_runtime_patched", False):
        return

    def _wrap_callable(obj: Any, name: str, arg_indexes: tuple[int, ...] = (0,), kw_names: tuple[str, ...] = ("label", "help", "placeholder", "caption", "text")) -> None:
        original = getattr(obj, name, None)
        if original is None or getattr(original, "_retention_i18n_wrapped", False):
            return

        def wrapped(*args: Any, **kwargs: Any):
            args_list = list(args)
            for idx in arg_indexes:
                if idx < len(args_list):
                    args_list[idx] = _translate_ui_arg(args_list[idx])
            for kw in kw_names:
                if kw in kwargs:
                    kwargs[kw] = _translate_ui_arg(kwargs[kw])
            return original(*args_list, **kwargs)

        wrapped._retention_i18n_wrapped = True  # type: ignore[attr-defined]
        setattr(obj, name, wrapped)

    # Streamlit text/widgets.
    for _name in [
        "markdown", "caption", "info", "warning", "error", "success", "write",
        "header", "subheader", "title", "toast", "spinner", "expander", "chat_input",
        "button", "checkbox", "toggle", "radio", "selectbox", "slider", "number_input",
        "text_input", "file_uploader", "metric", "image",
    ]:
        _wrap_callable(st, _name)

    # st.dataframe does not return edited values, so it is safe to localize a copy
    # of displayed data. Do not wrap st.data_editor because edited values are used
    # by the mapping workflow.
    _dataframe_original = getattr(st, "dataframe", None)
    if _dataframe_original is not None and not getattr(_dataframe_original, "_retention_i18n_wrapped", False):
        def _dataframe_wrapped(data: Any = None, *args: Any, **kwargs: Any):
            display_data = data
            try:
                if isinstance(display_data, pd.DataFrame):
                    display_data = _sanitize_display_dataframe(display_data)
                elif "data" in kwargs and isinstance(kwargs["data"], pd.DataFrame):
                    kwargs = dict(kwargs)
                    kwargs["data"] = _sanitize_display_dataframe(kwargs["data"])
                    display_data = data
            except Exception:
                display_data = data
            return _dataframe_original(display_data, *args, **kwargs)
        _dataframe_wrapped._retention_i18n_wrapped = True  # type: ignore[attr-defined]
        st.dataframe = _dataframe_wrapped  # type: ignore[assignment]

    # st.progress has a numeric first arg; translate only its text kwarg.
    _wrap_callable(st, "progress", arg_indexes=(), kw_names=("text",))

    # st.tabs receives a list of tab labels as the first argument.
    _wrap_callable(st, "tabs", arg_indexes=(0,), kw_names=())

    # Streamlit column_config labels/help.
    if hasattr(st, "column_config"):
        for _name in [
            "TextColumn", "NumberColumn", "SelectboxColumn", "CheckboxColumn", "DateColumn",
            "DatetimeColumn", "TimeColumn", "LinkColumn", "ListColumn", "ProgressColumn",
            "LineChartColumn", "BarChartColumn", "AreaChartColumn", "ImageColumn",
        ]:
            _wrap_callable(st.column_config, _name)

    # Plotly Express chart titles and human-readable label values.
    def _wrap_px(name: str) -> None:
        original = getattr(px, name, None)
        if original is None or getattr(original, "_retention_i18n_wrapped", False):
            return

        def wrapped(*args: Any, **kwargs: Any):
            if "title" in kwargs:
                kwargs["title"] = _translate_ui_arg(kwargs["title"])
            if "labels" in kwargs and isinstance(kwargs["labels"], dict):
                kwargs["labels"] = {k: _translate_ui_arg(v) for k, v in kwargs["labels"].items()}
            return original(*args, **kwargs)

        wrapped._retention_i18n_wrapped = True  # type: ignore[attr-defined]
        setattr(px, name, wrapped)

    for _name in ["bar", "line", "pie", "scatter", "histogram", "imshow", "area", "box", "violin"]:
        _wrap_px(_name)


    # Plotly figures often inherit raw dataframe column names as axis titles or legend values.
    # Localize them at the final render boundary so every chart uses dashboard language.
    _plotly_original = getattr(st, "plotly_chart", None)
    if _plotly_original is not None and not getattr(_plotly_original, "_retention_i18n_wrapped", False):
        def _plotly_wrapped(fig: Any, *args: Any, **kwargs: Any):
            return _plotly_original(_localize_plotly_figure(fig), *args, **kwargs)
        _plotly_wrapped._retention_i18n_wrapped = True  # type: ignore[attr-defined]
        st.plotly_chart = _plotly_wrapped  # type: ignore[assignment]

    st._retention_i18n_runtime_patched = True  # type: ignore[attr-defined]


_install_i18n_runtime_patches()

def _label_matches(label: str, *needles: str) -> bool:
    norm_label = _normalize_i18n_key(label)
    return any(_normalize_i18n_key(n) in norm_label for n in needles)


def _pick_existing_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    by_norm = {_normalize_i18n_key(c): c for c in df.columns}
    selected: list[str] = []
    for col in preferred:
        actual = by_norm.get(_normalize_i18n_key(col))
        if actual is not None and actual not in selected:
            selected.append(actual)
    return df[selected].copy() if selected else df.copy()


def _filter_display_columns_for_label(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Hide backend/debug columns and keep each core table focused for non-expert users."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    label = str(label or "")
    df = drop_duplicate_metric_columns(df)
    if _label_matches(label, "고객별 선택 이유", "customer level reasons", "reason caution", "顧客別選定理由", "顧客別選定理由注意事項"):
        return _pick_existing_columns(df, ["customer_id", "persona", "selection_reason", "reason_summary", "watchout", "caution", "next_best_action", "recommended_action"])
    if _label_matches(label, "최종 리텐션 타겟", "final retention target", "最終リテンション対象"):
        return _pick_existing_columns(df, ["customer_id", "persona", "uplift_segment", "churn_probability", "clv", "intervention_intensity", "recommended_action", "coupon_cost", "expected_incremental_profit", "expected_roi"])
    if _label_matches(label, "개인화 추천", "personalized recommendation", "パーソナライズ推薦"):
        return _pick_existing_columns(df, ["customer_id", "persona", "recommended_category", "recommendation_rank", "recommendation_score", "reason_tags"])
    if _label_matches(label, "이탈 위험 고객 목록", "at risk customer", "離脱リスク顧客"):
        return _pick_existing_columns(df, ["customer_id", "persona", "churn_probability", "clv"])
    if _label_matches(label, "세그먼트별 예산 배분 테이블", "segment budget allocation table", "セグメント別予算配分表"):
        return _pick_existing_columns(df, ["uplift_segment", "customer_count", "allocated_budget", "expected_profit", "intervention_intensity"])
    if _label_matches(label, "세그먼트별 예산 배분 후보", "candidate customers by segment", "候補顧客数"):
        return _pick_existing_columns(df, ["uplift_segment", "candidate_customer_count"])
    if _label_matches(label, "실시간 이탈 위험", "real time churn risk", "リアルタイム離脱リスク"):
        return _pick_existing_columns(df, ["customer_id", "persona", "realtime_churn_score", "churn_score", "churn_probability", "action_queue_status", "queued_recommended_action", "queued_expected_profit", "latest_trigger_reason"])
    if _label_matches(label, "실시간 액션 큐", "live action queue", "action queue", "アクションキュー"):
        return _pick_existing_columns(df, ["customer_id", "persona", "recommended_action", "queued_recommended_action", "intervention_intensity", "queued_intervention_intensity", "expected_profit", "queued_expected_profit", "expected_roi", "queued_expected_roi", "action_status", "latest_trigger_reason"])

    hidden_norms = {
        _normalize_i18n_key(c) for c in [
            "score_payload", "feature_payload", "source_payload", "raw_payload", "payload",
            "persona_source", "uplift_segment_source", "source_type", "queued_at", "updated_at", "created_at", "scored_at",
            "reoptimization_count", "customer_count_label", "index", "row_id", "internal_id", "model_version",
        ]
    }
    keep = []
    for col in df.columns:
        n = _normalize_i18n_key(col)
        if n in hidden_norms or "payload" in n:
            continue
        keep.append(col)
    return df[keep].copy() if keep else df.copy()


def _render_view_intro(view_key: str) -> None:
    key = str(view_key).split(".")[0]
    lines = VIEW_INTRO_LINES.get(key)
    if not lines:
        return
    labels = [T("이 화면을 보는 이유"), T("확인할 정보"), T("활용 목적")]
    body = "<br/>".join(
        f"<b>{html.escape(labels[i])}</b>: {html.escape(T(line))}"
        for i, line in enumerate(lines)
    )
    st.markdown(
        f"""
        <div style="background:#EEF6FF;border:1px solid #BFDBFE;border-radius:14px;padding:16px 18px;margin:10px 0 18px 0;line-height:1.65;color:#0F172A;">
            <div style="font-weight:800;margin-bottom:4px;">💡 {html.escape(T('뷰 안내'))}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def _is_churn_timing_view(current_view: str) -> bool:
    """True for the customer-level churn timing dashboard view."""
    normalized = str(current_view or "")
    return normalized.startswith("9.") and "이탈 시점" in normalized


def _format_churn_period(days: Any) -> str:
    days_num = pd.to_numeric(pd.Series([days]), errors="coerce").iloc[0]
    if pd.isna(days_num) or not np.isfinite(float(days_num)):
        return T("알 수 없음")
    days_int = max(1, int(math.ceil(float(days_num))))
    code = _language_code()
    if code == "en":
        return f"Within about {days_int} days"
    if code == "ja":
        return f"約{days_int}日以内"
    return f"약 {days_int}일 이내"


def _format_expected_churn_date(base_date: Any, days: Any) -> str:
    days_num = pd.to_numeric(pd.Series([days]), errors="coerce").iloc[0]
    base = pd.to_datetime(base_date, errors="coerce")
    if pd.isna(base) or pd.isna(days_num) or not np.isfinite(float(days_num)):
        return "-"
    return (base + pd.to_timedelta(int(math.ceil(float(days_num))), unit="D")).strftime("%Y-%m-%d")


def _merge_customer_value_columns(predictions: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """Attach CLV/spend/persona columns without exposing internal modeling fields."""
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return pd.DataFrame() if predictions is None else predictions.copy()
    out = predictions.copy()
    if not isinstance(customers_df, pd.DataFrame) or customers_df.empty or "customer_id" not in out.columns or "customer_id" not in customers_df.columns:
        return out

    candidate_cols = [
        "customer_id", "persona", "clv", "predicted_clv_12m", "monetary", "expected_incremental_profit"
    ]
    lookup_cols = [col for col in candidate_cols if col in customers_df.columns]
    lookup = customers_df[lookup_cols].copy()
    if "customer_id" not in lookup.columns:
        return out

    out["_merge_customer_id"] = out["customer_id"].astype(str)
    lookup["_merge_customer_id"] = lookup["customer_id"].astype(str)
    lookup = lookup.drop(columns=["customer_id"]).drop_duplicates("_merge_customer_id")
    out = out.merge(lookup, on="_merge_customer_id", how="left", suffixes=("", "_from_customer"))

    for col in ["persona", "clv", "predicted_clv_12m", "monetary", "expected_incremental_profit"]:
        src = f"{col}_from_customer"
        if src not in out.columns:
            continue
        if col not in out.columns:
            out[col] = out[src]
        else:
            out[col] = out[col].where(out[col].notna(), out[src])
        out = out.drop(columns=[src])
    return out.drop(columns=["_merge_customer_id"], errors="ignore")


def _count_churn_timing_candidates(
    predictions: pd.DataFrame,
    *,
    min_churn_probability: float = 0.0,
) -> int:
    """Count eligible churn-timing rows without formatting/rendering the full table."""
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return 0
    days_col = next(
        (col for col in ["predicted_median_time_to_churn_days", "expected_time_to_churn_days", "median_time_to_churn_days", "duration_days"] if col in predictions.columns),
        None,
    )
    if days_col is None:
        return 0
    mask = pd.to_numeric(predictions[days_col], errors="coerce").notna()
    try:
        probability_threshold = float(min_churn_probability)
    except (TypeError, ValueError):
        probability_threshold = 0.0
    probability_threshold = max(0.0, min(1.0, probability_threshold))
    if probability_threshold > 0:
        if "survival_prob_30d" in predictions.columns:
            survival_30 = pd.to_numeric(predictions["survival_prob_30d"], errors="coerce").clip(lower=0, upper=1)
            churn_30 = (1.0 - survival_30).clip(lower=0, upper=1)
        elif "churn_probability" in predictions.columns:
            churn_30 = pd.to_numeric(predictions["churn_probability"], errors="coerce").clip(lower=0, upper=1)
        else:
            churn_30 = pd.Series(np.nan, index=predictions.index)
        mask = mask & churn_30.notna() & (churn_30 >= probability_threshold)
    return int(mask.sum())


def _build_churn_timing_table(
    predictions: pd.DataFrame,
    customers_df: pd.DataFrame,
    metrics: dict[str, Any] | None,
    *,
    min_churn_probability: float = 0.0,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return a fast Korean table: customer, likely churn timing, and expected loss.

    속도 개선 포인트:
    - 전체 survival 결과를 고객 테이블과 통째로 merge하지 않는다.
    - 화면에 필요한 컬럼만 복사하고, 고객 속성은 customer_id 기준 map으로 붙인다.
    - 정렬/포맷팅은 표시 제한 후보에 대해서만 수행한다.
    """
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return pd.DataFrame()
    if "customer_id" not in predictions.columns:
        return pd.DataFrame()

    days_col = next(
        (col for col in ["predicted_median_time_to_churn_days", "expected_time_to_churn_days", "median_time_to_churn_days", "duration_days"] if col in predictions.columns),
        None,
    )
    if days_col is None:
        return pd.DataFrame()

    # 필요한 최소 컬럼만 사용한다. 큰 업로드 데이터에서 불필요한 merge/복사를 피하기 위함이다.
    base_cols = ["customer_id", days_col]
    for col in ["survival_prob_30d", "churn_probability", "predicted_hazard_ratio", "persona", "clv", "predicted_clv_12m", "monetary", "expected_incremental_profit"]:
        if col in predictions.columns and col not in base_cols:
            base_cols.append(col)
    out = predictions[base_cols].copy()
    out["_customer_id_key"] = out["customer_id"].astype(str)
    out["_expected_days"] = pd.to_numeric(out[days_col], errors="coerce")
    out = out[out["_expected_days"].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    if "survival_prob_30d" in out.columns:
        survival_30 = pd.to_numeric(out["survival_prob_30d"], errors="coerce").clip(lower=0, upper=1)
        out["_churn_30d"] = (1.0 - survival_30).clip(lower=0, upper=1)
    elif "churn_probability" in out.columns:
        out["_churn_30d"] = pd.to_numeric(out["churn_probability"], errors="coerce").clip(lower=0, upper=1)
    else:
        out["_churn_30d"] = np.nan

    try:
        probability_threshold = float(min_churn_probability)
    except (TypeError, ValueError):
        probability_threshold = 0.0
    probability_threshold = max(0.0, min(1.0, probability_threshold))
    if probability_threshold > 0:
        out = out[out["_churn_30d"].notna() & (out["_churn_30d"] >= probability_threshold)].copy()
        if out.empty:
            return pd.DataFrame()

    # 고객 테이블에서 필요한 표시 속성만 dictionary map으로 보강한다.
    customer_lookup = None
    if isinstance(customers_df, pd.DataFrame) and not customers_df.empty and "customer_id" in customers_df.columns:
        lookup_cols = [
            col for col in ["customer_id", "persona", "clv", "predicted_clv_12m", "monetary", "expected_incremental_profit"]
            if col in customers_df.columns
        ]
        if len(lookup_cols) > 1:
            customer_lookup = customers_df[lookup_cols].copy()
            customer_lookup["_customer_id_key"] = customer_lookup["customer_id"].astype(str)
            customer_lookup = customer_lookup.drop_duplicates("_customer_id_key", keep="first")

    if customer_lookup is not None:
        customer_lookup_indexed = customer_lookup.set_index("_customer_id_key")
        for col in ["persona", "clv", "predicted_clv_12m", "monetary", "expected_incremental_profit"]:
            if col not in customer_lookup_indexed.columns:
                continue
            mapped = out["_customer_id_key"].map(customer_lookup_indexed[col])
            if col in out.columns:
                out[col] = out[col].where(out[col].notna(), mapped)
            else:
                out[col] = mapped

    value_col = next((col for col in ["clv", "predicted_clv_12m", "monetary", "expected_incremental_profit"] if col in out.columns), None)
    if value_col is not None:
        out["_customer_value"] = pd.to_numeric(out[value_col], errors="coerce")
    else:
        out["_customer_value"] = np.nan

    out["_expected_loss"] = (out["_customer_value"].clip(lower=0) * out["_churn_30d"].fillna(1.0)).replace([np.inf, -np.inf], np.nan)
    if "predicted_hazard_ratio" in out.columns:
        out["_hazard_sort"] = pd.to_numeric(out["predicted_hazard_ratio"], errors="coerce")
    else:
        out["_hazard_sort"] = np.nan

    # 표시 후보만 남긴 뒤 문자열 포맷팅을 수행한다.
    out = out.sort_values(
        ["_expected_days", "_expected_loss", "_churn_30d", "_hazard_sort"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    if limit is not None:
        out = out.head(max(int(limit), 1)).copy()

    landmark_date = (metrics or {}).get("prediction_as_of_date") or (metrics or {}).get("landmark_as_of_date") or (metrics or {}).get("as_of_date")
    persona_series = out["persona"] if "persona" in out.columns else pd.Series(["-"] * len(out), index=out.index)

    display = pd.DataFrame({
        "customer_id": out["customer_id"].astype(str),
        "persona": persona_series.fillna("-").astype(str),
        "expected_churn_period": out["_expected_days"].map(_format_churn_period),
        "expected_churn_date": out["_expected_days"].map(lambda value: _format_expected_churn_date(landmark_date, value)),
        "churn_within_30d_probability": out["_churn_30d"].map(lambda value: pct(float(value)) if pd.notna(value) else "-"),
        "expected_loss_30d": out["_expected_loss"].map(lambda value: money(float(value)) if pd.notna(value) else "-"),
    })
    return display.reset_index(drop=True)

def _llm_language_name() -> str:
    return llm_language_name(_language_code())


def _llm_strict_language_instruction() -> str:
    return llm_language_instruction(_language_code())


def _wrap_llm_payload(payload_json: str) -> str:
    language = _llm_language_name()
    instruction = _llm_strict_language_instruction()
    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        payload = {"raw_payload": payload_json}
    return json.dumps(
        {
            "answer_language": language,
            "output_language_instruction": instruction,
            "important": instruction,
            "dashboard_payload": payload,
        },
        ensure_ascii=False,
    )


def _wrap_llm_question(question: str) -> str:
    return f"{_llm_strict_language_instruction()}\n\nUser question:\n{question}"


def _business_mode() -> str:
    mode = st.session_state.get("data_mode", "ecommerce") if hasattr(st, "session_state") else "ecommerce"
    return mode if mode in DOMAIN_DIRS else "ecommerce"


def _is_finance_display_mode() -> bool:
    try:
        return _business_mode() == "finance"
    except Exception:
        return False


def _domain_column_label(column: Any, code: str | None = None) -> str | None:
    if not _is_finance_display_mode():
        return None
    # Finance-facing table/axis labels are intentionally Korean even when the UI
    # language is English/Japanese. This prevents raw e-commerce terms from
    # appearing in finance mode screenshots and tables.
    lang = "ko" if (code is None or _data_label_language_code() == "ko") else code
    labels = FINANCE_COLUMN_LABELS.get(lang) or FINANCE_COLUMN_LABELS.get("ko", {})
    raw = str(column)
    if raw in labels:
        return labels[raw]
    raw_norm = _normalize_i18n_key(raw)
    for src, dst in labels.items():
        if _normalize_i18n_key(src) == raw_norm:
            return dst
    return None


def _domain_translate_value(column: Any, value: Any) -> Any:
    if not _is_finance_display_mode() or not isinstance(value, str):
        return value
    code = _data_label_language_code()
    mapping = FINANCE_VALUE_LABELS.get(code) or FINANCE_VALUE_LABELS.get("ko", {})
    out = value
    norm = _normalize_i18n_key(out)
    for src, dst in mapping.items():
        if norm == _normalize_i18n_key(src):
            return str(dst)
    compound_label = _humanize_finance_compound_label(out, column)
    if compound_label:
        return compound_label
    # Apply conservative phrase replacements to strings that are already localized.
    for src, dst in sorted(mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
        src_text = str(src)
        if len(_normalize_i18n_key(src_text)) <= 2:
            continue
        if src_text in out and str(dst) not in out:
            out = out.replace(src_text, str(dst))
    return out


def _domain_translate_text(text: Any) -> str:
    raw = str(text or "")
    if not raw or not _is_finance_display_mode():
        return raw
    code = _language_code()
    replacements = FINANCE_RUNTIME_REPLACEMENTS.get(code) or FINANCE_RUNTIME_REPLACEMENTS.get("ko", {})
    out = raw
    for src, dst in sorted(replacements.items(), key=lambda item: len(str(item[0])), reverse=True):
        out = out.replace(str(src), str(dst))
    # Also normalize a few common table/title tokens that come through runtime text.
    out = _domain_translate_value("__text__", out) if isinstance(out, str) else out
    return str(out)


def _domain_label(mode: str | None = None) -> str:
    mode = mode or _business_mode()
    code = _language_code()
    labels = DOMAIN_MODE_OPTIONS.get(mode, {})
    return labels.get(code) or labels.get("ko") or str(mode)


def _domain_paths(mode: str | None = None) -> dict[str, str]:
    return DOMAIN_DIRS.get(mode or _business_mode(), DOMAIN_DIRS["ecommerce"])


def _mode_metadata_path(mode: str | None = None) -> Path:
    return _project_root() / _domain_paths(mode).get("results", "results_ecommerce") / "dataset_metadata.json"


def _save_dataset_metadata(mode: str, filename: str, upload_path: str = "", row_count: int | None = None) -> None:
    meta = {
        "mode": mode,
        "domain_label_ko": DOMAIN_MODE_OPTIONS.get(mode, {}).get("ko", mode),
        "filename": filename,
        "upload_path": upload_path,
        "row_count": row_count,
        "saved_at": pd.Timestamp.now().isoformat(),
    }
    path = _mode_metadata_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_dataset_metadata(mode: str | None = None) -> dict[str, Any]:
    path = _mode_metadata_path(mode)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

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

    _mode = _st.session_state.get("data_mode", "ecommerce")
    if _mode not in BUSINESS_UPLOAD_MODES:
        return False

    _paths = _domain_paths(_mode)
    _has_user_data = (_P(_paths["data"]) / "customer_summary.csv").exists()
    _has_user_results = _P(_paths["results"]).exists() and any(_P(_paths["results"]).iterdir())
    if _has_user_data or _has_user_results:
        _st.info(T("현재 화면은 업로드된 CSV 산출물을 기준으로 표시합니다. 원본 CSV에 Treatment/Control이 없으면 전처리 단계의 자동 배정 및 쉬운 추정값이 사용됩니다."))
        return False

    _default_reason = (
        f"아직 {_domain_label(_mode)}에서 생성된 산출물이 없습니다. 첫 화면에서 CSV를 업로드하고 "
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
    return st.session_state.get("data_mode", "ecommerce") in BUSINESS_UPLOAD_MODES



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
        "unknown_persona",
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


@st.cache_data(show_spinner=False, ttl=3)
def _fetch_user_live_scores_cached(cache_key: str, limit: int, risk_threshold: float) -> tuple[dict, pd.DataFrame]:
    """Live scores는 summary는 전체 기준, records는 화면 후보 수만 조회한다.

    5만~10만 rows를 매 rerun마다 통째로 가져오면 Streamlit view switching이 느려진다.
    API/Redis cache는 전체 summary를 캐시하고, 화면은 위험도 정렬 상위 후보만 받는다.
    """
    return fetch_user_live_scores(limit=int(limit), risk_threshold=float(risk_threshold))


@st.cache_data(show_spinner=False, ttl=2)
def _fetch_user_live_health_cached(cache_key: str) -> dict:
    """Language/view reruns should not hit the health endpoint repeatedly."""
    return fetch_user_live_health()


@st.cache_data(show_spinner=False, ttl=10)
def _fetch_user_live_seed_status_cached(cache_key: str) -> dict:
    """Seed status changes only after training/seeding, so a short cache is safe."""
    return fetch_user_live_seed_status()


@st.cache_data(show_spinner=False, ttl=3)
def _fetch_user_live_actions_cached(cache_key: str, limit: int, status: str = "queued") -> tuple[dict, pd.DataFrame]:
    return fetch_user_live_actions(limit=limit, status=status)


@st.cache_data(show_spinner=False, ttl=3)
def _fetch_user_live_recommendations_cached(cache_key: str, limit: int) -> tuple[dict, pd.DataFrame]:
    return fetch_user_live_recommendations(limit=limit)

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

    # action_queue can contain expected_profit/expected_roi while coupon_cost is missing or 0.
    # If coupon_cost stays 0, the budget-target filter drops otherwise valid rows.
    if "coupon_cost" not in fixed.columns:
        fixed["coupon_cost"] = 0.0

    _profit_for_cost = pd.to_numeric(
        fixed.get(
            "expected_incremental_profit",
            fixed.get("expected_profit", pd.Series(0.0, index=fixed.index)),
        ),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    _roi_for_cost = pd.to_numeric(
        fixed.get("expected_roi", pd.Series(0.0, index=fixed.index)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    _cost_current = pd.to_numeric(
        fixed.get("coupon_cost", pd.Series(0.0, index=fixed.index)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    _derived_cost = (_profit_for_cost / _roi_for_cost.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _cost_missing = (_cost_current <= 0) & (_profit_for_cost > 0) & (_roi_for_cost > 0)
    fixed.loc[_cost_missing, "coupon_cost"] = _derived_cost.loc[_cost_missing]
    fixed["coupon_cost"] = pd.to_numeric(fixed["coupon_cost"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        # score table is authoritative for numeric score fields.
        # action_queue rows may have been normalized earlier with churn_probability=0.0.
        # If we treat that temporary 0.0 as a real value, every candidate fails
        # the dashboard threshold filter and final targets become 0.
        if col in {"churn_probability", "churn_score", "clv", "uplift_score"}:
            score_values = pd.to_numeric(merged[score_col], errors="coerce")
            if col not in merged.columns:
                merged[col] = score_values
            else:
                current_values = pd.to_numeric(merged[col], errors="coerce")
                merged[col] = score_values.where(score_values.notna(), current_values).fillna(0.0)
        elif col not in merged.columns:
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
    """Recompute live budget targets from the current score table.

    action_queue is an event-time operational queue and can be stale with
    respect to sidebar budget/threshold/cap.  This function treats the live
    score table as the current customer universe and reruns the same budget
    optimizer used by the offline dashboard, so changing the budget changes
    spent/target count immediately.
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
        summary["source"] = "postgresql_user_live_score_reoptimized_current_controls"
        summary["control_budget_sensitive"] = True
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
    covered = int(live_df["customer_id"].nunique()) if "customer_id" in live_df.columns else 0
    summary.update({
        "rows": int(len(live_df)),
        "customers_covered": covered,
        "per_customer": int(per_customer),
        "actual_per_customer": round(float(len(live_df) / covered), 3) if covered else 0.0,
        "candidate_limit": int(max_customers or 0),
        "budget_context": dict(optimize_summary or {}),
        "source": "postgresql_user_live_saved_recommendation_fallback",
        "is_fallback": True,
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
        budget_context = dict(optimize_summary or {})
        budget_context.update({
            "budget": int(budget),
            "threshold": float(threshold),
            "max_customers_cap": int(max_customers),
            "num_targeted": int(budget_context.get("num_targeted", 0) or 0),
        })
        return {
            "rows": 0,
            "customers_covered": 0,
            "per_customer": int(per_customer),
            "actual_per_customer": 0.0,
            "candidate_limit": int(max_customers),
            "eligible_target_customers": 0,
            "budget_context": budget_context,
            "source": "current_budget_threshold_targets",
            "warning": (
                "현재 예산/임계값 조건에서 최종 타겟 고객이 없어 새 개인화 추천을 생성하지 않았습니다. "
                "저장된 과거 후보와 현재 조건 결과가 섞이지 않도록 추천 테이블을 비워 둡니다."
            ),
        }, pd.DataFrame()

    _paths = _domain_paths(_business_mode())
    data_dir = _project_root() / _paths["data"]
    result_dir = _project_root() / _paths["results"]
    required_files = [data_dir / "customer_summary.csv", data_dir / "orders.csv", data_dir / "events.csv"]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        budget_context = dict(optimize_summary or {})
        budget_context.update({
            "budget": int(budget),
            "threshold": float(threshold),
            "max_customers_cap": int(max_customers),
            "num_targeted": int(len(selected_customers)),
        })
        summary = {
            "rows": 0,
            "customers_covered": 0,
            "per_customer": int(per_customer),
            "actual_per_customer": 0.0,
            "candidate_limit": int(max_customers),
            "eligible_target_customers": int(len(selected_customers)),
            "budget_context": budget_context,
            "source": "current_budget_threshold_targets",
            "error": "user raw data 파일이 없어 새 추천을 생성하지 못했습니다: " + ", ".join(missing_files),
            "warning": "필수 user raw data가 없어 저장된 과거 추천 후보를 현재 추천처럼 표시하지 않습니다.",
        }
        return summary, pd.DataFrame()

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
        if hasattr(artifacts, "summary") and isinstance(getattr(artifacts, "summary"), dict):
            summary = dict(artifacts.summary)
        elif hasattr(artifacts, "summary_path") and Path(artifacts.summary_path).exists():
            summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        else:
            summary = {}
    except Exception as exc:
        budget_context = dict(optimize_summary or {})
        budget_context.update({
            "budget": int(budget),
            "threshold": float(threshold),
            "max_customers_cap": int(max_customers),
            "num_targeted": int(len(selected_customers)),
        })
        return {
            "rows": 0,
            "customers_covered": 0,
            "per_customer": int(per_customer),
            "actual_per_customer": 0.0,
            "candidate_limit": int(max_customers),
            "eligible_target_customers": int(len(selected_customers)),
            "budget_context": budget_context,
            "source": "current_budget_threshold_targets",
            "error": f"새 추천 재생성에 실패했습니다({exc}).",
            "warning": "새 추천 재생성에 실패하여 저장된 과거 추천 후보를 현재 추천처럼 표시하지 않습니다.",
        }, pd.DataFrame()

    budget_context = dict(optimize_summary or {})
    budget_context.update({
        "budget": int(budget),
        "threshold": float(threshold),
        "max_customers_cap": int(max_customers),
    })
    covered = int(rec_df["customer_id"].nunique()) if not rec_df.empty and "customer_id" in rec_df.columns else 0
    summary.update({
        "rows": int(len(rec_df)),
        "customers_covered": covered,
        "per_customer": int(per_customer),
        "actual_per_customer": round(float(len(rec_df) / covered), 3) if covered else 0.0,
        "candidate_limit": int(max_customers),
        "eligible_target_customers": int(len(selected_customers)),
        "budget_context": budget_context,
        "source": summary.get("target_source", "current_budget_threshold_targets"),
    })
    try:
        (result_dir / "personalized_recommendations_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return summary, rec_df


def _load_user_live_tables(*, top_n: int, target_cap: int, threshold: float = 0.50, view: str = "") -> dict[str, Any]:
    """user mode 전용 live API 조회 묶음. 실패 시 빈 DataFrame fallback.

    성능 최적화:
    - health/seed는 짧은 TTL 캐시를 사용한다.
    - 전체 scores는 latest_event_time/seed 상태가 같으면 cache를 재사용한다.
    - actions/recommendations는 필요한 화면에서만 조회한다.
    - target_cap*10 같은 과도한 limit을 줄여 화면 전환 지연을 줄인다.
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

    safe_limit = min(max(int(top_n) * 8, int(target_cap) * 4, 2000), 20000)
    now_bucket_5s = str(int(pd.Timestamp.now().timestamp() // 5))
    now_bucket_10s = str(int(pd.Timestamp.now().timestamp() // 10))

    try:
        payload["health"] = _fetch_user_live_health_cached(now_bucket_5s)
    except Exception as exc:
        payload["health"] = {"status": "error", "error": str(exc)}
    try:
        payload["seed_status"] = _fetch_user_live_seed_status_cached(now_bucket_10s)
    except Exception as exc:
        payload["seed_status"] = {"success": False, "error": str(exc)}

    try:
        seed_status = payload.get("seed_status", {}) or {}
        seed_inner = seed_status.get("status", {}) if isinstance(seed_status, dict) else {}
        health = payload.get("health", {}) or {}
        score_cache_key = "|".join([
            str(health.get("latest_event_time") or "no_event"),
            str(health.get("latest_event_created_at") or "no_event_insert"),
            str(health.get("latest_feature_update_time") or "no_feature_update"),
            str(health.get("latest_score_time") or "no_score_update"),
            str(health.get("score_count") or seed_inner.get("score_count") or 0),
            str(seed_inner.get("latest_score_seeded_at") or "no_seed"),
            str(safe_limit),
            f"thr={float(threshold):.4f}",
        ])
        summary, scores = _fetch_user_live_scores_cached(score_cache_key, safe_limit, float(threshold))
        payload["score_summary"] = summary
        payload["scores"] = _rename_live_score_columns(scores)
    except Exception as exc:
        payload["score_summary"] = {"error": str(exc)}

    # Only budget/recommendation/real-time views need action_queue. View 1 can render from scores only.
    needs_actions = view in {
        "4. 예산 최적화 및 리텐션 타겟",
        "5. 개인화 추천",
        "6. 실시간 운영 모니터",
    }
    if needs_actions:
        try:
            health = payload.get("health", {}) or {}
            action_cache_key = "|".join([
                str(health.get("latest_event_time") or "no_event"),
                str(health.get("latest_event_created_at") or "no_event_insert"),
                str(health.get("latest_score_time") or "no_score_update"),
                str(health.get("latest_action_update_time") or "no_action_update"),
                str((payload.get("score_summary", {}) or {}).get("scored_customers") or 0),
                str(safe_limit),
            ])
            summary, actions = _fetch_user_live_actions_cached(action_cache_key, limit=safe_limit, status="queued")
            payload["action_summary"] = summary
            payload["actions"] = _normalize_live_actions_df(actions)
        except Exception as exc:
            payload["action_summary"] = {"error": str(exc)}

        # Recommendation summary is cheap and feeds the real-time KPI card. Fetch only
        # on action/recommendation views so normal churn view remains fast.
        try:
            health = payload.get("health", {}) or {}
            rec_cache_key = "|".join([
                str(health.get("latest_event_time") or "no_event"),
                str(health.get("latest_score_time") or "no_score_update"),
                str(health.get("latest_recommendation_update_time") or "no_rec_update"),
                str((payload.get("score_summary", {}) or {}).get("scored_customers") or 0),
                "rec",
                str(min(safe_limit, 5000)),
            ])
            rec_summary, rec_df = _fetch_user_live_recommendations_cached(rec_cache_key, limit=min(safe_limit, 5000))
            payload["recommendation_summary"] = rec_summary
            if view == "5. 개인화 추천":
                payload["recommendations"] = rec_df
        except Exception as exc:
            payload["recommendation_summary"] = {"error": str(exc)}

    return payload

def _render_user_live_status(live_payload: dict[str, Any]) -> None:
    if not live_payload.get("enabled"):
        return
    health = live_payload.get("health", {}) or {}
    if health.get("status") == "ok":
        st.success(
            f"{T('자사 데이터 Live DB 연결됨')} · {T('이벤트 수')} {int(health.get('event_count') or 0):,} · "
            f"{T('상태 보유 고객 수')} {int(health.get('feature_state_count') or 0):,} · "
            f"{T('최신 이벤트')} {health.get('latest_event_time') or '-'}"
        )
    else:
        st.warning(f"{T('자사 데이터 Live DB 상태 확인 실패')}: {health.get('error', 'unknown error')}")

    seed_status = live_payload.get("seed_status", {}) or {}
    status = seed_status.get("status", {}) if isinstance(seed_status, dict) else {}
    if status:
        st.caption(
            f"{T('Live DB 상태')} · "
            f"scores={int(status.get('score_count') or 0):,}, "
            f"{T('저장 추천후보')}={int(status.get('recommendation_count') or 0):,}, "
            f"queued actions={int(status.get('action_queue_count') or 0):,} "
            "(5번 화면은 저장 후보를 그대로 쓰지 않고 현재 예산·임계값 타겟 기준으로 새 추천을 만듭니다.)"
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
    if st.session_state.get("data_mode", "ecommerce") != "simulator":
        return False
    if has_data:
        return False
    _simulator_missing_result_box(feature_name, reason=reason, action_hint=action_hint)
    return True


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────

def _build_weekly_action_review(
    recommendations_df: pd.DataFrame,
    selected_customers_df: pd.DataFrame,
    execution_rate: float = 0.75,
    high_coupon_execution_rate: float = 0.50,
    noise_std: float = 0.15,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Build a simulated weekly execution log from recommendation data and analyze performance."""
    rng = np.random.default_rng(seed)

    df = recommendations_df.copy()
    if df.empty:
        empty_summary: dict[str, Any] = {
            "total_actions": 0, "total_executed": 0, "execution_rate": 0.0,
            "total_budget_spent": 0.0, "expected_profit_sum": 0.0,
            "actual_profit_sum": 0.0, "profit_gap": 0.0, "profit_gap_pct": 0.0,
            "loss_action_count": 0, "avg_expected_roi": 0.0, "avg_actual_roi": 0.0,
            "conversion_rate": 0.0, "over_investment_amount": 0.0,
            "underperformed_count": 0, "outcome_counts": {},
        }
        return empty_summary, pd.DataFrame(), []

    if not selected_customers_df.empty and "customer_id" in selected_customers_df.columns:
        opt_cols = ["customer_id"]
        for c in ["recommended_action", "intervention_intensity", "intervention_intensity_label"]:
            if c in selected_customers_df.columns and c not in df.columns:
                opt_cols.append(c)
        if len(opt_cols) > 1:
            df = df.merge(selected_customers_df[opt_cols], on="customer_id", how="left")

    if "intervention_intensity_label" not in df.columns:
        buckets = df.get("timing_priority_bucket", pd.Series(dtype=str))
        df["intervention_intensity_label"] = buckets.map(
            lambda b: "고강도" if str(b).startswith("immediate") else (
                "중강도" if str(b).startswith("soon") else "저강도"
            )
        )
    if "recommended_action" not in df.columns:
        df["recommended_action"] = df.apply(
            lambda r: f"{r.get('recommended_category', 'retention')} · {r.get('intervention_intensity_label', '중강도')} · {r.get('recommended_intervention_window', '')}",
            axis=1,
        )

    for col in ["coupon_cost", "expected_incremental_profit", "expected_roi", "churn_probability", "uplift_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    n = len(df)

    is_high_coupon = df["coupon_cost"] > df["coupon_cost"].quantile(0.75) if n > 4 else pd.Series([False] * n)
    base_probs = np.where(is_high_coupon, high_coupon_execution_rate, execution_rate)
    df["executed"] = rng.random(n) < base_probs

    df["execution_day"] = np.where(df["executed"], rng.choice([1, 2, 3, 4, 5], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]), 0)

    noise = rng.normal(0, noise_std, n)
    retention_base = 1 - df["churn_probability"].values
    uplift_boost = df["uplift_score"].values * 0.5

    df["actual_conversion"] = rng.random(n) < np.clip(retention_base + uplift_boost + noise * 0.3, 0, 1)
    df["actual_roi"] = np.where(df["executed"], df["expected_roi"] * np.clip(1 + noise, -0.3, 3.0), 0.0)
    perf_multiplier = np.clip(1 + noise * 1.5, 0.0, 2.5)
    converted = df["actual_conversion"].values.astype(float)
    gross_revenue = df["expected_incremental_profit"].values * perf_multiplier
    net_cost = df["coupon_cost"].values * (1 - converted * 0.7)
    df["actual_profit"] = np.where(
        df["executed"],
        gross_revenue - net_cost,
        0.0,
    )
    missed_mask = (~df["executed"]) & (df["expected_roi"] > 1.0)
    df.loc[missed_mask, "actual_profit"] = -df.loc[missed_mask, "expected_incremental_profit"] * 0.3

    df["actual_coupon_cost"] = np.where(df["executed"], df["coupon_cost"] * np.clip(1 + rng.normal(0, 0.05, n), 0.9, 1.1), 0.0)
    df["coupon_redeemed"] = df["executed"] & (rng.random(n) < np.clip(0.6 + df["uplift_score"] * 2, 0, 0.95))
    df["retained_30d"] = df["executed"] & df["actual_conversion"]

    median_cost = df.loc[df["executed"], "coupon_cost"].median() if df["executed"].any() else 0
    conditions = [
        df["executed"] & (df["actual_profit"] > 0) & (df["actual_roi"] >= df["expected_roi"] * 0.7),
        df["executed"] & (df["actual_profit"] > 0) & (df["actual_roi"] < df["expected_roi"] * 0.7),
        df["executed"] & (df["actual_profit"] <= 0) & (df["coupon_cost"] > median_cost),
        df["executed"] & (df["actual_profit"] <= 0) & (df["coupon_cost"] <= median_cost),
        missed_mask,
    ]
    labels = ["적정 판단", "기대 미달", "과잉 투자", "타겟 오류", "실행 누락"]
    df["outcome_label"] = np.select(conditions, labels, default="해당 없음")

    executed_df = df[df["executed"]]
    executed_count = int(executed_df.shape[0])
    expected_sum = float(executed_df["expected_incremental_profit"].sum())
    actual_sum = float(executed_df["actual_profit"].sum())
    gap = actual_sum - expected_sum
    loss_count = int((executed_df["actual_profit"] < 0).sum())
    over_inv = float(executed_df.loc[executed_df["outcome_label"] == "과잉 투자", "coupon_cost"].sum())
    underperf = int((df["outcome_label"] == "기대 미달").sum())

    summary: dict[str, Any] = {
        "total_actions": n,
        "total_executed": executed_count,
        "execution_rate": executed_count / max(n, 1),
        "total_budget_spent": float(executed_df["actual_coupon_cost"].sum()),
        "expected_profit_sum": expected_sum,
        "actual_profit_sum": actual_sum,
        "profit_gap": gap,
        "profit_gap_pct": gap / max(abs(expected_sum), 1),
        "loss_action_count": loss_count,
        "avg_expected_roi": float(executed_df["expected_roi"].mean()) if executed_count else 0.0,
        "avg_actual_roi": float(executed_df["actual_roi"].mean()) if executed_count else 0.0,
        "conversion_rate": float(executed_df["actual_conversion"].mean()) if executed_count else 0.0,
        "over_investment_amount": over_inv,
        "underperformed_count": underperf,
        "outcome_counts": df["outcome_label"].value_counts().to_dict(),
    }

    suggestions: list[dict] = []

    over_inv_df = executed_df[executed_df["outcome_label"] == "과잉 투자"]
    if not over_inv_df.empty:
        _oi_n = len(over_inv_df)
        _oi_loss = float(over_inv_df["actual_profit"].sum())
        _oi_cost = float(over_inv_df["coupon_cost"].sum())
        _oi_segments = over_inv_df["uplift_segment"].value_counts().head(2).to_dict() if "uplift_segment" in over_inv_df.columns else {}
        _oi_seg_str = ", ".join(f"{s} {c}명" for s, c in _oi_segments.items())
        suggestions.append({
            "title": "고비용 쿠폰 조정",
            "amount": _oi_loss,
            "what": f"{_oi_n}명에게 쿠폰 총 {_oi_cost:,.0f}원 지급 → 전환 실패",
            "who": _oi_seg_str or "-",
            "action": "고강도 쿠폰 기준을 expected_roi 2.0 이상으로 제한하거나, 쿠폰 대신 메시지/follow-up으로 전환",
            "severity": "warning",
        })

    wrong_df = executed_df[executed_df["outcome_label"] == "타겟 오류"]
    if not wrong_df.empty:
        _wt_n = len(wrong_df)
        _wt_loss = float(wrong_df["actual_profit"].sum())
        _wt_personas = wrong_df["persona"].value_counts().head(2).to_dict() if "persona" in wrong_df.columns else {}
        _wt_persona_str = ", ".join(f"{p} {c}명" for p, c in _wt_personas.items())
        suggestions.append({
            "title": "타겟 대상 재검토",
            "amount": _wt_loss,
            "what": f"{_wt_n}명에게 액션 실행했지만 반응 없음 (평균 ROI {float(wrong_df['actual_roi'].mean()):.2f})",
            "who": _wt_persona_str or "-",
            "action": "이 고객군을 다음 주 타겟에서 제외하거나 monitor_only로 전환",
            "severity": "warning",
        })

    under_df = executed_df[executed_df["outcome_label"] == "기대 미달"]
    if not under_df.empty:
        _ud_n = len(under_df)
        _ud_expected = float(under_df["expected_incremental_profit"].sum())
        _ud_actual = float(under_df["actual_profit"].sum())
        _ud_gap = _ud_actual - _ud_expected
        _ud_categories = under_df["recommended_category"].value_counts().head(2).to_dict() if "recommended_category" in under_df.columns else {}
        _ud_cat_str = ", ".join(f"{c} {n}건" for c, n in _ud_categories.items())
        suggestions.append({
            "title": "기대 미달 액션 점검",
            "amount": _ud_gap,
            "what": f"{_ud_n}명 이익 발생했지만 기대 대비 부족 (기대 {_ud_expected:,.0f}원 → 실제 {_ud_actual:,.0f}원)",
            "who": _ud_cat_str or "-",
            "action": "해당 카테고리의 쿠폰 강도를 한 단계 낮추거나 개입 타이밍을 앞당기세요",
            "severity": "warning",
        })

    missed_count = int(missed_mask.sum())
    if missed_count > 0:
        missed_df = df[missed_mask]
        _ms_expected = float(missed_df["expected_incremental_profit"].sum())
        _ms_loss = float(missed_df["actual_profit"].sum())
        _ms_segments = missed_df["uplift_segment"].value_counts().head(2).to_dict() if "uplift_segment" in missed_df.columns else {}
        _ms_seg_str = ", ".join(f"{s} {c}명" for s, c in _ms_segments.items())
        suggestions.append({
            "title": "실행 누락 고객 추가",
            "amount": _ms_loss,
            "what": f"기대 ROI 1.0 이상인 고객 {missed_count}명을 실행하지 않아 이탈 (원래 기대 이익 {_ms_expected:,.0f}원)",
            "who": _ms_seg_str or "-",
            "action": "다음 주 우선 실행 대상에 추가",
            "severity": "info",
        })

    seg_pnl = pd.Series(dtype=float)
    seg_cost = pd.Series(dtype=float)
    seg_cnt = pd.Series(dtype=int)
    if "uplift_segment" in df.columns and not executed_df.empty:
        seg_pnl = executed_df.groupby("uplift_segment")["actual_profit"].sum()
        seg_cost = executed_df.groupby("uplift_segment")["coupon_cost"].sum()
        seg_cnt = executed_df.groupby("uplift_segment").size()
        for seg, pnl in seg_pnl.items():
            if pnl < 0:
                _sc = float(seg_cost.get(seg, 0))
                _sn = int(seg_cnt.get(seg, 0))
                suggestions.append({
                    "title": f"{seg} 세그먼트 적자",
                    "amount": pnl,
                    "what": f"{seg} {_sn}명에 쿠폰 {_sc:,.0f}원 투입",
                    "who": f"{seg} 세그먼트 전체",
                    "action": "쿠폰 대신 CRM follow-up 또는 monitor_only로 전환",
                    "severity": "warning",
                })

    if not seg_pnl.empty:
        best_seg = seg_pnl.idxmax()
        if seg_pnl[best_seg] > 0:
            _bs_cnt = int(seg_cnt.get(best_seg, 0))
            _bs_cost = float(seg_cost.get(best_seg, 0))
            suggestions.append({
                "title": f"{best_seg} 세그먼트 유지",
                "amount": float(seg_pnl[best_seg]),
                "what": f"{best_seg} {_bs_cnt}명에 {_bs_cost:,.0f}원 투입하여 성과 달성",
                "who": f"{best_seg} 세그먼트",
                "action": "현재 전략 유지, 비슷한 프로필 고객을 추가 타겟으로 확대",
                "severity": "success",
            })

    if not suggestions:
        suggestions.append({
            "title": "전반적으로 양호",
            "amount": 0.0,
            "what": "주요 위험 신호 없음",
            "who": "-",
            "action": "현재 전략을 유지하세요",
            "severity": "success",
        })

    return summary, df, suggestions


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
    return CORE_VIEW_DISPLAY_LABELS.get(_language_code(), CORE_VIEW_DISPLAY_LABELS["ko"]).get(option, option)


def _set_query_param_if_changed(key: str, value: Any) -> None:
    """Avoid extra Streamlit reruns by writing query params only when changed."""
    try:
        value_s = str(value)
        if st.query_params.get(key) != value_s:
            st.query_params[key] = value_s
    except Exception:
        pass

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
            position: relative;
            z-index: 0;
            isolation: isolate;
            display: block;
            clear: both;
            overflow: auto;
            max-width: 100%;
            margin: 4px 0 18px 0;
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


def inject_rendering_guard_css():
    """Extra UI guard for Streamlit reruns and domain-specific rendering.

    Keep this separate from the general theme CSS so it is injected after the
    base rules.  The dashboard view selector is implemented with ``st.radio``
    rather than real ``st.tabs()``, so the radio label DOM must be styled
    directly.
    """
    st.markdown(
        """
        <style>
        /* Dashboard view selector: st.radio rendered as tab-like buttons. */
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"],
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] {
            display: flex !important;
            flex-wrap: wrap !important;
            align-items: center !important;
            gap: 12px 16px !important;
            margin-top: 10px !important;
            margin-bottom: 22px !important;
        }

        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] > label,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] > label,
        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] label,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] label {
            min-height: 44px !important;
            box-sizing: border-box !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            padding: 10px 18px !important;
            margin: 0 !important;
            border-radius: 14px !important;
            background: #ffffff !important;
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06) !important;
            color: #1f2937 !important;
            -webkit-text-fill-color: #1f2937 !important;
            cursor: pointer !important;
            transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease !important;
        }

        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] label:hover,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] label:hover {
            background: #dbeafe !important;
            border-color: #2563eb !important;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.14) !important;
            transform: translateY(-1px) !important;
        }

        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] label *,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] label * {
            color: #1f2937 !important;
            -webkit-text-fill-color: #1f2937 !important;
            font-weight: 800 !important;
            opacity: 1 !important;
        }

        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
            border-color: #2563eb !important;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.28) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        section[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) *,
        .main .block-container div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 900 !important;
            opacity: 1 !important;
        }

        /* Real st.tabs used elsewhere: keep inactive/active tabs visible too. */
        section[data-testid="stMain"] .stTabs [data-baseweb="tab-list"] {
            gap: 10px !important;
            flex-wrap: wrap !important;
            border-bottom: 0 !important;
        }
        section[data-testid="stMain"] .stTabs [data-baseweb="tab"] {
            min-height: 42px !important;
            padding: 10px 18px !important;
            border-radius: 14px !important;
            background: #ffffff !important;
            border: 1px solid rgba(148, 163, 184, 0.42) !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05) !important;
            color: #1f2937 !important;
            -webkit-text-fill-color: #1f2937 !important;
            font-weight: 800 !important;
        }
        section[data-testid="stMain"] .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
            border-color: #2563eb !important;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.28) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        section[data-testid="stMain"] .stTabs [data-baseweb="tab"] *,
        section[data-testid="stMain"] .stTabs [data-baseweb="tab"] p {
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
            font-weight: 800 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str):
    """Render the top hero in an isolated component.

    The finance mode turns on a runtime value-translation layer for product names
    such as card/credit_card.  Keeping the hero HTML/CSS inside an iframe and
    avoiding class names like ``*-card`` prevents finance-only text replacement or
    Streamlit markdown wrapper CSS from breaking the gradient background.
    """
    title_html = html.escape(_translate_runtime_text(title))
    subtitle_html = html.escape(_translate_runtime_text(subtitle))
    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <style>
            html, body {{
              margin: 0;
              padding: 0;
              background: transparent;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .rt-hero {{
              width: 100%;
              min-height: 184px;
              box-sizing: border-box;
              position: relative;
              overflow: hidden;
              padding: 32px 34px 28px 34px;
              border-radius: 28px;
              background-color: #0f172a;
              background-image:
                radial-gradient(circle at 92% 8%, rgba(255,255,255,0.20), transparent 30%),
                linear-gradient(135deg, #0f172a 0%, #2563eb 58%, #7c3aed 100%);
              box-shadow: 0 24px 60px rgba(15,23,42,0.24);
              border: 1px solid rgba(255,255,255,0.18);
              color: #ffffff;
            }}
            .rt-hero::after {{
              content: "";
              position: absolute;
              right: -72px;
              bottom: -92px;
              width: 230px;
              height: 230px;
              border-radius: 999px;
              background: rgba(255,255,255,0.16);
              pointer-events: none;
            }}
            .rt-kicker {{
              position: relative;
              z-index: 1;
              color: #dbeafe;
              font-size: 0.9rem;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              font-weight: 800;
              margin-bottom: 10px;
            }}
            .rt-title {{
              position: relative;
              z-index: 1;
              color: #ffffff;
              font-size: clamp(2.0rem, 3.1vw, 3.0rem);
              line-height: 1.08;
              font-weight: 900;
              margin: 0 0 12px 0;
              letter-spacing: -0.04em;
            }}
            .rt-subtitle {{
              position: relative;
              z-index: 1;
              color: #eff6ff;
              font-size: 1rem;
              font-weight: 600;
              line-height: 1.65;
              max-width: 980px;
            }}
          </style>
        </head>
        <body>
          <div class="rt-hero">
            <div class="rt-kicker">RETENTION INTELLIGENCE COPILOT</div>
            <div class="rt-title">{title_html}</div>
            <div class="rt-subtitle">{subtitle_html}</div>
          </div>
        </body>
        </html>
        """,
        height=224,
        scrolling=False,
    )


def render_step_title(title: str, caption: str | None = None):
    """Render wizard step titles with isolated styling."""
    title_html = html.escape(_translate_runtime_text(title))
    caption_block = ""
    if caption:
        caption_block = f'<div class="rt-step-caption">{html.escape(_translate_runtime_text(caption))}</div>'
    height = 104 if caption else 72
    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <style>
            html, body {{
              margin: 0;
              padding: 0;
              background: transparent;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .rt-step {{
              width: 100%;
              box-sizing: border-box;
              margin: 0;
              padding: 18px 22px;
              border-radius: 18px;
              background-color: #eff6ff;
              background-image: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%);
              border: 1px solid rgba(37,99,235,0.18);
              box-shadow: 0 10px 28px rgba(15,23,42,0.06);
            }}
            .rt-step-title {{
              color: #2563eb;
              font-size: 1.35rem;
              font-weight: 900;
              line-height: 1.25;
              letter-spacing: -0.02em;
            }}
            .rt-step-caption {{
              margin-top: 7px;
              color: #475569;
              font-size: 0.95rem;
              font-weight: 600;
              line-height: 1.55;
            }}
          </style>
        </head>
        <body>
          <div class="rt-step">
            <div class="rt-step-title">{title_html}</div>
            {caption_block}
          </div>
        </body>
        </html>
        """,
        height=height,
        scrolling=False,
    )


def render_status_pill(message: str, variant: str = "success"):
    message = _translate_runtime_text(message)
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


def _raw_data_token(mode: str | None = None) -> str:
    mode = mode or _business_mode()
    base = _domain_paths(mode)["data"]
    return _file_version_token([
        f"{base}/customer_summary.csv",
        f"{base}/cohort_retention.csv",
        f"{base}/events.csv",
        f"{base}/orders.csv",
    ])


def _result_data_token(mode: str | None = None) -> str:
    mode = mode or _business_mode()
    base = _domain_paths(mode)["results"]
    return _file_version_token([
        f"{base}/dataset_metadata.json",
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


def _live_seed_metadata_path() -> Path:
    return _project_root() / DOMAIN_DIRS["user"]["results"] / "live_seed_source.json"


def _copy_directory_contents(src: Path, dst: Path) -> dict[str, Any]:
    """현재 도메인 산출물을 user-live seed가 읽는 표준 경로로 동기화한다."""
    import shutil

    copied: list[str] = []
    if not src.exists():
        return {"source": str(src), "target": str(dst), "copied": copied, "missing": True}

    if src.resolve() == dst.resolve():
        return {"source": str(src), "target": str(dst), "copied": copied, "missing": False, "skipped_same_path": True}

    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        try:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
            copied.append(child.name)
        except Exception as exc:
            copied.append(f"{child.name}:ERROR:{exc}")
    return {"source": str(src), "target": str(dst), "copied": copied, "missing": False}


def _sync_domain_artifacts_for_live_seed(mode: str) -> dict[str, Any]:
    """금융/이커머스 모드 산출물을 PostgreSQL user-live seed 입력 경로로 복사한다.

    backend의 seed_user_live_from_artifacts()는 별도 mode 인자를 받지 않으므로,
    새 학습 결과를 DB에 반영하려면 현재 도메인 산출물을 user 표준 경로
    (results_user/models_user/data/feature_store_user/data/raw_user)에 먼저 맞춰야 한다.
    """
    root = _project_root()
    source = _domain_paths(mode)
    target = DOMAIN_DIRS["user"]
    sync_report: dict[str, Any] = {"mode": mode, "items": {}}
    for key in ["data", "results", "models", "features"]:
        sync_report["items"][key] = _copy_directory_contents(root / source[key], root / target[key])
    return sync_report


def _save_live_seed_metadata(mode: str, seed_result: Any, sync_report: dict[str, Any] | None = None) -> None:
    path = _live_seed_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_mode": mode,
        "source_raw_token": _raw_data_token(mode),
        "source_result_token": _result_data_token(mode),
        "seed_success": bool(isinstance(seed_result, dict) and seed_result.get("success")),
        "seed_result": seed_result if isinstance(seed_result, dict) else {"raw": str(seed_result)},
        "sync_report": sync_report or {},
        "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_live_seed_metadata() -> dict[str, Any]:
    path = _live_seed_metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _series_id_set(df: pd.DataFrame, column: str = "customer_id") -> set[str]:
    if df is None or df.empty or column not in df.columns:
        return set()
    return {str(value).strip() for value in df[column].dropna().astype(str).tolist() if str(value).strip()}


def _live_payload_matches_current_dataset(live_payload: dict[str, Any], customers_df: pd.DataFrame) -> bool:
    """Return True when the PostgreSQL live DB should drive dashboard-wide KPIs.

    The first implementation required a local seed metadata file to match the
    current CSV/result token exactly.  That was too strict for demos and patch
    applications: when the API DB had been seeded but the local metadata file was
    missing or stale, incoming live events updated PostgreSQL correctly while the
    dashboard silently fell back to static CSV artifacts.  The result was the
    symptom the user reported: top-level churn probability/customer counts never
    changed.

    We now prefer an exact metadata match when available, but also accept the
    live DB when its scored customer IDs substantially overlap the currently
    loaded customers.  New live customers may make live_ids larger than file_ids,
    which is expected and must not invalidate the live view.
    """
    if not _is_user_live_mode() or not isinstance(live_payload, dict):
        return False

    scores_df = live_payload.get("scores", pd.DataFrame())
    if not isinstance(scores_df, pd.DataFrame) or scores_df.empty:
        return False

    file_ids = _series_id_set(customers_df)
    live_ids = _series_id_set(scores_df)
    if not live_ids:
        return False

    mode = _business_mode()
    meta = _load_live_seed_metadata()
    exact_meta_match = (
        bool(meta)
        and meta.get("source_mode") == mode
        and meta.get("source_raw_token") == _raw_data_token(mode)
        and meta.get("source_result_token") == _result_data_token(mode)
        and bool(meta.get("seed_success"))
    )
    if exact_meta_match:
        return True

    if file_ids:
        seeded_coverage = len(file_ids & live_ids) / max(len(file_ids), 1)
        return seeded_coverage >= 0.80

    health = live_payload.get("health", {}) or {}
    return str(health.get("status", "")).lower() == "ok"


@st.cache_data(show_spinner=False)
def _load_app_bundle_cached(_token: str, data_dir: str = "data/raw"):
    return load_dashboard_bundle(data_dir=data_dir, include_optional=False)


def _resolve_data_dir_for_mode(mode: str) -> str:
    """도메인별 data 디렉토리. 금융/이커머스 모드에서는 다른 도메인 결과로 fallback하지 않는다."""
    return DOMAIN_DIRS.get(mode, DOMAIN_DIRS["ecommerce"])["data"]


def _resolve_result_dir_for_mode(mode: str) -> str:
    """도메인별 results 디렉토리. 새 업로드가 과거 결과를 섞어 보이지 않도록 fallback을 막는다."""
    return DOMAIN_DIRS.get(mode, DOMAIN_DIRS["ecommerce"])["results"]


@st.cache_data(show_spinner=False)
def _load_insight_bundle_cached(_raw_token: str, _result_token: str, data_dir: str = "data/raw", result_dir: str = "results"):
    return load_dashboard_insight_bundle(data_dir=data_dir, result_dir=result_dir)


def load_app_data():
    mode = _business_mode()
    data_dir = _resolve_data_dir_for_mode(mode)
    return _load_app_bundle_cached(_raw_data_token(), data_dir=data_dir)


def load_insight_data():
    mode = _business_mode()
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
        _fetch_user_live_health_cached.clear()
        _fetch_user_live_seed_status_cached.clear()
        _fetch_user_live_actions_cached.clear()
        _fetch_user_live_recommendations_cached.clear()
    except Exception:
        pass


def load_training_artifacts_api():
    mode = _business_mode()
    if mode in BUSINESS_UPLOAD_MODES:
        _paths = _domain_paths(mode)
        artifacts = load_dashboard_artifacts(
            result_dir=_paths["results"],
            model_dir=_paths["models"],
            feature_store_dir=_paths["features"],
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
    mode = _business_mode()
    if mode in BUSINESS_UPLOAD_MODES:
        return _load_saved_results_from_dir(_domain_paths(mode)["results"])
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
    customer_col = None
    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            if _normalize_i18n_key(str(col)) in {"customerid", "고객id", "顧客id"}:
                customer_col = col
                break
    if customer_col is not None:
        customers = int(df[customer_col].nunique())

    if _language_code() == "en":
        if customers is not None:
            if rows == customers:
                return f"{label}: {customers:,} customers"
            return f"{label}: {customers:,} customers / {rows:,} rows"
        return f"{label}: {rows:,} rows"
    if _language_code() == "ja":
        if customers is not None:
            if rows == customers:
                return f"{label}: 顧客 {customers:,}人"
            return f"{label}: 顧客 {customers:,}人 / {rows:,}行"
        return f"{label}: {rows:,}行"
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
    if isinstance(value, str):
        return _collapse_repeated_customer_words(value.strip())
    return value


def _translate_column_name(column: str) -> str:
    code = _data_label_language_code()
    domain_label = _domain_column_label(column, code)
    if domain_label:
        return domain_label
    labels = COLUMN_LABELS.get(code, COLUMN_LABELS.get("ko", {}))
    raw = str(column)

    if raw in labels:
        return labels[raw]

    normalized = _normalize_i18n_key(raw)
    for canonical, translated in labels.items():
        if _normalize_i18n_key(canonical) == normalized:
            return translated
        for labels_by_lang in COLUMN_LABELS.values():
            localized = labels_by_lang.get(canonical)
            if localized and _normalize_i18n_key(localized) == normalized:
                return translated

    # 흔한 수동/LLM 생성 컬럼명 보정
    alias_to_canonical = {
        "고객id": "customer_id",
        "고객아이디": "customer_id",
        "customerid": "customer_id",
        "고객유형": "persona",
        "이탈확률": "churn_probability",
        "이탈점수": "churn_score",
        "예상roi": "expected_roi",
        "추천액션": "recommended_action",
        "선정이유": "selection_reason",
        "selectionreason": "selection_reason",
        "watchout": "watchout",
        "주의사항": "caution",
        "다음추천액션": "next_best_action",
        "llm결과요약": "llm_result_summary",
    }
    canonical = alias_to_canonical.get(normalized)
    if canonical and canonical in labels:
        return labels[canonical]

    friendly = friendly_translate_column(raw, code)
    if friendly != raw:
        return _domain_translate_text(friendly)
    return T(raw.replace("_", " "))


def _term_caption_triggers() -> list[tuple[str, list[str]]]:
    return [
        ("CustomerType", ["persona", "customer type", "고객유형", "顧客タイプ"]),
        ("ChurnProbability", ["churn_probability", "churn score", "이탈확률", "이탈점수", "離脱確率", "離脱スコア"]),
        ("ChurnTiming", ["expected_churn_period", "expected_churn_date", "예상이탈시점", "예상이탈날짜", "予想離脱"]),
        ("ExpectedLoss", ["expected_loss", "expected_loss_30d", "예상손실액", "予想損失"]),
        ("CLV", ["clv", "생애가치", "lifetime value", "生涯価値"]),
        ("Uplift", ["uplift", "개입효과", "고객반응유형", "介入効果"]),
        ("ExpectedProfit", ["expected_incremental_profit", "expected_profit", "예상이익", "예상증분이익", "予想利益"]),
        ("ExpectedROI", ["expected_roi", "roi", "예상roi", "予想roi"]),
        ("InterventionIntensity", ["intervention_intensity", "개입강도", "介入強度"]),
        ("RecommendedAction", ["recommended_action", "queued_recommended_action", "추천액션", "큐추천액션", "推奨アクション"]),
        ("RecommendationReason", ["reason_tags", "selection_reason", "reason_summary", "추천이유", "선정이유", "推薦理由"]),
        ("ActionStatus", ["action_status", "action_queue_status", "액션상태", "액션큐상태", "アクション状態"]),
        ("CustomerValueScore", ["value_score", "고객가치점수", "顧客価値スコア"]),
        ("RecommendationScore", ["recommendation_score", "recommendation_priority", "추천점수", "추천우선순위", "推薦スコア"]),
        ("Priority", ["priority", "priority_score", "selection_score", "우선순위", "선정점수", "優先度"]),
    ]


@lru_cache(maxsize=512)
def _term_caption_html_cached(language_code: str, label: str, columns_key: str) -> str:
    captions = TERM_CAPTIONS.get(language_code, TERM_CAPTIONS.get("ko", {}))
    joined_norm = _normalize_i18n_key(f"{label} {columns_key}")
    ordered_keys: list[str] = []
    for key, aliases in _term_caption_triggers():
        if any(_normalize_i18n_key(alias) in joined_norm for alias in aliases):
            if key not in ordered_keys and captions.get(key):
                ordered_keys.append(key)
    if not ordered_keys:
        return ""
    lines = [captions[key] for key in ordered_keys[:8]]
    return (
        "<div style='margin:8px 0 18px 0;padding:12px 14px;border-radius:14px;background:#F8FAFC;border:1px solid #E2E8F0;color:#334155;line-height:1.6;font-size:0.92rem;'>"
        f"<b>{html.escape(T('용어 설명'))}</b><br/>"
        + "<br/>".join(f"• {html.escape(line)}" for line in lines)
        + "</div>"
    )


def _append_term_caption(df: pd.DataFrame, label: str = "") -> None:
    """Show plain-language explanations for any potentially unfamiliar table terms."""
    if df is None or df.empty:
        return
    columns_key = "|".join(str(c) for c in df.columns)
    caption_html = _term_caption_html_cached(_language_code(), str(label), columns_key)
    if caption_html:
        st.markdown(caption_html, unsafe_allow_html=True)




def _display_value_is_null(value):
    """Return True only for scalar null-like values.

    Pandas pd.isna(list/dict/array) can return array-like results, which should
    not be used as booleans. For display sanitization, non-scalar objects are
    treated as non-null and formatted safely.
    """
    try:
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
        try:
            return bool(result) if not hasattr(result, "__len__") else False
        except Exception:
            return False
    except Exception:
        return False


def _map_object_series_unique(series, formatter):
    """Format an object Series by formatting unique values once.

    This is a rendering optimization for large upload preview tables. It does
    not change the underlying dataframe used by the pipeline.
    """
    if series is None:
        return series

    def _safe_format(value):
        if _display_value_is_null(value):
            return ""
        try:
            return formatter(value)
        except Exception:
            return str(value)

    try:
        if getattr(series, "empty", False):
            return series

        # Fast path for hashable scalar/object values.
        mapping = {}
        for value in series.dropna().unique():
            try:
                hash(value)
            except Exception:
                raise TypeError("unhashable display value")
            mapping[value] = _safe_format(value)

        return series.map(lambda value: "" if _display_value_is_null(value) else mapping.get(value, _safe_format(value)))

    except Exception:
        # Safe fallback for lists/dicts/mixed objects.
        return series.apply(_safe_format)


def _sanitize_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    safe_df = df.copy().reset_index(drop=True)
    original_columns = _make_unique_columns([str(col) for col in safe_df.columns])
    safe_df.columns = original_columns

    for column in safe_df.columns:
        def _format_one(value: Any, column_name: str = column) -> Any:
            formatted = _format_table_value_by_column(column_name, value)
            return _normalize_table_cell(formatted)

        if pd.api.types.is_object_dtype(safe_df[column]) or pd.api.types.is_string_dtype(safe_df[column]):
            normalized = _map_object_series_unique(safe_df[column], _format_one)
        else:
            normalized = safe_df[column].map(_format_one)

        non_empty = [value for value in normalized.tolist() if value not in ("", None)]
        numeric_only = bool(non_empty) and all(isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)) for value in non_empty)
        if numeric_only:
            safe_df[column] = pd.to_numeric(normalized, errors="coerce")
        else:
            safe_df[column] = normalized.map(lambda value: "" if value is None else str(value))

    translated_columns = [_translate_column_name(_strip_duplicate_suffix(c)) for c in safe_df.columns]
    safe_df.columns = _make_unique_columns(translated_columns)
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
    """Render a compact, scrollable table without Streamlit dataframe JS.

    Performance note: the expensive operations are display-value translation,
    numeric formatting, and HTML serialization. Those now run only on the
    visible slice. Full row/customer counts and customer-ID search still use
    the original dataframe.
    """
    localized_label = T(label)

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.caption(_describe_table_count(pd.DataFrame(), label=localized_label))
        st.info(T("표시할 데이터가 없습니다."))
        return

    raw_df = _filter_display_columns_for_label(df, label)
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        st.caption(_describe_table_count(pd.DataFrame(), label=localized_label))
        st.info(T("표시할 데이터가 없습니다."))
        return

    raw_df = _dedupe_display_columns(raw_df.copy().reset_index(drop=True))
    total_rows = int(len(raw_df))
    view_raw = raw_df
    matched_rows: int | None = None
    _search_active = False

    if total_rows > 20:
        search_key = _table_widget_key(label, "search")
        _q = st.text_input(
            f"{localized_label} {T('검색')}",
            placeholder=T("고객 ID 검색"),
            key=search_key,
            label_visibility="collapsed",
        )
        if _q.strip():
            _search_active = True
            _ql = _q.strip().lower()
            customer_id_col = next(
                (
                    col for col in raw_df.columns
                    if str(col).lower() in {"customer_id", "customer id"}
                    or _normalize_i18n_key(str(col)) in {"customerid", "고객id", "顧客id"}
                ),
                None,
            )
            if customer_id_col is None:
                view_raw = raw_df.iloc[0:0].reset_index(drop=True)
            else:
                mask = (
                    raw_df[customer_id_col]
                    .astype(str)
                    .str.lower()
                    .str.contains(re.escape(_ql), na=False)
                )
                view_raw = raw_df[mask].reset_index(drop=True)
            matched_rows = int(len(view_raw))

    display_limit = max(50, int(TABLE_DISPLAY_ROW_LIMIT))
    truncated = int(len(view_raw)) > display_limit
    if truncated:
        view_raw = view_raw.head(display_limit).reset_index(drop=True)

    # Translate and format only the visible rows.
    safe_df = _sanitize_display_dataframe(view_raw)

    if _search_active:
        match_count = matched_rows if matched_rows is not None else int(len(view_raw))
        if _language_code() == "en":
            caption = f"{localized_label}: {match_count:,} matched of {total_rows:,} total"
            if truncated:
                caption += f" / showing first {len(safe_df):,} rows"
        elif _language_code() == "ja":
            caption = f"{localized_label}: 全体 {total_rows:,}件中 {match_count:,}件一致"
            if truncated:
                caption += f" / 先頭 {len(safe_df):,}行を表示"
        else:
            caption = f"{localized_label}: 전체 {total_rows:,}건 중 {match_count:,}건 일치"
            if truncated:
                caption += f" / 상위 {len(safe_df):,}행만 표시"
        st.caption(caption)
    else:
        caption = _describe_table_count(raw_df, label=localized_label)
        if truncated:
            if _language_code() == "en":
                caption += f" / showing first {len(safe_df):,} rows for speed"
            elif _language_code() == "ja":
                caption += f" / 速度のため先頭 {len(safe_df):,}行のみ表示"
            else:
                caption += f" / 속도 향상을 위해 상위 {len(safe_df):,}행만 표시"
        st.caption(caption)

    if safe_df.empty:
        st.info(T("표시할 데이터가 없습니다."))
        return

    try:
        _requested_height = int(max_height)
    except Exception:
        _requested_height = 420
    _table_height = min(420, max(220, _requested_height))

    # Ensure duplicate translated headers cannot break rendering or search.
    safe_df = safe_df.copy().reset_index(drop=True)
    safe_df.columns = _make_unique_columns([str(c) for c in safe_df.columns])
    html_table = safe_df.to_html(index=not hide_index, classes="oai-data-table", border=0, escape=True)
    table_doc = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  .table-frame {{
    height: {_table_height}px;
    overflow: auto;
    border: 1px solid rgba(148,163,184,0.32);
    border-radius: 14px;
    background: rgba(255,255,255,0.98);
  }}
  table.oai-data-table {{ width: max-content; min-width: 100%; border-collapse: collapse; font-size: 14px; line-height: 1.42; color: #0f172a; }}
  thead th {{ position: sticky; top: 0; z-index: 2; background: #f8fafc; color: #0f172a; text-align: left; font-weight: 800; border-bottom: 1px solid #cbd5e1; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: top; white-space: nowrap; }}
  tbody tr:nth-child(even) {{ background: rgba(248,250,252,0.92); }}
  tbody tr:hover {{ background: rgba(219,234,254,0.40); }}
</style>
</head>
<body>
<div class="table-frame">{html_table}</div>
</body>
</html>
"""
    components.html(table_doc, height=_table_height + 22, scrolling=False)
    _append_term_caption(raw_df, label=localized_label)



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






def sanitize_llm_markdown(text: str) -> str:
    """Remove Markdown/HTML strikethrough markers from LLM output before Streamlit renders it.

    LLMs often write numeric ranges such as ``3.65~~10.16`` when they mean
    ``3.65~10.16``. Streamlit interprets ``~~...~~`` as Markdown
    strikethrough, so we sanitize every LLM-rendered string at display time.
    """
    if text is None:
        return ""

    text = str(text)

    # Unicode combining strikethrough/overlay characters.
    text = re.sub(r"[\u0335-\u0338]", "", text)

    # HTML strikethrough tags: <s>, <strike>, <del>.
    text = re.sub(
        r"</?\s*(?:s|strike|del)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Inline style strikethrough spans.
    text = re.sub(
        r'<span\b[^>]*text-decoration\s*:\s*line-through[^>]*>(.*?)</span>',
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Numeric ranges: 3.65~~10.16 -> 3.65–10.16.
    text = re.sub(
        r"(?P<left>\d[\d,]*(?:\.\d+)?)\s*~{1,2}\s*(?P<right>\d[\d,]*(?:\.\d+)?)",
        r"\g<left>–\g<right>",
        text,
    )

    # Any remaining Markdown strikethrough delimiters are unsafe for this app.
    text = text.replace("~~", "")

    return text


def clear_llm_caches() -> None:
    """Remove cached LLM summaries/answers so old unsanitized text is not reused."""
    for key in list(st.session_state.keys()):
        key_str = str(key)
        if key_str.startswith("summary::") or key_str.startswith("qa::"):
            del st.session_state[key]


def _payload_hash(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def get_session_cached_summary(
    view_title: str,
    payload_json: str,
    api_key: str,
    model_name: str,
) -> str:
    payload_json = _wrap_llm_payload(payload_json)
    language = _llm_language_name()
    cache_key = f"summary::{_payload_hash(view_title, payload_json, model_name, language)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = generate_dashboard_summary(
            view_title=f"{view_title} | Answer language: {language} | {_llm_strict_language_instruction()}",
            payload_json=payload_json,
            user_api_key=api_key,
            model_name=model_name,
        )
    return _translate_runtime_text(sanitize_llm_markdown(st.session_state[cache_key]))


def get_session_cached_answer(
    view_title: str,
    payload_json: str,
    question: str,
    api_key: str,
    model_name: str,
) -> str:
    payload_json = _wrap_llm_payload(payload_json)
    question = _wrap_llm_question(question)
    language = _llm_language_name()
    cache_key = f"qa::{_payload_hash(view_title, payload_json, question, model_name, language)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = answer_dashboard_question(
            view_title=f"{view_title} | Answer language: {language} | {_llm_strict_language_instruction()}",
            payload_json=payload_json,
            question=question,
            user_api_key=api_key,
            model_name=model_name,
        )
    return _translate_runtime_text(sanitize_llm_markdown(st.session_state[cache_key]))


def get_chat_history_key(view_key: str) -> str:
    # 챗봇은 화면별로 새로 만들지 않고, 세션 전체에서 하나의 대화 기록을 공유한다.
    # view_key는 기존 호출부 호환을 위해 인자로만 유지한다.
    return "llm_chat_history"


def get_chat_input_key(view_key: str) -> str:
    # 화면 이동 시 chat_input widget key가 바뀌면 입력창/대화 UI가 새 위젯처럼 동작한다.
    return "llm_chat_input"


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
    # 닫았다가 다시 열면 그 시점의 현재 화면을 새 컨텍스트로 잡는다.
    for _key in (
        "llm_chat_view_key",
        "llm_chat_view_title",
        "llm_chat_payload",
        "llm_chat_model_name",
    ):
        st.session_state.pop(_key, None)


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
    st.subheader(T("LLM 결과 요약"))
    st.caption(T("현재 화면의 지표·표·그래프에서 추린 요약 컨텍스트만 바탕으로 응답합니다."))

    ready, status_message = get_llm_status(api_key)
    payload_json = build_payload_json(payload)

    if not ready:
        st.info(_translate_runtime_text(status_message))
        return

    with st.spinner(T("AI가 현재 화면의 결과를 요약하는 중입니다...")):
        try:
            summary = get_session_cached_summary(
                view_title=view_title,
                payload_json=payload_json,
                api_key=api_key or "",
                model_name=model_name,
            )
        except Exception as exc:
            st.error(f"{T('AI 요약 생성 중 오류가 발생했습니다')}: {exc}")
            return

    st.markdown(_translate_runtime_text(sanitize_llm_markdown(summary)))
    st.caption(T("추가 질문은 사이드바의 AI 챗봇 버튼을 눌러 이어서 대화할 수 있습니다."))


@st.fragment
def render_sidebar_chatbot_launcher(
    view_key: str,
    view_title: str,
    llm_enabled: bool,
    api_key: Optional[str],
    payload: Optional[Dict] = None,
    model_name: str = "gpt-4.1-mini",
):
    """사이드바 챗봇을 화면 전환과 독립적으로 유지한다.

    Streamlit은 화면 radio가 바뀌면 전체 스크립트를 다시 실행한다.
    따라서 챗봇을 "rerun 자체가 안 되게" 만들 수는 없지만,
    열림 상태/대화 기록/질문 컨텍스트를 session_state에 고정해
    다른 화면으로 이동해도 챗봇이 초기화되거나 새 화면 데이터로 자동 갱신되지 않게 한다.
    """
    st.divider()
    st.subheader(f"🤖 {T('AI 챗봇')}")

    ready, status_message = get_llm_status(api_key)
    is_open = bool(st.session_state.get("llm_chat_open", False))

    # 이미 열려 있는 챗봇은 처음 열었던 화면의 컨텍스트를 계속 사용한다.
    # 단, 구버전 세션처럼 컨텍스트가 비어 있으면 현재 화면으로 1회 보정한다.
    if is_open and st.session_state.get("llm_chat_payload") is None and payload is not None:
        st.session_state["llm_chat_view_key"] = view_key
        st.session_state["llm_chat_view_title"] = view_title
        st.session_state["llm_chat_payload"] = payload
        st.session_state["llm_chat_model_name"] = model_name

    btn_label = f"❌ {T('챗봇 닫기')}" if is_open else f"💬 {T('챗봇 열기')}"
    if st.button(
        btn_label,
        key="toggle_chatbot",
        use_container_width=True,
        disabled=(not llm_enabled) or (not ready),
    ):
        if is_open:
            close_llm_chat_dialog()
        else:
            # 챗봇을 여는 순간의 화면/데이터를 고정한다.
            st.session_state["llm_chat_open"] = True
            st.session_state["llm_chat_view_key"] = view_key
            st.session_state["llm_chat_view_title"] = view_title
            st.session_state["llm_chat_payload"] = payload or {}
            st.session_state["llm_chat_model_name"] = model_name
        st.rerun(scope="fragment")

    if not llm_enabled:
        st.caption(f"⚠️ {T('LLM 기능이 꺼져 있어 챗봇을 열 수 없습니다.')}")
        return
    if not ready:
        st.caption(f"⚠️ {_translate_runtime_text(status_message)}")
        return
    if not st.session_state.get("llm_chat_open", False):
        st.caption(f"📍 {T('현재 화면')}: **{view_title}**")
        st.caption(T("화면의 표·그래프를 보면서 질문할 수 있습니다."))
        return

    active_view_key = st.session_state.get("llm_chat_view_key") or view_key
    active_view_title = st.session_state.get("llm_chat_view_title") or view_title
    active_payload = st.session_state.get("llm_chat_payload") or payload or {}
    active_model_name = st.session_state.get("llm_chat_model_name") or model_name

    st.caption(f"📌 {T('고정된 챗봇 컨텍스트')}: **{active_view_title}**")
    if active_view_key != view_key:
        st.caption(T("화면을 이동해도 챗봇은 처음 열었던 화면의 데이터로 유지됩니다."))
        if st.button(T("현재 화면으로 컨텍스트 갱신"), key="refresh_chatbot_context", use_container_width=True):
            st.session_state["llm_chat_view_key"] = view_key
            st.session_state["llm_chat_view_title"] = view_title
            st.session_state["llm_chat_payload"] = payload or {}
            st.session_state["llm_chat_model_name"] = model_name
            st.rerun(scope="fragment")

    _render_sidebar_chatbot_inline(
        view_key=active_view_key,
        view_title=active_view_title,
        payload=active_payload,
        api_key=api_key,
        model_name=active_model_name,
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

    st.caption(f"📍 {T('컨텍스트')}: **{view_title}**")

    # 대화 지우기 버튼
    if st.button(f"🗑 {T('대화 지우기')}", key=f"clear_sidebar_chat_{view_key}", use_container_width=True):
        st.session_state[history_key] = []
        st.rerun(scope="fragment")

    history = st.session_state[history_key]

    # 대화 내역 (스크롤 가능 컨테이너 — height 제한)
    chat_container = st.container(height=400)
    with chat_container:
        if not history:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    f"{T('안녕하세요. 현재 보고 있는 화면 기준으로 답해드릴게요.')}\n\n"
                    f"- {T('왜 이 지표가 높/낮은지')}\n"
                    f"- {T('어떤 고객/세그먼트가 핵심인지')}\n"
                    f"- {T('예산·threshold에서 뭘 바꾸면 좋을지')}"
                )
        for item in history:
            role = item.get("role", "assistant")
            avatar = "🧑" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(_translate_runtime_text(sanitize_llm_markdown(item.get("content", ""))))

    # 입력창
    prompt = st.chat_input(
        T("현재 화면에 대해 질문하세요..."),
        key=input_key,
    )

    if prompt:
        history.append({"role": "user", "content": prompt})
        st.session_state[history_key] = history

        contextual_question = build_contextual_chat_question(
            view_title=view_title,
            history=history[:-1],
            latest_question=prompt,
        )

        with st.spinner(f"{T('AI 답변 생성 중')}..."):
            try:
                answer = get_session_cached_answer(
                    view_title=view_title,
                    payload_json=payload_json,
                    question=contextual_question,
                    api_key=api_key or "",
                    model_name=model_name,
                )
            except Exception as exc:
                answer = f"{T('AI 답변 생성 중 오류가 발생했습니다')}: {exc}"

        history.append({"role": "assistant", "content": answer})
        st.session_state[history_key] = history
        st.rerun(scope="fragment")


@st.dialog("AI Chatbot")
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
        f"""
        <div id="chatbot-drag-handle" class="chatbot-drag-handle">
            <span>🤖 {T('AI 분석 챗봇')}</span>
            <small>{T('드래그해서 이동')}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="chatbot-dialog-note">
            <strong>{T('현재 화면')}:</strong> {view_title}<br/>
            {T('현재 화면의 지표·표·그래프에서 추린 요약 컨텍스트만 바탕으로 응답합니다.')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_col1, top_col2 = st.columns([1, 1])
    if top_col1.button(T("대화 지우기"), key=f"clear_chat_{view_key}", use_container_width=True):
        st.session_state[history_key] = []
        st.rerun()
    if top_col2.button(T("닫기"), key=f"close_chat_{view_key}", use_container_width=True):
        close_llm_chat_dialog()
        st.rerun()

    if not ready:
        st.info(_translate_runtime_text(status_message))
        return

    history = st.session_state[history_key]

    if not history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                f"{T('안녕하세요. 현재 보고 있는 화면 기준으로 답해드릴게요.')}\n\n"
                f"- {T('왜 이 지표가 높/낮은지')}\n"
                f"- {T('어떤 고객/세그먼트가 핵심인지')}\n"
                f"- {T('예산·threshold에서 뭘 바꾸면 좋을지')}"
            )

    for item in history:
        role = item.get("role", "assistant")
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(sanitize_llm_markdown(item.get("content", "")))

    prompt = st.chat_input(
        T("현재 화면에 대해 질문하세요."),
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
            with st.spinner(T("AI가 답변하는 중입니다...")):
                try:
                    answer = get_session_cached_answer(
                        view_title=view_title,
                        payload_json=payload_json,
                        question=contextual_question,
                        api_key=api_key or "",
                        model_name=model_name,
                    )
                except Exception as exc:
                    answer = f"{T('AI 답변 생성 중 오류가 발생했습니다')}: {exc}"

            st.markdown(_translate_runtime_text(sanitize_llm_markdown(answer)))

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
inject_rendering_guard_css()


def _render_wizard_stepper(current: int, total: int = 6):
    labels = ["모드 선택", "CSV 업로드", "컬럼 매핑", "이벤트 매핑", "이탈 정의", "학습"]
    parts = []
    for i, label in enumerate(labels[:total]):
        if i < current:
            parts.append(f"<span style='color:#10B981;font-weight:700'>● {label}</span>")
        elif i == current:
            parts.append(f"<span style='color:#3B82F6;font-weight:700'>● {label}</span>")
        else:
            parts.append(f"<span style='color:#9CA3AF'>○ {label}</span>")
    st.markdown(
        "<div style='display:flex;gap:8px;align-items:center;margin:12px 0 20px 0;font-size:0.85rem'>"
        + " ─ ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def _wizard_nav(step_key: str, can_next: bool = True, can_prev: bool = True, next_label: str = "다음 →", prev_label: str = "← 이전"):
    col_l, col_r = st.columns(2)
    with col_l:
        if can_prev and st.button(prev_label, key=f"wiz_prev_{step_key}", use_container_width=True):
            st.session_state["wizard_step"] = max(st.session_state.get("wizard_step", 0) - 1, 0)
            st.rerun()
    with col_r:
        if can_next and st.button(next_label, key=f"wiz_next_{step_key}", use_container_width=True, type="primary"):
            st.session_state["wizard_step"] = st.session_state.get("wizard_step", 0) + 1
            st.rerun()


def _render_wizard() -> bool:
    """산업 도메인 선택 → CSV 업로드 → 매핑 → 학습 실행 마법사.

    기존 시뮬레이터/자사 데이터 선택을 제거하고 금융/이커머스 두 모드로 운영한다.
    학습 완료 후 mode/dashboard/view를 URL query parameter에 남겨 F5 새로고침 시
    첫 화면으로 되돌아가지 않도록 한다.
    """
    mode = _business_mode()
    paths = _domain_paths(mode)
    _root = _project_root()
    has_domain_data = (_root / paths["data"] / "customer_summary.csv").exists()
    _result_path = _root / paths["results"]
    has_domain_results = _result_path.exists() and any(_result_path.iterdir())

    if st.session_state.get("wizard_dismissed") or (st.query_params.get("dashboard") == "1" and (has_domain_data or has_domain_results)):
        st.session_state["wizard_dismissed"] = True
        return False

    st.session_state.setdefault("wizard_step", 0)
    step = int(st.session_state.get("wizard_step", 0))

    if step == 0:
        render_step_title(T("분석 모드 선택"), T("어떤 산업 데이터로 분석할지 선택하세요."))
        col_fin, col_ec = st.columns(2)

        with col_fin:
            st.markdown(
                "<div style='border:2px solid #DBEAFE;border-radius:14px;padding:22px;min-height:210px;background:#EFF6FF'>"
                "<div style='font-size:2.2rem'>🏦</div>"
                f"<div style='font-weight:800;margin:8px 0'>{T('금융 모드')}</div>"
                "<div style='font-size:0.88rem;color:#475569;line-height:1.55'>"
                "예금·대출·카드·거래·잔고·연체·상담 이력 기반 이탈/해지 위험과 캠페인 우선순위를 분석합니다."
                "</div></div>",
                unsafe_allow_html=True,
            )
            if st.button(T("금융 모드"), key="wiz_mode_finance", use_container_width=True, type="primary"):
                st.session_state["data_mode"] = "finance"
                st.session_state["domain_mode"] = "finance"
                st.session_state["wizard_step"] = 1
                st.query_params["mode"] = "finance"
                st.query_params["dashboard"] = "0"
                clear_dashboard_caches()
                st.rerun()

        with col_ec:
            st.markdown(
                "<div style='border:2px solid #DCFCE7;border-radius:14px;padding:22px;min-height:210px;background:#F0FDF4'>"
                "<div style='font-size:2.2rem'>🛒</div>"
                f"<div style='font-weight:800;margin:8px 0'>{T('이커머스 모드')}</div>"
                "<div style='font-size:0.88rem;color:#475569;line-height:1.55'>"
                "방문·검색·장바구니·구매·쿠폰·카테고리 선호 기반 이탈 위험과 개인화 추천을 분석합니다."
                "</div></div>",
                unsafe_allow_html=True,
            )
            if st.button(T("이커머스 모드"), key="wiz_mode_ecommerce", use_container_width=True, type="primary"):
                st.session_state["data_mode"] = "ecommerce"
                st.session_state["domain_mode"] = "ecommerce"
                st.session_state["wizard_step"] = 1
                st.query_params["mode"] = "ecommerce"
                st.query_params["dashboard"] = "0"
                clear_dashboard_caches()
                st.rerun()

        if has_domain_data or has_domain_results:
            st.divider()
            st.info(f"{_domain_label(mode)}에 이전 학습 결과가 있습니다.")
            if st.button(f"📊 {T('기존 결과로 대시보드 보기')}", key="wizard_skip_existing", use_container_width=True):
                st.session_state["wizard_dismissed"] = True
                st.query_params["mode"] = mode
                st.query_params["dashboard"] = "1"
                st.query_params["view"] = st.session_state.get("dashboard_view", DASHBOARD_VIEW_OPTIONS[0])
                clear_dashboard_caches()
                st.rerun()
        return True

    # Step 1: CSV 업로드 및 자동 미리보기
    if step == 1:
        import sys
        render_step_title(
            f"Step 2. CSV 업로드 — {_domain_label(mode)}",
            "금융/이커머스 원천 CSV를 업로드하세요. 고객 스냅샷, 거래, 이벤트 로그 형태를 모두 허용합니다.",
        )
        if mode == "finance":
            with st.expander("금융 데이터 권장 컬럼", expanded=False):
                st.markdown("customer_id, timestamp/transaction_date, event_type/transaction_type, balance, transaction_amount, product_type, loan_amount, delinquency_days, credit_score, tenure_months, channel 등")
        else:
            with st.expander("이커머스 데이터 권장 컬럼", expanded=False):
                st.markdown("customer_id, timestamp/event_time, event_type, order_id, order_amount, category, coupon_used, discount_amount, quantity, channel 등")

        uploaded_file = st.file_uploader("CSV/TSV 파일", type=["csv", "tsv"], key=f"wizard_csv_upload_{mode}")
        if uploaded_file is not None:
            import sys
            from src.ingestion.pipeline import prepare_mapping_preview as _prep
            root = _project_root()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            upload_dir = root / "data" / "uploads" / mode
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_path = upload_dir / uploaded_file.name
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["wizard_upload_path"] = str(upload_path)
            st.session_state["wizard_upload_filename"] = uploaded_file.name
            st.session_state["active_dataset_filename"] = uploaded_file.name

            prev_key = f"{mode}:{upload_path}:{upload_path.stat().st_mtime_ns}:{upload_path.stat().st_size}"
            if st.session_state.get("wizard_upload_token") != prev_key:
                st.session_state["wizard_upload_token"] = prev_key
                st.session_state.pop("wizard_mapping_preview", None)
                st.session_state.pop("wizard_column_mapping", None)
                st.session_state.pop("wizard_event_mapping", None)

            if "wizard_mapping_preview" not in st.session_state:
                with st.spinner("CSV 구조를 분석하고 자동 매핑하는 중입니다..."):
                    st.session_state["wizard_mapping_preview"] = _prep(upload_path, domain=mode)
            preview = st.session_state["wizard_mapping_preview"]
            st.success(f"업로드 완료: {uploaded_file.name} / {int(preview.total_rows):,}행")
            if preview.sample_rows is not None and not preview.sample_rows.empty:
                _render_dataframe_with_count(preview.sample_rows.head(10), label="업로드 샘플", height=360)
            _wizard_nav("domain_upload", can_next=True)
        else:
            st.info("분석할 CSV/TSV 파일을 업로드하면 다음 단계로 이동할 수 있습니다.")
            _wizard_nav("domain_upload_empty", can_next=False)
        return True

    # Step 2: 컬럼 매핑
    if step == 2:
        render_step_title("Step 3. 컬럼 매핑 검토")
        preview = st.session_state.get("wizard_mapping_preview")
        if preview is None:
            st.error("업로드 파일을 찾지 못했습니다. 이전 단계로 돌아가세요.")
            _wizard_nav("mapping_missing", can_next=False)
            return True
        all_cols = list(preview.validation.column_report and [c["original_name"] for c in preview.validation.column_report] or list(preview.column_mapping.values()))
        raw_to_display_col, display_to_raw_col = _unique_display_lookup(
            all_cols,
            lambda col: _column_display_label_for_wizard(col, mode),
        )
        options = ["(매핑 안 함)"] + [raw_to_display_col[str(col)] for col in all_cols]
        role_help = {
            "customer_id": "고객을 식별하는 값입니다.",
            "timestamp": "고객 행동이나 거래가 발생한 시각입니다.",
            "event_type": "고객이 한 행동의 종류입니다." if mode != "finance" else "계좌·상품 조회, 금융거래, 상담처럼 고객이 한 금융 행동의 종류입니다.",
            "amount": "주문·결제 금액입니다." if mode != "finance" else "거래금액, 잔액, 대출잔액처럼 분석에 사용할 금액성 값입니다.",
            "category": "상품 카테고리입니다." if mode != "finance" else "예금, 대출, 카드, 보험 같은 금융상품 유형입니다.",
            "churn_flag": "고객이 이탈했는지를 나타내는 라벨입니다." if mode != "finance" else "해지, 휴면, 연체 등 금융 이탈 상태를 나타내는 라벨입니다.",
            "persona": "고객 유형입니다." if mode != "finance" else "금융 고객 유형입니다.",
            "quantity": "상품 수량입니다." if mode != "finance" else "보유하거나 거래한 금융상품 수입니다.",
            "region": "고객 지역입니다.",
        }
        role_keys = sorted(set(list(preview.column_mapping.keys()) + ["customer_id", "timestamp", "event_type", "amount"]))
        rows = []
        for role in role_keys:
            detected = preview.column_mapping.get(role)
            detected_display = raw_to_display_col.get(str(detected), "(매핑 안 함)") if detected in all_cols else "(매핑 안 함)"
            rows.append({
                "시스템 역할": _schema_role_label_for_wizard(role, mode),
                "업로드 컬럼": detected_display,
                "설명": role_help.get(role, "분석 피처로 사용할 수 있는 컬럼입니다."),
            })
        editor_df = pd.DataFrame(rows)
        edited = st.data_editor(
            editor_df,
            use_container_width=True,
            hide_index=True,
            disabled=["시스템 역할", "설명"],
            column_order=["시스템 역할", "업로드 컬럼", "설명"],
            column_config={
                "시스템 역할": st.column_config.TextColumn("시스템 역할"),
                "업로드 컬럼": st.column_config.SelectboxColumn("업로드 컬럼", options=options, required=True),
                "설명": st.column_config.TextColumn("설명"),
            },
            key=f"wizard_col_map_editor_{mode}",
        )
        mapping = {}
        for row_pos, (_, r) in enumerate(edited.iterrows()):
            selected_display = str(r["업로드 컬럼"])
            if selected_display != "(매핑 안 함)":
                role_key = role_keys[row_pos] if row_pos < len(role_keys) else None
                raw_col = display_to_raw_col.get(selected_display, selected_display)
                if role_key:
                    mapping[str(role_key)] = str(raw_col)
        st.session_state["wizard_column_mapping"] = mapping
        _wizard_nav("domain_mapping", can_next=bool(mapping.get("customer_id")))
        return True

    # Step 3: 이벤트/거래 값 매핑
    if step == 3:
        render_step_title("Step 4. 이벤트·거래 타입 매핑")
        preview = st.session_state.get("wizard_mapping_preview")
        if preview is None:
            st.error("업로드 파일을 찾지 못했습니다.")
            _wizard_nav("event_missing", can_next=False)
            return True
        if preview.has_event_data and preview.event_value_mapping:
            from src.ingestion.preprocessor import INTERNAL_EVENT_TYPES as _STD
            std_values = list(_STD) + ["other", "ignore"]
            std_raw_to_display, std_display_to_raw = _unique_display_lookup(
                std_values,
                lambda value: _event_display_label_for_wizard(value, mode, internal=True),
            )
            raw_values = [raw for raw, _std in sorted(preview.event_value_mapping.items(), key=lambda x: -preview.event_value_counts.get(x[0], 0))]
            raw_to_display_event, display_to_raw_event = _unique_display_lookup(
                raw_values,
                lambda value: _event_display_label_for_wizard(value, mode, internal=False),
            )
            std_options = [std_raw_to_display[str(value)] for value in std_values]
            e_rows = []
            for raw in raw_values:
                std = preview.event_value_mapping.get(raw)
                e_rows.append({
                    "원본 값": raw_to_display_event.get(str(raw), str(raw)),
                    "빈도": preview.event_value_counts.get(raw, 0),
                    "내부 표준 값": std_raw_to_display.get(str(std), std_raw_to_display.get("other", "기타")),
                })
            edited_ev = st.data_editor(
                pd.DataFrame(e_rows),
                use_container_width=True,
                hide_index=True,
                disabled=["원본 값", "빈도"],
                column_config={
                    "원본 값": st.column_config.TextColumn("원본 값"),
                    "빈도": st.column_config.NumberColumn("빈도", format="%d"),
                    "내부 표준 값": st.column_config.SelectboxColumn("내부 표준 값", options=std_options, required=True),
                },
                key=f"wizard_ev_map_editor_{mode}",
            )
            mapped_events = {}
            for _, r in edited_ev.iterrows():
                raw_event = display_to_raw_event.get(str(r["원본 값"]), str(r["원본 값"]))
                std_event = std_display_to_raw.get(str(r["내부 표준 값"]), str(r["내부 표준 값"]))
                mapped_events[str(raw_event)] = str(std_event)
            st.session_state["wizard_event_mapping"] = mapped_events
            st.session_state["wizard_synthetic_fallback"] = False
            st.info(f"자동 매핑 커버리지: {float(preview.coverage_rate):.0%}")
        else:
            st.warning("event_type/timestamp 조합이 부족합니다. 스냅샷 데이터로 진행하면 일부 실시간·행동 시계열 분석은 제한됩니다.")
            st.session_state["wizard_event_mapping"] = None
            st.session_state["wizard_synthetic_fallback"] = st.checkbox("스냅샷 데이터로 진행", value=True, key=f"wizard_synthetic_{mode}")
        _wizard_nav("domain_event", can_next=(preview.has_event_data or st.session_state.get("wizard_synthetic_fallback", False)))
        return True

    # Step 4: 이탈 기준과 학습
    if step >= 4:
        render_step_title(T("Step 5. 이탈 기준·학습"))
        preview = st.session_state.get("wizard_mapping_preview")
        if preview is None:
            st.error("업로드 파일을 찾지 못했습니다.")
            _wizard_nav("train_missing", can_next=False)
            return True
        recommended = int(getattr(preview, "recommended_churn_days", None) or (60 if mode == "finance" else 30))
        st.info(f"**{T('이탈 기준 설정 안내')}**  \n{T('이 슬라이더는 고객을 언제부터 이탈로 볼지 정하는 기준입니다. 예를 들어 30일로 두면 마지막 활동 후 30일 이상 지난 고객을 이탈 사례로 학습합니다.')}  \n{T('이 기준은 이탈 모델 학습, 생존분석, 이탈 시점 예측의 기준이 됩니다. 업종별 방문·구매 주기에 맞게 조절하세요.')}")
        churn_days = st.slider(T("이탈 기준: N일 이상 비활성"), 7, 180, recommended, 1, key=f"wizard_churn_days_{mode}")
        w_budget = int(st.session_state.get("control_budget", 5_000_000))
        w_threshold = float(st.session_state.get("control_threshold", 0.50))
        w_cap = int(st.session_state.get("control_target_cap", 1500))
        st.info(T("학습 단계에서는 예산과 이탈 임계값을 조절하지 않습니다. 학습이 끝난 뒤 대시보드의 분석 컨트롤에서 운영 조건을 바꿔 비교하세요."))

        st.caption(f"{T('학습 대상')}: {_domain_label(mode)} / {T('파일')}: {st.session_state.get('wizard_upload_filename', '-')}")
        if st.button(f"🚀 {T('학습 시작')}", key=f"wizard_train_{mode}", use_container_width=True, type="primary"):
            from src.ingestion.pipeline import run_ingestion_pipeline as _run_pipeline
            import threading, time as _t
            root = _project_root()
            paths = _domain_paths(mode)
            upload_path = st.session_state.get("wizard_upload_path")
            filename = st.session_state.get("wizard_upload_filename", Path(str(upload_path)).name)
            progress_bar = st.progress(0, text=T("시작 중..."))
            holder: dict[str, Any] = {}

            def _train():
                try:
                    holder["result"] = _run_pipeline(
                        file_path=upload_path,
                        data_dir=root / paths["data"],
                        model_dir=root / paths["models"],
                        result_dir=root / paths["results"],
                        feature_store_dir=root / paths["features"],
                        budget=int(w_budget),
                        threshold=float(w_threshold),
                        max_customers=int(w_cap),
                        backup_existing=True,
                        column_mapping_override=st.session_state.get("wizard_column_mapping") or None,
                        event_value_mapping=st.session_state.get("wizard_event_mapping"),
                        allow_synthetic_fallback=bool(st.session_state.get("wizard_synthetic_fallback", False)),
                        churn_inactivity_days=int(churn_days),
                        domain=mode,
                    )
                except Exception as exc:
                    holder["error"] = exc

            th = threading.Thread(target=_train, daemon=True)
            th.start()
            msgs = [T("CSV 검증"), T("전처리"), T("피처 생성"), T("이탈 모델 학습"), T("Uplift/CLV 계산"), T("예산 최적화"), T("추천/설명 생성")]
            start_time = _t.time()
            while th.is_alive():
                elapsed = _t.time() - start_time
                progress_bar.progress(min(95, max(5, int(elapsed * 2))), text=f"{msgs[min(int(elapsed // 12), len(msgs)-1)]} 중... ({int(elapsed)}초)")
                _t.sleep(0.5)
            th.join()
            if "error" in holder:
                progress_bar.progress(100, text="오류")
                st.error(f"{T('학습 실패')}: {holder['error']}")
            else:
                result = holder["result"]
                if result.success:
                    _save_dataset_metadata(mode, filename=filename, upload_path=str(upload_path), row_count=int(getattr(preview, "total_rows", 0) or 0))

                    live_seed_result = None
                    live_seed_error = None
                    live_sync_report = None
                    try:
                        progress_bar.progress(96, text="PostgreSQL user-live 테이블 초기 적재 준비 중...")
                        live_sync_report = _sync_domain_artifacts_for_live_seed(mode)
                        live_seed_result = seed_user_live_from_artifacts(reset=True)
                        _save_live_seed_metadata(mode, live_seed_result, live_sync_report)
                        st.session_state["user_live_seed_result"] = live_seed_result
                        st.session_state.pop("user_live_seed_error", None)
                    except Exception as seed_exc:
                        live_seed_error = seed_exc
                        st.session_state["user_live_seed_error"] = str(seed_exc)

                    progress_bar.progress(100, text="완료")
                    st.success(T("학습 완료. 대시보드로 이동합니다."))
                    if isinstance(live_seed_result, dict) and live_seed_result.get("success"):
                        st.success(T("PostgreSQL user-live DB 초기 적재 완료"))
                    elif live_seed_error is not None:
                        st.warning(f"{T('PostgreSQL user-live DB 자동 적재 실패')}: {live_seed_error}")
                    else:
                        st.warning(T("PostgreSQL user-live DB 자동 적재 실패"))
                    st.session_state["wizard_dismissed"] = True
                    st.session_state["data_mode"] = mode
                    st.session_state["active_dataset_filename"] = filename
                    # 학습 단계에서는 예산/임계값을 사용자가 조절하지 않는다. 대시보드 분석 컨트롤의 기존 값을 유지한다.
                    st.session_state["control_target_cap"] = int(w_cap)
                    st.session_state["dashboard_view"] = "1. 이탈현황"
                    st.query_params["mode"] = mode
                    st.query_params["dashboard"] = "1"
                    st.query_params["view"] = "1. 이탈현황"
                    clear_dashboard_caches()
                    _t.sleep(1)
                    st.rerun()
                else:
                    progress_bar.progress(100, text="부분 완료")
                    st.warning(f"일부 단계 실패: {result.error or '산출물을 확인하세요.'}")
        if st.button("← 이전 단계로", key=f"wizard_train_prev_{mode}"):
            st.session_state["wizard_step"] = 3
            st.rerun()
        return True

    return False


CONTROL_DEFAULTS = {
    "control_threshold": 0.50,
    "control_budget": 5_000_000,
    "control_top_n": 25,
    "control_target_cap": 1500,
    "control_recommendation_per_customer": 3,
}
for _state_key, _state_value in CONTROL_DEFAULTS.items():
    st.session_state.setdefault(_state_key, _state_value)


def _get_control_value(*keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in st.session_state:
            return st.session_state.get(key)
    return default


def _snapshot_analysis_controls() -> None:
    """Persist analysis controls into non-widget shadow keys before language-only reruns.

    Streamlit may drop a widget key when an early st.rerun() happens before the widget is
    rendered. The shadow keys below are never used as widget keys, so language switching
    cannot reset threshold/budget/cap/top_n to widget defaults.
    """
    st.session_state["control_threshold_shadow"] = float(
        _get_control_value("control_threshold_widget", "control_threshold", "control_threshold_shadow", default=CONTROL_DEFAULTS["control_threshold"])
    )
    st.session_state["control_budget_shadow"] = int(
        _get_control_value("control_budget", "control_budget_shadow", default=CONTROL_DEFAULTS["control_budget"])
    )
    st.session_state["control_target_cap_shadow"] = int(
        _get_control_value("control_target_cap", "control_target_cap_shadow", default=CONTROL_DEFAULTS["control_target_cap"])
    )
    st.session_state["control_top_n_shadow"] = int(
        _get_control_value("control_top_n", "control_top_n_shadow", default=CONTROL_DEFAULTS["control_top_n"])
    )
    st.session_state["control_recommendation_per_customer_shadow"] = int(
        _get_control_value("control_recommendation_per_customer", "control_recommendation_per_customer_shadow", default=CONTROL_DEFAULTS["control_recommendation_per_customer"])
    )


def _restore_analysis_controls_from_shadow() -> None:
    if "control_threshold_shadow" in st.session_state and "control_threshold_widget" not in st.session_state:
        st.session_state["control_threshold_widget"] = float(st.session_state["control_threshold_shadow"])
    if "control_threshold_shadow" in st.session_state:
        st.session_state["control_threshold"] = float(st.session_state["control_threshold_shadow"])
    if "control_budget_shadow" in st.session_state:
        st.session_state["control_budget"] = int(st.session_state["control_budget_shadow"])
        st.session_state.setdefault("control_budget_text", str(int(st.session_state["control_budget_shadow"])))
    if "control_target_cap_shadow" in st.session_state:
        st.session_state["control_target_cap"] = int(st.session_state["control_target_cap_shadow"])
        st.session_state.setdefault("control_target_cap_text", str(int(st.session_state["control_target_cap_shadow"])))
    if "control_top_n_shadow" in st.session_state:
        st.session_state["control_top_n"] = int(st.session_state["control_top_n_shadow"])
    if "control_recommendation_per_customer_shadow" in st.session_state:
        st.session_state["control_recommendation_per_customer"] = int(st.session_state["control_recommendation_per_customer_shadow"])


def _init_url_state() -> None:
    """URL query parameter를 최초 1회만 session_state로 복원한다.

    이전 구현은 매 rerun마다 URL의 old view 값을 session_state에 다시 덮어써서,
    사용자가 radio에서 2/3/4번 화면을 눌러도 다음 rerun 시작 시 다시 1번으로
    회귀하는 현상이 발생했다.
    """
    try:
        qp = st.query_params
    except Exception:
        return

    already_initialized = bool(st.session_state.get("_url_state_initialized"))

    lang = qp.get("lang")
    if not already_initialized and lang in LANGUAGE_LABEL_BY_CODE:
        st.session_state["language_code"] = lang
    else:
        st.session_state.setdefault("language_code", "ko")

    mode = qp.get("mode")
    if not already_initialized and mode in {"finance", "ecommerce", "user"}:
        st.session_state["data_mode"] = mode
        st.session_state["domain_mode"] = mode
    else:
        st.session_state.setdefault("data_mode", "ecommerce")
        st.session_state.setdefault("domain_mode", st.session_state.get("data_mode", "ecommerce"))

    view_q = qp.get("view")
    if not already_initialized and view_q:
        view_q = LEGACY_VIEW_REDIRECTS.get(view_q, view_q)
        if view_q in DASHBOARD_VIEW_OPTIONS:
            st.session_state["dashboard_view"] = view_q

    if not already_initialized and qp.get("dashboard") == "1":
        st.session_state["wizard_dismissed"] = True

    st.session_state["_url_state_initialized"] = True


_init_url_state()

bundle = load_app_data()

customers = bundle.customer_summary
cohort_df = bundle.cohort_retention

render_hero(
    T("고객 이탈 예측·개입 최적화·ROI 분석 플랫폼"),
    T("누가 이탈할 가능성이 높은지뿐 아니라, 언제 개입해야 하는지, 누구에게 예산을 우선 배분할지, 어떤 액션을 추천할지까지 연결해 보여주는 운영형 리텐션 분석 플랫폼입니다."),
)

if bundle.used_mock:
    render_status_pill("실제 data/raw 산출물을 찾지 못해 mock data로 실행 중입니다.", "warn")

_wizard_active = _render_wizard()

with st.sidebar:
    st.header(T("제어 패널"))
    _current_lang_label = LANGUAGE_LABEL_BY_CODE.get(st.session_state.get("language_code", "ko"), "한국어")
    _selected_lang_label = st.selectbox(T("언어"), options=list(LANGUAGE_OPTIONS.keys()), index=list(LANGUAGE_OPTIONS.keys()).index(_current_lang_label), key="language_selector")
    _new_lang_code = LANGUAGE_OPTIONS[_selected_lang_label]
    if _new_lang_code != st.session_state.get("language_code"):
        # Preserve analysis controls before language-only rerun.
        _snapshot_analysis_controls()
        st.session_state["language_code"] = _new_lang_code
        _set_query_param_if_changed("lang", _new_lang_code)
        st.rerun()

    if _wizard_active:
        uploaded_file = None
        selected_mode = _business_mode()
    else:
        if st.button(f"🏠 {T('모드/데이터셋 변경')}", key="reset_wizard_btn", use_container_width=True):
            st.session_state["wizard_dismissed"] = False
            st.session_state["wizard_step"] = 0
            st.session_state.pop("wizard_mapping_preview", None)
            st.query_params["dashboard"] = "0"
            st.rerun()

        selected_mode = _business_mode()
        st.subheader(T("현재 분석 모드"))
        st.caption(_domain_label(selected_mode))
        _meta = _load_dataset_metadata(selected_mode)
        _dataset_name = st.session_state.get("active_dataset_filename") or _meta.get("filename") or T("미선택")
        st.subheader(T("사용 데이터셋"))
        st.caption(str(_dataset_name))

        uploaded_file = None

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
            st.session_state.pop("churn_inactivity_days", None)

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

            st.markdown(f"### ⚙️ {T('학습 설정')}")
            upload_budget = int(st.session_state.get("control_budget", 5_000_000))
            upload_threshold = float(st.session_state.get("control_threshold", 0.50))
            st.info(T("학습 단계에서는 예산과 이탈 임계값을 조절하지 않습니다. 학습이 끝난 뒤 대시보드의 분석 컨트롤에서 운영 조건을 바꿔 비교하세요."))

            st.markdown(f"### 📛 {T('이탈 고객 정의')}")
            recommended_churn_days = int(getattr(preview, "recommended_churn_days", None) or 30)
            st.info(f"**{T('이탈 기준 설정 안내')}**  \n{T('이 슬라이더는 고객을 언제부터 이탈로 볼지 정하는 기준입니다. 예를 들어 30일로 두면 마지막 활동 후 30일 이상 지난 고객을 이탈 사례로 학습합니다.')}  \n{T('이 기준은 이탈 모델 학습, 생존분석, 이탈 시점 예측의 기준이 됩니다. 업종별 방문·구매 주기에 맞게 조절하세요.')}")
            if getattr(preview, "recommended_churn_days", None):
                st.info(
                    f"업로드 데이터의 평균 활동/구매 주기를 기준으로 "
                    f"**{recommended_churn_days}일**을 추천합니다."
                )
            churn_inactivity_days = st.slider(
                T("이탈 기준: N일 이상 비활성"),
                min_value=7,
                max_value=180,
                value=recommended_churn_days,
                step=1,
                key="churn_inactivity_days",
                help=(
                    "**서비스 성격별 권장 기준:**\n\n"
                    "- **7~14일:** 데일리 앱 (게임, SNS)\n"
                    "- **30일:** 일반 커머스, 라이프스타일\n\n"
                    "- **60~90일:** 정기 구독 서비스 (OTT, 멤버십)\n\n"
                    "설정한 기간 동안 접속 기록이 없으면 '이탈'로 간주합니다."
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

                progress_bar = st.progress(0, text=T("시작 중..."))
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
                            live_sync_report = _sync_domain_artifacts_for_live_seed(_business_mode())
                            live_seed_result = seed_user_live_from_artifacts(reset=True)
                            _save_live_seed_metadata(_business_mode(), live_seed_result, live_sync_report)
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

    if _wizard_active:
        st.stop()

    _restore_analysis_controls_from_shadow()

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

selected_group = "핵심 화면"
group_options = list(DASHBOARD_VIEW_OPTIONS)
if st.session_state.get("dashboard_view") not in group_options:
    st.session_state["dashboard_view"] = group_options[0]

view = st.radio(
    f"📌 {T('분석 화면')}",
    options=group_options,
    format_func=_view_title_from_option,
    horizontal=True,
    key="dashboard_view",
)
_set_query_param_if_changed("view", view)
_set_query_param_if_changed("mode", _business_mode())
_set_query_param_if_changed("dashboard", "1" if st.session_state.get("wizard_dismissed") else st.query_params.get("dashboard", "0"))
_set_query_param_if_changed("lang", st.session_state.get("language_code", "ko"))

with st.sidebar:
    st.divider()
    st.markdown(f"#### ⚙️ {T('분석 컨트롤')}")

    if "control_threshold_widget" not in st.session_state:
        st.session_state["control_threshold_widget"] = float(
            st.session_state.get("control_threshold", CONTROL_DEFAULTS["control_threshold"])
        )
    threshold = st.slider(
        T("이탈 임계값"),
        min_value=0.10,
        max_value=0.90,
        step=0.01,
        key="control_threshold_widget",
        help=T("이 값 이상인 고객을 이탈 위험군으로 간주합니다. 모든 화면에서 동일하게 유지됩니다."),
    )
    st.session_state["control_threshold"] = float(threshold)
    st.session_state["control_threshold_shadow"] = float(threshold)

    budget_raw = st.text_input(
        T("총 마케팅 예산"),
        key="control_budget_text",
        help="상한 없이 입력 가능합니다. 쉼표 없이 숫자만 입력해도 됩니다.",
    )

    try:
        budget = parse_unlimited_nonnegative_int(
        budget_raw,
        default=int(st.session_state.get("control_budget", 5_000_000)),
    )
        st.session_state["control_budget"] = budget
        st.session_state["control_budget_shadow"] = int(budget)
    except ValueError:
        st.warning("총 마케팅 예산은 0 이상의 정수로 입력해야 합니다.")
        budget = int(st.session_state.get("control_budget", 5_000_000))
    
    if "control_target_cap_text" not in st.session_state:
        st.session_state["control_target_cap_text"] = str(
            int(st.session_state.get("control_target_cap", 1500))
        )

    target_cap_raw = st.text_input(
        T("최대 타겟 고객 수"),
        key="control_target_cap_text",
        help="상한 없이 입력 가능합니다. 1 이상의 정수만 입력하세요.",
    )

    try:
        target_cap = parse_unlimited_nonnegative_int(
            target_cap_raw,
            default=int(st.session_state.get("control_target_cap", 1500)),
        )
        if target_cap <= 0:
            raise ValueError("최대 타겟 고객 수는 1 이상의 정수여야 합니다.")
        st.session_state["control_target_cap"] = target_cap
        st.session_state["control_target_cap_shadow"] = int(target_cap)
    except ValueError:
        st.warning("최대 타겟 고객 수는 1 이상의 정수로 입력해야 합니다.")
        target_cap = int(st.session_state.get("control_target_cap", 1500))

    # top_n은 실시간/설명가능성/리스크 화면에서 쓰는 표시 개수입니다.
    # 이탈 시점 예측 화면은 별도의 30일 이탈 가능성 필터로 전체 고객을 걸러 보여주므로 여기서는 숨깁니다.
    if _is_churn_timing_view(view):
        top_n = int(st.session_state.get("control_top_n", CONTROL_DEFAULTS["control_top_n"]))
    else:
        top_n = st.slider(
            T("표시 고객 수"),
            min_value=5,
            max_value=200,
            step=5,
            key="control_top_n",
        )
        st.session_state["control_top_n_shadow"] = int(top_n)

    if view == "5. 개인화 추천":
        st.caption("최종 리텐션 타겟 고객군(예산/임계값 적용)에게만 추천을 생성합니다.")
        if "control_recommendation_per_customer_widget" not in st.session_state:
            st.session_state["control_recommendation_per_customer_widget"] = int(
                st.session_state.get("control_recommendation_per_customer", CONTROL_DEFAULTS["control_recommendation_per_customer"])
            )
        recommendation_per_customer = st.slider(
            T("고객당 추천 개수"),
            min_value=1,
            max_value=5,
            step=1,
            key="control_recommendation_per_customer_widget",
        )
        st.session_state["control_recommendation_per_customer"] = int(recommendation_per_customer)
    else:
        recommendation_per_customer = int(st.session_state.get("control_recommendation_per_customer", CONTROL_DEFAULTS["control_recommendation_per_customer"]))

    preview_selected_customers, preview_optimize_summary, preview_segment_allocation = get_budget_result(
        customers,
        budget=budget,
        threshold=threshold,
        max_customers=target_cap,
    )
    st.session_state["_last_preview_budget_key"] = (
        _business_mode(), float(threshold), int(budget), int(target_cap), _raw_data_token(_business_mode())
    )
    st.session_state["_last_preview_selected_customers"] = preview_selected_customers
    st.session_state["_last_preview_optimize_summary"] = preview_optimize_summary
    st.session_state["_last_preview_segment_allocation"] = preview_segment_allocation
    st.caption(
        f"현재 공통 조건: threshold={float(threshold):.2f} / "
        f"예산={int(budget):,}원 / 최종 타겟 고객 수={int(len(preview_selected_customers)):,}명"
    )

with st.sidebar:
    st.divider()
    st.subheader(T("실행 / 새로고침"))
    if notice := st.session_state.pop("dashboard_refresh_notice", None):
        st.success(notice)
    if warning := st.session_state.pop("dashboard_refresh_warning", None):
        st.warning(warning)

    if st.button(T("데이터/결과 새로고침"), use_container_width=True):
        refresh_notice = None
        refresh_warning = None
        if view in REALTIME_REFRESH_VIEWS and not _is_user_live_mode():
            try:
                tick_payload = advance_realtime_stream(batch_size=250, top_n=max(int(top_n), 50), reset_when_exhausted=True)
                tick_summary = tick_payload.get("summary", {}) if isinstance(tick_payload, dict) else {}
                refresh_notice = (
                    f"실시간 스냅샷을 {int(tick_summary.get('last_tick_advanced', 0) or 0):,}건 갱신했습니다. "
                    f"누적 처리 이벤트 수: {int(tick_summary.get('processed_events', 0) or 0):,}건"
                )
            except Exception as exc:
                refresh_warning = f"실시간 갱신 호출에는 실패했지만 화면 캐시는 새로고침했습니다: {exc}"
        clear_dashboard_caches()
        clear_llm_caches()
        if refresh_notice:
            st.session_state["dashboard_refresh_notice"] = refresh_notice
        if refresh_warning:
            st.session_state["dashboard_refresh_warning"] = refresh_warning
        st.rerun()

    st.caption(T("실시간 화면에서는 새로고침 시 최신 DB/캐시 상태를 다시 읽습니다. 나머지 화면도 캐시를 비우고 다시 계산합니다."))

    st.divider()
    st.subheader(T("LLM 설정"))
    st.caption(T("권장: API 키는 코드에 쓰지 말고 환경변수 OPENAI_API_KEY 또는 Streamlit secrets로 관리하세요."))

    llm_enabled = st.toggle(
        T("LLM 요약/질문 기능 사용"),
        value=bool(os.getenv("OPENAI_API_KEY")),
        key="llm_enabled",
    )
    llm_api_key = st.text_input(
        T("OpenAI API Key (선택)"),
        type="password",
        help=T("비워두면 OPENAI_API_KEY 환경변수를 사용합니다."),
    )
    st.caption(T("모델이 목록에 없으면 '직접 입력'을 선택해서 모델명을 넣어주세요."))
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
    llm_model_choice = st.selectbox(T("LLM 모델 선택"), options=_llm_preset_labels, index=_llm_preset_labels.index(_default_label))
    _chosen_model = _llm_preset_models.get(llm_model_choice, DEFAULT_MODEL_NAME)
    if _chosen_model == "__custom__":
        llm_model = st.text_input(T("LLM 모델명 (직접 입력)"), value=DEFAULT_MODEL_NAME)
    else:
        llm_model = _chosen_model

    env_key_configured = bool(os.getenv("OPENAI_API_KEY"))
    if env_key_configured and not llm_api_key:
        st.caption(T("현재 OPENAI_API_KEY 환경변수를 사용하도록 설정되어 있습니다."))

live_payload = _load_user_live_tables(
    top_n=int(top_n),
    target_cap=int(target_cap),
    threshold=float(threshold),
    view=view,
)

_use_live_payload = _live_payload_matches_current_dataset(live_payload, customers)

if _is_user_live_mode():
    _render_user_live_status(live_payload)
    if not _use_live_payload and not live_payload.get("scores", pd.DataFrame()).empty:
        st.info(T("현재 데이터셋과 Live DB가 일치하지 않아 CSV/결과 파일 기준으로 표시합니다."))

    # 시연 실행 중에는 어느 뷰에 있든 최신 event/score/action 지표를 보여줘야 한다.
    # 브라우저 location.reload()는 Streamlit 세션을 새로 만들어 사이드바 분석 컨트롤을
    # 기본값으로 되돌릴 수 있다. 따라서 세션을 유지하는 st.rerun() 방식으로만 갱신한다.
    try:
        _global_demo_status = fetch_demo_status()
    except Exception:
        _global_demo_status = {}
    _global_demo_autorefresh_active = bool(_global_demo_status.get("running")) and view != "6. 실시간 운영 모니터"
    if _global_demo_autorefresh_active:
        st.caption(T("시연 실행 중: 10초마다 live 지표를 자동 갱신합니다."))
else:
    _global_demo_autorefresh_active = False

if _use_live_payload and not live_payload.get("scores", pd.DataFrame()).empty:
    customers = _rename_live_score_columns(live_payload["scores"])

churn_summary, risk_customers = get_churn_status(customers, threshold)
if _use_live_payload:
    _score_summary = live_payload.get("score_summary", {}) or {}
    _total_live = int(_score_summary.get("scored_customers") or churn_summary.get("total_customers", 0) or 0)
    _risk_live = int(_score_summary.get("high_risk_customers") or 0)
    churn_summary.update({
        "total_customers": _total_live,
        "at_risk_customers": _risk_live,
        "risk_rate": float(_risk_live / max(_total_live, 1)),
        "avg_churn_prob": float(_score_summary.get("avg_churn_score") or churn_summary.get("avg_churn_prob", 0.0) or 0.0),
    })
cohort_curve = pd.DataFrame()
top_customers = pd.DataFrame()
if view == "2. 코호트 리텐션 분석":
    cohort_curve = get_cohort_curve(cohort_df)
if view == "3. Uplift + CLV 상위 고객":
    top_customers = get_top_high_value_customers(customers, top_n=None)

if _use_live_payload:
    # Budget view must be recomputed from the current sidebar controls.
    # action_queue is an operational queue generated at event time; using it as
    # the only source makes spend/target count look fixed when the user changes
    # the budget.  Prefer the current live score table and use action_queue only
    # as a fallback for older seeded deployments where score columns are sparse.
    score_selected, score_summary, score_allocation = _build_score_based_live_budget_payload(
        live_payload.get("scores", pd.DataFrame()),
        budget=budget,
        threshold=threshold,
        max_customers=target_cap,
    )
    action_selected, action_summary, action_allocation = (pd.DataFrame(), {}, pd.DataFrame())
    if not live_payload.get("actions", pd.DataFrame()).empty:
        action_selected, action_summary, action_allocation = _build_live_optimize_payload(
            live_payload["actions"],
            budget=budget,
            threshold=threshold,
            max_customers=target_cap,
            scores_df=live_payload.get("scores", pd.DataFrame()),
        )

    if not score_selected.empty:
        selected_customers, optimize_summary, segment_allocation = score_selected, score_summary, score_allocation
        optimize_summary = dict(optimize_summary or {})
        optimize_summary.setdefault("source", "postgresql_user_live_score_reoptimized_current_controls")
        optimize_summary["action_queue_candidate_customers"] = int(
            action_summary.get("candidate_customers", 0) if isinstance(action_summary, dict) else 0
        )
    else:
        selected_customers, optimize_summary, segment_allocation = action_selected, action_summary, action_allocation
else:
    _preview_key = (
        _business_mode(), float(threshold), int(budget), int(target_cap), _raw_data_token(_business_mode())
    )
    if st.session_state.get("_last_preview_budget_key") == _preview_key:
        selected_customers = st.session_state.get("_last_preview_selected_customers", pd.DataFrame())
        optimize_summary = st.session_state.get("_last_preview_optimize_summary", {})
        segment_allocation = st.session_state.get("_last_preview_segment_allocation", pd.DataFrame())
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

retention_targets = pd.DataFrame()

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
    if _use_live_payload:
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
        if _is_user_live_mode():
            try:
                _bundle = load_insight_data()
                realtime_scores = _bundle.realtime_scores.copy().head(max(int(top_n), 500))
                realtime_summary = {
                    "tracked_customers": int(len(realtime_scores)),
                    "high_risk_customers": int((pd.to_numeric(realtime_scores.get("realtime_churn_score", realtime_scores.get("churn_score", pd.Series(dtype=float))), errors="coerce") >= threshold).sum()) if not realtime_scores.empty else 0,
                    "processed_events": 0,
                    "source": "current_mode_result_files",
                }
                realtime_error = None
            except Exception as exc:
                realtime_summary, realtime_scores = {}, pd.DataFrame()
                realtime_error = str(exc)
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

if _is_churn_timing_view(view) or view == "13. 고객별 대응 전략 비교":
    if _business_mode() in BUSINESS_UPLOAD_MODES:
        _mode_result_dir = Path(_resolve_result_dir_for_mode(_business_mode()))
        _bundle = load_insight_data()
        survival_metrics = {}
        _metrics_path = _mode_result_dir / "survival_metrics.json"
        if _metrics_path.exists():
            try:
                survival_metrics = json.loads(_metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                survival_metrics = {}
        survival_predictions = _bundle.survival_predictions.copy()
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

if view in INSIGHT_HEAVY_VIEWS and not (view == "6. 실시간 운영 모니터" and _use_live_payload):
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

    if view in {"4. 예산 최적화 및 리텐션 타겟", "11. 설명가능성 / 고객별 개입 이유"}:
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
c1.metric(T("전체 고객 수"), f"{churn_summary['total_customers']:,}")
c2.metric(T("이탈 위험 고객 수"), f"{churn_summary['at_risk_customers']:,}")
c3.metric(T("위험 고객 비율"), pct(churn_summary["risk_rate"]))
c4.metric(T("평균 이탈 확률"), pct(churn_summary["avg_churn_prob"]))

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
    st.subheader(T("이탈 현황"))
    _render_view_intro("1")

    hist_fig = px.histogram(
        customers,
        x="churn_probability",
        nbins=30,
        title="고객별 이탈 확률 분포" if _language_code() == "ko" else ("Customer Churn Probability Distribution" if _language_code() == "en" else "顧客別離脱確率分布"),
    )
    hist_fig.update_traces(
        marker_line_color="rgba(255,255,255,0.95)",
        marker_line_width=1.2,
        opacity=0.9,
    )
    hist_fig.update_layout(bargap=0.02)
    hist_fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"{T('이탈 기준값')}={threshold:.2f}")
    st.plotly_chart(hist_fig, use_container_width=True)

    # 페르소나별 그래프는 해커톤 발표용 핵심 화면 단순화를 위해 제거했다.
    # 단, LLM 요약과 내부 해석에는 사용할 수 있도록 집계값은 유지한다.
    persona_risk = (
        risk_customers.groupby("persona", as_index=False)
        .agg(at_risk_count=("customer_id", "count"))
        .sort_values("at_risk_count", ascending=False)
    ) if "persona" in risk_customers.columns and not risk_customers.empty else pd.DataFrame()

    st.markdown(f"### {T('이탈 위험 고객 목록')}")
    display_df = risk_customers[
        ["customer_id", "persona", "churn_probability", "clv", "uplift_score", "uplift_segment"]
    ].copy()
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.3f}")
    display_df["clv"] = display_df["clv"].map(money)
    display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{x:.3f}")
    _render_dataframe_with_count(display_df, label=T("이탈 위험 고객 목록"))

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
        line_fig.update_layout(xaxis_title=T("경과 기간(개월)"), yaxis_title=T("리텐션율"))
        st.plotly_chart(line_fig, use_container_width=True)

        if not heatmap_df.empty:
            heatmap_fig = px.imshow(
                heatmap_df,
                text_auto=".0%",
                aspect="auto",
                labels={"x": T("경과 기간(개월)"), "y": T("코호트"), "color": T("리텐션율")},
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
        "CSV를 업로드해 학습을 실행한 뒤 새로고침하세요.",
    ):
        st.stop()
    st.subheader(T("예산 최적화 및 리텐션 타겟"))
    _render_view_intro("4")
    st.caption(T("예산 배분 후보, 최종 선정 고객, 고객별 선택 이유만 남긴 핵심 운영 화면입니다."))
    st.markdown(budget_formula_html(_language_code()), unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(T("총 예산"), money(optimize_summary.get("budget", budget)))
    m2.metric(T("집행 예산"), money(optimize_summary.get("spent", 0)))
    m3.metric(T("잔여 예산"), money(optimize_summary.get("remaining", 0)))
    m4.metric(T("타겟 고객 수"), f"{int(optimize_summary.get('num_targeted', len(selected_customers))):,}")
    m5.metric(T("예상 증분 이익"), money(optimize_summary.get("expected_incremental_profit", 0)))

    st.markdown("### 예산 민감도 지도")
    st.caption("예산을 100만 원 단위로 조정했을 때 타깃 고객 수, 기대 순이익, 평균 ROI, 한계 ROI가 어떻게 바뀌는지 표로 비교합니다.")
    try:
        budget_sensitivity_summary, budget_sensitivity_table = build_budget_sensitivity_map(
            customers,
            budget=int(budget),
            threshold=float(threshold),
            max_customers=target_cap,
            budget_step=1_000_000,
        )

        # user-live 점수 테이블에 비용 컬럼이 아직 충분히 없거나,
        # 액션 큐 기준 후보만 존재하는 경우에는 현재 액션 큐를 후보 풀로 삼아 한 번 더 계산한다.
        # 기존 customers 기반 계산 결과가 있으면 그대로 사용하므로 오프라인/업로드 기능은 건드리지 않는다.
        if budget_sensitivity_table.empty and _use_live_payload:
            live_actions_for_sensitivity = _normalize_live_actions_df(
                _merge_live_score_dimensions(
                    live_payload.get("actions", pd.DataFrame()),
                    live_payload.get("scores", pd.DataFrame()),
                )
            )
            if not live_actions_for_sensitivity.empty:
                budget_sensitivity_summary, budget_sensitivity_table = build_budget_sensitivity_map(
                    live_actions_for_sensitivity,
                    budget=int(budget),
                    threshold=float(threshold),
                    max_customers=target_cap,
                    budget_step=1_000_000,
                )
    except Exception as exc:
        budget_sensitivity_summary, budget_sensitivity_table = {}, pd.DataFrame()
        st.warning(f"예산 민감도 지도를 계산하지 못했습니다: {exc}")

    if not budget_sensitivity_table.empty:
        sensitivity_summary_df = pd.DataFrame(
            [
                {"항목": "현재 입력 예산", "값": money(budget_sensitivity_summary.get("current_budget", budget))},
                {"항목": "현재 집행 예산", "값": money(budget_sensitivity_summary.get("current_spent", 0))},
                {"항목": "현재 타깃 고객 수", "값": f"{int(budget_sensitivity_summary.get('current_target_count', 0)):,}명"},
                {"항목": "현재 기대 순이익", "값": money(budget_sensitivity_summary.get("current_expected_profit", 0))},
                {"항목": "현재 평균 ROI", "값": _format_roi_display(budget_sensitivity_summary.get("current_average_roi", 0))},
                {"항목": "현재 구간 한계 ROI", "값": _format_roi_display(budget_sensitivity_summary.get("current_marginal_roi", 0))},
                {"항목": "예산 100만 원 추가 시 기대 순이익 증가", "값": money(budget_sensitivity_summary.get("next_1m_expected_profit_gain", 0))},
                {"항목": "예산 포화점", "값": str(budget_sensitivity_summary.get("saturation_label", "확인되지 않음"))},
                {"항목": "저효율 예산 구간", "값": str(budget_sensitivity_summary.get("low_efficiency_label", "확인되지 않음"))},
            ]
        )
        _render_dataframe_with_count(
            sensitivity_summary_df,
            label="예산 민감도 핵심 지표",
            prefer_static=True,
            height=360,
        )

        sensitivity_display = budget_sensitivity_table.copy()
        sensitivity_display["예산 구간"] = sensitivity_display["budget"].map(
            lambda x: "현재 선택 예산" if int(x) == int(budget) else f"예산 {int(x):,}원"
        )
        sensitivity_display["입력 예산"] = sensitivity_display["budget"].map(money)
        sensitivity_display["집행 예산"] = sensitivity_display["spent"].map(money)
        sensitivity_display["잔여 예산"] = sensitivity_display["remaining"].map(money)
        sensitivity_display["타깃 고객 수"] = sensitivity_display["target_count"].map(lambda x: f"{int(x):,}명")
        sensitivity_display["기대 순이익"] = sensitivity_display["expected_incremental_profit"].map(money)
        sensitivity_display["평균 ROI"] = sensitivity_display["average_roi"].map(_format_roi_display)
        sensitivity_display["직전 구간 대비 추가 예산"] = sensitivity_display["added_budget"].map(money)
        sensitivity_display["직전 구간 대비 추가 집행액"] = sensitivity_display["added_spend"].map(money)
        sensitivity_display["추가 타깃 고객 수"] = sensitivity_display["added_target_count"].map(lambda x: f"{int(x):,}명")
        sensitivity_display["직전 구간 대비 추가 순이익"] = sensitivity_display["added_profit"].map(money)
        sensitivity_display["예산 100만 원당 추가 순이익"] = sensitivity_display["marginal_profit_per_1m"].map(money)
        sensitivity_display["한계 ROI"] = sensitivity_display["marginal_roi"].map(_format_roi_display)
        sensitivity_display["예산 상태"] = sensitivity_display["budget_status"].astype(str)
        sensitivity_display["운영 해석"] = sensitivity_display["operator_message"].astype(str)
        sensitivity_display = sensitivity_display[
            [
                "예산 구간",
                "입력 예산",
                "집행 예산",
                "잔여 예산",
                "타깃 고객 수",
                "기대 순이익",
                "평균 ROI",
                "직전 구간 대비 추가 예산",
                "직전 구간 대비 추가 집행액",
                "추가 타깃 고객 수",
                "직전 구간 대비 추가 순이익",
                "예산 100만 원당 추가 순이익",
                "한계 ROI",
                "예산 상태",
                "운영 해석",
            ]
        ]
        _render_dataframe_with_count(
            sensitivity_display,
            label="예산 구간별 민감도 지도",
            prefer_static=True,
            height=min(760, 220 + 34 * len(sensitivity_display)),
        )
    else:
        st.info("예산 민감도 지도를 만들 후보 고객이 없습니다. 이탈 임계값을 낮추거나 예산 조건을 조정해 보세요.")

    selected_customers = _ensure_retention_target_schema(selected_customers)
    optimized_targets = selected_customers.sort_values(
        ["priority_score", "selection_score", "expected_incremental_profit", "customer_id"],
        ascending=[False, False, False, True],
    ).copy() if not selected_customers.empty else pd.DataFrame()

    st.markdown(f"### {T('세그먼트별 예산 배분 후보 고객 수')}")
    candidate_by_segment = pd.DataFrame(
        {
            "uplift_segment": list(optimize_summary.get("candidate_segment_counts", {}).keys()),
            "candidate_customer_count": list(optimize_summary.get("candidate_segment_counts", {}).values()),
        }
    )
    if candidate_by_segment.empty and not segment_allocation.empty and "uplift_segment" in segment_allocation.columns:
        candidate_by_segment = (
            segment_allocation.groupby("uplift_segment", as_index=False)
            .agg(candidate_customer_count=("customer_count", "sum"))
            .sort_values("candidate_customer_count", ascending=False)
        )
    if not candidate_by_segment.empty:
        candidate_by_segment = _translate_dataframe_values_for_display(candidate_by_segment)
        _render_dataframe_with_count(candidate_by_segment, label=T("세그먼트별 예산 배분 후보 고객 수"), prefer_static=True)
    else:
        st.info(T("세그먼트별 후보 고객 수를 계산할 데이터가 없습니다."))

    if segment_allocation.empty or int(optimize_summary.get("num_targeted", 0)) == 0:
        st.warning(T("현재 조건에서 예산 배분 대상 고객이 없습니다."))
    else:
        st.markdown(f"### {T('세그먼트별 예산 배분 테이블')}")
        display_df = _translate_dataframe_values_for_display(segment_allocation.copy())
        if "allocated_budget" in display_df.columns:
            display_df["allocated_budget"] = display_df["allocated_budget"].map(money)
        if "expected_profit" in display_df.columns:
            display_df["expected_profit"] = display_df["expected_profit"].map(money)
        _render_dataframe_with_count(display_df, label=T("세그먼트별 예산 배분 테이블"))

    st.markdown(f"### {T('최종 리텐션 타겟 고객 테이블')}")
    if optimized_targets.empty:
        st.warning(T("현재 조건에서 리텐션 타겟 고객이 없습니다."))
    else:
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
        display_df = _translate_dataframe_values_for_display(display_df)
        if "churn_probability" in display_df.columns:
            display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
        if "uplift_score" in display_df.columns:
            display_df["uplift_score"] = display_df["uplift_score"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
        if "clv" in display_df.columns:
            display_df["clv"] = display_df["clv"].map(lambda x: money(float(x)) if pd.notna(x) else "")
        if "coupon_cost" in display_df.columns:
            display_df["coupon_cost"] = display_df["coupon_cost"].map(lambda x: money(float(x)) if pd.notna(x) else "")
        if "expected_incremental_profit" in display_df.columns:
            display_df["expected_incremental_profit"] = display_df["expected_incremental_profit"].map(lambda x: money(float(x)) if pd.notna(x) else "")
        if "expected_roi" in display_df.columns:
            display_df["expected_roi"] = display_df["expected_roi"].map(lambda x: _format_roi_display(x) if pd.notna(x) else "")
        if "priority_score" in display_df.columns:
            display_df["priority_score"] = display_df["priority_score"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
        _render_dataframe_with_count(
            display_df,
            label=T("최종 리텐션 타겟 고객 테이블"),
            height=min(1100, 180 + 32 * len(display_df)),
        )

    st.markdown(f"### {T('고객별 선택 이유 / 주의사항')}")
    if not customer_explanations.empty:
        explain_df = customer_explanations.copy()
        for col in ["churn_probability", "realtime_churn_score", "uplift_score", "expected_roi", "survival_prob_30d"]:
            if col in explain_df.columns:
                explain_df[col] = explain_df[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
        for col in ["clv", "expected_incremental_profit"]:
            if col in explain_df.columns:
                explain_df[col] = explain_df[col].map(lambda x: money(float(x)) if pd.notna(x) else "")
        _render_dataframe_with_count(
            explain_df,
            label=T("고객별 선택 이유 / 주의사항"),
            height=min(760, 220 + 34 * len(explain_df)),
        )
    else:
        st.info(T("고객별 설명 테이블을 만들 데이터가 부족합니다. 학습 파이프라인의 explainability 단계가 생성한 산출물을 확인하세요."))

    llm_payload = {
        "threshold": threshold,
        "budget": budget,
        "optimize_summary": optimize_summary,
        "budget_sensitivity_summary": budget_sensitivity_summary if isinstance(budget_sensitivity_summary, dict) else {},
        "budget_sensitivity_table": budget_sensitivity_table.head(20).round(4).to_dict(orient="records") if isinstance(budget_sensitivity_table, pd.DataFrame) and not budget_sensitivity_table.empty else [],
        "candidate_by_segment": candidate_by_segment.to_dict(orient="records") if not candidate_by_segment.empty else [],
        "segment_allocation": segment_allocation.round(4).to_dict(orient="records") if not segment_allocation.empty else [],
        "target_count": int(len(optimized_targets)),
        "customer_explanations": customer_explanations.head(20).to_dict(orient="records") if not customer_explanations.empty else [],
        "segment_distribution": series_distribution(optimized_targets, "uplift_segment") if not optimized_targets.empty else {},
        "target_numeric_summary": numeric_summary(
            optimized_targets,
            ["priority_score", "selection_score", "churn_probability", "uplift_score", "clv", "coupon_cost", "expected_incremental_profit", "expected_roi"],
        ),
    }

elif view == "13. 고객별 대응 전략 비교":
    if _user_mode_unavailable("고객별 대응 전략 비교", "반사실 손익 비교는 churn·uplift·CLV·survival 신호를 사용합니다. 업로드 CSV에 Treatment/Control이 없으면 전처리 단계의 휴리스틱 개입효과 추정값으로 표시됩니다."):
        st.stop()

    st.subheader("고객별 대응 전략 비교")
    _render_view_intro("13")
    st.caption("이 화면은 실제 집행 결과가 아니라, 이탈 가능성·고객가치·개입 반응 가능성·예상 이탈 시점을 조합해 만든 의사결정 비교표입니다. 실제 효과는 A/B 검증이나 검증용 미개입군으로 확인해야 합니다.")

    counterfactual_display_limit = 500
    counterfactual_summary, counterfactual_lab, counterfactual_scenarios = build_counterfactual_retention_lab(
        customers=customers,
        selected_customers=selected_customers,
        survival_predictions=survival_predictions,
        top_n=counterfactual_display_limit,
        threshold=float(threshold),
    )

    def _ko_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        if _is_finance_display_mode():
            finance_value = _domain_translate_value("__counterfactual__", text)
            if isinstance(finance_value, str) and finance_value != text:
                return finance_value
        humanized = _humanize_business_display_value("recommended_action", text)
        if isinstance(humanized, str) and humanized != text:
            return humanized
        ko_map = VALUE_LABELS.get("ko", {})
        return ko_map.get(text, ko_map.get(text.lower(), text))

    def _format_ko_money(value: Any) -> str:
        return money(float(value)) if pd.notna(value) else ""

    def _format_ko_probability(value: Any) -> str:
        return f"{float(value) * 100:.1f}%" if pd.notna(value) else ""

    counterfactual_column_rename = {
        "customer_id": "고객 ID",
        "persona": "고객 유형",
        "churn_probability": "현재 이탈 가능성",
        "expected_churn_period": "예상 이탈 시점",
        "clv": "고객 생애가치",
        "recommended_action": "기존 추천 액션",
        "expected_no_action_net_profit": "무개입 예상 순이익",
        "expected_net_profit_coupon_5000": "5,000원 혜택 예상 순이익",
        "expected_net_profit_consult_call": "상담 전화 예상 순이익",
        "expected_net_profit_push_email": "푸시/이메일 예상 순이익",
        "expected_net_profit_wait_7d": "7일 대기 예상 순이익",
        "final_recommendation": "최종 추천 전략",
        "best_expected_net_profit": "최선 전략 예상 순이익",
        "incremental_vs_no_action": "무개입 대비 개선액",
        "confidence": "신뢰도",
        "ab_test_recommended": "검증 필요 여부",
        "recommendation_reason": "추천 근거",
    }
    scenario_column_rename = {
        "action_label": "비교 전략",
        "action_cost": "개입 비용",
        "expected_net_profit": "예상 순이익",
        "incremental_vs_no_action": "무개입 대비 개선액",
        "treated_churn_probability": "전략 적용 후 이탈 가능성",
        "estimated_retention_lift": "이탈 가능성 감소폭",
        "description": "전략 설명",
    }

    if counterfactual_lab.empty:
        st.warning("현재 조건에서 반사실 시나리오를 계산할 고객이 없습니다.")
        llm_payload = {
            "threshold": float(threshold),
            "budget": int(budget),
            "counterfactual_summary": counterfactual_summary,
        }
    else:
        summary_df = pd.DataFrame(
            [
                {"항목": "분석 고객 수", "값": f"{int(counterfactual_summary.get('customer_count', len(counterfactual_lab))):,}명"},
                {"항목": "무개입 대비 평균 개선액", "값": money(counterfactual_summary.get("avg_incremental_vs_no_action", 0.0))},
                {"항목": "무개입보다 나은 추천이 나온 고객", "값": f"{int(counterfactual_summary.get('positive_recommendation_count', 0)):,}명"},
                {"항목": "A/B 검증 또는 미개입군 검증이 필요한 고객", "값": f"{int(counterfactual_summary.get('ab_test_recommended_count', 0)):,}명"},
            ]
        )
        _render_dataframe_with_count(summary_df, label="반사실 실험 요약", prefer_static=True, height=260)

        action_counts = pd.DataFrame(
            [
                {"최종 추천 전략": _ko_value(action), "고객 수": f"{int(count):,}명"}
                for action, count in (counterfactual_summary.get("best_action_counts", {}) or {}).items()
            ]
        )
        if not action_counts.empty:
            st.markdown("### 최종 추천 전략 분포")
            _render_dataframe_with_count(action_counts, label="최종 추천 전략 분포", prefer_static=True, height=min(360, 180 + 34 * len(action_counts)))

        st.markdown("### 고객별 시나리오 상세")
        customer_options = counterfactual_lab["customer_id"].astype(str).tolist() if "customer_id" in counterfactual_lab.columns else []
        selected_customer_for_lab = st.selectbox(
            "상세 비교 고객 선택",
            options=customer_options,
            index=0,
            key="counterfactual_customer_selector",
        ) if customer_options else None

        if selected_customer_for_lab is not None and not counterfactual_scenarios.empty:
            one_customer_scenarios = counterfactual_scenarios[
                counterfactual_scenarios["customer_id"].astype(str) == str(selected_customer_for_lab)
            ].copy()
            if not one_customer_scenarios.empty:
                detail_cols = ["action_label", "action_cost", "expected_net_profit", "incremental_vs_no_action", "treated_churn_probability", "estimated_retention_lift", "description"]
                detail_df = one_customer_scenarios[[c for c in detail_cols if c in one_customer_scenarios.columns]].copy()
                if "action_label" in detail_df.columns:
                    detail_df["action_label"] = detail_df["action_label"].map(_ko_value)
                if "description" in detail_df.columns:
                    detail_df["description"] = detail_df["description"].map(_ko_value)
                for money_col in ["action_cost", "expected_net_profit", "incremental_vs_no_action"]:
                    if money_col in detail_df.columns:
                        detail_df[money_col] = detail_df[money_col].map(_format_ko_money)
                for prob_col in ["treated_churn_probability", "estimated_retention_lift"]:
                    if prob_col in detail_df.columns:
                        detail_df[prob_col] = detail_df[prob_col].map(_format_ko_probability)
                detail_df = detail_df.rename(columns=scenario_column_rename)
                _render_dataframe_with_count(detail_df, label="고객별 시나리오 상세", prefer_static=True, height=min(520, 220 + 34 * len(detail_df)))

        st.markdown("### 고객별 반사실 손익 비교")
        display_columns = [
            "customer_id",
            "persona",
            "churn_probability",
            "expected_churn_period",
            "clv",
            "recommended_action",
            "expected_no_action_net_profit",
            "expected_net_profit_coupon_5000",
            "expected_net_profit_consult_call",
            "expected_net_profit_push_email",
            "expected_net_profit_wait_7d",
            "final_recommendation",
            "best_expected_net_profit",
            "incremental_vs_no_action",
            "confidence",
            "ab_test_recommended",
            "recommendation_reason",
        ]
        display_df = counterfactual_lab[[col for col in display_columns if col in counterfactual_lab.columns]].head(counterfactual_display_limit).copy()
        if "persona" in display_df.columns:
            display_df["persona"] = display_df["persona"].map(_ko_value)
        for text_col in ["recommended_action", "final_recommendation", "confidence", "recommendation_reason"]:
            if text_col in display_df.columns:
                display_df[text_col] = display_df[text_col].map(_ko_value)
        if "churn_probability" in display_df.columns:
            display_df["churn_probability"] = display_df["churn_probability"].map(_format_ko_probability)
        if "expected_churn_period" in display_df.columns:
            display_df["expected_churn_period"] = display_df["expected_churn_period"].map(lambda x: _format_churn_period(x) if pd.notna(x) else "")
        for money_col in [
            "clv",
            "expected_no_action_net_profit",
            "expected_net_profit_coupon_5000",
            "expected_net_profit_consult_call",
            "expected_net_profit_push_email",
            "expected_net_profit_wait_7d",
            "best_expected_net_profit",
            "incremental_vs_no_action",
        ]:
            if money_col in display_df.columns:
                display_df[money_col] = display_df[money_col].map(_format_ko_money)
        if "ab_test_recommended" in display_df.columns:
            display_df["ab_test_recommended"] = display_df["ab_test_recommended"].map(lambda x: "검증 권장" if bool(x) else "바로 집행 가능")
        display_df = display_df.rename(columns=counterfactual_column_rename)
        _render_dataframe_with_count(
            display_df,
            label="고객별 반사실 손익 비교",
            prefer_static=True,
            height=min(760, 220 + 34 * len(display_df)),
        )

        st.info("권장 해석: 신뢰도가 낮거나 중간인 고객은 바로 전체 집행하지 말고 A/B 검증 또는 검증용 미개입군에 포함해 실제 추가 이익을 확인하세요.")

        llm_payload = {
            "threshold": float(threshold),
            "budget": int(budget),
            "counterfactual_summary": counterfactual_summary,
            "best_action_counts": counterfactual_summary.get("best_action_counts", {}),
            "top_counterfactual_customers": dataframe_snapshot(
                counterfactual_lab,
                columns=[
                    "customer_id",
                    "churn_probability",
                    "clv",
                    "final_recommendation",
                    "best_expected_net_profit",
                    "incremental_vs_no_action",
                    "confidence",
                    "ab_test_recommended",
                    "recommendation_reason",
                ],
                max_rows=20,
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
    st.subheader(T("최종 타겟 고객 대상 개인화 추천"))
    _render_view_intro("5")
    st.caption(T("현재 예산·이탈 임계값으로 선별된 최종 타겟 고객에게만 새 추천을 생성합니다. 추천 점수는 고객 구매 이력, 최근 관심, 세그먼트 인기, 전역 인기를 혼합해 계산합니다."))

    budget_context = recommendation_summary.get('budget_context', {}) if isinstance(recommendation_summary, dict) else {}
    current_target_count = int(
        budget_context.get(
            'num_targeted',
            recommendation_summary.get('eligible_target_customers', 0) if isinstance(recommendation_summary, dict) else 0,
        ) or 0
    )

    if isinstance(recommendation_summary, dict) and recommendation_summary.get("warning"):
        st.warning(str(recommendation_summary.get("warning")))

    if isinstance(recommendation_summary, dict):
        st.caption(
            f"{T('추천 기준')}: {T('예산')} {money(budget_context.get('budget', budget))}, "
            f"{T('이탈 임계값')} {float(budget_context.get('threshold', threshold)):.2f}, "
            f"{T('최대 타겟')} {int(budget_context.get('max_customers_cap', target_cap) or 0):,}{T('명')}"
        )

    if recommendation_error:
        st.error(f"추천 API 호출 실패: {recommendation_error}")
    elif personalized_recommendations.empty:
        st.info(
            "현재 조건에서 생성된 추천이 없습니다. 최종 타겟 고객 수가 0명이면 예산을 늘리거나 "
            "이탈 임계값을 낮춰야 합니다. 저장된 과거 후보를 현재 추천처럼 표시하지 않습니다."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T("표시 추천 행 수"), "0")
        m2.metric(T("추천 대상 고객 수"), "0")
        m3.metric(T("평균 추천 수/고객"), "0.00")
        m4.metric(T("현재 최종 타겟 고객 수"), f"{current_target_count:,}")
    else:
        covered_customers = int(recommendation_summary.get('customers_covered', personalized_recommendations['customer_id'].nunique()))
        displayed_rows = int(recommendation_summary.get('rows', len(personalized_recommendations)))
        actual_per_customer = float(recommendation_summary.get('actual_per_customer', displayed_rows / max(covered_customers, 1)))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T("표시 추천 행 수"), f"{displayed_rows:,}")
        m2.metric(T("추천 대상 고객 수"), f"{covered_customers:,}")
        m3.metric(T("평균 추천 수/고객"), f"{actual_per_customer:.2f}")
        m4.metric(T("현재 최종 타겟 고객 수"), f"{current_target_count:,}")

        category_counts = (
            _translate_dataframe_values_for_display(personalized_recommendations).groupby('recommended_category', as_index=False)
            .agg(recommend_count=('customer_id', 'count'))
            .sort_values('recommend_count', ascending=False)
        )
        fig = px.bar(
            category_counts,
            x='recommended_category',
            y='recommend_count',
            title=T('추천 카테고리 분포'),
        )
        st.plotly_chart(fig, use_container_width=True)

        display_df = _translate_dataframe_values_for_display(personalized_recommendations.copy())
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
            display_df['expected_roi'] = display_df['expected_roi'].map(lambda x: _format_roi_display(x) if pd.notna(x) else '')
        if 'recommendation_priority' in display_df.columns:
            display_df['recommendation_priority'] = display_df['recommendation_priority'].map(lambda x: f"{x:.3f}")
        if 'target_priority_score' in display_df.columns:
            display_df['target_priority_score'] = display_df['target_priority_score'].map(lambda x: f"{x:.3f}")
        if 'recommendation_score' in display_df.columns:
            display_df['recommendation_score'] = display_df['recommendation_score'].map(lambda x: f"{x:.3f}")
        _render_dataframe_with_count(display_df, label=T("개인화 추천 테이블"))

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
        st.subheader(T("실시간 운영 모니터"))
        _render_view_intro("6")
        st.caption(f"{_domain_label()} {T('기준 PostgreSQL live DB 운영 모니터입니다.')}")

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

        st.caption(T("시연을 시작하면 설정된 간격마다 가상 고객 이벤트(방문, 구매 등)가 자동 생성되고, 이탈 점수 재산정 및 액션 큐가 갱신됩니다."))
        _demo_bar = st.container()
        with _demo_bar:
            if _page_demo_running:
                _ev = _page_demo.get("total_events_sent", 0)
                _new = _page_demo.get("new_customers_created", 0)
                _exist = _page_demo.get("existing_customers_updated", 0)
                st.success(f"{T('시연 실행 중')}  |  {T('이벤트 수')} {_ev}{T('건')}  |  {T('신규')} {_new}{T('명')}  |  {T('기존')} {_exist}{T('명')}")
                _dc1, _dc2, _dc3 = st.columns(3)
                with _dc1:
                    if st.button(T("시연 중지"), use_container_width=True, key="pg_demo_stop"):
                        _page_stop_demo()
                        clear_dashboard_caches()
                        st.rerun()
                with _dc2:
                    if st.button(T("시연 초기화"), use_container_width=True, type="secondary", key="pg_demo_reset_running"):
                        _page_reset_demo()
                        clear_dashboard_caches()
                        st.rerun()
                with _dc3:
                    st.caption(T("10초마다 자동 새로고침"))
            else:
                _dc1, _dc2, _dc3, _dc4 = st.columns([1.5, 1.5, 1, 1])
                with _dc1:
                    st.caption(T("N초마다 이벤트 1건 생성"))
                    _pg_interval = st.number_input(T("간격(초)"), min_value=0.5, max_value=30.0, value=2.0, step=0.5, key="pg_demo_interval")
                with _dc2:
                    st.caption(T("새 고객 vs 기존 고객 비율"))
                    _pg_ratio = st.number_input(T("신규 비율"), min_value=0.0, max_value=1.0, value=0.3, step=0.1, key="pg_demo_ratio")
                with _dc3:
                    if st.button(T("시연 시작"), use_container_width=True, type="primary", key="pg_demo_start"):
                        _page_start_demo(interval_seconds=_pg_interval, new_customer_ratio=_pg_ratio)
                        clear_dashboard_caches()
                        st.rerun()
                with _dc4:
                    if st.button(T("시연 초기화"), use_container_width=True, type="secondary", key="pg_demo_reset_idle"):
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
                    _log_label = f"{T('이벤트 로그')} ({len(_log_data)}{T('건')})" if _page_demo_running else f"{T('이벤트 로그')} ({len(_log_data)}{T('건')}, {T('중지됨')})"
                    with st.expander(_log_label, expanded=True):
                        _lines = []
                        for _r in reversed(_log_data):
                            _label = T("NEW") if _r.get("is_new") else T("UPD")
                            _score_str = f"risk={_r['churn_score']:.2f}" if _r.get("churn_score") is not None else ""
                            _action_str = "→ " + T("큐 적재 수") if _r.get("action_queued") else ""
                            _event = _translate_cell_value(_r.get('event_type', ''))
                            _lines.append(f"[{_label}] #{_r['customer_id']}  {_event}  {_score_str}  {_action_str}")
                        _render_dataframe_with_count(pd.DataFrame({"log": _lines}), label=T("이벤트 로그"), height=300, hide_index=True)
            elif st.session_state.get("_demo_last_log"):
                _log_data = st.session_state["_demo_last_log"]
                with st.expander(f"{T('이벤트 로그')} ({len(_log_data)}{T('건')}, {T('중지됨')})", expanded=False):
                    _lines = []
                    for _r in reversed(_log_data):
                        _label = T("NEW") if _r.get("is_new") else T("UPD")
                        _score_str = f"risk={_r['churn_score']:.2f}" if _r.get("churn_score") is not None else ""
                        _action_str = "→ " + T("큐 적재 수") if _r.get("action_queued") else ""
                        _event = _translate_cell_value(_r.get('event_type', ''))
                        _lines.append(f"[{_label}] #{_r['customer_id']}  {_event}  {_score_str}  {_action_str}")
                    _render_dataframe_with_count(pd.DataFrame({"log": _lines}), label=T("이벤트 로그"), height=300, hide_index=True)

        st.divider()

        health = live_payload.get("health", {}) or {}
        score_summary = live_payload.get("score_summary", {}) or {}
        action_summary = live_payload.get("action_summary", {}) or {}
        rec_summary = live_payload.get("recommendation_summary", {}) or {}

        total_live_customers = max(
            int(health.get('feature_state_count') or 0),
            int(score_summary.get('scored_customers') or 0),
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T("이벤트 수"), f"{int(health.get('event_count') or 0):,}")
        c2.metric(T("전체 고객 수"), f"{total_live_customers:,}", help=T("현재 live DB에서 상태 또는 이탈 점수를 보유한 고유 고객 수입니다."))
        c3.metric(T("액션 큐"), f"{int(action_summary.get('queued_actions') or 0):,}", help=T("현재 후속 조치 대기열에 올라간 고객 단위 액션 후보 수입니다."))
        c4.metric(T("최신 점수 갱신"), str(score_summary.get("latest_scored_at") or "-"))
        st.caption(T("액션 큐는 실시간 이벤트 반영 후 이탈 위험과 기대 효과 조건을 만족해 쿠폰, 상담, 알림 등 후속 조치 대상으로 대기 중인 고객 단위 후보 목록입니다."))

        scores_df = live_payload.get("scores", pd.DataFrame()).copy()
        actions_df = live_payload.get("actions", pd.DataFrame()).copy()

        # 요청 반영: 실시간 운영 모니터의 첫 번째 "Live 이탈 점수 Top 고객" 표는 화면에서 숨긴다.
        # 점수 데이터는 상단 지표와 LLM 컨텍스트에는 그대로 유지하므로 기존 운영/요약 기능은 훼손하지 않는다.
        if scores_df.empty:
            st.info(T("표시할 live score 데이터가 없습니다."))

        if not actions_df.empty:
            display_cols = [
                col for col in [
                    "customer_id",
                    "recommended_action",
                    "intervention_intensity",
                    "coupon_cost",
                    "expected_profit",
                    "expected_incremental_profit",
                    "expected_roi",
                    "action_status",
                    "trigger_reason",
                    "updated_at",
                ]
                if col in actions_df.columns
            ]
            _queue_total = int(action_summary.get("queued_actions") or action_summary.get("total_actions") or len(actions_df))
            st.caption(
                f"{T('Live Action Queue')}: 전체 queued action {_queue_total:,}건 중 "
                f"현재 표는 우선순위 상위 {len(actions_df):,}건을 표시합니다."
            )

            _render_dataframe_with_count(
                actions_df[display_cols],
                label=T("Live Action Queue"),
                height=520,
            )
        else:
            st.info(T("현재 queued action이 없습니다. action_threshold를 낮춰 테스트하거나 새 이벤트를 입력하세요."))

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
            _snapshot_analysis_controls()
            _placeholder = st.empty()
            _placeholder.caption(T("다음 자동 새로고침까지 10초..."))
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
    st.subheader(T("실시간 운영 모니터"))
    _render_view_intro("6")
    st.caption(T("이벤트 스트림을 재생하며 고객별 실시간 위험 점수와 액션 큐 상태를 함께 갱신합니다."))

    if realtime_error:
        st.error(f"{T('실시간 스코어 API 호출 실패')}: {realtime_error}")
        st.info(T("먼저 Redis를 실행한 뒤 realtime-bootstrap / realtime-produce / realtime-consume(또는 realtime-replay) 명령을 수행하세요."))
    elif realtime_scores.empty:
        st.warning(T("실시간 스코어 스냅샷이 없습니다. 스트림 소비 결과가 아직 생성되지 않았을 수 있습니다."))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T("추적 고객 수"), f"{int(realtime_summary.get('tracked_customers', 0)):,}")
        m2.metric(T("현재 기준 이탈 위험 고객 수"), f"{int(realtime_summary.get('high_risk_customers', 0)):,}", help=f"{T('이탈 임계값')} ≥ {float(threshold):.2f}")
        m3.metric(T("재최적화 트리거 수"), f"{int(realtime_summary.get('triggered_reoptimizations', 0)):,}")
        m4.metric(T("액션 큐 적재 수"), f"{int(realtime_summary.get('action_queue_size', 0)):,}")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric(T("임계 위험 고객 수"), f"{int(realtime_summary.get('critical_risk_customers', 0)):,}")
        q2.metric(T("처리 이벤트 수"), f"{int(realtime_summary.get('processed_events', 0)):,}")
        q3.metric(T("폐쇄루프 예산 사용"), money(int(realtime_summary.get('closed_loop_budget_spent', 0))))
        q4.metric(T("채널 할당 수"), f"{int(realtime_summary.get('daily_channel_allocated', 0)):,} / {int(realtime_summary.get('daily_channel_capacity', 0)):,}")

        st.caption(T("실시간 운영 모니터 그래프는 제거하고 표 중심으로 표시합니다."))
        # 요청 반영: 실시간 운영 모니터의 첫 번째 점수 Top 고객 표는 숨기고,
        # 아래 액션 큐/상태 테이블 중심으로 운영 화면을 단순화한다.

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
                queue_display['queued_expected_roi'] = queue_display['queued_expected_roi'].map(lambda x: _format_roi_display(x) if pd.notna(x) else '')
            _render_dataframe_with_count(queue_display, label=T("실시간 부분 재최적화 액션 큐"), height=min(520, 180 + 32 * len(queue_display)))

        display_df = realtime_scores.copy()
        for col in ['base_churn_probability', 'realtime_churn_score', 'score_delta', 'behavioral_risk', 'inactivity_signal', 'queued_expected_roi']:
            if col in display_df.columns:
                formatter = (lambda x: f"{float(x):.2%}") if col == 'queued_expected_roi' else (lambda x: f"{float(x):.3f}")
                display_df[col] = display_df[col].map(formatter)
        for money_col in ['clv', 'coupon_cost', 'queued_coupon_cost', 'queued_expected_profit']:
            if money_col in display_df.columns:
                display_df[money_col] = display_df[money_col].map(money)
        if 'expected_roi' in display_df.columns:
            display_df['expected_roi'] = display_df['expected_roi'].map(lambda x: _format_roi_display(x) if pd.notna(x) else '')
        _render_dataframe_with_count(display_df, label=T("실시간 이탈 위험 테이블"))

    realtime_summary_display = realtime_monitor_overview.get("summary", realtime_summary) if realtime_monitor_overview else realtime_summary
    st.markdown(f"### {T('운영 모니터')}")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric(T("처리 이벤트 수"), f"{int(realtime_summary_display.get('processed_events', 0) or 0):,}")
    q2.metric(T("재최적화 횟수"), f"{int(realtime_summary_display.get('triggered_reoptimizations', 0) or 0):,}")
    q3.metric(T("큐 적재 수"), f"{int(realtime_summary_display.get('queued_actions_total', realtime_summary_display.get('action_queue_size', 0)) or 0):,}")
    cap = int(realtime_summary_display.get('daily_channel_capacity', 0) or 0)
    alloc = int(realtime_summary_display.get('daily_channel_allocated', 0) or 0)
    utilization = alloc / cap if cap > 0 else 0.0
    q4.metric(T("채널 용량 사용률"), pct(utilization))
    q5.metric(T("고우선순위 큐"), f"{int(realtime_summary_display.get('high_priority_queue_size', 0) or 0):,}")

    if realtime_monitor_overview:
        tab1, tab2, tab3 = st.tabs([T("큐 상태"), T("트리거 이유"), T("행동 신호")])
        with tab1:
            status_df = _translate_dataframe_values_for_display(realtime_monitor_overview.get("status_df", pd.DataFrame()))
            queue_df = realtime_monitor_overview.get("queue_df", pd.DataFrame())
            if not status_df.empty:
                _render_dataframe_with_count(status_df, label=T("액션 큐 상태 구성"), prefer_static=True)
            if not queue_df.empty:
                display_df = _translate_dataframe_values_for_display(queue_df.copy())
                for col in ["queued_coupon_cost", "queued_expected_profit"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: money(float(x)) if pd.notna(x) else "")
                for col in ["queued_expected_roi", "realtime_churn_score"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
                _render_dataframe_with_count(display_df, label=T("실시간 액션 큐 상세"), height=min(1200, 220 + 28 * len(display_df)))
        with tab2:
            trigger_df = _translate_dataframe_values_for_display(realtime_monitor_overview.get("trigger_df", pd.DataFrame()))
            if not trigger_df.empty:
                _render_dataframe_with_count(trigger_df.head(15), label=T("트리거 이유 빈도"), prefer_static=True)
        with tab3:
            signal_df = _translate_dataframe_values_for_display(realtime_monitor_overview.get("signal_df", pd.DataFrame()))
            if not signal_df.empty:
                _render_dataframe_with_count(signal_df, label=T("행동 신호 평균"), prefer_static=True)

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

elif _is_churn_timing_view(view):
    st.subheader(T("이탈 시점 예측"))
    _render_view_intro("9")

    _sv_data_span = (survival_metrics or {}).get("data_span_days")
    _sv_horizon = (survival_metrics or {}).get("horizon_days")
    _sv_auto_adjusted = (survival_metrics or {}).get("horizon_auto_adjusted", False)
    if survival_error and _sv_data_span is not None and int(_sv_data_span) < 60:
        st.error(
            f"📊 입력한 데이터 기간은 **{_sv_data_span}일**로, "
            f"이탈 시점 예측에 필요한 최소 기간(60일)에 미달하여 "
            f"**생존분석이 비활성화**되었습니다. "
            f"이탈 확률 등 다른 분석은 정상 제공됩니다."
        )
    elif _sv_data_span is not None and _sv_horizon is not None:
        st.info(
            f"📊 입력한 데이터 기간은 **{_sv_data_span}일**이며, "
            f"예측 범위는 **{_sv_horizon}일**로 학습되었습니다."
        )

    st.caption(T("고객별로 언제쯤 이탈할 가능성이 큰지와 ..."))

    st.caption(T("고객별로 언제쯤 이탈할 가능성이 큰지와 그때 잃을 수 있는 금액만 표로 보여줍니다."))
    churn_timing_probability_threshold_pct = st.slider(
        T("30일 내 이탈 가능성 기준"),
        min_value=0,
        max_value=100,
        value=int(st.session_state.get("churn_timing_probability_threshold_pct", 0)),
        step=5,
        key="churn_timing_probability_threshold_pct",
        help=T("0%로 두어도 전체 행을 한 번에 렌더링하지 않고, 운영 우선순위가 높은 고객부터 제한된 수만 빠르게 표시합니다."),
    )
    _churn_timing_display_limit = int(CHURN_TIMING_DISPLAY_ROW_LIMIT)
    st.caption(
        f"{T('이 표는 선택한 기준 이상 고객 중 운영 우선순위가 높은 고객부터 빠르게 보여줍니다.')} "
        f"{T('표시 고객 수 제한')}: {_churn_timing_display_limit:,}{T('명')}"
    )

    if survival_error or survival_predictions.empty:
        _simulator_missing_result_box(
            T("이탈 시점 예측 결과가 없습니다."),
            survival_error or T("survival_predictions.csv가 없거나 survival 분석이 아직 실행되지 않았습니다."),
            T("시뮬레이터 데모에서는 python src/main.py --mode survival 실행 후 대시보드를 새로고침하세요."),
        )
        timing_display = pd.DataFrame()
    else:
        _churn_threshold = float(churn_timing_probability_threshold_pct) / 100.0
        _churn_candidate_count = _count_churn_timing_candidates(
            survival_predictions,
            min_churn_probability=_churn_threshold,
        )
        timing_display = _build_churn_timing_table(
            survival_predictions,
            customers,
            survival_metrics,
            min_churn_probability=_churn_threshold,
            limit=_churn_timing_display_limit,
        )
        if timing_display.empty:
            st.info(T("이탈 시점 예측 결과가 없습니다."))
        else:
            st.caption(
                f"{T('현재 기준 이상 고객')}: {_churn_candidate_count:,}{T('명')} / "
                f"{T('표시 고객 수 제한')}: {len(timing_display):,}{T('명')} "
                f"({T('현재 표시는 운영 우선순위 상위 고객만 보여줍니다.')})"
            )
            st.caption(T("예상 손실액은 고객 생애가치(CLV)에 30일 내 이탈 가능성을 곱해 계산합니다. CLV가 없으면 최근 구매금액을 보수적 대체값으로 사용합니다."))
            _render_dataframe_with_count(
                timing_display,
                label=T("고객별 이탈 시점과 예상 손실"),
                height=min(720, 220 + 34 * len(timing_display)),
            )

    llm_payload = {
        "survival_metrics": survival_metrics,
        "churn_timing_table": dataframe_snapshot(
            timing_display,
            columns=[
                "customer_id",
                "persona",
                "expected_churn_period",
                "expected_churn_date",
                "churn_within_30d_probability",
                "expected_loss_30d",
            ],
            max_rows=20,
        ) if isinstance(timing_display, pd.DataFrame) and not timing_display.empty else [],
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
        _p_val_raw = exp_metrics.get('p_value', np.nan)
        if pd.notna(_p_val_raw):
            _p_val_float = float(_p_val_raw)
            _p_display = "< 0.000001" if _p_val_float < 1e-6 else f"{_p_val_float:.6f}"
        else:
            _p_display = "-"
        m5.metric("Z-test p-value", _p_display)

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

# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
elif view == "14. 주간 액션 성과 리뷰":
    _review_reco_df = insight_bundle.personalized_recommendations if insight_bundle else recommendation_context_df
    _review_sel_df = insight_bundle.optimization_selected_customers if insight_bundle else selected_customers
    _review_has_data = isinstance(_review_reco_df, pd.DataFrame) and not _review_reco_df.empty
    if _simulator_mode_unavailable(
        "주간 액션 성과 리뷰", _review_has_data,
        "개인화 추천 또는 최적화 선정 고객 산출물이 없습니다.",
        "시뮬레이터에서 recommend 모드를 먼저 실행하세요.",
    ):
        st.stop()
    if not _review_has_data:
        st.warning(T("개인화 추천 또는 최적화 선정 고객 산출물이 없습니다."))
        st.stop()

    if "show_report_14" not in st.session_state:
        st.session_state["show_report_14"] = False
    if "review_memo_area" not in st.session_state:
        st.session_state["review_memo_area"] = ""

    _title_c, _btn_c = st.columns([5, 1])
    with _title_c:
        st.subheader(T("주간 액션 성과 리뷰"))
    with _btn_c:
        st.write("")
        if st.button("📄 " + T("보고서 보기"), key="report_btn_14"):
            st.session_state["show_report_14"] = True
            st.rerun()

    st.caption(T("추천 기반 시뮬레이션으로 지난주 리텐션 액션 성과를 리뷰합니다."))
    with st.expander(T("이 화면 설명 보기"), expanded=False):
        for _intro_line in VIEW_INTRO_LINES.get("14", []):
            st.markdown(f"- {T(_intro_line)}")
        st.info(T("이 화면은 실제 집행 결과가 아닌, 추천 데이터 기반의 시뮬레이션 리뷰입니다. 실행률과 성과 노이즈 슬라이더로 가상 시나리오를 조정할 수 있습니다."))

    _today = pd.Timestamp.now().normalize()
    _last_sunday = _today - pd.Timedelta(days=_today.dayofweek + 1)
    _last_monday = _last_sunday - pd.Timedelta(days=6)
    _week_options = [_last_monday - pd.Timedelta(weeks=i) for i in range(8)]
    _week_labels = [f"{d.strftime('%Y-%m-%d')} ~ {(d + pd.Timedelta(days=6)).strftime('%Y-%m-%d')}" for d in _week_options]
    _wk_col1, _wk_col2 = st.columns([1, 3])
    with _wk_col1:
        _selected_idx = st.selectbox(
            T("주 선택"), range(len(_week_labels)),
            format_func=lambda i: _week_labels[i],
            key="review_week_idx",
        )
    _week_start = _week_options[_selected_idx]
    _week_end = _week_start + pd.Timedelta(days=6)
    with _wk_col2:
        st.markdown("")
        st.markdown(f"📅 **{T('분석 기간')}:** {_week_start.strftime('%Y-%m-%d')} ~ {_week_end.strftime('%Y-%m-%d')}")

    with st.sidebar.expander(T("시뮬레이션 설정"), expanded=False):
        _exec_rate = st.slider(T("전체 실행률"), 0.0, 1.0, 0.75, 0.05, key="review_exec_rate",
                               help=T("CRM 담당자가 추천 액션 중 실제 실행하는 비율"))
        _hc_exec_rate = st.slider(T("고쿠폰 실행률"), 0.0, 1.0, 0.50, 0.05, key="review_hc_exec_rate",
                                  help=T("고비용 쿠폰 추천의 실행 비율 (보통 더 낮음)"))
        _noise = st.slider(T("성과 노이즈"), 0.0, 0.50, 0.15, 0.05, key="review_noise",
                           help=T("실제 성과가 예상에서 벗어나는 정도"))
        _seed = st.number_input(T("시뮬레이션 시드"), value=42, min_value=0, max_value=9999, key="review_seed")

    _week_hash = int(_week_start.strftime("%Y%m%d"))
    _effective_seed = int(_seed) ^ _week_hash

    review_summary, action_log, policy_suggestions = _build_weekly_action_review(
        _review_reco_df, _review_sel_df,
        execution_rate=_exec_rate,
        high_coupon_execution_rate=_hc_exec_rate,
        noise_std=_noise,
        seed=_effective_seed,
    )

    _data_dir = _project_root() / _domain_paths()["data"]
    _events_path = _data_dir / "events.csv"
    _orders_path = _data_dir / "orders.csv"
    _campaigns_path = _data_dir / "campaign_exposures.csv"

    @st.cache_data(show_spinner=False)
    def _load_review_events(_token: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        _evts = pd.read_csv(_events_path) if _events_path.exists() else pd.DataFrame()
        _ords = pd.read_csv(_orders_path) if _orders_path.exists() else pd.DataFrame()
        _camps = pd.read_csv(_campaigns_path) if _campaigns_path.exists() else pd.DataFrame()
        return _evts, _ords, _camps

    _review_events, _review_orders, _review_campaigns = _load_review_events(_raw_data_token())

    _total = review_summary["total_actions"]
    _executed_cnt = review_summary["total_executed"]
    _exec_pct = review_summary["execution_rate"]
    _expected = review_summary["expected_profit_sum"]
    _actual = review_summary["actual_profit_sum"]
    _gap = review_summary["profit_gap"]
    _budget = review_summary["total_budget_spent"]
    _gap_positive = _gap >= 0

    _m1, _m2, _m3 = st.columns(3)
    _m1.metric(T("총 추천 건수"), f"{_total:,}")
    _m2.metric(T("총 집행 건수"), f"{_executed_cnt:,}")
    _m3.metric(T("집행률"), f"{_exec_pct:.1%}")

    _gap_color = "#2e7d32" if _gap_positive else "#c62828"
    _gap_bg = "#e8f5e9" if _gap_positive else "#fbe9e7"
    _gap_icon = "📈" if _gap_positive else "📉"

    def _budget_card(label: str, value: str, color: str = "#333", bg: str = "#f8f9fa") -> str:
        return f"""<div style="background:{bg};border-radius:10px;padding:14px 16px;text-align:center;min-height:85px;display:flex;flex-direction:column;justify-content:center;">
        <div style="font-size:12px;color:#888;margin-bottom:4px;">{label}</div>
        <div style="font-size:20px;font-weight:700;color:{color};">{value}</div>
        </div>"""

    _b1, _b2, _b3, _b4 = st.columns(4)
    _b1.markdown(_budget_card(T("총 집행 예산"), money(_budget)), unsafe_allow_html=True)
    _b2.markdown(_budget_card(T("기대 이익"), money(_expected)), unsafe_allow_html=True)
    _b3.markdown(_budget_card(T("실제 이익"), money(_actual)), unsafe_allow_html=True)
    _b4.markdown(_budget_card(T("예상 대비 손익"), f"{_gap_icon} {_gap:+,.0f}{T('원')}", color=_gap_color, bg=_gap_bg), unsafe_allow_html=True)

    st.divider()

    _outcome_config = [
        ("적정 판단", "#2ecc71", "적절한 비용으로 전환에 성공한 건강한 액션 — 유사 고객 확대 근거"),
        ("기대 미달", "#f39c12", "이익은 발생했으나 기대 ROI 대비 70% 미만 — 쿠폰 강도 또는 타이밍 점검"),
        ("과잉 투자", "#e74c3c", "높은 쿠폰 비용 대비 전환 실패 — 비용 상한 재설정 필요"),
        ("타겟 오류", "#e67e22", "반응 가능성이 낮은 대상에 실행 — 세그먼트 필터 재검토"),
        ("실행 누락", "#9b59b6", "기대 ROI 1.0+ 고객을 미실행하여 이탈 — 다음 주 우선 실행 대상"),
    ]
    _oc = review_summary.get("outcome_counts", {})

    st.markdown(f"### {T('판정 분포 차트')}")
    _chart_col, _legend_col = st.columns([1, 1])
    with _chart_col:
        _donut_labels = []
        _donut_values = []
        _color_map = {T(_lbl): _clr for _lbl, _clr, _ in _outcome_config}
        for _lbl, _clr, _ in _outcome_config:
            _cnt = _oc.get(_lbl, 0)
            if _cnt > 0:
                _donut_labels.append(T(_lbl))
                _donut_values.append(_cnt)
        if _donut_values:
            _donut_df = pd.DataFrame({"label": _donut_labels, "count": _donut_values})
            _fig = px.pie(
                _donut_df, names="label", values="count",
                hole=0.45,
                color="label",
                color_discrete_map=_color_map,
            )
            _fig.update_traces(
                textposition="inside", textinfo="label",
                customdata=_donut_df["count"].values,
                hovertemplate="%{label}: %{customdata[0]:,}건 (%{percent})<extra></extra>",
            )
            _fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(_fig, use_container_width=True)
        else:
            st.info(T("실행된 액션이 없습니다."))
    with _legend_col:
        for _lbl, _clr, _crm_desc in _outcome_config:
            _cnt = _oc.get(_lbl, 0)
            st.markdown(
                f"""<div style="display:flex;align-items:flex-start;gap:10px;padding:8px 12px;border-radius:8px;margin-bottom:6px;background:#fafafa;">
                <div style="width:12px;height:12px;border-radius:50%;background:{_clr};flex-shrink:0;margin-top:3px;"></div>
                <div>
                    <div style="font-weight:600;font-size:14px;">{T(_lbl)} <span style="color:{_clr};font-weight:700;">{_cnt}</span></div>
                    <div style="font-size:11px;color:#666;line-height:1.4;">{T(_crm_desc)}</div>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    _persona_kr = {
        "price_sensitive": "가격 민감형",
        "vip_loyal": "VIP 충성 고객",
        "churn_progressing": "이탈 진행 고객",
        "explorer": "탐색형 고객",
        "coupon_sensitive": "쿠폰 민감형",
        "loyal_regular": "충성 일반 고객",
    }

    def _kr_persona(v: str) -> str:
        return _persona_kr.get(str(v).strip(), str(v))

    _exec_decision_kr = {
        "executed_as_recommended": "추천대로 실행",
        "executed_with_lower_intensity": "강도 낮춰 실행",
        "executed_with_higher_intensity": "강도 높여 실행",
        "skipped": "미실행",
        "manual_override": "수동 변경",
    }

    _detail_col_rename = {
        "customer_id": "고객 ID",
        "persona": "페르소나",
        "uplift_segment": "반응 유형",
        "coupon_cost": "쿠폰 비용",
        "actual_profit": "실제 이익",
        "actual_roi": "ROI",
        "outcome_label": "결과 분류",
        "recommended_action": "추천 액션",
        "executed": "실행 여부",
    }
    _detail_cols_order = ["customer_id", "persona", "outcome_label", "uplift_segment", "coupon_cost", "actual_profit", "actual_roi", "executed"]

    def _format_detail_df(src: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in _detail_cols_order if c in src.columns]
        out = src[cols].copy()
        if "persona" in out.columns:
            out["persona"] = out["persona"].map(_kr_persona)
        if "executed" in out.columns:
            out["executed"] = out["executed"].map({True: "실행", False: "미실행"})
        out = out.rename(columns=_detail_col_rename)
        return out

    _memo_json_path = _project_root() / "results_user" / "weekly_action_memos.json"

    def _load_memos() -> list[dict]:
        if _memo_json_path.exists():
            try:
                import json as _j
                data = _j.loads(_memo_json_path.read_text())
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def _save_memos(memos: list[dict]) -> None:
        import json as _j
        _memo_json_path.parent.mkdir(parents=True, exist_ok=True)
        _memo_json_path.write_text(_j.dumps(memos, ensure_ascii=False, indent=2))

    st.markdown(f"### {T('다음 주 정책 조정 제안')}")
    for _sug_idx, sug in enumerate(policy_suggestions):
        _sev = sug.get("severity", "info")
        _sug_amt = float(sug.get("amount", 0))
        _sug_action_text = sug.get("action", "")
        _sug_bg = {"warning": "#fff8e1", "info": "#e3f2fd", "success": "#e8f5e9"}.get(_sev, "#e3f2fd")
        _sug_border = {"warning": "#f9a825", "info": "#1976d2", "success": "#388e3c"}.get(_sev, "#1976d2")
        if _sug_amt < 0:
            _status = T("손실")
            _status_color = "#c62828"
            _amt_str = f"-{abs(_sug_amt):,.0f}원"
        elif _sug_amt > 0 and _sev == "success":
            _status = T("유지추천")
            _status_color = "#388e3c"
            _amt_str = f"+{_sug_amt:,.0f}원"
        elif _sug_amt > 0:
            _status = T("이득")
            _status_color = "#2e7d32"
            _amt_str = f"+{_sug_amt:,.0f}원"
        else:
            _status = T("개선기회")
            _status_color = "#1976d2"
            _amt_str = "-"

        _sug_title = sug["title"]
        _outcome_map = {"고비용 쿠폰 조정": "과잉 투자", "타겟 대상 재검토": "타겟 오류", "기대 미달 액션 점검": "기대 미달", "실행 누락 고객 추가": "실행 누락"}
        _mapped = _outcome_map.get(_sug_title, "")
        _sug_cust = pd.DataFrame()
        if _mapped and not action_log.empty:
            _sug_cust = action_log[action_log["outcome_label"] == _mapped]
        elif "세그먼트" in _sug_title and not action_log.empty and "uplift_segment" in action_log.columns:
            _seg_n = _sug_title.replace(" 세그먼트 적자", "").replace(" 세그먼트 유지", "")
            _sug_cust = action_log[action_log["uplift_segment"] == _seg_n]
        _sug_cust_cnt = len(_sug_cust)

        _card_c, _memo_c = st.columns([9, 1])
        with _card_c:
            st.markdown(
                f"""<div style="border-left:4px solid {_sug_border};background:{_sug_bg};padding:10px 16px;border-radius:0 8px 8px 0;margin-bottom:2px;">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                    <span style="font-size:14px;font-weight:700;">{T(_sug_title)}</span>
                    <span style="font-size:18px;font-weight:800;color:{_status_color};">{_amt_str}</span>
                    <span style="font-size:11px;background:{_status_color}15;color:{_status_color};padding:2px 8px;border-radius:10px;font-weight:600;">{_status}</span>
                    <span style="font-size:12px;color:#666;">→ {T(_sug_action_text)}</span>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with _memo_c:
            if st.button("📝", key=f"add_memo_{_sug_idx}", help=T("메모에 추가")):
                _top_ids = []
                if not _sug_cust.empty and "customer_id" in _sug_cust.columns:
                    _sorted_cust = _sug_cust.sort_values("actual_profit", ascending=True)
                    _top_n = max(1, int(len(_sorted_cust) * 0.2))
                    _top_ids = _sorted_cust["customer_id"].head(_top_n).tolist()
                _new_memo = {
                    "week_start": _week_start.strftime("%Y-%m-%d"),
                    "week_end": _week_end.strftime("%Y-%m-%d"),
                    "title": _sug_title,
                    "impact_type": _status,
                    "impact_amount": _sug_amt,
                    "customer_count": _sug_cust_cnt,
                    "recommended_action": _sug_action_text,
                    "top_customers": _top_ids,
                    "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                }
                _existing = _load_memos()
                _existing.append(_new_memo)
                _save_memos(_existing)
                _memo_line = f"[{_status}] {_sug_title} {_amt_str} | {_sug_cust_cnt}명 대상 | 조치: {_sug_action_text}"
                if _top_ids:
                    _top_str = ", ".join(str(x) for x in _top_ids[:5])
                    _memo_line += f"\n  └ 상위 {len(_top_ids)}명(상위20%): {_top_str}" + ("..." if len(_top_ids) > 5 else "")
                st.session_state["review_memo_area"] = st.session_state.get("review_memo_area", "") + f"• {_memo_line}\n"
                st.rerun()

        with st.expander(T("상세 보기"), expanded=False):
            st.markdown(f"**{T('결정 근거')}:** {T(sug.get('what', ''))}")
            st.markdown(f"**{T('추천 조치')}:** {T(_sug_action_text)}")
            if not _sug_cust.empty:
                st.caption(f"{T('관련 고객')} — {len(_sug_cust)}{T('건')}")
                st.dataframe(_format_detail_df(_sug_cust.sort_values("actual_profit")), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown(f"### {T('운영 메모')}")
    if st.session_state.pop("_reset_review_memo_flag", False):
        st.session_state["review_memo_area"] = ""
    st.text_area(
        T("운영 메모"), key="review_memo_area", height=100, label_visibility="collapsed",
        placeholder=T("매주 월요일 리뷰 후 다음 주 액션을 메모하세요."),
    )

    _memo_btn1, _memo_btn2, _memo_btn3, _memo_btn4 = st.columns(4)
    with _memo_btn1:
        if st.button("💾 " + T("메모 저장"), key="save_memo_14"):
            _text_memo = st.session_state.get("review_memo_area", "")
            if _text_memo.strip():
                _existing = _load_memos()
                _existing.append({
                    "week_start": _week_start.strftime("%Y-%m-%d"),
                    "week_end": _week_end.strftime("%Y-%m-%d"),
                    "title": "수동 메모",
                    "impact_type": "-",
                    "impact_amount": 0,
                    "customer_count": 0,
                    "recommended_action": _text_memo.strip(),
                    "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                })
                _save_memos(_existing)
                st.success(T("저장 완료"))
            else:
                st.warning(T("메모 내용이 비어있습니다."))
    with _memo_btn2:
        _saved_memos = _load_memos()
        if _saved_memos:
            with st.expander(f"📂 {T('저장된 메모 보기')} ({len(_saved_memos)}{T('건')})"):
                for _sm in reversed(_saved_memos[-20:]):
                    _sm_icon = "🔴" if _sm.get("impact_type") == "손실" else ("🟢" if _sm.get("impact_type") in ("이득", "유지추천") else "🔵")
                    _sm_amt = float(_sm.get("impact_amount", 0))
                    _sm_line = f"{_sm_icon} **{_sm.get('title', '')}** {_sm_amt:+,.0f}원 · {_sm.get('customer_count', 0)}명 · {_sm.get('week_start', '')}"
                    st.markdown(f"<div style='font-size:13px;padding:4px 0;'>{_sm_line}</div>", unsafe_allow_html=True)
                    st.caption(f"  → {_sm.get('recommended_action', '')} ({_sm.get('created_at', '')[:16]})")
        else:
            st.caption(T("저장된 메모가 없습니다."))
    with _memo_btn3:
        _saved_for_dl = _load_memos()
        if _saved_for_dl:
            import json as _json_dl
            st.download_button(
                "📥 " + T("메모 JSON 다운로드"),
                _json_dl.dumps(_saved_for_dl, ensure_ascii=False, indent=2).encode("utf-8"),
                "weekly_action_memos.json", "application/json",
                key="dl_memo_json_14",
            )
        else:
            st.caption("-")
    with _memo_btn4:
        _confirm_reset = st.checkbox(T("정말 초기화하시겠습니까?"), key="confirm_memo_reset_14")
        if _confirm_reset:
            if st.button("🗑️ " + T("메모 초기화"), key="reset_memo_14"):
                _save_memos([])
                st.session_state["_reset_review_memo_flag"] = True
                st.rerun()

    st.divider()

    _target_outcomes = ["실행 누락", "타겟 오류", "과잉 투자", "기대 미달"]
    _csv_customers = action_log[action_log["outcome_label"].isin(_target_outcomes)].copy() if not action_log.empty else pd.DataFrame()
    if not _csv_customers.empty:
        _csv_customers = _csv_customers.sort_values("actual_profit", ascending=True)
        _priority_map = {"실행 누락": "높음", "과잉 투자": "높음", "타겟 오류": "중간", "기대 미달": "낮음"}
        _next_action_map = {
            "실행 누락": "다음 주 우선 실행 대상에 추가",
            "과잉 투자": "쿠폰 금액 하향 또는 메시지 전환",
            "타겟 오류": "타겟에서 제외 또는 모니터링 전환",
            "기대 미달": "쿠폰 강도 또는 타이밍 조정",
        }
        _reason_map = {
            "실행 누락": "기대 ROI가 높았지만 지난주 실행되지 않아 기회손실 발생",
            "과잉 투자": "쿠폰 비용 대비 전환 실패로 손실 발생",
            "타겟 오류": "액션 실행했으나 고객 반응 없음",
            "기대 미달": "이익 발생했으나 기대 ROI 대비 부족",
        }
        _csv_out = pd.DataFrame()
        _csv_out["고객 ID"] = _csv_customers.get("customer_id", "")
        _csv_out["페르소나"] = _csv_customers["persona"].map(_kr_persona) if "persona" in _csv_customers.columns else "-"
        _csv_out["결과 분류"] = _csv_customers.get("outcome_label", "-")
        _csv_out["지난주 추천 액션"] = _csv_customers.get("recommended_action", "-") if "recommended_action" in _csv_customers.columns else "-"
        _csv_out["지난주 실행 판단"] = _csv_customers["executed"].map({True: "실행", False: "미실행"}) if "executed" in _csv_customers.columns else "-"
        _csv_out["지난주 쿠폰 비용"] = _csv_customers.get("coupon_cost", 0)
        _csv_out["지난주 실제 이익"] = _csv_customers.get("actual_profit", 0).round(0).astype(int)
        _csv_out["지난주 ROI"] = _csv_customers.get("actual_roi", 0).round(2) if "actual_roi" in _csv_customers.columns else 0
        _csv_out["다음 주 권장 액션"] = _csv_customers["outcome_label"].map(_next_action_map).fillna("-")
        _csv_out["액션 사유"] = _csv_customers["outcome_label"].map(_reason_map).fillna("-")
        _csv_out["우선순위"] = _csv_customers["outcome_label"].map(_priority_map).fillna("-")
        _csv_out["예상 개선 금액"] = np.where(
            _csv_customers["actual_profit"].values < 0,
            (-_csv_customers["actual_profit"].values).round(0).astype(int),
            (_csv_customers["expected_incremental_profit"].values * 0.5).round(0).astype(int) if "expected_incremental_profit" in _csv_customers.columns else 0,
        )
        _csv_fname = f"next_week_customer_actions_{_week_end.strftime('%Y-%m-%d')}.csv"
        st.download_button(
            f"📥 {T('다음 주 고객 액션 CSV')} ({len(_csv_out)}{T('건')})",
            _csv_out.to_csv(index=False).encode("utf-8-sig"),
            _csv_fname, "text/csv",
            key="dl_customer_csv_14",
        )
    else:
        st.caption(T("대상 고객이 없습니다."))


    st.markdown(f"### {T('예상과 다른 반응을 보인 고객')}")
    _all_actioned = action_log[action_log["outcome_label"] != "해당 없음"] if not action_log.empty else pd.DataFrame()
    if _all_actioned.empty:
        st.info(T("실행된 액션이 없습니다."))
    else:
        _cs1, _cs2, _cs3, _cs4 = st.columns(4)
        _cs1.metric(T("대상 고객 수"), f"{len(_all_actioned):,}")
        _cs2.metric(T("평균 기대 ROI"), f"{float(_all_actioned['expected_roi'].mean()):.2f}" if "expected_roi" in _all_actioned.columns else "-")
        _cs3.metric(T("평균 실제 ROI"), f"{float(_all_actioned['actual_roi'].mean()):.2f}" if "actual_roi" in _all_actioned.columns else "-")
        if "uplift_segment" in _all_actioned.columns:
            _cs4.metric(T("주요 세그먼트"), T(str(_all_actioned["uplift_segment"].value_counts().idxmax())))

        def _on_filter_change():
            st.session_state["review_search_cid_14"] = T("선택 안 함")

        _fc1, _fc2 = st.columns([1, 1])
        with _fc1:
            _filter_options = [T("전체")] + [T(lbl) for lbl, _, _ in _outcome_config]
            _selected_filter = st.selectbox(
                T("판정 필터"), _filter_options, key="review_filter_14",
                on_change=_on_filter_change,
            )

        _t_reverse = {T(lbl): lbl for lbl, _, _ in _outcome_config}
        if _selected_filter == T("전체"):
            _filtered_log = _all_actioned.copy()
        else:
            _orig_label = _t_reverse.get(_selected_filter, _selected_filter)
            _filtered_log = _all_actioned[_all_actioned["outcome_label"] == _orig_label].copy()
        _filtered_log = _filtered_log.sort_values("actual_profit", ascending=True)

        _filtered_cid_list = sorted(_filtered_log["customer_id"].unique().tolist()) if not _filtered_log.empty and "customer_id" in _filtered_log.columns else []
        with _fc2:
            _search_cid = st.selectbox(
                T("고객 ID 검색"), [T("선택 안 함")] + _filtered_cid_list, key="review_search_cid_14",
            )

        if _search_cid != T("선택 안 함"):
            _match = _all_actioned[_all_actioned["customer_id"] == _search_cid]
            if not _match.empty:
                _row = _match.iloc[0]
                _cid = _row.get("customer_id", "?")
                _persona = _row.get("persona", "-")
                _seg = _row.get("uplift_segment", "-")
                _outcome = _row.get("outcome_label", "-")
                _a_profit = _row.get("actual_profit", 0)
                _e_profit = _row.get("expected_incremental_profit", 0)
                _coupon = _row.get("coupon_cost", 0)
                _e_roi = _row.get("expected_roi", 0)
                _a_roi = _row.get("actual_roi", 0)
                _category = _row.get("recommended_category", "-")
                _executed = _row.get("executed", False)
                _converted = _row.get("actual_conversion", False)
                _redeemed = _row.get("coupon_redeemed", False)
                _oc_emoji = {"적정 판단": "🟢", "기대 미달": "🟡", "과잉 투자": "🔴", "타겟 오류": "🟠", "실행 누락": "🟣"}
                _emoji = _oc_emoji.get(_outcome, "⚪")
                _exec_badge = "실행" if _executed else "미실행"
                st.markdown(f"#### {_emoji} {_cid}  ·  {T(_outcome)}  ·  {T(_persona)} / {T(_seg)}")
                _s1, _s2, _s3, _s4, _s5 = st.columns(5)
                _s1.metric(T("실행 여부"), T(_exec_badge))
                _s2.metric(T("쿠폰 비용"), money(_coupon))
                _s3.metric(T("기대 이익"), money(_e_profit))
                _s4.metric(T("실제 이익"), money(_a_profit), delta=f"{_a_profit - _e_profit:+,.0f}")
                _s5.metric(T("ROI"), f"{_a_roi:.2f}", delta=f"{_a_roi - _e_roi:+.2f}")
                _info_cols = st.columns(4)
                _info_cols[0].markdown(f"**{T('추천 카테고리')}:** {T(_category)}")
                _info_cols[1].markdown(f"**{T('쿠폰 사용')}:** {'O' if _redeemed else 'X'}")
                _info_cols[2].markdown(f"**{T('전환')}:** {'O' if _converted else 'X'}")
                _intensity = _row.get("intervention_intensity_label", "-")
                _info_cols[3].markdown(f"**{T('개입 강도')}:** {T(_intensity)}")
                _cid_val = _cid
                _cust_events = _review_events[_review_events["customer_id"] == _cid_val] if not _review_events.empty and "customer_id" in _review_events.columns else pd.DataFrame()
                _cust_orders = _review_orders[_review_orders["customer_id"] == _cid_val] if not _review_orders.empty and "customer_id" in _review_orders.columns else pd.DataFrame()
                _cust_campaigns = _review_campaigns[_review_campaigns["customer_id"] == _cid_val] if not _review_campaigns.empty and "customer_id" in _review_campaigns.columns else pd.DataFrame()
                _timeline_tab1, _timeline_tab2, _timeline_tab3 = st.tabs([
                    f"{T('이벤트 로그')} ({len(_cust_events)})",
                    f"{T('주문 내역')} ({len(_cust_orders)})",
                    f"{T('쿠폰 이력')} ({len(_cust_campaigns)})",
                ])
                with _timeline_tab1:
                    if not _cust_events.empty:
                        _evt_display = _cust_events.sort_values("timestamp", ascending=False).head(15)
                        _evt_cols = [c for c in ["timestamp", "event_type", "item_category", "quantity"] if c in _evt_display.columns]
                        st.dataframe(_evt_display[_evt_cols], use_container_width=True, hide_index=True)
                        if len(_cust_events) > 15:
                            st.caption(f"{T('최근')} 15{T('건만 표시')} (전체 {len(_cust_events)}건)")
                    else:
                        st.caption(T("이벤트 기록 없음"))
                with _timeline_tab2:
                    if not _cust_orders.empty:
                        _ord_display = _cust_orders.sort_values("order_time", ascending=False)
                        _ord_cols = [c for c in ["order_time", "item_category", "net_amount", "discount_amount", "coupon_used"] if c in _ord_display.columns]
                        st.dataframe(_ord_display[_ord_cols], use_container_width=True, hide_index=True)
                        _total_spend = _cust_orders["net_amount"].sum() if "net_amount" in _cust_orders.columns else 0
                        _coupon_used_cnt = int(_cust_orders["coupon_used"].sum()) if "coupon_used" in _cust_orders.columns else 0
                        st.caption(f"{T('총 구매')} {money(_total_spend)} · {T('쿠폰 사용')} {_coupon_used_cnt}{T('회')}")
                    else:
                        st.caption(T("주문 기록 없음"))
                with _timeline_tab3:
                    if not _cust_campaigns.empty:
                        _camp_display = _cust_campaigns.sort_values("exposure_time", ascending=False)
                        _camp_cols = [c for c in ["exposure_time", "campaign_type", "coupon_cost"] if c in _camp_display.columns]
                        st.dataframe(_camp_display[_camp_cols], use_container_width=True, hide_index=True)
                        _total_coupon = _cust_campaigns["coupon_cost"].sum() if "coupon_cost" in _cust_campaigns.columns else 0
                        st.caption(f"{T('총 쿠폰 지급')} {money(_total_coupon)} · {len(_cust_campaigns)}{T('회')}")
                    else:
                        st.caption(T("쿠폰 이력 없음"))
                st.divider()

        _display_limit = 30
        _filtered_display = _filtered_log.head(_display_limit)
        if _filtered_display.empty:
            st.info(T("해당 판정의 고객이 없습니다."))
        else:
            for _, _row in _filtered_display.iterrows():
                _cid = _row.get("customer_id", "?")
                _persona = _row.get("persona", "-")
                _seg = _row.get("uplift_segment", "-")
                _outcome = _row.get("outcome_label", "-")
                _a_profit = _row.get("actual_profit", 0)
                _coupon = _row.get("coupon_cost", 0)
                _oc_emoji = {"적정 판단": "🟢", "기대 미달": "🟡", "과잉 투자": "🔴", "타겟 오류": "🟠", "실행 누락": "🟣"}
                _emoji = _oc_emoji.get(_outcome, "⚪")
                _profit_sign = "+" if _a_profit >= 0 else ""
                _header = (
                    f"{_emoji} **{_cid}** [{T(_outcome)}]  ·  "
                    f"{T(_persona)} / {T(_seg)}  ·  "
                    f"{T('쿠폰')} {money(_coupon)} → {T('손익')} **{_profit_sign}{money(_a_profit)}**"
                )
                with st.expander(_header, expanded=False):
                    _e_profit = _row.get("expected_incremental_profit", 0)
                    _e_roi = _row.get("expected_roi", 0)
                    _a_roi = _row.get("actual_roi", 0)
                    _executed = _row.get("executed", False)
                    _exec_badge = "실행" if _executed else "미실행"
                    _s1, _s2, _s3, _s4, _s5 = st.columns(5)
                    _s1.metric(T("실행 여부"), T(_exec_badge))
                    _s2.metric(T("쿠폰 비용"), money(_coupon))
                    _s3.metric(T("기대 이익"), money(_e_profit))
                    _s4.metric(T("실제 이익"), money(_a_profit), delta=f"{_a_profit - _e_profit:+,.0f}")
                    _s5.metric(T("ROI"), f"{_a_roi:.2f}", delta=f"{_a_roi - _e_roi:+.2f}")
            if len(_filtered_log) > _display_limit:
                st.caption(f"{T('상위')} {_display_limit}{T('건만 표시')} (전체 {len(_filtered_log)}건)")

    if st.session_state.get("show_report_14", False):
        @st.dialog(T("주간 리텐션 액션 성과 보고서"), width="large")
        def _report_dialog():
            st.caption(f"{T('분석 기간')}: {_week_start.strftime('%Y-%m-%d')} ~ {_week_end.strftime('%Y-%m-%d')}")
            st.markdown("## 1. 지난주 마케팅 현황")
            st.markdown("### 1-1. 추천 고객과 실행 고객의 일치도")
            st.markdown(f"- 전체 추천 대상: **{_total:,}명**")
            st.markdown(f"- 실제 집행: **{_executed_cnt:,}명** (집행률 {_exec_pct:.1%})")
            st.markdown(f"- 미집행: **{_total - _executed_cnt:,}명**")
            st.markdown("### 1-2. 예산 사용 내역")
            st.markdown(f"- 총 집행 예산: **{money(_budget)}**")
            if not action_log.empty and "uplift_segment" in action_log.columns:
                _rpt_exec = action_log[action_log["executed"]]
                if not _rpt_exec.empty:
                    _rpt_seg_budget = _rpt_exec.groupby("uplift_segment")["actual_coupon_cost"].sum().sort_values(ascending=False)
                    for _rs, _ra in _rpt_seg_budget.items():
                        st.markdown(f"  - {_rs}: {money(_ra)}")
            st.markdown("### 1-3. 결과 및 기대 수익과의 차이")
            st.markdown(f"- 기대 이익: **{money(_expected)}**")
            st.markdown(f"- 실제 이익: **{money(_actual)}**")
            st.markdown(f"- 차이: **{_gap:+,.0f}원** ({'초과 달성' if _gap_positive else '미달'})")
            st.markdown("## 2. 분석")
            st.markdown("### 2-1. 기대 수익이 나오지 않은 이유")
            for _nl in ["과잉 투자", "타겟 오류", "기대 미달", "실행 누락"]:
                _nl_df = action_log[action_log["outcome_label"] == _nl] if not action_log.empty else pd.DataFrame()
                if _nl_df.empty:
                    continue
                st.markdown(f"**{_nl}** — {len(_nl_df)}건, {float(_nl_df['actual_profit'].sum()):+,.0f}원")
                if "uplift_segment" in _nl_df.columns:
                    _nl_segs = _nl_df["uplift_segment"].value_counts().head(3)
                    st.markdown(f"  - 주요 세그먼트: {', '.join(f'{s} {c}건' for s, c in _nl_segs.items())}")
                st.markdown(f"  - 평균 쿠폰 비용: {money(float(_nl_df['coupon_cost'].mean()))}")
            st.markdown("### 2-2. 기대 수익보다 높았던 고객들의 특성")
            _rpt_pos = action_log[action_log["outcome_label"] == "적정 판단"] if not action_log.empty else pd.DataFrame()
            if _rpt_pos.empty:
                st.markdown("해당 고객 없음")
            else:
                st.markdown(f"- 대상: **{len(_rpt_pos)}명**, 총 이익: **+{float(_rpt_pos['actual_profit'].sum()):,.0f}원**")
                if "uplift_segment" in _rpt_pos.columns:
                    st.markdown(f"- 주요 세그먼트: {_rpt_pos['uplift_segment'].value_counts().idxmax()}")
                if "persona" in _rpt_pos.columns:
                    st.markdown(f"- 주요 페르소나: {_rpt_pos['persona'].value_counts().idxmax()}")
                st.markdown(f"- 평균 ROI: {float(_rpt_pos['actual_roi'].mean()):.2f}")
            st.markdown("## 3. 다음 주 마케팅 전략 방안")
            for sug in policy_suggestions:
                _sa = float(sug.get("amount", 0))
                st.markdown(f"**{sug['title']}** ({_sa:+,.0f}원)")
                st.markdown(f"  - 결정: {sug.get('what', '')}")
                st.markdown(f"  - 대상: {sug.get('who', '')}")
                st.markdown(f"  - 조치: {sug.get('action', '')}")
            _memo_content = st.session_state.get("review_memo_area", "")
            if _memo_content:
                st.markdown("## 4. 담당자 실행 메모")
                st.markdown(_memo_content)
            st.divider()
            _rpt_plain = (
                f"[주간 리텐션 액션 성과 보고서]\n"
                f"기간: {_week_start.strftime('%Y-%m-%d')} ~ {_week_end.strftime('%Y-%m-%d')}\n\n"
                f"■ 마케팅 현황\n"
                f"  추천 대상: {_total:,}명 / 실제 집행: {_executed_cnt:,}명 (집행률 {_exec_pct:.1%})\n"
                f"  총 집행 예산: {money(_budget)}\n"
                f"  기대 이익: {money(_expected)} → 실제 이익: {money(_actual)} (Gap: {_gap:+,.0f}원)\n\n"
                f"■ 분석\n"
            )
            for _nl in ["과잉 투자", "타겟 오류", "기대 미달", "실행 누락"]:
                _nl_df = action_log[action_log["outcome_label"] == _nl] if not action_log.empty else pd.DataFrame()
                if not _nl_df.empty:
                    _rpt_plain += f"  {_nl}: {len(_nl_df)}건, {float(_nl_df['actual_profit'].sum()):+,.0f}원\n"
            _rpt_plain += f"\n■ 다음 주 전략\n"
            for sug in policy_suggestions:
                _sa = float(sug.get("amount", 0))
                _rpt_plain += f"  • {sug['title']} ({_sa:+,.0f}원): {sug.get('action', '')}\n"
            if _memo_content:
                _rpt_plain += f"\n■ 담당자 메모\n{_memo_content}\n"
            with st.expander("📋 " + T("복사하기") + " — " + T("보고서 내용 (복사용)")):
                st.code(_rpt_plain, language=None)
            if st.button("✕ " + T("닫기"), key="close_report_dialog_14"):
                st.session_state["show_report_14"] = False
                st.rerun()
        _report_dialog()

    llm_payload = {
        "review_summary": review_summary,
        "outcome_counts": review_summary.get("outcome_counts", {}),
        "policy_suggestions": [{"title": s["title"], "amount": s.get("amount", 0), "what": s.get("what", ""), "action": s.get("action", "")} for s in policy_suggestions],
        "top_loss_actions": (
            action_log[action_log["actual_profit"] < 0]
            .nsmallest(10, "actual_profit")[["customer_id", "persona", "recommended_category", "expected_roi", "actual_roi", "actual_profit", "outcome_label"]]
            .to_dict(orient="records")
        ) if not action_log.empty and (action_log["actual_profit"] < 0).any() else [],
    }



current_view_key = view.split(".")[0]
current_model_name = llm_model.strip() or DEFAULT_MODEL_NAME

_llm_summary_ready, _llm_summary_status = get_llm_status(llm_api_key_value)
if llm_enabled and _llm_summary_ready:
    render_llm_summary(
        view_key=current_view_key,
        view_title=llm_view_title,
        payload=llm_payload,
        api_key=llm_api_key_value,
        model_name=current_model_name,
    )
elif llm_enabled and not _llm_summary_ready:
    # Do not render the main LLM summary block when the API key is missing.
    # Keeping the notice in the sidebar prevents it from overlapping tables.
    pass

with st.sidebar:
    render_sidebar_chatbot_launcher(
        view_key=current_view_key,
        view_title=llm_view_title,
        llm_enabled=llm_enabled,
        api_key=llm_api_key_value,
        payload=llm_payload,
        model_name=current_model_name,
        )

if globals().get("_global_demo_autorefresh_active", False):
            import time as _demo_time
            _snapshot_analysis_controls()
            _demo_time.sleep(10)
            clear_dashboard_caches()
            st.rerun()
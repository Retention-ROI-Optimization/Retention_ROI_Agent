"""UI labeling, table cleanup, and i18n helpers for the retention dashboard.

This module intentionally contains only presentation-layer helpers. It does not
change model outputs or optimization logic; it only makes backend column names,
event names, and generated action labels readable in Korean/English/Japanese.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


def _norm(value: Any) -> str:
    return re.sub(r"[\s_\-:：/\.()\[\]{}]+", "", str(value or "")).lower()


COLUMN_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "customer_id": "고객 ID",
        "persona": "고객 유형",
        "customer_segment": "고객 유형",
        "client_no": "고객 번호",
        "account_no": "계좌 번호",
        "transaction_date": "거래일",
        "transaction_time": "거래 시각",
        "transaction_amount": "거래금액",
        "txn_amount": "거래금액",
        "product_type": "금융상품 유형",
        "banking_event": "금융 이벤트 유형",
        "account_status": "계좌/거래 상태",
        "churn_probability": "이탈 위험도",
        "churn_score": "이탈 점수",
        "realtime_churn_score": "실시간 이탈 점수",
        "base_churn_probability": "기준 이탈 위험도",
        "score_delta": "점수 변화",
        "clv": "고객 생애가치",
        "uplift_score": "개입 반응 가능성",
        "uplift_segment": "개입 반응 유형",
        "risk_segment": "위험 등급",
        "expected_roi": "예상 ROI",
        "expected_roi_2": "예상 ROI",
        "queued_expected_roi": "큐 예상 ROI",
        "expected_profit": "예상 이익",
        "expected_incremental_profit": "예상 추가 이익",
        "queued_expected_profit": "큐 예상 이익",
        "coupon_cost": "추천 투자액",
        "queued_coupon_cost": "큐 투자액",
        "recommended_investment_amount": "추천 투자액",
        "allocated_budget": "배정 예산",
        "customer_count": "선정 고객 수",
        "candidate_customer_count": "후보 고객 수",
        "intervention_intensity": "개입 강도",
        "queued_intervention_intensity": "큐 개입 강도",
        "recommended_action": "추천 액션",
        "queued_recommended_action": "큐 추천 액션",
        "priority_score": "우선순위 점수",
        "selection_score": "선정 점수",
        "recommended_intervention_window": "추천 개입 시점",
        "recommended_category": "추천 카테고리",
        "recommendation_rank": "추천 순위",
        "recommendation_score": "추천 점수",
        "reason_tags": "추천 이유",
        "selection_reason": "선정 이유",
        "reason_summary": "선정 이유",
        "action_status": "액션 상태",
        "action_queue_status": "액션 큐 상태",
        "source_type": "발생 경로",
        "trigger_reason": "트리거 이유",
        "latest_trigger_reason": "최근 트리거 이유",
        "updated_at": "갱신 시각",
        "scored_at": "점수 산출 시각",
        "last_event_type": "최근 이벤트",
        # finance-friendly labels
        "deposit_balance": "예금 잔액",
        "balance": "잔액",
        "balance_drop": "잔액 감소",
        "loan_balance": "대출 잔액",
        "loan_repayment_delay": "대출 상환 지연",
        "delinquency_days": "연체 일수",
        "card_spend": "카드 사용액",
        "card_spend_drop": "카드 사용 감소",
        "transaction_count": "거래 횟수",
        "large_transfer": "대규모 이체",
        "deposit_withdrawal": "예금 인출",
        "support_complaint": "상담/불만 접수",
        "support_complaints": "상담/불만 건수",
    },
    "en": {
        "customer_id": "Customer ID",
        "persona": "Customer Type",
        "customer_segment": "Customer Type",
        "churn_probability": "Churn Risk",
        "churn_score": "Churn Score",
        "realtime_churn_score": "Real-time Churn Score",
        "base_churn_probability": "Baseline Churn Risk",
        "score_delta": "Score Change",
        "clv": "Customer Lifetime Value",
        "uplift_score": "Response Likelihood",
        "uplift_segment": "Response Type",
        "risk_segment": "Risk Level",
        "expected_roi": "Expected ROI",
        "expected_roi_2": "Expected ROI",
        "queued_expected_roi": "Queued Expected ROI",
        "expected_profit": "Expected Profit",
        "expected_incremental_profit": "Expected Incremental Profit",
        "queued_expected_profit": "Queued Expected Profit",
        "coupon_cost": "Recommended Investment",
        "queued_coupon_cost": "Queued Investment",
        "recommended_investment_amount": "Recommended Investment",
        "allocated_budget": "Allocated Budget",
        "customer_count": "Selected Customers",
        "candidate_customer_count": "Candidate Customers",
        "intervention_intensity": "Intervention Intensity",
        "queued_intervention_intensity": "Queued Intensity",
        "recommended_action": "Recommended Action",
        "queued_recommended_action": "Queued Action",
        "priority_score": "Priority Score",
        "selection_score": "Selection Score",
        "recommended_intervention_window": "Recommended Timing",
        "recommended_category": "Recommended Category",
        "recommendation_rank": "Rank",
        "recommendation_score": "Recommendation Score",
        "reason_tags": "Recommendation Reasons",
        "selection_reason": "Reason Selected",
        "reason_summary": "Reason Selected",
        "action_status": "Action Status",
        "action_queue_status": "Action Queue Status",
        "source_type": "Source",
        "trigger_reason": "Trigger Reason",
        "latest_trigger_reason": "Latest Trigger Reason",
        "updated_at": "Updated At",
        "scored_at": "Scored At",
        "last_event_type": "Latest Event",
        "deposit_balance": "Deposit Balance",
        "balance": "Balance",
        "balance_drop": "Balance Drop",
        "loan_balance": "Loan Balance",
        "loan_repayment_delay": "Loan Repayment Delay",
        "delinquency_days": "Days Delinquent",
        "card_spend": "Card Spend",
        "card_spend_drop": "Card Spend Drop",
        "transaction_count": "Transaction Count",
        "large_transfer": "Large Transfer",
        "deposit_withdrawal": "Deposit Withdrawal",
        "support_complaint": "Support Complaint",
        "support_complaints": "Support Complaints",
    },
    "ja": {
        "customer_id": "顧客ID",
        "persona": "顧客タイプ",
        "customer_segment": "顧客タイプ",
        "churn_probability": "離脱リスク",
        "churn_score": "離脱スコア",
        "realtime_churn_score": "リアルタイム離脱スコア",
        "base_churn_probability": "基準離脱リスク",
        "score_delta": "スコア変化",
        "clv": "顧客生涯価値",
        "uplift_score": "介入反応見込み",
        "uplift_segment": "介入反応タイプ",
        "risk_segment": "リスク等級",
        "expected_roi": "予想ROI",
        "expected_roi_2": "予想ROI",
        "queued_expected_roi": "キュー予想ROI",
        "expected_profit": "予想利益",
        "expected_incremental_profit": "予想追加利益",
        "queued_expected_profit": "キュー予想利益",
        "coupon_cost": "推奨投資額",
        "queued_coupon_cost": "キュー投資額",
        "recommended_investment_amount": "推奨投資額",
        "allocated_budget": "配分予算",
        "customer_count": "選定顧客数",
        "candidate_customer_count": "候補顧客数",
        "intervention_intensity": "介入強度",
        "queued_intervention_intensity": "キュー介入強度",
        "recommended_action": "推奨アクション",
        "queued_recommended_action": "キュー推奨アクション",
        "priority_score": "優先度スコア",
        "selection_score": "選定スコア",
        "recommended_intervention_window": "推奨介入タイミング",
        "recommended_category": "推薦カテゴリ",
        "recommendation_rank": "推薦順位",
        "recommendation_score": "推薦スコア",
        "reason_tags": "推薦理由",
        "selection_reason": "選定理由",
        "reason_summary": "選定理由",
        "action_status": "アクション状態",
        "action_queue_status": "アクションキュー状態",
        "source_type": "発生経路",
        "trigger_reason": "トリガー理由",
        "latest_trigger_reason": "最新トリガー理由",
        "updated_at": "更新時刻",
        "scored_at": "スコア算出時刻",
        "last_event_type": "最新イベント",
        "deposit_balance": "預金残高",
        "balance": "残高",
        "balance_drop": "残高減少",
        "loan_balance": "融資残高",
        "loan_repayment_delay": "融資返済遅延",
        "delinquency_days": "延滞日数",
        "card_spend": "カード利用額",
        "card_spend_drop": "カード利用減少",
        "transaction_count": "取引回数",
        "large_transfer": "大口送金",
        "deposit_withdrawal": "預金引き出し",
        "support_complaint": "相談・苦情受付",
        "support_complaints": "相談・苦情件数",
    },
}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "generic_retention_offer": "기본 리텐션 혜택",
        "generic retention offer": "기본 리텐션 혜택",
        "personalized_retention_offer": "개인화 리텐션 혜택",
        "high_value_retention_coupon": "고가치 고객 리텐션 쿠폰",
        "coupon_offer": "쿠폰 혜택",
        "discount_offer": "할인 혜택",
        "service_recovery": "서비스 회복 안내",
        "loyalty_reward": "충성 고객 보상",
        "medium": "중강도",
        "high": "고강도",
        "low": "저강도",
        "medium_intensity": "중강도",
        "high_intensity": "고강도",
        "low_intensity": "저강도",
        "중강도": "중강도",
        "고강도": "고강도",
        "저강도": "저강도",
        "queued": "큐에 적재됨",
        "not_queued": "미적재",
        "pending": "대기 중",
        "sent": "발송 완료",
        "completed": "완료",
        "failed": "실패",
        "own_purchase_history": "본인 구매 이력",
        "recent_browse_signal": "최근 탐색 신호",
        "segment_popularity": "유사 고객군 인기",
        "global_popularity": "전체 인기",
        "churn_progressing": "이탈 진행 고객",
        "new_signup": "신규 가입 고객",
        "price_sensitive": "가격 민감 고객",
        "loyal_vip_customer": "충성 VIP 고객",
        "로열VIP고객": "충성 VIP 고객",
        "로열일반고객": "충성 일반 고객",
        "loyal regular customer": "충성 일반 고객",
        "Monitor": "관찰",
        "monitor": "관찰",
        "page_view": "페이지 방문",
        "purchase": "구매",
        "cart": "장바구니",
        "add_to_cart": "장바구니 담기",
        "search": "검색",
        "login": "로그인",
        "large_transfer": "대규모 이체",
        "loan_repayment_delay": "대출 상환 지연",
        "card_spend_drop": "카드 사용 감소",
        "balance_drop": "잔액 감소",
        "support_complaint": "상담/불만 접수",
        "deposit_withdrawal": "예금 인출",
    },
    "en": {
        "generic_retention_offer": "Basic retention offer",
        "generic retention offer": "Basic retention offer",
        "personalized_retention_offer": "Personalized retention offer",
        "high_value_retention_coupon": "High-value retention coupon",
        "coupon_offer": "Coupon offer",
        "discount_offer": "Discount offer",
        "service_recovery": "Service recovery message",
        "loyalty_reward": "Loyalty reward",
        "medium": "Medium intensity",
        "high": "High intensity",
        "low": "Low intensity",
        "medium_intensity": "Medium intensity",
        "high_intensity": "High intensity",
        "low_intensity": "Low intensity",
        "중강도": "Medium intensity",
        "고강도": "High intensity",
        "저강도": "Low intensity",
        "queued": "Queued",
        "not_queued": "Not queued",
        "pending": "Pending",
        "sent": "Sent",
        "completed": "Completed",
        "failed": "Failed",
        "own_purchase_history": "Own purchase history",
        "recent_browse_signal": "Recent browsing signal",
        "segment_popularity": "Segment popularity",
        "global_popularity": "Overall popularity",
        "churn_progressing": "Churn-risk customer",
        "new_signup": "New signup",
        "price_sensitive": "Price-sensitive customer",
        "loyal_vip_customer": "Loyal VIP customer",
        "로열VIP고객": "Loyal VIP customer",
        "로열일반고객": "Loyal regular customer",
        "Monitor": "Monitor",
        "monitor": "Monitor",
        "page_view": "Page visit",
        "purchase": "Purchase",
        "cart": "Cart",
        "add_to_cart": "Add to cart",
        "search": "Search",
        "login": "Login",
        "large_transfer": "Large transfer",
        "loan_repayment_delay": "Loan repayment delay",
        "card_spend_drop": "Card spend drop",
        "balance_drop": "Balance drop",
        "support_complaint": "Support complaint",
        "deposit_withdrawal": "Deposit withdrawal",
    },
    "ja": {
        "generic_retention_offer": "基本リテンション特典",
        "generic retention offer": "基本リテンション特典",
        "personalized_retention_offer": "個別リテンション特典",
        "high_value_retention_coupon": "高価値顧客向けリテンションクーポン",
        "coupon_offer": "クーポン特典",
        "discount_offer": "割引特典",
        "service_recovery": "サービス回復案内",
        "loyalty_reward": "ロイヤル顧客特典",
        "medium": "中強度",
        "high": "高強度",
        "low": "低強度",
        "medium_intensity": "中強度",
        "high_intensity": "高強度",
        "low_intensity": "低強度",
        "중강도": "中強度",
        "고강도": "高強度",
        "저강도": "低強度",
        "queued": "キュー登録済み",
        "not_queued": "未登録",
        "pending": "待機中",
        "sent": "送信済み",
        "completed": "完了",
        "failed": "失敗",
        "own_purchase_history": "本人の購入履歴",
        "recent_browse_signal": "最近の閲覧シグナル",
        "segment_popularity": "類似顧客群での人気",
        "global_popularity": "全体人気",
        "churn_progressing": "離脱進行顧客",
        "new_signup": "新規登録顧客",
        "price_sensitive": "価格敏感顧客",
        "loyal_vip_customer": "ロイヤルVIP顧客",
        "로열VIP고객": "ロイヤルVIP顧客",
        "로열일반고객": "ロイヤル一般顧客",
        "Monitor": "観察",
        "monitor": "観察",
        "page_view": "ページ訪問",
        "purchase": "購入",
        "cart": "カート",
        "add_to_cart": "カート追加",
        "search": "検索",
        "login": "ログイン",
        "large_transfer": "大口送金",
        "loan_repayment_delay": "融資返済遅延",
        "card_spend_drop": "カード利用減少",
        "balance_drop": "残高減少",
        "support_complaint": "相談・苦情受付",
        "deposit_withdrawal": "預金引き出し",
    },
}


# Korean data-label supplements used by business-mode tables/charts. These are
# presentation-only labels; backend CSV schemas and model features remain stable.
_DATA_KO_COLUMN_SUPPLEMENTS: dict[str, str] = {
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
    "actual_profit": "실제 이익",
    "actual_roi": "실제 ROI",
    "actual_conversion": "실제 전환 여부",
    "coupon_redeemed": "쿠폰 사용 여부",
    "outcome_label": "결과 분류",
    "executed": "실행 여부",
    "intervention_intensity_label": "개입 강도",
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
COLUMN_LABELS.setdefault("ko", {}).update(_DATA_KO_COLUMN_SUPPLEMENTS)

_DATA_KO_VALUE_SUPPLEMENTS: dict[str, str] = {
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
}
VALUE_LABELS.setdefault("ko", {}).update(_DATA_KO_VALUE_SUPPLEMENTS)


# Extra plain Korean labels for customer types, generated actions, and finance products.
_PLAIN_KO_VALUE_SUPPLEMENTS: dict[str, str] = {
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
    "vip_at_risk": "이탈 위험이 큰 VIP 고객",
    "regular_at_risk": "이탈 위험이 큰 일반 고객",
    "coupon_sensitive": "혜택에 민감한 고객",
    "high_uplift": "개입하면 반응할 가능성이 높은 고객",
    "very_high_uplift": "개입 반응 가능성이 매우 높은 고객",
    "medium_uplift": "개입 반응 가능성이 보통인 고객",
    "low_uplift": "개입 반응 가능성이 낮은 고객",
    "negative_uplift": "개입을 권장하지 않는 고객",
    "Generic retention offer": "기본 고객 유지 혜택 안내",
    "generic_retention_offer": "기본 고객 유지 혜택 안내",
    "generic retention offer": "기본 고객 유지 혜택 안내",
    "personalized_retention_offer": "고객 맞춤 혜택 안내",
    "high_value_retention_coupon": "고가치 고객 맞춤 혜택 안내",
    "high_value_retention_benefit": "고가치 고객 맞춤 혜택 안내",
    "coupon_offer": "혜택 안내",
    "5,000원 혜택": "금리·수수료 우대 혜택",
    "discount_offer": "할인/우대 혜택 안내",
    "retention_message": "고객 유지 안내 메시지",
    "light_retention_message": "가벼운 안내 메시지",
    "priority_human_followup": "담당자 우선 상담",
    "low_risk_upsell_offer": "관심 상품 추가 안내",
    "monitor_only": "추가 행동 관찰",
    "benefit": "혜택",
    "Immediate (<=14d)": "14일 이내 즉시 연락",
    "Near-term (15-30d)": "15~30일 안에 연락",
    "Planned (31-60d)": "31~60일 안에 계획적으로 연락",
    "Monitor (>60d)": "60일 이후 관찰",
    "deposit": "예금",
    "savings": "적금",
    "checking_account": "입출금계좌",
    "credit_card": "신용카드",
    "debit_card": "체크카드",
    "loan": "대출",
    "mortgage": "주택담보대출",
    "insurance": "보험",
    "pension": "연금",
    "fund": "펀드",
    "investment": "투자상품",
    "wealth_management": "자산관리",
    "digital_banking": "디지털금융",
    "mobile_banking": "모바일뱅킹",
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
VALUE_LABELS.setdefault("ko", {}).update(_PLAIN_KO_VALUE_SUPPLEMENTS)

PHRASE_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "이탈 위험이 높음": "High churn risk",
        "개입 반응 가능성이 큼": "High response likelihood",
        "고객 가치가 높음": "High customer value",
        "예상 ROI가 양호함": "Good expected ROI",
        "단기 이탈 가속 주의": "Watch short-term churn acceleration",
        "가격·서비스·타이밍 리스크를 함께 점검": "Check price, service, and timing risks together",
    },
    "ja": {
        "이탈 위험이 높음": "離脱リスクが高い",
        "개입 반응 가능성이 큼": "介入反応の可能性が高い",
        "고객 가치가 높음": "顧客価値が高い",
        "예상 ROI가 양호함": "予想ROIが良好",
        "단기 이탈 가속 주의": "短期離脱の加速に注意",
        "가격·서비스·타이밍 리스크를 함께 점검": "価格・サービス・タイミングリスクを一緒に確認",
    },
    "ko": {},
}


def translate_column(column: Any, lang: str = "ko") -> str:
    raw = str(column)
    labels = COLUMN_LABELS.get(lang, COLUMN_LABELS["ko"])
    if raw in labels:
        return labels[raw]

    norm = _norm(raw)
    # remove Pandas duplicate suffixes such as expected_roi_2, expected roi 2
    canonical_norms = { _norm(k): k for k in COLUMN_LABELS["ko"].keys() }
    if norm in canonical_norms:
        return labels.get(canonical_norms[norm], COLUMN_LABELS["ko"].get(canonical_norms[norm], raw))

    suffixless = re.sub(r"(?:_?\d+|x|y)$", "", norm)
    if suffixless in canonical_norms:
        return labels.get(canonical_norms[suffixless], COLUMN_LABELS["ko"].get(canonical_norms[suffixless], raw))

    return raw.replace("_", " ")


def _replace_token_variant(text: str, src: str, dst: str) -> str:
    if not src:
        return text
    if re.search(r"[A-Za-z]", src):
        return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])", dst, text)
    return text.replace(src, dst)


def _replace_known_tokens(text: str, mapping: dict[str, str]) -> str:
    out = text
    # Replace longest values first.
    for src, dst in sorted(mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
        src = str(src)
        if not src:
            continue

        # Code-like English tokens must not match inside longer words such as
        # "follow-up", while Korean/Japanese phrases still use plain matching.
        dst = str(dst)
        out = _replace_token_variant(out, src, dst)
        out = _replace_token_variant(out, src.replace("_", " ").title(), dst)
        out = _replace_token_variant(out, src.replace("_", " "), dst)
    return out


_COMPOUND_VALUE_EXACT_KO = {
    "credit_revolver": "카드 리볼빙 이용 고객",
    "card_revolver": "카드 리볼빙 이용 고객",
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
}

_COMPOUND_VALUE_TOKEN_KO = {
    "credit": "신용",
    "card": "카드",
    "revolver": "리볼빙 이용",
    "revolving": "리볼빙 이용",
    "loan": "대출",
    "loans": "대출",
    "mortgage": "주택담보대출",
    "watch": "관찰 대상",
    "watchlist": "관찰 대상",
    "risk": "위험",
    "dormant": "활동 감소",
    "inactive": "활동 감소",
    "churn": "이탈",
    "high": "높은",
    "medium": "보통",
    "low": "낮은",
}


def _translate_compound_code_value(raw: str, lang: str) -> str | None:
    if lang != "ko":
        return None
    text = str(raw or "").strip()
    if not text:
        return None
    norm = _norm(text)
    for src, dst in _COMPOUND_VALUE_EXACT_KO.items():
        if norm == _norm(src):
            return dst
    if not ("_" in text or "-" in text or re.search(r"[A-Za-z]", text)):
        return None
    parts = [part for part in re.split(r"[\s_\-/]+", text) if part]
    labels: list[str] = []
    for part in parts:
        part_norm = part.lower()
        label = _COMPOUND_VALUE_TOKEN_KO.get(part_norm)
        if not label and re.search(r"[가-힣]", part):
            label = part
        if not label:
            return None
        if not labels or labels[-1] != label:
            labels.append(label)
    phrase = " ".join(labels).strip()
    if any(token in norm for token in ["risk", "watch", "revolver", "dormant", "inactive", "churn"]) and not phrase.endswith(("고객", "대상")):
        phrase = f"{phrase} 고객"
    return phrase if phrase and phrase != text else None


def translate_value(value: Any, lang: str = "ko") -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        try:
            if math.isnan(float(value)) or math.isinf(float(value)):
                return ""
        except Exception:
            pass
        return value
    raw = str(value).strip()
    if not raw:
        return ""

    mapping = VALUE_LABELS.get(lang, VALUE_LABELS["ko"])
    norm = _norm(raw)
    for src, dst in mapping.items():
        if raw == src or norm == _norm(src):
            return dst

    compound = _translate_compound_code_value(raw, lang)
    if compound:
        return compound

    out = raw
    out = _replace_known_tokens(out, mapping)
    out = _replace_known_tokens(out, PHRASE_LABELS.get(lang, {}))

    # Cleanup common code-format remnants after token replacement.
    out = out.replace("_", " ")
    out = re.sub(r"\s*,\s*", ", ", out)
    out = re.sub(r"\s*·\s*", " · ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def translate_text(text: Any, lang: str = "ko") -> str:
    raw = str(text or "")
    if not raw:
        return ""
    col = translate_column(raw, lang)
    if col != raw.replace("_", " "):
        return col
    out = _replace_known_tokens(raw, COLUMN_LABELS.get(lang, {}))
    out = _replace_known_tokens(out, VALUE_LABELS.get(lang, {}))
    out = _replace_known_tokens(out, PHRASE_LABELS.get(lang, {}))
    return out


def _canonical_metric_group(col: str) -> str | None:
    n = _norm(col)
    # Remove suffixes created by Pandas/HTML display code.
    base = re.sub(r"(?:_?\d+|x|y)$", "", n)

    if base in {"expectedroi", "expectedroiaction"} or n in {"expectedroi2", "expectedroi02", "예상roi2", "予想roi2"}:
        return "expected_roi"
    if base in {"expectedprofit", "expectedincrementalprofit", "incrementalprofit", "expectedprofitaction"}:
        return base
    if base in {"couponcost", "interventioncost", "recommendedinvestmentamount", "queuedcouponcost"}:
        return base
    if base in {"churnprobability", "churnscore", "realtimechurnscore"}:
        return base
    return None


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    try:
        lnum = pd.to_numeric(left, errors="coerce")
        rnum = pd.to_numeric(right, errors="coerce")
        if lnum.notna().any() or rnum.notna().any():
            return lnum.fillna(-999999999.123456).round(8).equals(
                rnum.fillna(-999999999.123456).round(8)
            )
    except Exception:
        pass
    return left.astype(str).fillna("").equals(right.astype(str).fillna(""))


def drop_duplicate_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate metric columns caused by merge/suffix/display steps.

    Works even when a DataFrame contains duplicate column names. It keeps the
    first meaningful metric and removes exact duplicates or suffix-generated
    variants such as expected_roi_2 / expected roi 2 / expected_roi_action.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df

    columns = list(df.columns)
    keep_indices: list[int] = []
    seen_exact: dict[str, int] = {}
    seen_group: dict[str, int] = {}

    for idx, col in enumerate(columns):
        col_str = str(col)
        exact_key = _norm(col_str)
        group_key = _canonical_metric_group(col_str)
        current = df.iloc[:, idx]

        drop = False
        if exact_key in seen_exact:
            drop = True

        if not drop and group_key is not None and group_key in seen_group:
            first_idx = seen_group[group_key]
            first = df.iloc[:, first_idx]
            # For expected ROI duplicates, suffix/action versions are display noise.
            suffix_like = bool(re.search(r"(_?\d+| action|_action| expected roi 2)$", col_str, flags=re.I))
            if suffix_like or _series_values_equal(first, current):
                drop = True

        if not drop:
            keep_indices.append(idx)
            seen_exact.setdefault(exact_key, idx)
            if group_key is not None:
                seen_group.setdefault(group_key, idx)

    return df.iloc[:, keep_indices].copy()


def localize_plotly_figure(fig: Any, lang: str = "ko") -> Any:
    """Translate Plotly axis titles, chart titles, legend labels, and trace names."""
    if fig is None:
        return fig

    def tr(v: Any) -> Any:
        return translate_text(v, lang) if isinstance(v, str) else v

    try:
        if getattr(fig.layout, "title", None) and fig.layout.title.text:
            fig.update_layout(title_text=tr(fig.layout.title.text))
    except Exception:
        pass

    try:
        fig.for_each_xaxis(lambda axis: axis.update(title_text=tr(axis.title.text)) if axis.title and axis.title.text else None)
        fig.for_each_yaxis(lambda axis: axis.update(title_text=tr(axis.title.text)) if axis.title and axis.title.text else None)
    except Exception:
        pass

    try:
        if fig.layout.legend and fig.layout.legend.title and fig.layout.legend.title.text:
            fig.update_layout(legend_title_text=tr(fig.layout.legend.title.text))
    except Exception:
        pass

    try:
        for trace in fig.data:
            if getattr(trace, "name", None):
                trace.name = tr(trace.name)
            if getattr(trace, "hovertemplate", None):
                trace.hovertemplate = translate_text(trace.hovertemplate, lang)
    except Exception:
        pass

    try:
        for ann in fig.layout.annotations or []:
            if getattr(ann, "text", None):
                ann.text = tr(ann.text)
    except Exception:
        pass

    return fig

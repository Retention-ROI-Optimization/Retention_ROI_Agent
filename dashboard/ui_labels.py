"""Dashboard display-language helpers.

This module keeps non-business logic out of dashboard/app.py:
- beginner-friendly column names
- display value translation for table cells
- duplicate metric-column cleanup
- Plotly axis/title localization
- LLM output-language instructions
- budget/ROI formula explanation snippets
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


def norm_key(text: Any) -> str:
    return re.sub(r"[\s_\-:：/\.()\[\]{}%]+", "", str(text or "")).lower()


FRIENDLY_COLUMN_LABELS: dict[str, dict[str, str]] = {
    "ko": {
        "customer_id": "고객 ID",
        "persona": "고객 유형",
        "customer_type": "고객 유형",
        "churn_probability": "이탈 위험도",
        "churn_score": "이탈 위험 점수",
        "realtime_churn_score": "실시간 이탈 위험 점수",
        "base_churn_probability": "기준 이탈 위험도",
        "score_delta": "위험도 변화",
        "clv": "예상 고객 가치(CLV)",
        "uplift_score": "개입 반응 가능성",
        "uplift_segment": "개입 반응 유형",
        "risk_segment": "위험 등급",
        "risk_group": "위험 그룹",
        "expected_roi": "예상 ROI",
        "expected_incremental_profit": "예상 추가 이익",
        "expected_profit": "예상 이익",
        "coupon_cost": "개입 비용",
        "allocated_budget": "배정 예산",
        "customer_count": "고객 수",
        "candidate_customer_count": "후보 고객 수",
        "intervention_intensity": "개입 강도",
        "recommended_action": "추천 액션",
        "queued_recommended_action": "대기 중 추천 액션",
        "priority_score": "우선순위 점수",
        "selection_score": "선정 점수",
        "recommended_intervention_window": "추천 개입 시점",
        "recommended_category": "추천 카테고리",
        "recommendation_rank": "추천 순위",
        "recommendation_score": "추천 점수",
        "recommendation_priority": "추천 우선순위",
        "target_priority_score": "타겟 우선순위",
        "reason_tags": "추천 이유",
        "selection_reason": "선정 이유",
        "reason_summary": "선정 이유",
        "caution": "주의사항",
        "watchout": "주의사항",
        "next_best_action": "다음 추천 액션",
        "action_status": "액션 상태",
        "action_queue_status": "액션 큐 상태",
        "source_type": "발생 경로",
        "trigger_reason": "트리거 이유",
        "latest_trigger_reason": "최근 트리거 이유",
        "event_type": "이벤트 유형",
        "transaction_type": "거래 유형",
        "amount": "금액",
        "balance": "잔고",
        "deposit_balance": "예금 잔고",
        "loan_balance": "대출 잔액",
        "card_usage": "카드 이용액",
        "delinquency_days": "연체 일수",
        "support_ticket_count": "상담/문의 건수",
        "last_active_days": "마지막 활동 후 경과일",
        "visit_count": "방문 횟수",
        "purchase_count": "구매 횟수",
        "cart_count": "장바구니 횟수",
        "search_count": "검색 횟수",
        "recent_browse_signal": "최근 탐색 신호",
        "segment_popularity": "유사 고객군 인기",
        "own_purchase_history": "본인 구매 이력",
        "global_popularity": "전체 인기",
        "feature": "변수",
        "feature_display": "변수명",
        "importance": "중요도",
        "importance_share": "중요도 비중",
        "count": "건수",
    },
    "en": {
        "customer_id": "Customer ID",
        "persona": "Customer Type",
        "customer_type": "Customer Type",
        "churn_probability": "Churn Risk",
        "churn_score": "Churn Risk Score",
        "realtime_churn_score": "Real-time Churn Risk Score",
        "base_churn_probability": "Base Churn Risk",
        "score_delta": "Risk Change",
        "clv": "Expected Customer Value (CLV)",
        "uplift_score": "Response Potential",
        "uplift_segment": "Response Type",
        "risk_segment": "Risk Level",
        "risk_group": "Risk Group",
        "expected_roi": "Expected ROI",
        "expected_incremental_profit": "Expected Incremental Profit",
        "expected_profit": "Expected Profit",
        "coupon_cost": "Intervention Cost",
        "allocated_budget": "Allocated Budget",
        "customer_count": "Customers",
        "candidate_customer_count": "Candidate Customers",
        "intervention_intensity": "Intervention Intensity",
        "recommended_action": "Recommended Action",
        "queued_recommended_action": "Queued Recommended Action",
        "priority_score": "Priority Score",
        "selection_score": "Selection Score",
        "recommended_intervention_window": "Recommended Timing",
        "recommended_category": "Recommended Category",
        "recommendation_rank": "Recommendation Rank",
        "recommendation_score": "Recommendation Score",
        "recommendation_priority": "Recommendation Priority",
        "target_priority_score": "Target Priority",
        "reason_tags": "Recommendation Basis",
        "selection_reason": "Reason Selected",
        "reason_summary": "Reason Selected",
        "caution": "Caution",
        "watchout": "Caution",
        "next_best_action": "Next Action",
        "action_status": "Action Status",
        "action_queue_status": "Action Queue Status",
        "source_type": "Source",
        "trigger_reason": "Trigger Reason",
        "latest_trigger_reason": "Latest Trigger Reason",
        "event_type": "Event Type",
        "transaction_type": "Transaction Type",
        "amount": "Amount",
        "balance": "Balance",
        "deposit_balance": "Deposit Balance",
        "loan_balance": "Loan Balance",
        "card_usage": "Card Spending",
        "delinquency_days": "Days Past Due",
        "support_ticket_count": "Support Contacts",
        "last_active_days": "Days Since Last Activity",
        "visit_count": "Visits",
        "purchase_count": "Purchases",
        "cart_count": "Cart Adds",
        "search_count": "Searches",
        "recent_browse_signal": "Recent Browsing Signal",
        "segment_popularity": "Segment Popularity",
        "own_purchase_history": "Own Purchase History",
        "global_popularity": "Overall Popularity",
        "feature": "Feature",
        "feature_display": "Feature",
        "importance": "Importance",
        "importance_share": "Importance Share",
        "count": "Count",
    },
    "ja": {
        "customer_id": "顧客ID",
        "persona": "顧客タイプ",
        "customer_type": "顧客タイプ",
        "churn_probability": "離脱リスク",
        "churn_score": "離脱リスクスコア",
        "realtime_churn_score": "リアルタイム離脱リスクスコア",
        "base_churn_probability": "基準離脱リスク",
        "score_delta": "リスク変化",
        "clv": "予想顧客価値（CLV）",
        "uplift_score": "介入反応見込み",
        "uplift_segment": "介入反応タイプ",
        "risk_segment": "リスク等級",
        "risk_group": "リスクグループ",
        "expected_roi": "予想ROI",
        "expected_incremental_profit": "予想追加利益",
        "expected_profit": "予想利益",
        "coupon_cost": "介入費用",
        "allocated_budget": "配分予算",
        "customer_count": "顧客数",
        "candidate_customer_count": "候補顧客数",
        "intervention_intensity": "介入強度",
        "recommended_action": "推奨アクション",
        "queued_recommended_action": "キュー内推奨アクション",
        "priority_score": "優先度スコア",
        "selection_score": "選定スコア",
        "recommended_intervention_window": "推奨介入タイミング",
        "recommended_category": "推薦カテゴリ",
        "recommendation_rank": "推薦順位",
        "recommendation_score": "推薦スコア",
        "recommendation_priority": "推薦優先度",
        "target_priority_score": "対象優先度",
        "reason_tags": "推薦根拠",
        "selection_reason": "選定理由",
        "reason_summary": "選定理由",
        "caution": "注意事項",
        "watchout": "注意事項",
        "next_best_action": "次の推奨アクション",
        "action_status": "アクション状態",
        "action_queue_status": "アクションキュー状態",
        "source_type": "発生経路",
        "trigger_reason": "トリガー理由",
        "latest_trigger_reason": "最新トリガー理由",
        "event_type": "イベント種別",
        "transaction_type": "取引種別",
        "amount": "金額",
        "balance": "残高",
        "deposit_balance": "預金残高",
        "loan_balance": "融資残高",
        "card_usage": "カード利用額",
        "delinquency_days": "延滞日数",
        "support_ticket_count": "相談・問い合わせ件数",
        "last_active_days": "最終活動からの日数",
        "visit_count": "訪問回数",
        "purchase_count": "購入回数",
        "cart_count": "カート投入回数",
        "search_count": "検索回数",
        "recent_browse_signal": "最近の閲覧シグナル",
        "segment_popularity": "類似顧客群での人気",
        "own_purchase_history": "本人の購入履歴",
        "global_popularity": "全体人気",
        "feature": "変数",
        "feature_display": "変数名",
        "importance": "重要度",
        "importance_share": "重要度比率",
        "count": "件数",
    },
}

COLUMN_ALIASES: dict[str, str] = {}
for canonical, label in FRIENDLY_COLUMN_LABELS["ko"].items():
    COLUMN_ALIASES[norm_key(canonical)] = canonical
    COLUMN_ALIASES[norm_key(label)] = canonical
for lang_map in FRIENDLY_COLUMN_LABELS.values():
    for canonical, label in lang_map.items():
        COLUMN_ALIASES[norm_key(label)] = canonical
# common duplicate/merged names
COLUMN_ALIASES.update({
    "expectedroi2": "expected_roi",
    "expectedroiaction": "expected_roi",
    "queuedexpectedroi": "expected_roi",
    "expectedprofit2": "expected_profit",
    "expectedprofitaction": "expected_profit",
    "queuedexpectedprofit": "expected_profit",
    "churnprobability2": "churn_probability",
    "churnscore2": "churn_score",
    "couponcost2": "coupon_cost",
    "couponcostaction": "coupon_cost",
    "고객id": "customer_id",
    "顧客id": "customer_id",
    "추천이유": "reason_tags",
    "推薦理由": "reason_tags",
    "推奨理由": "reason_tags",
})

VALUE_LABELS_EXT: dict[str, dict[str, str]] = {
    "ko": {
        "churn_progressing": "이탈 진행 고객",
        "price_sensitive": "가격 민감 고객",
        "new_signup": "신규 가입 고객",
        "loyal_vip": "충성 VIP 고객",
        "loyal_regular": "충성 일반 고객",
        "로열VIP고객": "충성 VIP 고객",
        "로열일반고객": "충성 일반 고객",
        "generic_retention_offer": "기본 리텐션 혜택",
        "Generic retention offer": "기본 리텐션 혜택",
        "coupon_offer": "쿠폰 혜택",
        "personalized_coupon": "개인 맞춤 쿠폰",
        "service_recovery": "서비스 회복 안내",
        "loyalty_reward": "충성 고객 보상",
        "own_purchase_history": "본인 구매 이력",
        "recent_browse_signal": "최근 탐색 신호",
        "segment_popularity": "유사 고객군 인기",
        "global_popularity": "전체 인기",
        "Monitor": "관찰",
        "중강도": "중강도",
        "고강도": "고강도",
        "저강도": "저강도",
        "medium": "중강도",
        "high": "고강도",
        "low": "저강도",
        "deposit_withdrawal": "예금 인출",
        "large_transfer": "고액 이체",
        "loan_repayment_delay": "대출 상환 지연",
        "card_spend_drop": "카드 이용 감소",
        "balance_drop": "잔고 감소",
        "support_complaint": "상담/불만 접수",
        "branch_visit": "지점 방문",
        "mobile_login": "모바일 로그인",
        "page_view": "페이지 방문",
        "purchase": "구매",
        "add_to_cart": "장바구니 담기",
        "cart": "장바구니",
        "search": "검색",
        "login": "로그인",
    },
    "en": {
        "churn_progressing": "Churn-risk customer",
        "price_sensitive": "Price-sensitive customer",
        "new_signup": "Newly signed-up customer",
        "loyal_vip": "Loyal VIP customer",
        "loyal_regular": "Loyal regular customer",
        "로열VIP고객": "Loyal VIP customer",
        "로열일반고객": "Loyal regular customer",
        "generic_retention_offer": "Basic retention offer",
        "Generic retention offer": "Basic retention offer",
        "coupon_offer": "Coupon offer",
        "personalized_coupon": "Personalized coupon",
        "service_recovery": "Service recovery message",
        "loyalty_reward": "Loyalty reward",
        "own_purchase_history": "Own purchase history",
        "recent_browse_signal": "Recent browsing signal",
        "segment_popularity": "Segment popularity",
        "global_popularity": "Overall popularity",
        "Monitor": "Monitor",
        "중강도": "Medium intensity",
        "고강도": "High intensity",
        "저강도": "Low intensity",
        "medium": "Medium intensity",
        "high": "High intensity",
        "low": "Low intensity",
        "deposit_withdrawal": "Deposit withdrawal",
        "large_transfer": "Large transfer",
        "loan_repayment_delay": "Delayed loan repayment",
        "card_spend_drop": "Card spending drop",
        "balance_drop": "Balance drop",
        "support_complaint": "Support complaint",
        "branch_visit": "Branch visit",
        "mobile_login": "Mobile login",
        "page_view": "Page visit",
        "purchase": "Purchase",
        "add_to_cart": "Add to cart",
        "cart": "Cart",
        "search": "Search",
        "login": "Login",
    },
    "ja": {
        "churn_progressing": "離脱進行顧客",
        "price_sensitive": "価格敏感顧客",
        "new_signup": "新規登録顧客",
        "loyal_vip": "ロイヤルVIP顧客",
        "loyal_regular": "ロイヤル一般顧客",
        "로열VIP고객": "ロイヤルVIP顧客",
        "로열일반고객": "ロイヤル一般顧客",
        "generic_retention_offer": "基本リテンション特典",
        "Generic retention offer": "基本リテンション特典",
        "coupon_offer": "クーポン特典",
        "personalized_coupon": "個別クーポン",
        "service_recovery": "サービス回復メッセージ",
        "loyalty_reward": "ロイヤル顧客特典",
        "own_purchase_history": "本人の購入履歴",
        "recent_browse_signal": "最近の閲覧シグナル",
        "segment_popularity": "類似顧客群での人気",
        "global_popularity": "全体人気",
        "Monitor": "観察",
        "중강도": "中強度",
        "고강도": "高強度",
        "저강도": "低強度",
        "medium": "中強度",
        "high": "高強度",
        "low": "低強度",
        "deposit_withdrawal": "預金引き出し",
        "large_transfer": "高額振込",
        "loan_repayment_delay": "融資返済遅延",
        "card_spend_drop": "カード利用減少",
        "balance_drop": "残高減少",
        "support_complaint": "相談・苦情受付",
        "branch_visit": "店舗訪問",
        "mobile_login": "モバイルログイン",
        "page_view": "ページ訪問",
        "purchase": "購入",
        "add_to_cart": "カート追加",
        "cart": "カート",
        "search": "検索",
        "login": "ログイン",
    },
}

PHRASE_LABELS_EXT: dict[str, dict[str, str]] = {
    "ko": {
        "이탈 위험이 높음": "이탈 위험이 높음",
        "개입 반응 가능성이 큼": "개입 반응 가능성이 큼",
        "고객 가치가 높음": "고객 가치가 높음",
        "예상 ROI가 양호함": "예상 ROI가 양호함",
        "가격·서비스·타이밍 리스크를 함께 점검": "가격·서비스·타이밍 리스크를 함께 점검",
    },
    "en": {
        "이탈 위험이 높음": "high churn risk",
        "개입 반응 가능성이 큼": "high response potential",
        "고객 가치가 높음": "high customer value",
        "예상 ROI가 양호함": "good expected ROI",
        "단기 이탈 가속 주의": "watch for short-term churn acceleration",
        "가격·서비스·타이밍 리스크를 함께 점검": "check price, service, and timing risks together",
    },
    "ja": {
        "이탈 위험이 높음": "離脱リスクが高い",
        "개입 반응 가능성이 큼": "介入反応の可能性が高い",
        "고객 가치가 높음": "顧客価値が高い",
        "예상 ROI가 양호함": "予想ROIが良好",
        "단기 이탈 가속 주의": "短期離脱の加速に注意",
        "가격·서비스·타이밍 리스크를 함께 점검": "価格・サービス・タイミングリスクを一緒に確認",
    },
}

UI_TEXT_EXT: dict[str, dict[str, str]] = {
    "ko": {
        "예산·이익·ROI 산출식": "예산·이익·ROI 산출식",
        "예상 추가 이익 = 고객 가치 × 개입 반응 가능성 × 이탈 위험도 - 개입 비용": "예상 추가 이익 = 고객 가치 × 개입 반응 가능성 × 이탈 위험도 - 개입 비용",
        "예상 ROI = 예상 추가 이익 ÷ 개입 비용": "예상 ROI = 예상 추가 이익 ÷ 개입 비용",
        "예산 최적화는 예상 추가 이익과 ROI가 높은 고객부터 예산 한도 안에서 선택합니다.": "예산 최적화는 예상 추가 이익과 ROI가 높은 고객부터 예산 한도 안에서 선택합니다.",
    },
    "en": {
        "예산·이익·ROI 산출식": "Budget, Profit, and ROI Formula",
        "예상 추가 이익 = 고객 가치 × 개입 반응 가능성 × 이탈 위험도 - 개입 비용": "Expected incremental profit = Customer value × Response potential × Churn risk - Intervention cost",
        "예상 ROI = 예상 추가 이익 ÷ 개입 비용": "Expected ROI = Expected incremental profit ÷ Intervention cost",
        "예산 최적화는 예상 추가 이익과 ROI가 높은 고객부터 예산 한도 안에서 선택합니다.": "Budget optimization selects customers with higher expected incremental profit and ROI within the budget limit.",
        "고객별 이탈 확률 분포": "Customer churn-risk distribution",
        "Threshold": "Threshold",
    },
    "ja": {
        "예산·이익·ROI 산출식": "予算・利益・ROIの算出式",
        "예상 추가 이익 = 고객 가치 × 개입 반응 가능성 × 이탈 위험도 - 개입 비용": "予想追加利益 = 顧客価値 × 介入反応見込み × 離脱リスク - 介入費用",
        "예상 ROI = 예상 추가 이익 ÷ 개입 비용": "予想ROI = 予想追加利益 ÷ 介入費用",
        "예산 최적화는 예상 추가 이익과 ROI가 높은 고객부터 예산 한도 안에서 선택합니다.": "予算最適化では、予算上限の中で予想追加利益とROIが高い顧客から選定します。",
        "고객별 이탈 확률 분포": "顧客別離脱リスク分布",
        "Threshold": "しきい値",
    },
}


def canonical_column_name(column: Any) -> str | None:
    raw = str(column or "")
    if raw in FRIENDLY_COLUMN_LABELS["ko"]:
        return raw
    return COLUMN_ALIASES.get(norm_key(raw))


def translate_column(column: Any, lang: str = "ko") -> str:
    raw = str(column or "")
    canonical = canonical_column_name(raw)
    if canonical:
        return FRIENDLY_COLUMN_LABELS.get(lang, FRIENDLY_COLUMN_LABELS["ko"]).get(canonical, raw)
    return raw.replace("_", " ")


def translate_text(text: Any, lang: str = "ko") -> str:
    raw = str(text or "")
    if not raw:
        return ""
    direct = UI_TEXT_EXT.get(lang, {}).get(raw)
    if direct is not None:
        return direct
    n = norm_key(raw)
    for src, dst in UI_TEXT_EXT.get(lang, {}).items():
        if norm_key(src) == n:
            return dst
    canonical = canonical_column_name(raw)
    if canonical:
        return translate_column(canonical, lang)
    return raw


def _replace_known_tokens(text: str, mapping: Mapping[str, str]) -> str:
    out = text
    # Longest first prevents partial replacement.
    for src, dst in sorted(mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
        src_s = str(src)
        if not src_s:
            continue
        # Short generic intensity tokens are handled by exact matching only.
        # Replacing them inside translated phrases can create strings such as
        # "Medium intensity intensity".
        if src_s.lower() in {"low", "medium", "high"}:
            continue
        if src_s in out:
            out = out.replace(src_s, str(dst))
        # normalized human form: Generic retention offer -> generic_retention_offer
        human = src_s.replace("_", " ").title()
        if human in out:
            out = out.replace(human, str(dst))
        human_lower = src_s.replace("_", " ")
        if human_lower in out:
            out = out.replace(human_lower, str(dst))
    return out


def translate_value(value: Any, lang: str = "ko") -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if math.isnan(float(value)) or math.isinf(float(value)):
                return ""
        except Exception:
            pass
        return value
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    raw = str(value).strip()
    if raw == "":
        return ""
    mapping = VALUE_LABELS_EXT.get(lang, VALUE_LABELS_EXT["ko"])
    phrase_map = PHRASE_LABELS_EXT.get(lang, {})
    n = norm_key(raw)
    for src, dst in mapping.items():
        if raw == src or n == norm_key(src):
            return dst
    out = raw
    out = _replace_known_tokens(out, mapping)
    out = _replace_known_tokens(out, phrase_map)
    # Split comma-separated recommendation reasons and dot-separated actions.
    if "," in out:
        parts = [p.strip() for p in out.split(",")]
        out = ", ".join(_replace_known_tokens(_replace_known_tokens(p, mapping), phrase_map) for p in parts)
    if "·" in out:
        parts = [p.strip() for p in out.split("·")]
        out = " · ".join(_replace_known_tokens(_replace_known_tokens(p, mapping), phrase_map) for p in parts)
    return out


def drop_duplicate_metric_columns(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    fixed = df.copy()
    duplicate_groups = [
        ["expected_roi", "expected roi", "expected_roi_2", "expected roi 2", "expected_roi_action", "queued_expected_roi"],
        ["expected_profit", "expected profit", "expected_profit_2", "expected profit 2", "queued_expected_profit"],
        ["expected_incremental_profit", "expected incremental profit", "expected_incremental_profit_2"],
        ["coupon_cost", "coupon cost", "coupon_cost_2", "queued_coupon_cost"],
        ["churn_probability", "churn probability", "churn_probability_2"],
        ["churn_score", "churn score", "churn_score_2"],
    ]
    norm_to_cols: dict[str, list[str]] = {}
    for col in fixed.columns:
        norm_to_cols.setdefault(norm_key(col), []).append(col)

    to_drop: list[str] = []
    for group in duplicate_groups:
        existing: list[str] = []
        for key in group:
            existing.extend(norm_to_cols.get(norm_key(key), []))
        existing = list(dict.fromkeys(existing))
        if len(existing) <= 1:
            continue
        keep = existing[0]
        for col in existing[1:]:
            left = pd.to_numeric(fixed[keep], errors="coerce")
            right = pd.to_numeric(fixed[col], errors="coerce")
            both_numeric = left.notna().any() or right.notna().any()
            if both_numeric:
                same = left.fillna(-999999999).round(8).equals(right.fillna(-999999999).round(8))
            else:
                same = fixed[keep].astype(str).fillna("").equals(fixed[col].astype(str).fillna(""))
            # For queued columns, drop anyway when the visible base metric exists.
            if same or norm_key(col).startswith("queued") or norm_key(col).endswith("action") or norm_key(col).endswith("2"):
                to_drop.append(col)
    if to_drop:
        fixed = fixed.drop(columns=[c for c in dict.fromkeys(to_drop) if c in fixed.columns])
    return fixed


def translate_dataframe(df: pd.DataFrame | None, lang: str = "ko") -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = drop_duplicate_metric_columns(df.copy())
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("string"):
            out[col] = out[col].map(lambda v: translate_value(v, lang))
    out.columns = [translate_column(c, lang) for c in out.columns]
    return out


def localize_plotly_figure(fig: Any, lang: str = "ko") -> Any:
    if fig is None:
        return fig
    label_map = FRIENDLY_COLUMN_LABELS.get(lang, FRIENDLY_COLUMN_LABELS["ko"])
    # Axis titles
    try:
        for axis_name in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
            axis = getattr(fig.layout, axis_name, None)
            if axis is not None and getattr(axis, "title", None) is not None:
                txt = getattr(axis.title, "text", None)
                if txt:
                    axis.title.text = translate_column(txt, lang)
    except Exception:
        pass
    # Figure title and annotations
    try:
        if fig.layout.title and fig.layout.title.text:
            fig.layout.title.text = translate_text(fig.layout.title.text, lang)
    except Exception:
        pass
    try:
        for ann in fig.layout.annotations or []:
            if getattr(ann, "text", None):
                ann.text = _replace_known_tokens(str(ann.text), UI_TEXT_EXT.get(lang, {}))
                ann.text = _replace_known_tokens(str(ann.text), label_map)
    except Exception:
        pass
    # Trace names and hover labels
    try:
        for trace in fig.data:
            if getattr(trace, "name", None):
                trace.name = translate_value(trace.name, lang)
    except Exception:
        pass
    return fig


def llm_language_name(lang: str = "ko") -> str:
    return {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(lang, "Korean")


def llm_language_instruction(lang: str = "ko") -> str:
    if lang == "en":
        return (
            "You must write the entire response in English. Do not write Korean or Japanese. "
            "Translate all dashboard labels and generated explanations into clear business English. "
            "Do not copy Korean table cell values verbatim; paraphrase them in English."
        )
    if lang == "ja":
        return (
            "必ず回答全体を日本語で書いてください。韓国語や英語の文章を混ぜないでください。"
            "ダッシュボードのラベルや生成された説明も、分かりやすい日本語に言い換えてください。"
            "韓国語のテーブル値をそのまま引用せず、日本語で説明してください。"
        )
    return (
        "반드시 전체 답변을 한국어로 작성하세요. 영어/일본어 문장을 섞지 말고, "
        "비즈니스 담당자가 이해하기 쉬운 표현을 사용하세요."
    )


def budget_formula_html(lang: str = "ko") -> str:
    title = translate_text("예산·이익·ROI 산출식", lang)
    profit = translate_text("예상 추가 이익 = 고객 가치 × 개입 반응 가능성 × 이탈 위험도 - 개입 비용", lang)
    roi = translate_text("예상 ROI = 예상 추가 이익 ÷ 개입 비용", lang)
    note = translate_text("예산 최적화는 예상 추가 이익과 ROI가 높은 고객부터 예산 한도 안에서 선택합니다.", lang)
    return f"""
<div style="background:#F8FAFC;border:1px solid #CBD5E1;border-radius:14px;padding:14px 16px;margin:10px 0 18px 0;line-height:1.65;color:#0F172A;">
  <div style="font-weight:800;margin-bottom:6px;">📌 {title}</div>
  <div><b>1)</b> {profit}</div>
  <div><b>2)</b> {roi}</div>
  <div style="color:#64748B;font-size:13px;margin-top:6px;">{note}</div>
</div>
"""

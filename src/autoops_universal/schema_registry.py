from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Canonical fields that the add-on tries to create regardless of user CSV schema.
CANONICAL_CUSTOMER_FIELDS = [
    "customer_id",
    "label",
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "avg_order_amount",
    "tenure_days",
    "signup_date",
    "last_activity_date",
    "segment",
    "region",
    "channel",
    "device",
    "category",
    "campaign_exposed",
    "converted",
]

# Columns required by legacy dashboard/optimization screens. The canonicalizer
# always creates them, either from user data, derived values, or safe defaults.
LEGACY_COMPATIBILITY_FIELDS = [
    "assigned_at",
    "last_purchase_date",
    "treatment_group",
    "treatment_flag",
    "campaign_type",
    "coupon_cost",
    "customer_segment",
    "persona",
    "device_type",
    "acquisition_channel",
    "preferred_category",
    "price_sensitivity",
    "coupon_affinity",
    "discount_pressure_score",
    "brand_sensitivity",
    "support_contact_propensity",
]

# Rich alias table. Add synonyms here rather than changing downstream models.
FIELD_ALIASES: dict[str, list[str]] = {
    "customer_id": [
        "customer_id", "customerid", "cust_id", "custid", "cust_no", "customer_no", "client_id", "account_id",
        "user_id", "userid", "member_id", "memberid", "member_no", "subscriber_id", "id", "uid",
        "household_key", "household_id", "household", "hh_id", "가구번호",
        "customer_name", "member_name", "user_name", "username", "name", "full_name", "email", "phone", "mobile", "mobile_no",
        "고객id", "고객_id", "고객번호", "고객번호", "고객", "고객명", "회원id", "회원_id", "회원번호", "회원명", "사용자id", "사용자_id", "사용자명", "이름", "성명", "이메일", "전화번호", "휴대폰",
    ],
    "transaction_id": [
        "transaction_id", "transactionid", "txn_id", "tx_id", "order_id", "orderid", "purchase_id", "receipt_id",
        "basket_id", "basketid", "basket", "cart_id", "cartid", "session_order_id",
        "주문번호", "거래번호", "결제번호", "영수증번호",
    ],
    "transaction_date": [
        "transaction_date", "transaction_time", "txn_date", "order_date", "order_time", "purchase_date", "purchase_time",
        "payment_date", "paid_at", "event_date", "event_time", "timestamp", "created_at", "created_time", "datetime", "date",
        "day", "trans_day", "transaction_day", "purchase_day", "order_day", "week_no", "week_number",
        "구매일", "구매일자", "주문일", "주문일자", "결제일", "결제일자", "거래일", "거래일자", "이벤트일",
    ],
    "signup_date": [
        "signup_date", "join_date", "joined_at", "registration_date", "registered_at", "created_customer_at", "first_seen_at",
        "가입일", "가입일자", "등록일", "등록일자", "회원가입일",
    ],
    "last_activity_date": [
        "last_activity_date", "last_active_date", "last_seen_at", "last_login", "last_login_date", "last_purchase_date",
        "last_order_date", "last_event_date", "최근활동일", "최근구매일", "최종구매일", "마지막구매일", "최종접속일",
    ],
    "amount": [
        "amount", "payment", "payment_amount", "paid_amount", "price", "sales", "revenue", "gross_revenue", "net_amount",
        "order_amount", "purchase_amount", "total_price", "total_amount", "gmv", "spend", "total_spend", "monetary", "monetary_90d",
        "구매금액", "결제금액", "주문금액", "매출", "총구매액", "총결제금액", "금액",
    ],
    "quantity": [
        "quantity", "qty", "item_qty", "items", "units", "unit_count", "total_qty", "total_quantity", "total_qty_365d",
        "상품수량", "수량", "구매수량",
    ],
    "recency_days": [
        "recency", "recency_days", "days_since_last_purchase", "days_since_last_order", "days_since_last_event",
        "inactivity_days", "inactive_days", "last_purchase_days", "미구매일수", "비활성일수", "휴면일수",
    ],
    "frequency_90d": [
        "frequency", "frequency_90d", "purchase_count", "order_count", "orders", "transactions", "total_orders", "visits", "visit_count",
        "num_visits", "num_visits_365d", "basket_count",
        "구매횟수", "주문횟수", "거래횟수", "방문횟수",
    ],
    "monetary_90d": [
        "monetary", "monetary_90d", "revenue", "sales", "sales_value", "total_spend", "total_spend_365d",
        "total_amount", "amount", "net_amount", "clv", "ltv", "gmv",
        "총구매액", "매출", "결제금액합계", "구매금액합계",
    ],
    "label": [
        "label", "target", "churn", "is_churn", "churned", "is_churned", "churn_label", "attrition", "left", "cancelled", "canceled", "탈퇴여부", "이탈여부", "이탈", "해지", "해지여부",
    ],
    "converted": [
        "converted", "conversion", "is_converted", "purchase_after_campaign", "responded", "response", "clicked", "coupon_used", "전환", "전환여부", "반응여부", "구매전환",
    ],
    "campaign_exposed": [
        "campaign_exposed", "exposed", "treatment", "treatment_flag", "is_treated", "campaign_sent", "message_sent", "coupon_sent", "캠페인노출", "발송여부", "쿠폰발송",
    ],
    "treatment_group": [
        "treatment_group", "group", "ab_group", "variant", "control_treatment", "실험군", "대조군", "그룹",
    ],
    "segment": ["segment", "persona", "customer_segment", "cluster", "grade", "tier", "등급", "세그먼트", "고객군"],
    "region": ["region", "city", "location", "area", "country", "province", "state", "지역", "도시", "국가"],
    "channel": ["acquisition_channel", "channel", "source", "medium", "utm_source", "유입채널", "채널", "소스"],
    "device": ["device", "device_type", "platform", "os", "browser", "기기", "디바이스", "플랫폼"],
    "category": ["category", "item_category", "product_category", "favorite_category", "preferred_category", "카테고리", "상품군", "품목"],
    "coupon_cost": ["coupon_cost", "discount", "discount_amount", "coupon_value", "benefit_amount", "할인금액", "쿠폰금액", "혜택금액"],
}

REQUIRED_ANY_FIELDS = {
    "customer_id": "A stable customer/member/user identifier is required. If missing, row-based pseudo IDs will be created, but customer aggregation quality will be weaker.",
}


@dataclass
class MappingCandidate:
    canonical_field: str
    source_column: str | None
    confidence: float
    reason: str


@dataclass
class SchemaMapping:
    mapping: dict[str, str | None]
    candidates: dict[str, MappingCandidate] = field(default_factory=dict)
    manual_overrides: dict[str, str] = field(default_factory=dict)

    def get(self, field: str) -> str | None:
        return self.mapping.get(field)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping,
            "manual_overrides": self.manual_overrides,
            "candidates": {
                k: {
                    "canonical_field": v.canonical_field,
                    "source_column": v.source_column,
                    "confidence": v.confidence,
                    "reason": v.reason,
                }
                for k, v in self.candidates.items()
            },
        }

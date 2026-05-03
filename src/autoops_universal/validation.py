from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .mapper import infer_schema_mapping, normalize_column_name
from .profiler import infer_grain, profile_dataframe
from .schema_registry import SchemaMapping


class InvalidAutoOpsDataset(ValueError):
    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class DatasetValidationResult:
    allowed: bool
    dataset_type: str
    confidence_score: float
    mapped_fields: dict[str, str]
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    grain_detection: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "dataset_type": self.dataset_type,
            "confidence_score": self.confidence_score,
            "mapped_fields": self.mapped_fields,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "errors": self.errors,
            "profile": self.profile,
            "grain_detection": self.grain_detection,
        }


TRANSACTION_FIELDS = {"transaction_id", "transaction_date", "amount", "quantity"}
BEHAVIOR_FIELDS = {"label", "recency_days", "frequency_90d", "monetary_90d", "amount", "quantity", "signup_date", "last_activity_date", "transaction_date", "converted", "campaign_exposed", "coupon_cost"}
CONTEXT_FIELDS = {"segment", "region", "channel", "device", "category", "treatment_group"}
TRANSACTION_ID_NAMES = {"basket_id", "basketid", "order_id", "orderid", "transaction_id", "transactionid", "txn_id", "tx_id", "receipt_id", "purchase_id", "cart_id", "session_id", "주문번호", "거래번호", "결제번호", "영수증번호"}


def _mapped_fields(mapping: SchemaMapping, df: pd.DataFrame) -> dict[str, str]:
    cols = {str(c) for c in df.columns}
    return {field: str(col) for field, col in mapping.mapping.items() if col is not None and str(col) in cols}


def _confidence(mapping: SchemaMapping, mapped: dict[str, str]) -> float:
    scores = [float(mapping.candidates[field].confidence) for field in mapped if field in mapping.candidates]
    return round(float(sum(scores) / len(scores)), 4) if scores else 0.0


def validate_uploaded_dataset(df: pd.DataFrame, *, manual_mapping: dict[str, str] | None = None, raise_on_error: bool = True) -> DatasetValidationResult:
    profile = profile_dataframe(df)
    mapping = infer_schema_mapping(df, manual_mapping=manual_mapping)
    mapped = _mapped_fields(mapping, df)
    grain = infer_grain(
        df,
        customer_col=mapped.get("customer_id"),
        transaction_col=mapped.get("transaction_id"),
        transaction_date_col=mapped.get("transaction_date"),
        amount_col=mapped.get("amount") or mapped.get("monetary_90d"),
    )
    errors: list[str] = []
    warnings: list[str] = []
    reasons: list[str] = []

    if df.empty:
        errors.append("업로드한 CSV에 데이터 행이 없습니다.")
    if len(df.columns) < 2:
        errors.append("업로드한 CSV의 컬럼 수가 너무 적습니다. 고객 식별자와 구매/활동/라벨 관련 컬럼이 필요합니다.")

    customer_col = mapped.get("customer_id")
    has_identity = bool(customer_col)
    if customer_col and normalize_column_name(customer_col) in TRANSACTION_ID_NAMES:
        errors.append(f"'{customer_col}' 컬럼은 거래/주문 ID로 보이며 고객 식별자로 사용할 수 없습니다.")
        has_identity = False
    if has_identity:
        reasons.append(f"고객 식별 컬럼 감지: {customer_col}")

    behavior_hits = sorted((BEHAVIOR_FIELDS | TRANSACTION_FIELDS) & set(mapped))
    context_hits = sorted(CONTEXT_FIELDS & set(mapped))
    transaction_hits = sorted(TRANSACTION_FIELDS & set(mapped))
    customer_behavior_hits = sorted(BEHAVIOR_FIELDS & set(mapped))
    if behavior_hits:
        reasons.append("대시보드 산출에 필요한 행동/거래/라벨 컬럼 감지: " + ", ".join(behavior_hits))
    if context_hits:
        reasons.append("세그먼트 보조 컬럼 감지: " + ", ".join(context_hits))

    duplicate_ratio = float(grain.get("duplicate_customer_ratio", 0.0) or 0.0)
    transaction_like = grain.get("grain") == "transaction" or len(transaction_hits) >= 2 or duplicate_ratio >= 0.10

    allowed = False
    dataset_type = "unsupported"
    if not errors:
        if has_identity and customer_behavior_hits:
            allowed = True
            dataset_type = "transaction" if transaction_like else "customer"
        elif has_identity and transaction_hits:
            allowed = True
            dataset_type = "transaction"
        elif has_identity and len(context_hits) >= 2:
            allowed = True
            dataset_type = "customer"
            warnings.append("구매/활동/이탈 라벨이 부족해 일부 지표는 보수적인 기본값 또는 프록시로 생성됩니다.")
        elif not has_identity:
            errors.append("고객 식별 컬럼이 없습니다. 회원번호/고객ID/household_key/고객명/이메일/전화번호 중 하나가 필요합니다.")
        else:
            errors.append("고객 식별자는 감지됐지만 구매금액, 거래일, 구매횟수, 최근활동일, 이탈라벨 등 대시보드 관련 컬럼이 부족합니다.")

    if allowed and len(mapped) < 2:
        allowed = False
        dataset_type = "unsupported"
        errors.append("대시보드 스키마와 연결되는 컬럼이 너무 적습니다.")

    score = _confidence(mapping, mapped)
    if allowed:
        score = round(min(1.0, score + 0.05 * len(behavior_hits) + 0.05), 4)
        return DatasetValidationResult(True, dataset_type, score, mapped, reasons, warnings, [], profile, grain)

    message = "잘못된 CSV 파일입니다. 이 대시보드는 고객 이탈/리텐션 분석용이므로 고객 식별자와 구매·거래·활동·이탈라벨 중 최소 하나 이상의 관련 컬럼이 필요합니다."
    result = DatasetValidationResult(False, "unsupported", score, mapped, reasons, warnings, [message, *errors], profile, grain)
    if raise_on_error:
        raise InvalidAutoOpsDataset(message + (" " + " ".join(errors) if errors else ""), diagnostics=result.as_dict())
    return result

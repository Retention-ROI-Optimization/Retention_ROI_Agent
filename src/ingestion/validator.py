"""
validator.py — CSV relevance & quality validation.

Determines whether an uploaded CSV contains customer/transactional data
relevant to churn/retention analysis, or is completely unrelated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ── Semantic column families ──────────────────────────────────────────

CUSTOMER_ID_SYNONYMS: Set[str] = {
    "customer_id", "cust_id", "user_id", "userid", "member_id", "memberid",
    "account_id", "accountid", "client_id", "clientid", "id", "고객id",
    "고객번호", "회원번호", "customer_no", "cust_no", "subscriber_id",
}

TIMESTAMP_SYNONYMS: Set[str] = {
    "timestamp", "date", "datetime", "event_date", "event_time", "created_at",
    "updated_at", "order_date", "order_time", "purchase_date", "transaction_date",
    "signup_date", "registration_date", "일시", "날짜", "주문일", "가입일",
    "exposure_time", "assigned_at", "snapshot_date", "time", "ts",
}

EVENT_TYPE_SYNONYMS: Set[str] = {
    "event_type", "event", "action", "activity", "action_type", "activity_type",
    "이벤트유형", "행동유형", "behavior", "behaviour",
}

MONETARY_SYNONYMS: Set[str] = {
    "amount", "revenue", "price", "total", "net_amount", "gross_amount",
    "monetary", "spend", "payment", "금액", "매출", "결제금액", "주문금액",
    "transaction_amount", "order_amount", "sales", "net_revenue",
}

CHURN_SYNONYMS: Set[str] = {
    "churn", "churned", "is_churn", "churn_flag", "churn_label", "이탈",
    "이탈여부", "attrition", "left", "cancelled", "canceled", "unsubscribed",
    "status", "current_status", "customer_status",
}

CATEGORY_SYNONYMS: Set[str] = {
    "category", "item_category", "product_category", "카테고리", "상품분류",
    "product_type", "item_type", "department", "section",
}

QUANTITY_SYNONYMS: Set[str] = {
    "quantity", "qty", "수량", "items", "item_count", "count",
}

PERSONA_SYNONYMS: Set[str] = {
    "persona", "segment", "customer_segment", "group", "tier", "등급",
    "고객유형", "membership_tier", "customer_type",
}

REGION_SYNONYMS: Set[str] = {
    "region", "city", "state", "country", "지역", "도시", "location",
    "area", "province", "zip", "zipcode", "postal_code",
}

# Columns that are clearly unrelated to customer/retention analytics
IRRELEVANT_INDICATORS: Set[str] = {
    "latitude", "longitude", "lat", "lng", "lon", "pixel",
    "image_url", "photo", "dna", "genome", "gene", "chromosome",
    "temperature", "humidity", "pressure", "altitude", "speed",
    "voltage", "current", "resistance", "wavelength",
}

# Minimum relevance thresholds
MIN_RELEVANCE_SCORE = 0.25
MIN_ROWS_FOR_ANALYSIS = 10
MAX_PREVIEW_ROWS = 5


@dataclass
class ValidationResult:
    """Result of CSV validation."""
    is_valid: bool
    relevance_score: float  # 0.0 ~ 1.0
    detected_schema: Dict[str, str]  # internal_name -> original_column
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    column_report: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    preview: Optional[pd.DataFrame] = None
    data_type_summary: Dict[str, int] = field(default_factory=dict)
    missing_rate: Dict[str, float] = field(default_factory=dict)


def _normalize_column_name(name: str) -> str:
    """Lowercase, strip, remove special characters."""
    normalized = str(name).strip().lower()
    normalized = re.sub(r'[^a-z0-9가-힣_]', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    return normalized


def _best_match(column: str, synonym_set: Set[str]) -> float:
    """Return match score 0~1 for a column name against a synonym set."""
    normalized = _normalize_column_name(column)
    if normalized in synonym_set:
        return 1.0
    # Partial match
    for syn in synonym_set:
        if syn in normalized or normalized in syn:
            return 0.7
    return 0.0


def _detect_column_role(column: str, sample_values: pd.Series) -> Tuple[str, float]:
    """Detect what role a column plays (customer_id, timestamp, etc.)."""
    norm = _normalize_column_name(column)

    checks = [
        ("customer_id", CUSTOMER_ID_SYNONYMS),
        ("timestamp", TIMESTAMP_SYNONYMS),
        ("event_type", EVENT_TYPE_SYNONYMS),
        ("amount", MONETARY_SYNONYMS),
        ("churn_flag", CHURN_SYNONYMS),
        ("category", CATEGORY_SYNONYMS),
        ("quantity", QUANTITY_SYNONYMS),
        ("persona", PERSONA_SYNONYMS),
        ("region", REGION_SYNONYMS),
    ]

    best_role = "unknown"
    best_score = 0.0

    for role, synonyms in checks:
        score = _best_match(column, synonyms)
        if score > best_score:
            best_score = score
            best_role = role

    # Heuristic: if name doesn't match, check data patterns
    if best_score < 0.5 and not sample_values.dropna().empty:
        sample = sample_values.dropna().head(100)

        # Check if it looks like an ID column (unique integers)
        if sample.dtype in (np.int64, np.float64, 'int64', 'float64'):
            nunique_ratio = sample.nunique() / max(len(sample), 1)
            if nunique_ratio > 0.9:
                if best_role == "unknown":
                    best_role = "potential_id"
                    best_score = 0.3

        # Check if it looks like a timestamp
        if sample.dtype == 'object':
            try:
                parsed = pd.to_datetime(sample.head(20), errors='coerce')
                if parsed.notna().mean() > 0.7:
                    best_role = "timestamp"
                    best_score = max(best_score, 0.6)
            except Exception:
                pass

        # Check if it looks like a binary churn flag
        if set(sample.unique()) <= {0, 1, '0', '1', True, False, 'yes', 'no', 'Y', 'N'}:
            if best_role == "unknown":
                best_role = "potential_binary"
                best_score = 0.3

    return best_role, best_score


def _check_irrelevance(columns: List[str]) -> Tuple[bool, str]:
    """Check if the data is clearly irrelevant (e.g., weather, genome data)."""
    normalized = {_normalize_column_name(c) for c in columns}
    overlap = normalized & IRRELEVANT_INDICATORS
    if len(overlap) >= 3:
        return True, f"이 CSV는 고객/리텐션 분석과 관련이 없는 데이터로 보입니다. 감지된 비관련 컬럼: {', '.join(overlap)}"
    return False, ""


def _compute_relevance_score(detected: Dict[str, Tuple[str, float]]) -> float:
    """Compute overall relevance score based on detected columns."""
    weights = {
        "customer_id": 0.30,
        "timestamp": 0.20,
        "event_type": 0.15,
        "amount": 0.15,
        "churn_flag": 0.10,
        "category": 0.05,
        "quantity": 0.03,
        "persona": 0.01,
        "region": 0.01,
    }
    score = 0.0
    for role, (col, confidence) in detected.items():
        weight = weights.get(role, 0.0)
        score += weight * confidence
    return min(score, 1.0)


def validate_csv(
    file_path: str | Path,
    *,
    max_preview_rows: int = MAX_PREVIEW_ROWS,
    encoding: str = "utf-8",
    chunk_size: int | None = None,
) -> ValidationResult:
    """
    Validate an uploaded CSV file for relevance to customer retention analysis.

    Returns a ValidationResult with:
    - is_valid: whether the file can be used
    - relevance_score: 0~1 how relevant the data appears
    - detected_schema: mapping of internal names to original column names
    - warnings/errors: issues found
    """
    path = Path(file_path)
    warnings: List[str] = []
    errors: List[str] = []

    # ── Basic file checks ──
    if not path.exists():
        return ValidationResult(
            is_valid=False, relevance_score=0.0, detected_schema={},
            errors=[f"파일을 찾을 수 없습니다: {path}"],
        )

    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        return ValidationResult(
            is_valid=False, relevance_score=0.0, detected_schema={},
            errors=[f"지원하지 않는 파일 형식입니다: {path.suffix}. CSV 파일만 지원합니다."],
        )

    # ── Read file ──
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        # Try multiple encodings
        df = None
        for enc in [encoding, "utf-8", "cp949", "euc-kr", "latin-1"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if df is None:
            return ValidationResult(
                is_valid=False, relevance_score=0.0, detected_schema={},
                errors=["파일 인코딩을 감지할 수 없습니다. UTF-8, CP949, EUC-KR, Latin-1 인코딩을 모두 시도했습니다."],
            )
    except pd.errors.EmptyDataError:
        return ValidationResult(
            is_valid=False, relevance_score=0.0, detected_schema={},
            errors=["빈 CSV 파일입니다."],
        )
    except Exception as exc:
        return ValidationResult(
            is_valid=False, relevance_score=0.0, detected_schema={},
            errors=[f"CSV 파일 읽기 실패: {exc}"],
        )

    row_count = len(df)
    col_count = len(df.columns)

    if row_count < MIN_ROWS_FOR_ANALYSIS:
        errors.append(f"데이터 행 수가 너무 적습니다 ({row_count}행). 최소 {MIN_ROWS_FOR_ANALYSIS}행 이상 필요합니다.")

    if col_count < 2:
        errors.append("컬럼이 2개 미만입니다. 분석 가능한 데이터가 아닙니다.")

    # ── Irrelevance check ──
    is_irrelevant, irrelevance_msg = _check_irrelevance(df.columns.tolist())
    if is_irrelevant:
        return ValidationResult(
            is_valid=False, relevance_score=0.0, detected_schema={},
            errors=[irrelevance_msg],
            row_count=row_count, column_count=col_count,
            preview=df.head(max_preview_rows),
        )

    # ── Detect column roles ──
    detected: Dict[str, Tuple[str, float]] = {}
    column_report: List[Dict[str, Any]] = []
    missing_rates: Dict[str, float] = {}

    for col in df.columns:
        role, confidence = _detect_column_role(col, df[col])
        missing_rate = float(df[col].isna().mean())
        missing_rates[col] = missing_rate

        report_entry = {
            "original_name": col,
            "detected_role": role,
            "confidence": round(confidence, 3),
            "dtype": str(df[col].dtype),
            "missing_rate": round(missing_rate, 4),
            "nunique": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist(),
        }
        column_report.append(report_entry)

        # Keep the best match for each role
        if role not in {"unknown", "potential_id", "potential_binary"}:
            if role not in detected or confidence > detected[role][1]:
                detected[role] = (col, confidence)
        elif role == "potential_id" and "customer_id" not in detected:
            detected["customer_id"] = (col, confidence)

    # ── Compute relevance ──
    relevance = _compute_relevance_score(detected)

    # ── Build schema mapping ──
    schema_mapping: Dict[str, str] = {}
    for role, (col, conf) in detected.items():
        if conf >= 0.3:
            schema_mapping[role] = col

    # ── Quality warnings ──
    high_missing = [col for col, rate in missing_rates.items() if rate > 0.5]
    if high_missing:
        warnings.append(f"결측률 50% 초과 컬럼: {', '.join(high_missing[:5])}")

    duplicate_ids = False
    if "customer_id" in schema_mapping:
        id_col = schema_mapping["customer_id"]
        if df[id_col].duplicated().any():
            dup_count = int(df[id_col].duplicated().sum())
            warnings.append(f"고객 ID 컬럼({id_col})에 중복 값 {dup_count:,}개가 있습니다. 트랜잭션 레벨 데이터일 수 있습니다.")
            duplicate_ids = True

    if "customer_id" not in schema_mapping:
        warnings.append("고객 ID로 사용할 수 있는 컬럼을 찾지 못했습니다. 첫 번째 고유 컬럼을 ID로 사용합니다.")
        # Fallback: use first column with high uniqueness
        for col in df.columns:
            nunique_ratio = df[col].nunique() / max(len(df), 1)
            if nunique_ratio > 0.5:
                schema_mapping["customer_id"] = col
                break

    # ── Data type summary ──
    dtype_summary = {
        "numeric": int(df.select_dtypes(include=[np.number]).shape[1]),
        "categorical": int(df.select_dtypes(include=['object', 'category']).shape[1]),
        "datetime": int(df.select_dtypes(include=['datetime']).shape[1]),
        "boolean": int(df.select_dtypes(include=['bool']).shape[1]),
    }

    is_valid = relevance >= MIN_RELEVANCE_SCORE and not errors and row_count >= MIN_ROWS_FOR_ANALYSIS

    if not is_valid and relevance < MIN_RELEVANCE_SCORE and not errors:
        errors.append(
            f"이 CSV 파일은 고객 리텐션 분석과의 관련성이 낮습니다 (관련성 점수: {relevance:.1%}). "
            "고객 ID, 날짜/시간, 거래 금액, 이벤트 유형 등의 컬럼이 포함된 데이터를 업로드해 주세요."
        )

    return ValidationResult(
        is_valid=is_valid,
        relevance_score=relevance,
        detected_schema=schema_mapping,
        warnings=warnings,
        errors=errors,
        column_report=column_report,
        row_count=row_count,
        column_count=col_count,
        preview=df.head(max_preview_rows),
        data_type_summary=dtype_summary,
        missing_rate=missing_rates,
    )


def validate_multiple_csvs(
    file_paths: List[str | Path],
) -> Dict[str, ValidationResult]:
    """Validate multiple CSV files and return results keyed by filename."""
    results = {}
    for path in file_paths:
        p = Path(path)
        results[p.name] = validate_csv(p)
    return results

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from .schema_registry import FIELD_ALIASES, MappingCandidate, SchemaMapping


def normalize_column_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-\.\/\\]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_가-힣]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    ambiguous_short_tokens = {"id", "day", "date", "time", "no", "key"}
    if a in b or b in a:
        shorter = a if len(a) <= len(b) else b
        if len(shorter) <= 4 and shorter in ambiguous_short_tokens:
            return SequenceMatcher(None, a, b).ratio()
        return 0.86
    return SequenceMatcher(None, a, b).ratio()


def _score_column(canonical: str, column: str) -> tuple[float, str]:
    norm_col = normalize_column_name(column)
    aliases = [canonical, *FIELD_ALIASES.get(canonical, [])]
    best = 0.0
    best_alias = canonical
    for alias in aliases:
        norm_alias = normalize_column_name(alias)
        score = _similarity(norm_alias, norm_col)
        if score > best:
            best = score
            best_alias = alias
    if best >= 1.0:
        reason = f"exact alias match: {best_alias}"
    elif best >= 0.86:
        reason = f"substring/strong alias match: {best_alias}"
    else:
        reason = f"fuzzy match: {best_alias}"
    return float(best), reason


def infer_schema_mapping(
    df: pd.DataFrame,
    *,
    manual_mapping: dict[str, str] | None = None,
    min_confidence: float = 0.72,
) -> SchemaMapping:
    """Infer source-to-canonical column mapping.

    Manual mapping has priority and should use canonical field names as keys and
    uploaded CSV columns as values, e.g. `{ "customer_id": "회원번호" }`.
    """
    manual_mapping = manual_mapping or {}
    columns = [str(col) for col in df.columns]
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    candidates: dict[str, MappingCandidate] = {}

    for canonical in FIELD_ALIASES:
        if canonical in manual_mapping and manual_mapping[canonical] in columns:
            col = manual_mapping[canonical]
            mapping[canonical] = col
            used.add(col)
            candidates[canonical] = MappingCandidate(canonical, col, 1.0, "manual override")
            continue

        best_col: str | None = None
        best_score = 0.0
        best_reason = "no candidate"
        for col in columns:
            if col in used:
                continue
            score, reason = _score_column(canonical, col)
            if score > best_score:
                best_score = score
                best_col = col
                best_reason = reason

        field_threshold = 0.92 if canonical == "transaction_id" else min_confidence
        if best_col is not None and best_score >= field_threshold:
            mapping[canonical] = best_col
            used.add(best_col)
            candidates[canonical] = MappingCandidate(canonical, best_col, best_score, best_reason)
        else:
            mapping[canonical] = None
            candidates[canonical] = MappingCandidate(canonical, None, best_score, best_reason)

    return SchemaMapping(mapping=mapping, candidates=candidates, manual_overrides=manual_mapping)


def mapping_table(mapping: SchemaMapping) -> pd.DataFrame:
    rows = []
    for canonical, candidate in mapping.candidates.items():
        rows.append({
            "canonical_field": canonical,
            "source_column": candidate.source_column,
            "confidence": round(candidate.confidence, 4),
            "reason": candidate.reason,
            "manual_override": canonical in mapping.manual_overrides,
        })
    return pd.DataFrame(rows).sort_values(["source_column", "canonical_field"], na_position="last")

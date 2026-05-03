from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.api.services.analytics import (
    DEFAULT_SEGMENT_ORDER,
    budget_allocation_by_segment,
    get_budget_result as _get_budget_result,
)
from src.optimization.timing import load_survival_predictions


def _candidate_result_dirs() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / 'results',
        project_root / 'results',
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _load_first_survival_predictions() -> pd.DataFrame:
    for result_dir in _candidate_result_dirs():
        df = load_survival_predictions(result_dir)
        if not df.empty:
            return df
    return pd.DataFrame()


def get_budget_result(
    customers: pd.DataFrame,
    budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
):
    return _get_budget_result(
        customers=customers,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
        survival_predictions=_load_first_survival_predictions(),
    )


__all__ = [
    "DEFAULT_SEGMENT_ORDER",
    "budget_allocation_by_segment",
    "get_budget_result",
]

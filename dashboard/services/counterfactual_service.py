from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.optimization.counterfactual import ACTION_CATALOG, build_counterfactual_retention_lab as _build_counterfactual_retention_lab
from src.optimization.timing import load_survival_predictions


def _candidate_result_dirs() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / "results_user",
        Path.cwd() / "results_ecommerce",
        Path.cwd() / "results_finance",
        Path.cwd() / "results",
        Path.cwd() / "results_simulator",
        project_root / "results_user",
        project_root / "results_ecommerce",
        project_root / "results_finance",
        project_root / "results",
        project_root / "results_simulator",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
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


def build_counterfactual_retention_lab(
    customers: pd.DataFrame,
    selected_customers: Optional[pd.DataFrame] = None,
    survival_predictions: Optional[pd.DataFrame] = None,
    *,
    top_n: Optional[int] = 100,
    threshold: float = 0.50,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    resolved_survival = survival_predictions
    if resolved_survival is None or resolved_survival.empty:
        resolved_survival = _load_first_survival_predictions()
    return _build_counterfactual_retention_lab(
        customers=customers,
        selected_customers=selected_customers,
        survival_predictions=resolved_survival,
        top_n=top_n,
        threshold=threshold,
    )


__all__ = ["ACTION_CATALOG", "build_counterfactual_retention_lab"]

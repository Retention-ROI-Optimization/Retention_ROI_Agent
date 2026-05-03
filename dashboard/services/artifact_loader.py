from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.autoops_universal.dashboard_contract import ensure_dashboard_contract


@dataclass(frozen=True)
class DashboardArtifactsBundle:
    result_dir: str
    model_dir: str
    feature_store_dir: str

    feature_summary: Optional[Dict]
    customer_features: pd.DataFrame
    customer_features_metadata: Optional[Dict]

    churn_metrics: Optional[Dict]
    threshold_analysis: Optional[Dict]
    top_feature_importance: pd.DataFrame

    uplift_summary: Optional[Dict]
    uplift_segmentation: pd.DataFrame

    optimization_summary: Optional[Dict]
    optimization_segment_budget: pd.DataFrame
    optimization_selected_customers: pd.DataFrame

    image_paths: Dict[str, Optional[str]]
    model_paths: Dict[str, Optional[str]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _safe_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])

    return pd.DataFrame()


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "customer_id" in df.columns:
        return ensure_dashboard_contract(df)
    return df


def _safe_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def load_dashboard_artifacts(
    result_dir: str = "results",
    model_dir: str = "models",
    feature_store_dir: str = "data/feature_store",
) -> DashboardArtifactsBundle:
    result_base = _resolve_dir(result_dir)
    model_base = _resolve_dir(model_dir)
    feature_base = _resolve_dir(feature_store_dir)

    churn_model = None
    model_candidates = sorted(model_base.glob("churn_model_*.joblib"))
    if model_candidates:
        churn_model = str(model_candidates[0])

    image_paths = {
        "churn_auc_roc": _safe_path(result_base / "churn_auc_roc.png"),
        "churn_precision_recall_tradeoff": _safe_path(
            result_base / "churn_precision_recall_tradeoff.png"
        ),
        "churn_shap_summary": _safe_path(result_base / "churn_shap_summary.png"),
        "churn_shap_local": _safe_path(result_base / "churn_shap_local.png"),
    }

    model_paths = {
        "churn_model": churn_model,
    }

    return DashboardArtifactsBundle(
        result_dir=str(result_base),
        model_dir=str(model_base),
        feature_store_dir=str(feature_base),
        feature_summary=_safe_json(result_base / "feature_engineering_summary.json"),
        customer_features=_safe_csv(feature_base / "customer_features.csv"),
        customer_features_metadata=_safe_json(feature_base / "customer_features_metadata.json"),
        churn_metrics=_safe_json(result_base / "churn_metrics.json"),
        threshold_analysis=_safe_json(result_base / "churn_threshold_analysis.json"),
        top_feature_importance=_safe_json_df(result_base / "churn_top10_feature_importance.json"),
        uplift_summary=_safe_json(result_base / "uplift_summary.json"),
        uplift_segmentation=_safe_csv(result_base / "uplift_segmentation.csv"),
        optimization_summary=_safe_json(result_base / "optimization_summary.json"),
        optimization_segment_budget=_safe_csv(result_base / "optimization_segment_budget.csv"),
        optimization_selected_customers=_safe_csv(
            result_base / "optimization_selected_customers.csv"
        ),
        image_paths=image_paths,
        model_paths=model_paths,
    )
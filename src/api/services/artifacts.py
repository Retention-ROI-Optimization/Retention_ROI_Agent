from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.api.settings import ApiSettings
from src.workflows.pipeline_runner import (
    run_churn_training_pipeline,
    run_optimize_pipeline,
    run_uplift_pipeline,
)


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
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
    return pd.read_csv(path, low_memory=False)


def _safe_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _find_model_path(model_dir: Path) -> Optional[str]:
    candidates = sorted(model_dir.glob("churn_model_*.joblib"))
    if not candidates:
        candidates = sorted(model_dir.glob("churn_model*.joblib"))
    return str(candidates[0]) if candidates else None


def training_artifacts_missing(settings: ApiSettings) -> bool:
    result_dir = settings.resolved_result_dir
    feature_store_dir = settings.resolved_feature_store_dir
    required = [
        result_dir / "churn_metrics.json",
        result_dir / "churn_threshold_analysis.json",
        feature_store_dir / "customer_features.csv",
        feature_store_dir / "customer_features_metadata.json",
    ]
    if any(not path.exists() for path in required):
        return True
    return _find_model_path(settings.resolved_model_dir) is None


def uplift_artifacts_missing(settings: ApiSettings) -> bool:
    result_dir = settings.resolved_result_dir
    return not ((result_dir / "uplift_summary.json").exists() and (result_dir / "uplift_segmentation.csv").exists())


def optimization_artifacts_missing(settings: ApiSettings) -> bool:
    result_dir = settings.resolved_result_dir
    return not (
        (result_dir / "optimization_summary.json").exists()
        and (result_dir / "optimization_segment_budget.csv").exists()
        and (result_dir / "optimization_selected_customers.csv").exists()
    )


def optimization_budget_mismatch(settings: ApiSettings, budget: int) -> bool:
    summary_path = settings.resolved_result_dir / "optimization_summary.json"
    if not summary_path.exists():
        return True
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return int(summary.get("budget", -1)) != int(budget)


def ensure_training_artifacts(settings: ApiSettings, rebuild: bool = False) -> None:
    if rebuild or training_artifacts_missing(settings):
        run_churn_training_pipeline(
            data_dir=settings.resolved_data_dir,
            model_dir=settings.resolved_model_dir,
            result_dir=settings.resolved_result_dir,
            feature_store_dir=settings.resolved_feature_store_dir,
        )


def ensure_saved_results_artifacts(settings: ApiSettings, budget: int, rebuild: bool = False) -> None:
    if rebuild or uplift_artifacts_missing(settings):
        run_uplift_pipeline(
            data_dir=settings.resolved_data_dir,
            result_dir=settings.resolved_result_dir,
        )
    if rebuild or optimization_artifacts_missing(settings) or optimization_budget_mismatch(settings, budget):
        run_optimize_pipeline(
            data_dir=settings.resolved_data_dir,
            result_dir=settings.resolved_result_dir,
            budget=int(budget),
        )


def load_training_artifacts_payload(settings: ApiSettings) -> Dict[str, Any]:
    result_dir = settings.resolved_result_dir
    model_dir = settings.resolved_model_dir
    feature_store_dir = settings.resolved_feature_store_dir

    return {
        "directories": {
            "result_dir": str(result_dir),
            "model_dir": str(model_dir),
            "feature_store_dir": str(feature_store_dir),
        },
        "feature_summary": _safe_json(result_dir / "feature_engineering_summary.json"),
        "customer_features": _safe_csv(feature_store_dir / "customer_features.csv").head(200).to_dict(orient="records"),
        "customer_features_metadata": _safe_json(feature_store_dir / "customer_features_metadata.json"),
        "churn_metrics": _safe_json(result_dir / "churn_metrics.json"),
        "threshold_analysis": _safe_json(result_dir / "churn_threshold_analysis.json"),
        "top_feature_importance": _safe_json_df(result_dir / "churn_top10_feature_importance.json").to_dict(orient="records"),
        "image_paths": {
            "churn_auc_roc": _safe_path(result_dir / "churn_auc_roc.png"),
            "churn_precision_recall_tradeoff": _safe_path(result_dir / "churn_precision_recall_tradeoff.png"),
            "churn_shap_summary": _safe_path(result_dir / "churn_shap_summary.png"),
            "churn_shap_local": _safe_path(result_dir / "churn_shap_local.png"),
        },
        "model_paths": {
            "churn_model": _find_model_path(model_dir),
        },
    }


def load_saved_results_payload(settings: ApiSettings) -> Dict[str, Any]:
    result_dir = settings.resolved_result_dir
    return {
        "result_dir": str(result_dir),
        "uplift_summary": _safe_json(result_dir / "uplift_summary.json"),
        "uplift_segmentation": _safe_csv(result_dir / "uplift_segmentation.csv").head(200).to_dict(orient="records"),
        "optimization_summary": _safe_json(result_dir / "optimization_summary.json"),
        "optimization_segment_budget": _safe_csv(result_dir / "optimization_segment_budget.csv").to_dict(orient="records"),
        "optimization_selected_customers": _safe_csv(result_dir / "optimization_selected_customers.csv").head(200).to_dict(orient="records"),
    }

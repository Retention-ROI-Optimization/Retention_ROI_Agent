from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.api.settings import ApiSettings
from src.survival.modeling import run_survival_pipeline


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def ensure_survival_artifacts(
    settings: ApiSettings,
    *,
    rebuild: bool = False,
) -> None:
    metrics_path = settings.resolved_result_dir / 'survival_metrics.json'
    predictions_path = settings.resolved_result_dir / 'survival_predictions.csv'
    model_path = settings.resolved_model_dir / 'survival_cox_model.joblib'
    if rebuild or any(not path.exists() for path in [metrics_path, predictions_path, model_path]):
        run_survival_pipeline(
            data_dir=settings.resolved_data_dir,
            model_dir=settings.resolved_model_dir,
            result_dir=settings.resolved_result_dir,
            feature_store_dir=settings.resolved_feature_store_dir,
            horizon_days=settings.survival_horizon_days,
            test_size=settings.survival_test_size,
            random_state=settings.survival_random_state,
            penalizer=settings.survival_penalizer,
        )


def load_survival_payload(settings: ApiSettings, *, top_n: int = 50) -> Dict[str, Any]:
    metrics = _safe_json(settings.resolved_result_dir / 'survival_metrics.json')
    predictions = _safe_csv(settings.resolved_result_dir / 'survival_predictions.csv')
    coefficients = _safe_csv(settings.resolved_result_dir / 'survival_top_coefficients.csv')
    risk_plot_path = settings.resolved_result_dir / 'survival_risk_stratification.png'

    if not predictions.empty:
        top_predictions = predictions.sort_values(['predicted_hazard_ratio', 'customer_id'], ascending=[False, True]).head(int(top_n))
    else:
        top_predictions = pd.DataFrame()

    return {
        'metrics': metrics,
        'predictions': top_predictions.to_dict(orient='records') if not top_predictions.empty else [],
        'coefficients': coefficients.head(20).to_dict(orient='records') if not coefficients.empty else [],
        'image_paths': {
            'risk_stratification': str(risk_plot_path) if risk_plot_path.exists() else None,
        },
    }

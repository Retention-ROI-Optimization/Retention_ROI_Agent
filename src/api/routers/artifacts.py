from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_settings
from src.api.schemas import SavedResultsArtifactsResponse, TrainingArtifactsResponse
from src.api.services.artifacts import (
    ensure_saved_results_artifacts,
    ensure_training_artifacts,
    load_saved_results_payload,
    load_training_artifacts_payload,
    resolve_training_request,
)
from src.api.settings import ApiSettings

router = APIRouter(prefix='/artifacts', tags=['artifacts'])


@router.get('/training', response_model=TrainingArtifactsResponse)
def training_artifacts(
    rebuild: bool = Query(default=False),
    test_size: float = Query(default=0.20, ge=0.05, le=0.50),
    random_state: int = Query(default=42, ge=0),
    shap_sample_size: int = Query(default=300, ge=20, le=5000),
    models: str = Query(default='xgboost,lightgbm'),
    threshold_tp_value: float = Query(default=120000.0, ge=0.0),
    threshold_fp_cost: float = Query(default=18000.0, ge=0.0),
    threshold_fn_cost: float = Query(default=60000.0, ge=0.0),
    settings: ApiSettings = Depends(get_settings),
) -> TrainingArtifactsResponse:
    training_request = resolve_training_request(
        test_size=test_size,
        random_state=random_state,
        shap_sample_size=shap_sample_size,
        models=models,
        threshold_tp_value=threshold_tp_value,
        threshold_fp_cost=threshold_fp_cost,
        threshold_fn_cost=threshold_fn_cost,
    )
    ensure_training_artifacts(settings, rebuild=rebuild, training_request=training_request)
    return TrainingArtifactsResponse(**load_training_artifacts_payload(settings, training_request=training_request))


@router.get('/saved-results', response_model=SavedResultsArtifactsResponse)
def saved_results_artifacts(
    budget: int = Query(default=5000000, ge=1),
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    max_customers: int | None = Query(default=None, ge=1),
    rebuild: bool = Query(default=False),
    settings: ApiSettings = Depends(get_settings),
) -> SavedResultsArtifactsResponse:
    ensure_saved_results_artifacts(settings, budget=budget, rebuild=rebuild)
    return SavedResultsArtifactsResponse(
        **load_saved_results_payload(
            settings,
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
        )
    )

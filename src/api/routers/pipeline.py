from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_repository, get_settings
from src.api.schemas import PipelineRunRequest, PipelineRunResponse
from src.api.services.pipeline import run_mode
from src.api.services.repository import DataRepository
from src.api.settings import ApiSettings

router = APIRouter(prefix='/pipeline', tags=['pipeline'])


@router.post('/train', response_model=PipelineRunResponse)
def run_train(
    request: PipelineRunRequest,
    settings: ApiSettings = Depends(get_settings),
    repository: DataRepository = Depends(get_repository),
) -> PipelineRunResponse:
    result = run_mode(
        'train',
        settings.resolved_data_dir,
        settings.resolved_model_dir,
        settings.resolved_result_dir,
        feature_store_dir=settings.resolved_feature_store_dir,
        force_simulation=request.force_simulation,
    )
    repository.reload_all()
    return PipelineRunResponse(**result)


@router.post('/uplift', response_model=PipelineRunResponse)
def run_uplift(
    request: PipelineRunRequest,
    settings: ApiSettings = Depends(get_settings),
    repository: DataRepository = Depends(get_repository),
) -> PipelineRunResponse:
    result = run_mode(
        'uplift',
        settings.resolved_data_dir,
        settings.resolved_model_dir,
        settings.resolved_result_dir,
        force_simulation=request.force_simulation,
    )
    repository.reload_all()
    return PipelineRunResponse(**result)


@router.post('/optimize', response_model=PipelineRunResponse)
def run_optimize(
    request: PipelineRunRequest,
    budget: int = Query(default=50000000, ge=1),
    settings: ApiSettings = Depends(get_settings),
    repository: DataRepository = Depends(get_repository),
) -> PipelineRunResponse:
    result = run_mode(
        'optimize',
        settings.resolved_data_dir,
        settings.resolved_model_dir,
        settings.resolved_result_dir,
        budget=request.budget or budget,
        force_simulation=request.force_simulation,
    )
    repository.reload_all()
    return PipelineRunResponse(**result)



@router.post('/abtest', response_model=PipelineRunResponse)
def run_abtest(
    request: PipelineRunRequest,
    settings: ApiSettings = Depends(get_settings),
    repository: DataRepository = Depends(get_repository),
) -> PipelineRunResponse:
    result = run_mode(
        'abtest',
        settings.resolved_data_dir,
        settings.resolved_model_dir,
        settings.resolved_result_dir,
        force_simulation=request.force_simulation,
    )
    repository.reload_all()
    return PipelineRunResponse(**result)


@router.post('/survival', response_model=PipelineRunResponse)
def run_survival(
    request: PipelineRunRequest,
    settings: ApiSettings = Depends(get_settings),
    repository: DataRepository = Depends(get_repository),
) -> PipelineRunResponse:
    result = run_mode(
        'survival',
        settings.resolved_data_dir,
        settings.resolved_model_dir,
        settings.resolved_result_dir,
        feature_store_dir=settings.resolved_feature_store_dir,
        force_simulation=request.force_simulation,
    )
    repository.reload_all()
    return PipelineRunResponse(**result)

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api import __version__
from src.api.dependencies import get_repository
from src.api.schemas import HealthResponse, TableListResponse
from src.api.services.repository import DataRepository

router = APIRouter(tags=['health'])


@router.get('/health', response_model=HealthResponse)
def health(repository: DataRepository = Depends(get_repository)) -> HealthResponse:
    return HealthResponse(
        status='ok',
        api_version=__version__,
        data_dir=str(repository.data_dir),
        available_tables=[name for name, present in repository.available_tables().items() if present],
    )


@router.get('/tables', response_model=TableListResponse)
def table_status(repository: DataRepository = Depends(get_repository)) -> TableListResponse:
    return TableListResponse(tables=repository.available_tables())

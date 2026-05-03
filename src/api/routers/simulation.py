from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, Depends

from src.api.dependencies import get_repository, get_settings
from src.api.schemas import SimulationRunRequest, SimulationRunResponse
from src.api.services.repository import DataRepository
from src.api.settings import ApiSettings
from src.simulator.config import DEFAULT_CONFIG
from src.simulator.pipeline import run_simulation

router = APIRouter(prefix='/simulation', tags=['simulation'])


@router.post('/run', response_model=SimulationRunResponse)
def run_simulation_endpoint(
    request: SimulationRunRequest,
    settings: ApiSettings = Depends(get_settings),
    repository: DataRepository = Depends(get_repository),
) -> SimulationRunResponse:
    config_kwargs = request.config.model_dump(exclude_none=True)
    config = replace(DEFAULT_CONFIG, **config_kwargs)

    target_dir = Path(request.output_dir) if request.output_dir else settings.resolved_data_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    tables = run_simulation(
        config=config,
        export=request.export,
        output_dir=str(target_dir),
        file_format=request.file_format or config.default_file_format,
    )

    if request.persist_to_api_data_dir and target_dir.resolve() != settings.resolved_data_dir:
        run_simulation(
            config=config,
            export=True,
            output_dir=str(settings.resolved_data_dir),
            file_format=request.file_format or config.default_file_format,
        )

    repository.reload_all()

    return SimulationRunResponse(
        exported=bool(request.export),
        output_dir=str(target_dir),
        file_format=request.file_format or config.default_file_format,
        tables={name: int(len(df)) for name, df in tables.items()},
        customer_count=int(len(tables.get('customer_summary', []))),
        event_count=int(len(tables.get('events', []))),
        order_count=int(len(tables.get('orders', []))),
        cohort_row_count=int(len(tables.get('cohort_retention', []))),
    )

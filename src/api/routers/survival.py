from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_settings
from src.api.schemas import SurvivalArtifactsResponse
from src.api.services.survival import ensure_survival_artifacts, load_survival_payload
from src.api.settings import ApiSettings

router = APIRouter(prefix='/survival', tags=['survival'])


@router.get('/summary', response_model=SurvivalArtifactsResponse)
def survival_summary(
    top_n: int = Query(default=50, ge=1, le=500),
    rebuild: bool = Query(default=False),
    settings: ApiSettings = Depends(get_settings),
) -> SurvivalArtifactsResponse:
    ensure_survival_artifacts(settings, rebuild=rebuild)
    return SurvivalArtifactsResponse(**load_survival_payload(settings, top_n=top_n))

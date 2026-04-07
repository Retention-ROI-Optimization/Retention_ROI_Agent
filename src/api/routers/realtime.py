from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_settings
from src.api.schemas import RealtimeScoringResponse
from src.api.services.realtime import load_realtime_payload
from src.api.settings import ApiSettings

router = APIRouter(prefix='/realtime', tags=['realtime'])


@router.get('/scores', response_model=RealtimeScoringResponse)
def realtime_scores(
    top_n: int = Query(default=50, ge=1, le=500),
    settings: ApiSettings = Depends(get_settings),
) -> RealtimeScoringResponse:
    payload = load_realtime_payload(settings, top_n=top_n)
    return RealtimeScoringResponse(**payload)

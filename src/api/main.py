from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_settings
from src.api.routers.analytics import router as analytics_router
from src.api.routers.artifacts import router as artifacts_router
from src.api.routers.health import router as health_router
from src.api.routers.pipeline import router as pipeline_router
from src.api.routers.realtime import router as realtime_router
from src.api.routers.recommendations import router as recommendation_router
from src.api.routers.simulation import router as simulation_router
from src.api.routers.survival import router as survival_router
from src.api.services.pipeline import bootstrap_data

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_data(settings.resolved_data_dir)
    yield


app = FastAPI(
    title=settings.app_name,
    description='FastAPI backend for the Retention ROI simulator, pipelines, and dashboard.',
    version='0.3.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins or ['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health_router)
app.include_router(analytics_router, prefix=settings.api_prefix)
app.include_router(simulation_router, prefix=settings.api_prefix)
app.include_router(pipeline_router, prefix=settings.api_prefix)
app.include_router(recommendation_router, prefix=settings.api_prefix)
app.include_router(artifacts_router, prefix=settings.api_prefix)
app.include_router(realtime_router, prefix=settings.api_prefix)
app.include_router(survival_router, prefix=settings.api_prefix)

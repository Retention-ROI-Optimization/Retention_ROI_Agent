from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api_router import router as universal_autoops_router

app = FastAPI(
    title="Retention ROI Universal AutoOps API",
    version="4.0.0",
    description="Additive API for arbitrary user dataset schema mapping, canonicalization, retraining, and dashboard artifact refresh.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "retention-roi-universal-autoops", "health": "/api/v1/universal-autoops/health"}


app.include_router(universal_autoops_router)

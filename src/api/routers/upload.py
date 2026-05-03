"""
Upload router — handles CSV file uploads via API.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.api.dependencies import get_repository, get_settings
from src.api.services.repository import DataRepository
from src.api.settings import ApiSettings
from src.ingestion.validator import validate_csv
from src.ingestion.pipeline import run_ingestion_pipeline


router = APIRouter(prefix='/upload', tags=['upload'])


class UploadValidationResponse(BaseModel):
    is_valid: bool
    relevance_score: float
    detected_schema: Dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    data_type_summary: Dict[str, int] = Field(default_factory=dict)


class UploadProcessResponse(BaseModel):
    success: bool
    stage: str
    error: Optional[str] = None
    validation: UploadValidationResponse
    preprocessing_metadata: Dict[str, Any] = Field(default_factory=dict)
    training_metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post('/validate', response_model=UploadValidationResponse)
async def validate_upload(
    file: UploadFile = File(...),
) -> UploadValidationResponse:
    """Validate an uploaded CSV file without processing it."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = validate_csv(tmp_path)
        return UploadValidationResponse(
            is_valid=result.is_valid,
            relevance_score=result.relevance_score,
            detected_schema=result.detected_schema,
            warnings=result.warnings,
            errors=result.errors,
            row_count=result.row_count,
            column_count=result.column_count,
            data_type_summary=result.data_type_summary,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post('/process', response_model=UploadProcessResponse)
async def process_upload(
    file: UploadFile = File(...),
    budget: int = Query(default=5000000, ge=1),
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    max_customers: int = Query(default=1500, ge=1),
    skip_training: bool = Query(default=False),
    settings: ApiSettings = Depends(get_settings),
    repository: DataRepository = Depends(get_repository),
) -> UploadProcessResponse:
    """Upload, validate, preprocess, and train on a CSV file."""
    # Save uploaded file
    upload_dir = settings.resolved_data_dir.parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / file.filename

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pipeline_result = run_ingestion_pipeline(
            file_path=upload_path,
            data_dir=settings.resolved_data_dir,
            model_dir=settings.resolved_model_dir,
            result_dir=settings.resolved_result_dir,
            feature_store_dir=settings.resolved_feature_store_dir,
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
            skip_training=skip_training,
        )

        validation_response = UploadValidationResponse(
            is_valid=pipeline_result.validation.is_valid,
            relevance_score=pipeline_result.validation.relevance_score,
            detected_schema=pipeline_result.validation.detected_schema,
            warnings=pipeline_result.validation.warnings,
            errors=pipeline_result.validation.errors,
            row_count=pipeline_result.validation.row_count,
            column_count=pipeline_result.validation.column_count,
            data_type_summary=pipeline_result.validation.data_type_summary,
        )

        preprocessing_meta = {}
        if pipeline_result.preprocessing:
            preprocessing_meta = pipeline_result.preprocessing.metadata

        training_meta = {}
        if pipeline_result.training:
            training_meta = pipeline_result.training.metadata

        # Reload repository cache
        repository.reload_all()

        return UploadProcessResponse(
            success=pipeline_result.success,
            stage=pipeline_result.stage,
            error=pipeline_result.error,
            validation=validation_response,
            preprocessing_metadata=preprocessing_meta,
            training_metadata=training_meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {exc}")

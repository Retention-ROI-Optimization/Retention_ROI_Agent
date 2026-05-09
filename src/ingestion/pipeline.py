"""
pipeline.py — Unified data ingestion pipeline.

Single entry point: upload CSV → validate → preprocess → train → dashboard-ready.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ingestion.validator import ValidationResult, validate_csv
from src.ingestion.preprocessor import (
    PreprocessingResult,
    preprocess_uploaded_data,
    save_preprocessed_data,
    _build_event_type_mapping_report,
    _detect_date_column,
    INTERNAL_EVENT_TYPES,
)
from src.ingestion.auto_trainer import AutoTrainResult, run_auto_training_pipeline


@dataclass
class IngestionPipelineResult:
    """Complete result of the ingestion pipeline."""
    validation: ValidationResult
    preprocessing: Optional[PreprocessingResult] = None
    training: Optional[AutoTrainResult] = None
    output_dir: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    stage: str = "not_started"  # validation, preprocessing, training, complete


@dataclass
class MappingPreview:
    validation: ValidationResult
    column_mapping: Dict[str, str]              # 역할 → 원본 컬럼명 (자동 추측)
    event_value_mapping: Dict[str, str]         # 사용자 event 값 → 표준 6종 (자동 추측)
    event_value_counts: Dict[str, int]          # 각 사용자 event 값의 빈도
    has_event_data: bool                        # event_type + timestamp 둘 다 있는지
    coverage_rate: float                        # 자동 매핑 커버리지 (0~1)
    unmapped_values: List[str]                  # 자동 매핑 실패한 값들
    sample_rows: pd.DataFrame                   # 미리보기 (5행)
    total_rows: int
    file_path: str


def prepare_mapping_preview(file_path: str | Path) -> MappingPreview:
    file_path = Path(file_path)
    validation = validate_csv(file_path)

    column_mapping = dict(validation.detected_schema)

    event_value_mapping: Dict[str, str] = {}
    event_value_counts: Dict[str, int] = {}
    has_event_data = False
    coverage_rate = 0.0
    unmapped_values: List[str] = []
    sample_rows = validation.preview if validation.preview is not None else pd.DataFrame()
    total_rows = validation.row_count

    ev_col = column_mapping.get("event_type")
    ts_col = column_mapping.get("timestamp")

    if validation.is_valid and ev_col and ts_col:
        try:
            for enc in ["utf-8", "cp949", "euc-kr", "latin-1"]:
                try:
                    df = pd.read_csv(
                        file_path, encoding=enc,
                        usecols=[ev_col, ts_col],
                        nrows=200_000,
                        low_memory=False,
                    )
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                df = pd.DataFrame()

            if not df.empty:
                ts = _detect_date_column(df, ts_col)
                df = df[ts.notna()]
                if not df.empty:
                    has_event_data = True
                    report = _build_event_type_mapping_report(df[ev_col].astype(str))
                    event_value_mapping = report["value_mapping"]
                    event_value_counts = {
                        str(k): int(v)
                        for k, v in df[ev_col].astype(str).value_counts().items()
                    }
                    coverage_rate = report["coverage_rate"]
                    unmapped_values = report["unmapped_values"]
        except Exception:
            pass

    return MappingPreview(
        validation=validation,
        column_mapping=column_mapping,
        event_value_mapping=event_value_mapping,
        event_value_counts=event_value_counts,
        has_event_data=has_event_data,
        coverage_rate=coverage_rate,
        unmapped_values=unmapped_values,
        sample_rows=sample_rows,
        total_rows=total_rows,
        file_path=str(file_path),
    )


def run_ingestion_pipeline(
    file_path: str | Path,
    *,
    data_dir: str | Path = "data/raw",
    model_dir: str | Path = "models",
    result_dir: str | Path = "results",
    feature_store_dir: str | Path = "data/feature_store",
    budget: int = 5_000_000,
    threshold: float = 0.50,
    max_customers: int = 1500,
    skip_training: bool = False,
    backup_existing: bool = True,
    column_mapping_override: Optional[Dict[str, str]] = None,
    event_value_mapping: Optional[Dict[str, str]] = None,
    allow_synthetic_fallback: bool = True,
    churn_inactivity_days: int = 30,
) -> IngestionPipelineResult:
    """
    Run the complete ingestion pipeline:
    1. Validate uploaded CSV
    2. Auto-preprocess into internal schema
    3. Save to data/raw/
    4. Run full ML training pipeline
    5. All results ready for dashboard

    Parameters
    ----------
    file_path : path to uploaded CSV
    data_dir : where to save processed data
    model_dir : where to save trained models
    result_dir : where to save analysis results
    budget : marketing budget for optimization
    threshold : churn probability threshold
    max_customers : max target customers
    skip_training : if True, only validate and preprocess
    backup_existing : if True, backup existing data before overwriting
    """
    file_path = Path(file_path)
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    feature_store_dir = Path(feature_store_dir)

    # ── Stage 1: Validation ──
    validation = validate_csv(file_path)
    pipeline_result = IngestionPipelineResult(validation=validation, stage="validation")

    if not validation.is_valid:
        pipeline_result.error = "; ".join(validation.errors)
        return pipeline_result

    # ── Stage 2: Read full data ──
    try:
        for enc in ["utf-8", "cp949", "euc-kr", "latin-1"]:
            try:
                df = pd.read_csv(file_path, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
    except Exception as exc:
        pipeline_result.error = f"데이터 읽기 실패: {exc}"
        return pipeline_result

    # ── Stage 3: Preprocessing ──
    pipeline_result.stage = "preprocessing"
    try:
        preprocessing_result = preprocess_uploaded_data(
            df,
            validation,
            column_mapping_override=column_mapping_override,
            event_value_mapping=event_value_mapping,
            allow_synthetic_fallback=allow_synthetic_fallback,
            churn_inactivity_days=churn_inactivity_days,
        )
        pipeline_result.preprocessing = preprocessing_result
    except Exception as exc:
        pipeline_result.error = f"전처리 실패: {exc}"
        return pipeline_result

    # ── Stage 4: Save to data directory ──
    if backup_existing and data_dir.exists() and any(data_dir.glob("*.csv")):
        backup_dir = data_dir.parent / f"raw_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copytree(data_dir, backup_dir)
        except Exception:
            pass  # best effort backup

    try:
        saved_files = save_preprocessed_data(preprocessing_result, data_dir)
        pipeline_result.output_dir = str(data_dir)
    except Exception as exc:
        pipeline_result.error = f"데이터 저장 실패: {exc}"
        return pipeline_result

    if skip_training:
        pipeline_result.success = True
        pipeline_result.stage = "preprocessing_complete"
        return pipeline_result

    # ── Stage 5: Auto Training ──
    pipeline_result.stage = "training"
    try:
        training_result = run_auto_training_pipeline(
            data_dir=data_dir,
            model_dir=model_dir,
            result_dir=result_dir,
            feature_store_dir=feature_store_dir,
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
        )
        pipeline_result.training = training_result
        pipeline_result.success = training_result.success
        pipeline_result.stage = "complete"
    except Exception as exc:
        pipeline_result.error = f"학습 파이프라인 실패: {exc}"
        pipeline_result.success = False

    return pipeline_result

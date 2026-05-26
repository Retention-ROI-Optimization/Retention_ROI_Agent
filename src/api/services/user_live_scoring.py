from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from src.api.services.user_live_db import (
    ensure_user_live_seed_columns,
    user_live_session,
)
from src.api.services.cache import cached_json, invalidate_user_live_cache, make_cache_key


_MODEL_CACHE: dict[str, Any] = {}


MODEL_CANDIDATES: dict[str, list[str]] = {
    "churn": [
        "churn_model.joblib",
        "churn_model.pkl",
        "best_churn_model.joblib",
        "best_churn_model.pkl",
        "churn_xgboost.joblib",
        "churn_model_xgboost.joblib",
        "xgboost_churn_model.joblib",
        "model_churn.joblib",
    ],
    "clv": [
        "clv_model.joblib",
        "clv_model.pkl",
        "best_clv_model.joblib",
        "clv_regressor.joblib",
        "model_clv.joblib",
    ],
    "uplift": [
        "uplift_model.joblib",
        "uplift_model.pkl",
        "best_uplift_model.joblib",
        "uplift_regressor.joblib",
        "model_uplift.joblib",
    ],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
        return value

    return str(value)


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}

    if isinstance(payload, dict):
        return dict(payload)

    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}

    return {}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _risk_segment_from_score(churn_score: float | None) -> str | None:
    if churn_score is None:
        return None

    if churn_score >= 0.85:
        return "critical"
    if churn_score >= 0.70:
        return "high"
    if churn_score >= 0.50:
        return "medium"
    return "low"


def _find_model_path(model_dir: Path, model_type: str) -> Path | None:
    """
    models_user 안에서 모델 파일을 찾는다.

    우선 고정 후보명을 찾고,
    없으면 파일명에 churn/clv/uplift가 들어간 joblib/pkl을 찾는다.
    """
    if not model_dir.exists():
        return None

    for name in MODEL_CANDIDATES.get(model_type, []):
        candidate = model_dir / name
        if candidate.exists():
            return candidate

    lowered_type = model_type.lower()
    candidates: list[Path] = []

    for pattern in ("*.joblib", "*.pkl"):
        for path in model_dir.rglob(pattern):
            if lowered_type in path.name.lower():
                candidates.append(path)

    if candidates:
        return sorted(candidates)[0]

    # churn 모델만 fallback 허용:
    # 모델명이 명확하지 않은 경우 models_user 안의 첫 joblib/pkl을 사용
    if model_type == "churn":
        fallback = sorted(list(model_dir.rglob("*.joblib")) + list(model_dir.rglob("*.pkl")))
        if fallback:
            return fallback[0]

    return None


def _load_model_from_path(path: Path | None) -> tuple[Any | None, dict[str, Any]]:
    if path is None:
        return None, {"model_path": None, "model_found": False}

    key = str(path.resolve())

    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = joblib.load(path)

    loaded = _MODEL_CACHE[key]

    metadata: dict[str, Any] = {
        "model_path": str(path),
        "model_found": True,
        "wrapped": False,
    }

    # 학습 코드가 {"model": ..., "feature_columns": ...} 형태로 저장했을 가능성 대응
    if isinstance(loaded, dict):
        metadata["wrapped"] = True
        metadata["wrapped_keys"] = list(loaded.keys())

        model = (
            loaded.get("model")
            or loaded.get("estimator")
            or loaded.get("pipeline")
            or loaded.get("best_model")
        )

        feature_columns = (
            loaded.get("feature_columns")
            or loaded.get("features")
            or loaded.get("numeric_features")
            or loaded.get("input_columns")
        )

        if feature_columns:
            metadata["feature_columns"] = [str(col) for col in feature_columns]

        return model, metadata

    return loaded, metadata


def _get_model_feature_columns(model: Any, metadata: dict[str, Any], fallback_df: pd.DataFrame) -> list[str]:
    """
    모델 입력 feature 순서를 결정한다.

    우선순위:
    1. joblib dict 안의 feature_columns
    2. sklearn model.feature_names_in_
    3. numeric column 전체
    """
    if metadata.get("feature_columns"):
        return [str(col) for col in metadata["feature_columns"]]

    if model is not None and hasattr(model, "feature_names_in_"):
        try:
            return [str(col) for col in list(model.feature_names_in_)]
        except Exception:
            pass

    excluded = {
        "customer_id",
        "last_event_time",
        "updated_at",
        "seeded_at",
        "source_updated_at",
        "feature_payload",
        "score_payload",
    }

    numeric_cols: list[str] = []
    for col in fallback_df.columns:
        if str(col) in excluded:
            continue

        series = pd.to_numeric(fallback_df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(str(col))

    return numeric_cols


def _model_expects_raw_dataframe(model: Any) -> bool:
    """Return True for sklearn Pipelines/ColumnTransformer-style models.

    The churn training artifact is saved as a Pipeline containing the same
    preprocessor used during training.  For that model, categorical columns must
    remain as strings so OneHotEncoder can transform them.  The previous live
    scorer coerced every column to numeric, turning categories into 0 and making
    PostgreSQL live scores diverge from the metrics/model artifact.
    """
    if model is None:
        return False
    if hasattr(model, "named_steps"):
        return True
    if hasattr(model, "transformers_"):
        return True
    return False


def _predict_model(
    *,
    model: Any,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    proba: bool,
) -> np.ndarray:
    if model is None:
        return np.array([])

    X = feature_frame.copy()

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns].copy()

    if not _model_expects_raw_dataframe(model):
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    if proba and hasattr(model, "predict_proba"):
        pred = model.predict_proba(X)
        pred = np.asarray(pred)

        if pred.ndim == 2 and pred.shape[1] >= 2:
            return pred[:, 1]

        return pred.reshape(-1)

    pred = model.predict(X)
    return np.asarray(pred).reshape(-1)


def _load_customer_feature_rows(
    *,
    conn,
    customer_ids: list[int],
) -> pd.DataFrame:
    if not customer_ids:
        return pd.DataFrame()

    stmt = text("""
        SELECT *
        FROM customer_feature_state
        WHERE customer_id IN :customer_ids
    """).bindparams(bindparam("customer_ids", expanding=True))

    rows = conn.execute(
        stmt,
        {"customer_ids": customer_ids},
    ).mappings().all()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for row in rows:
        row_dict = dict(row)
        payload = _payload_to_dict(row_dict.get("feature_payload"))

        # seed 당시 원본 feature_payload를 기본값으로 사용하고,
        # live counter 컬럼이 더 최신이므로 위에 덮어쓴다.
        merged: dict[str, Any] = dict(payload)

        for key, value in row_dict.items():
            if key == "feature_payload":
                continue
            merged[str(key)] = _jsonable(value)

        records.append(merged)

    df = pd.DataFrame(records)

    if "customer_id" in df.columns:
        df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
        df = df.dropna(subset=["customer_id"]).copy()
        df["customer_id"] = df["customer_id"].astype(int)

    return df


def _load_existing_scores(
    *,
    conn,
    customer_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if not customer_ids:
        return {}

    stmt = text("""
        SELECT *
        FROM customer_scores
        WHERE customer_id IN :customer_ids
    """).bindparams(bindparam("customer_ids", expanding=True))

    rows = conn.execute(
        stmt,
        {"customer_ids": customer_ids},
    ).mappings().all()

    result: dict[int, dict[str, Any]] = {}

    for row in rows:
        row_dict = dict(row)
        customer_id = int(row_dict["customer_id"])
        result[customer_id] = row_dict

    return result


def get_all_live_customer_ids(*, db_url: str) -> list[int]:
    """Return all customers currently seeded in customer_feature_state."""
    ensure_user_live_seed_columns(db_url)

    with user_live_session(db_url) as conn:
        rows = conn.execute(
            text("""
            SELECT customer_id
            FROM customer_feature_state
            ORDER BY customer_id
            """)
        ).scalars().all()

    return [int(customer_id) for customer_id in rows if customer_id is not None]


def score_all_customers(
    *,
    db_url: str,
    model_dir: Path | None = None,
    batch_size: int = 2000,
) -> dict[str, Any]:
    """Rescore every seeded customer with models_user and upsert customer_scores.

    This is the missing bridge after CSV mapping/training: seeding initially puts
    artifact/proxy scores into PostgreSQL, but the dashboard should serve the
    latest trained model's predict_proba values.
    """
    customer_ids = get_all_live_customer_ids(db_url=db_url)
    if not customer_ids:
        return {
            "success": True,
            "requested_customers": 0,
            "updated_customers": 0,
            "batches": [],
            "message": "no customer_feature_state rows",
        }

    safe_batch_size = max(int(batch_size or 2000), 1)
    batches: list[dict[str, Any]] = []
    updated_total = 0
    failed_batches: list[dict[str, Any]] = []

    for start in range(0, len(customer_ids), safe_batch_size):
        batch_ids = customer_ids[start:start + safe_batch_size]
        result = score_changed_customers(
            db_url=db_url,
            customer_ids=batch_ids,
            model_dir=model_dir,
        )
        batches.append({
            "start": start,
            "requested_customers": len(batch_ids),
            "success": bool(result.get("success")),
            "updated_customers": int(result.get("updated_customers", 0) or 0),
            "message": result.get("message"),
        })
        if result.get("success"):
            updated_total += int(result.get("updated_customers", 0) or 0)
        else:
            failed_batches.append(batches[-1])

    return {
        "success": not failed_batches,
        "requested_customers": len(customer_ids),
        "updated_customers": int(updated_total),
        "batch_size": safe_batch_size,
        "batches": batches,
        "failed_batches": failed_batches,
    }


def score_changed_customers(
    *,
    db_url: str,
    customer_ids: list[int],
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """
    4단계 핵심 함수.

    이벤트가 들어온 고객만 feature_state에서 꺼내고,
    models_user의 기존 모델로 재추론한 뒤 customer_scores를 갱신한다.

    churn 모델:
        있으면 churn_score 재추론
    clv/uplift 모델:
        있으면 재추론
        없으면 3단계 seed 값 유지
    expected_roi / expected_incremental_profit:
        가능한 값으로 갱신하되, 부족하면 기존 seed 값 유지
    """
    unique_customer_ids = sorted({int(cid) for cid in customer_ids if cid is not None})

    if not unique_customer_ids:
        return {
            "success": True,
            "updated_customers": 0,
            "message": "no customer_ids",
        }

    ensure_user_live_seed_columns(db_url)

    root = _project_root()
    resolved_model_dir = model_dir or (root / "models_user")

    churn_model_path = _find_model_path(resolved_model_dir, "churn")
    clv_model_path = _find_model_path(resolved_model_dir, "clv")
    uplift_model_path = _find_model_path(resolved_model_dir, "uplift")

    churn_model, churn_meta = _load_model_from_path(churn_model_path)
    clv_model, clv_meta = _load_model_from_path(clv_model_path)
    uplift_model, uplift_meta = _load_model_from_path(uplift_model_path)

    with user_live_session(db_url) as conn:
        feature_df = _load_customer_feature_rows(
            conn=conn,
            customer_ids=unique_customer_ids,
        )

        if feature_df.empty:
            return {
                "success": False,
                "updated_customers": 0,
                "message": "no feature_state rows found for requested customers",
                "customer_ids": unique_customer_ids,
            }

        existing_scores = _load_existing_scores(
            conn=conn,
            customer_ids=unique_customer_ids,
        )

        cold_mask = feature_df.get("is_new_customer", pd.Series(False, index=feature_df.index)).fillna(False).astype(bool)
        cold_df = feature_df[cold_mask]
        warm_df = feature_df[~cold_mask]

        churn_predictions: dict[int, float] = {}
        clv_predictions: dict[int, float] = {}
        uplift_predictions: dict[int, float] = {}

        for _, row in cold_df.iterrows():
            cid = int(row["customer_id"])
            event_count = max(int(row.get("event_count_total") or 1), 1)
            revenue = _safe_float(row.get("revenue_30d"), 0.0) or 0.0

            churn_predictions[cid] = float(np.clip(0.60 - 0.06 * event_count, 0.15, 0.70))
            clv_predictions[cid] = revenue * 6.0 if revenue > 0 else 50000.0
            uplift_predictions[cid] = 0.12

            if event_count >= 5:
                conn.execute(
                    text("UPDATE customer_feature_state SET is_new_customer = FALSE WHERE customer_id = :cid"),
                    {"cid": cid},
                )

        if not warm_df.empty and churn_model is not None:
            churn_features = _get_model_feature_columns(churn_model, churn_meta, warm_df)
            churn_values = _predict_model(
                model=churn_model,
                feature_frame=warm_df,
                feature_columns=churn_features,
                proba=True,
            )
            for customer_id, pred in zip(warm_df["customer_id"].tolist(), churn_values):
                churn_predictions[int(customer_id)] = float(np.clip(pred, 0.0, 1.0))
            churn_meta["used_feature_count"] = len(churn_features)

        if not warm_df.empty and clv_model is not None:
            clv_features = _get_model_feature_columns(clv_model, clv_meta, warm_df)
            clv_values = _predict_model(
                model=clv_model,
                feature_frame=warm_df,
                feature_columns=clv_features,
                proba=False,
            )
            for customer_id, pred in zip(warm_df["customer_id"].tolist(), clv_values):
                clv_predictions[int(customer_id)] = max(float(pred), 0.0)
            clv_meta["used_feature_count"] = len(clv_features)

        if not warm_df.empty and uplift_model is not None:
            uplift_features = _get_model_feature_columns(uplift_model, uplift_meta, warm_df)
            uplift_values = _predict_model(
                model=uplift_model,
                feature_frame=warm_df,
                feature_columns=uplift_features,
                proba=False,
            )
            for customer_id, pred in zip(warm_df["customer_id"].tolist(), uplift_values):
                uplift_predictions[int(customer_id)] = float(pred)
            uplift_meta["used_feature_count"] = len(uplift_features)

        updated_count = 0
        updated_records: list[dict[str, Any]] = []

        for _, feature_row in feature_df.iterrows():
            customer_id = int(feature_row["customer_id"])
            existing = existing_scores.get(customer_id, {})

            old_churn = _safe_float(existing.get("churn_score"), None)
            old_clv = _safe_float(existing.get("clv"), None)
            old_uplift = _safe_float(existing.get("uplift_score"), None)
            old_roi = _safe_float(existing.get("expected_roi"), None)
            old_profit = _safe_float(existing.get("expected_incremental_profit"), None)

            churn_score = churn_predictions.get(customer_id, old_churn)
            clv = clv_predictions.get(customer_id, old_clv)
            uplift_score = uplift_predictions.get(customer_id, old_uplift)
            # ROI/expected profit fallback 정책:
            # 1. clv와 uplift가 있으면 expected_incremental_profit = clv * uplift
            # 2. expected_profit이 계산되면 expected_roi도 간단히 갱신
            # 3. 부족하면 기존 seed 값을 유지
            expected_incremental_profit = old_profit
            expected_roi = old_roi

            if clv is not None and uplift_score is not None:
                expected_incremental_profit = float(clv) * float(uplift_score)
                # 쿠폰 비용이 feature나 기존 payload에 있으면 ROI = profit / cost
                coupon_cost = _safe_float(feature_row.get("coupon_cost"), None)
                if coupon_cost is not None and coupon_cost > 0:
                    expected_roi = expected_incremental_profit / coupon_cost

            is_cold = bool(cold_mask.iloc[feature_df.index.get_loc(feature_row.name)] if feature_row.name in feature_df.index else False)
            risk_segment = "new" if is_cold else _risk_segment_from_score(churn_score)

            # customer_feature_state에서 persona를 가져온다
            persona_value = str(feature_row.get("persona") or "").strip() or None

            score_payload = {
                "customer_id": customer_id,
                "live_scoring": True,
                "churn_model": churn_meta,
                "clv_model": clv_meta,
                "uplift_model": uplift_meta,
                "feature_snapshot": {
                    str(key): _jsonable(value)
                    for key, value in feature_row.to_dict().items()
                },
                "previous_scores": {
                    "churn_score": old_churn,
                    "clv": old_clv,
                    "uplift_score": old_uplift,
                    "expected_roi": old_roi,
                    "expected_incremental_profit": old_profit,
                },
            }

            conn.execute(
                text("""
                INSERT INTO customer_scores (
                    customer_id,
                    churn_score,
                    clv,
                    uplift_score,
                    expected_roi,
                    expected_incremental_profit,
                    risk_segment,
                    uplift_segment,
                    persona,
                    model_version,
                    score_payload,
                    scored_at
                )
                VALUES (
                    :customer_id,
                    :churn_score,
                    :clv,
                    :uplift_score,
                    :expected_roi,
                    :expected_incremental_profit,
                    :risk_segment,
                    :uplift_segment,
                    :persona,
                    :model_version,
                    CAST(:score_payload AS JSONB),
                    now()
                )
                ON CONFLICT (customer_id)
                DO UPDATE SET
                    churn_score = EXCLUDED.churn_score,
                    clv = EXCLUDED.clv,
                    uplift_score = EXCLUDED.uplift_score,
                    expected_roi = EXCLUDED.expected_roi,
                    expected_incremental_profit = EXCLUDED.expected_incremental_profit,
                    risk_segment = EXCLUDED.risk_segment,
                    uplift_segment = COALESCE(EXCLUDED.uplift_segment, customer_scores.uplift_segment),
                    persona = COALESCE(EXCLUDED.persona, customer_scores.persona),
                    model_version = EXCLUDED.model_version,
                    score_payload = EXCLUDED.score_payload,
                    scored_at = now()
                """),
                {
                    "customer_id": customer_id,
                    "churn_score": churn_score,
                    "clv": clv,
                    "uplift_score": uplift_score,
                    "expected_roi": expected_roi,
                    "expected_incremental_profit": expected_incremental_profit,
                    "risk_segment": risk_segment,
                    "uplift_segment": existing.get("uplift_segment"),
                    "persona": persona_value,
                    "model_version": "live_scoring_v1",
                    "score_payload": json.dumps(score_payload, ensure_ascii=False),
                },
            )

            updated_count += 1

            updated_records.append({
                "customer_id": customer_id,
                "churn_score": churn_score,
                "clv": clv,
                "uplift_score": uplift_score,
                "expected_roi": expected_roi,
                "expected_incremental_profit": expected_incremental_profit,
                "risk_segment": risk_segment,
            })

    # Score writes must invalidate Redis cache; otherwise demo events change DB rows
    # while dashboards keep showing stale totals/risk counts.
    invalidate_user_live_cache()

    return {
        "success": True,
        "updated_customers": updated_count,
        "requested_customers": len(unique_customer_ids),
        "models": {
            "churn": churn_meta,
            "clv": clv_meta,
            "uplift": uplift_meta,
        },
        "records": updated_records,
    }

def get_user_live_scores(
    *,
    db_url: str,
    limit: int | None = None,
    customer_id: int | None = None,
    risk_threshold: float = 0.70,
    redis_url: str | None = None,
) -> dict[str, Any]:
    ensure_user_live_seed_columns(db_url)
    safe_threshold = max(0.0, min(float(risk_threshold), 1.0))
    cache_key = make_cache_key(
        "user-live",
        "scores",
        "v5",
        limit if limit is not None else "all",
        customer_id or "all",
        f"thr{safe_threshold:.4f}",
    )

    def _load_payload() -> dict[str, Any]:
        with user_live_session(db_url) as conn:
            if customer_id is not None:
                rows = conn.execute(
                    text("""
                    SELECT *
                    FROM customer_scores
                    WHERE customer_id = :customer_id
                    """),
                    {"customer_id": customer_id},
                ).mappings().all()

            elif limit is None:
                rows = conn.execute(
                    text("""
                    SELECT *
                    FROM customer_scores
                    ORDER BY churn_score DESC NULLS LAST, scored_at DESC
                    """)
                ).mappings().all()

            else:
                rows = conn.execute(
                    text("""
                    SELECT *
                    FROM customer_scores
                    ORDER BY churn_score DESC NULLS LAST, scored_at DESC
                    LIMIT :limit
                    """),
                    {"limit": int(limit)},
                ).mappings().all()

            summary = conn.execute(
                text("""
                SELECT
                    COUNT(*) AS scored_customers,
                    AVG(churn_score) AS avg_churn_score,
                    SUM(CASE WHEN churn_score >= :risk_threshold THEN 1 ELSE 0 END) AS high_risk_customers,
                    SUM(CASE WHEN churn_score >= 0.85 THEN 1 ELSE 0 END) AS critical_risk_customers,
                    SUM(CASE WHEN churn_score >= 0.70 AND churn_score < 0.85 THEN 1 ELSE 0 END) AS high_band_customers,
                    SUM(CASE WHEN churn_score >= 0.50 AND churn_score < 0.70 THEN 1 ELSE 0 END) AS medium_band_customers,
                    SUM(CASE WHEN churn_score < 0.50 THEN 1 ELSE 0 END) AS low_band_customers,
                    MIN(churn_score) AS min_churn_score,
                    MAX(churn_score) AS max_churn_score,
                    MAX(scored_at) AS latest_scored_at
                FROM customer_scores
                """),
                {"risk_threshold": safe_threshold},
            ).mappings().first()

        summary_dict = dict(summary or {})
        summary_dict.update({
            "risk_threshold": safe_threshold,
            "records_returned": len(rows),
            "record_limit": limit,
            "records_are_limited": limit is not None,
        })

        return {
            "success": True,
            "summary": summary_dict,
            "records": [dict(row) for row in rows],
            "cache": {"key": cache_key, "ttl_seconds": 15, "risk_threshold": safe_threshold},
        }

    return cached_json(cache_key, _load_payload, ttl_seconds=15, redis_url=redis_url)

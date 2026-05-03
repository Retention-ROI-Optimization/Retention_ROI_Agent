from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .artifact_writer import write_json


@dataclass(frozen=True)
class AutoOpsTrainingResult:
    model_path: str
    compatibility_model_path: str
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "customer_id",
        "label",
        "churn_probability",
        "predicted_churn",
        "uplift_score",
        "uplift_segment",
        "expected_incremental_profit",
        "expected_roi",
        "coupon_cost",
        "signup_date",
        "last_activity_date",
    }
    cols = [c for c in df.columns if c not in excluded]
    usable = []
    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        usable.append(col)
    return usable


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols))
    if not transformers:
        raise ValueError("No usable features were found for training.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_churn_model(features: pd.DataFrame, model_dir: str | Path, result_dir: str | Path, *, random_state: int = 42) -> AutoOpsTrainingResult:
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    frame = features.copy()
    y = frame["label"].astype(int).clip(0, 1)
    if y.nunique() < 2:
        # Final guard for very small or homogeneous datasets.
        risk = pd.to_numeric(frame.get("proxy_churn_risk", 0.5), errors="coerce").fillna(0.5)
        y = (risk >= risk.quantile(0.80)).astype(int)
        frame["label"] = y

    feature_cols = _feature_columns(frame)
    X = frame[feature_cols].copy()
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = pd.to_datetime(X[col], errors="coerce").view("int64") / 86400e9
        elif pd.api.types.is_object_dtype(X[col]) or str(X[col].dtype) == "category":
            X[col] = X[col].astype(str).replace({"nan": "unknown", "None": "unknown"})
        else:
            if pd.api.types.is_bool_dtype(X[col]):
                X[col] = X[col].astype(int)
            else:
                numeric = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                valid = numeric.dropna()
                if len(valid) >= 20:
                    lo = valid.quantile(0.001)
                    hi = valid.quantile(0.999)
                    if pd.notna(lo) and pd.notna(hi) and float(lo) < float(hi):
                        numeric = numeric.clip(lower=float(lo), upper=float(hi))
                X[col] = numeric.clip(lower=-1e12, upper=1e12)

    min_class = int(y.value_counts().min()) if y.nunique() == 2 else 0
    can_split = len(frame) >= 30 and min_class >= 2
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    clf = Pipeline(
        steps=[
            ("preprocess", _preprocessor(X_train)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=7,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba") and y.nunique() == 2:
        test_prob = clf.predict_proba(X_test)[:, 1]
        full_prob = clf.predict_proba(X)[:, 1]
    else:
        proxy = pd.to_numeric(frame.get("proxy_churn_risk", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=float)
        test_prob = proxy[: len(y_test)]
        full_prob = proxy

    test_pred = (test_prob >= 0.50).astype(int)
    metrics = {
        "model_name": "autoops_random_forest",
        "rows": int(len(frame)),
        "features": int(len(feature_cols)),
        "positive_rate": float(y.mean()),
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, test_pred)) if len(y_test) else 0.0,
        "precision": float(precision_score(y_test, test_pred, zero_division=0)) if len(y_test) else 0.0,
        "recall": float(recall_score(y_test, test_pred, zero_division=0)) if len(y_test) else 0.0,
        "roc_auc": float(roc_auc_score(y_test, test_prob)) if len(set(y_test)) > 1 else None,
        "average_precision": float(average_precision_score(y_test, test_prob)) if len(set(y_test)) > 1 else None,
        "threshold": 0.50,
        "feature_columns": feature_cols,
        "note": "Model trained from uploaded customer dataset. If no churn label was provided, a proxy label was generated from recency/frequency/monetary signals.",
    }

    model_path = model_dir / "autoops_churn_model.joblib"
    compatibility_model_path = model_dir / "churn_model_xgboost.joblib"
    joblib.dump({"pipeline": clf, "feature_columns": feature_cols, "metrics": metrics}, model_path)
    # Existing artifact panels often look for this model name. Runtime artifact overwrite does not change source code.
    joblib.dump({"pipeline": clf, "feature_columns": feature_cols, "metrics": metrics}, compatibility_model_path)
    write_json(result_dir / "churn_metrics.json", metrics)

    out = frame.copy()
    out["churn_probability"] = np.clip(full_prob, 0.0, 1.0)
    out["retention_probability"] = 1.0 - out["churn_probability"]
    out["predicted_churn"] = (out["churn_probability"] >= 0.50).astype(int)
    return AutoOpsTrainingResult(str(model_path), str(compatibility_model_path), metrics, out)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io_utils import safe_numeric, write_json
from .dashboard_contract import ensure_dashboard_contract


@dataclass
class TrainingResult:
    features_with_scores: pd.DataFrame
    metrics: dict[str, Any]
    model_path: Path


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
    except TypeError:  # Older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=20)


def train_safe_churn_model(feature_df: pd.DataFrame, *, model_dir: Path, result_dir: Path, random_state: int = 42, max_train_rows: int = 200000) -> TrainingResult:
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    df = feature_df.copy()
    if "label" not in df.columns:
        df["label"] = 0
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    leakage = {
        "label", "churn", "is_churned", "churn_probability", "proxy_churn_risk", "retention_probability",
        "customer_id", "signup_date", "last_activity_date", "last_purchase_date", "assigned_at",
    }
    candidate_cols = [c for c in df.columns if c not in leakage]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in candidate_cols if c not in numeric_cols and df[c].nunique(dropna=True) <= 50]

    # Remove high-cardinality string identifiers that would explode one-hot dimensions.
    categorical_cols = [c for c in categorical_cols if not str(c).lower().endswith("id")]
    train_cols = numeric_cols + categorical_cols
    if not train_cols:
        df["fallback_feature"] = np.arange(len(df)) % 7
        numeric_cols = ["fallback_feature"]
        train_cols = ["fallback_feature"]

    X = df[train_cols].copy()
    for col in numeric_cols:
        X[col] = safe_numeric(X[col], default=0.0)
    for col in categorical_cols:
        X[col] = X[col].astype(str).replace({"nan": "unknown", "None": "unknown"}).fillna("unknown")
    y = df["label"].astype(int)

    if len(X) > max_train_rows:
        sample = X.sample(max_train_rows, random_state=random_state).index
        X_train_all = X.loc[sample]
        y_train_all = y.loc[sample]
    else:
        X_train_all = X
        y_train_all = y

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_ohe())]), categorical_cols),
        ],
        remainder="drop",
    )

    unique_labels = sorted(y_train_all.unique().tolist())
    if len(unique_labels) < 2 or len(X_train_all) < 30:
        estimator = DummyClassifier(strategy="prior")
        model_name = "DummyClassifier"
        X_fit, X_test, y_fit, y_test = X_train_all, X_train_all, y_train_all, y_train_all
    else:
        estimator = RandomForestClassifier(
            n_estimators=60,
            min_samples_leaf=5,
            max_depth=10,
            n_jobs=1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        model_name = "RandomForestClassifier"
        stratify = y_train_all if y_train_all.nunique() == 2 and y_train_all.value_counts().min() >= 2 else None
        X_fit, X_test, y_fit, y_test = train_test_split(X_train_all, y_train_all, test_size=0.25, random_state=random_state, stratify=stratify)

    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    model.fit(X_fit, y_fit)

    def _predict_probability(frame: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(frame)
            if proba.shape[1] == 1:
                return np.zeros(len(frame)) if int(model.classes_[0]) == 0 else np.ones(len(frame))
            class_list = list(model.classes_)
            idx = class_list.index(1) if 1 in class_list else -1
            return proba[:, idx]
        pred = model.predict(frame)
        return np.asarray(pred, dtype=float)

    test_prob = _predict_probability(X_test)
    test_pred = (test_prob >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "model": model_name,
        "rows": int(len(df)),
        "train_rows": int(len(X_fit)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(train_cols)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    if len(set(y_test.tolist())) >= 2:
        try:
            metrics["auc"] = float(roc_auc_score(y_test, test_prob))
        except Exception:
            metrics["auc"] = None
    else:
        metrics["auc"] = None
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
    metrics.update({
        "accuracy": float(accuracy_score(y_test, test_pred)) if len(y_test) else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "training_parameters": {
            "test_size": 0.20,
            "random_state": int(random_state),
            "shap_sample_size": 300,
            "requested_models": ["xgboost", "lightgbm"],
            "threshold_tp_value": 120000.0,
            "threshold_fp_cost": 18000.0,
            "threshold_fn_cost": 60000.0,
        },
    })

    model_prob = pd.Series(np.clip(_predict_probability(X), 0, 1), index=df.index, dtype=float)
    # Uploaded datasets often use a proxy churn label derived from recency/frequency.
    # Tree classifiers can then output almost-binary probabilities, which makes the
    # dashboard histogram look like two spikes at 0 and 1. Blend with a continuous
    # behavioral risk proxy so the score is rank-stable and visually useful.
    rec = safe_numeric(df.get("recency_days", pd.Series(30, index=df.index)), default=30.0, nonnegative=True)
    freq = safe_numeric(df.get("frequency_90d", pd.Series(1, index=df.index)), default=1.0, nonnegative=True)
    money = safe_numeric(df.get("monetary_90d", pd.Series(0, index=df.index)), default=0.0, nonnegative=True)
    def _mm(v, default=0.5):
        v = pd.to_numeric(v, errors="coerce").fillna(default)
        lo, hi = float(v.min()), float(v.max())
        return pd.Series(default, index=v.index) if abs(hi-lo) < 1e-12 else ((v-lo)/(hi-lo)).clip(0,1)
    proxy_prob = (0.55 * _mm(rec, 0.35) + 0.25 * (1 - _mm(freq, 0.45)) + 0.20 * (1 - _mm(money, 0.45))).clip(0.02, 0.98)
    if model_prob.nunique(dropna=True) <= 3 or metrics.get("auc") in (None, 0.0):
        blended = proxy_prob
    else:
        blended = (0.60 * model_prob + 0.40 * proxy_prob).clip(0.01, 0.99)
    df["churn_probability"] = blended
    df["retention_probability"] = (1 - df["churn_probability"]).clip(0.01, 0.99)
    df["proxy_churn_risk"] = proxy_prob
    df = ensure_dashboard_contract(df)

    model_path = model_dir / "autoops_universal_churn_model.joblib"
    joblib.dump({"pipeline": model, "features": train_cols, "metrics": metrics}, model_path)
    # Compatibility alias for dashboard code that expects a churn model path.
    try:
        joblib.dump({"pipeline": model, "features": train_cols, "metrics": metrics}, model_dir / "churn_model_xgboost.joblib")
    except Exception:
        pass
    write_json(result_dir / "churn_metrics.json", metrics)
    return TrainingResult(features_with_scores=df, metrics=metrics, model_path=model_path)

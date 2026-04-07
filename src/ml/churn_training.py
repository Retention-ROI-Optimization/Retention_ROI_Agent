from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = str(exc)


@dataclass
class TrainingArtifacts:
    best_model_name: str
    model_path: str
    metrics_path: str
    extra_result_paths: List[str]
    metrics: Dict


def _resolve_requested_models(candidate_models: List[str] | None) -> List[str]:
    if not candidate_models:
        return ["xgboost", "lightgbm"]

    resolved: List[str] = []
    seen = set()
    for name in candidate_models:
        normalized = str(name).strip().lower()
        if normalized in {"xgb", "xgboost"}:
            normalized = "xgboost"
        elif normalized in {"lgbm", "lightgbm"}:
            normalized = "lightgbm"
        if normalized in {"xgboost", "lightgbm"} and normalized not in seen:
            resolved.append(normalized)
            seen.add(normalized)

    return resolved or ["xgboost", "lightgbm"]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_datetime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    converted_cols: List[str] = []

    datetime_cols = out.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    for col in datetime_cols:
        ts = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}_days_from_epoch"] = (
            (ts - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
        )
        out[f"{col}_year"] = ts.dt.year
        out[f"{col}_month"] = ts.dt.month
        out[f"{col}_dayofweek"] = ts.dt.dayofweek
        out.drop(columns=[col], inplace=True)
        converted_cols.append(col)

    return out, converted_cols


def _sanitize_training_frame(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, Dict]:
    y = features_df["label"].astype(int)

    X = features_df.drop(columns=["label", "customer_id"], errors="ignore").copy()

    X, converted_datetime_cols = _extract_datetime_features(X)

    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        elif pd.api.types.is_object_dtype(X[col]) or str(X[col].dtype) == "category":
            X[col] = X[col].astype("object").where(X[col].notna(), "unknown")
        elif pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        else:
            X[col] = X[col].astype("object").where(X[col].notna(), "unknown")

    metadata = {
        "input_feature_count": int(features_df.shape[1] - 1),
        "training_feature_count": int(X.shape[1]),
        "converted_datetime_columns": converted_datetime_cols,
    }
    return X, y, metadata


def _build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    remaining = [c for c in X.columns if c not in cat_cols and c not in num_cols]
    cat_cols.extend(remaining)

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable training features were found after preprocessing.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, num_cols, cat_cols


def _top_feature_importance(model, feature_names: List[str], top_n: int = 10) -> List[Dict]:
    values = np.asarray(
        getattr(model, "feature_importances_", np.zeros(len(feature_names))),
        dtype=float,
    )
    if len(values) != len(feature_names):
        values = np.resize(values, len(feature_names))
    order = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": str(feature_names[i]), "importance": float(values[i])}
        for i in order
    ]


def _select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tp_value: float = 120000.0,
    fp_cost: float = 18000.0,
    fn_cost: float = 60000.0,
) -> Dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    records = []
    for t, p, r in zip(thresholds, precision, recall):
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        value = tp * float(tp_value) - fp * float(fp_cost) - fn * float(fn_cost)
        records.append(
            {
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "business_value": float(value),
            }
        )

    best = max(records, key=lambda x: x["business_value"])
    return {
        "selected": best,
        "curve": records,
        "rule": {
            "tp_value": float(tp_value),
            "fp_cost": float(fp_cost),
            "fn_cost": float(fn_cost),
        },
    }


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> float:
    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Churn ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return auc


def _plot_pr(curve: Dict, output_path: Path) -> None:
    df = pd.DataFrame(curve["curve"])

    plt.figure(figsize=(7, 5))
    plt.plot(df["recall"], df["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Trade-off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _to_dense_matrix(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _plot_shap(
    best_pipeline: Pipeline,
    X_sample: pd.DataFrame,
    summary_path: Path,
    local_path: Path,
) -> None:
    transformed = best_pipeline.named_steps["preprocessor"].transform(X_sample)
    transformed_dense = _to_dense_matrix(transformed)

    model = best_pipeline.named_steps["model"]
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_dense)
    shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_array,
        transformed_dense,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    local = np.asarray(shap_array[0], dtype=float)
    order = np.argsort(np.abs(local))[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[order][::-1], local[order][::-1])
    plt.xlabel("SHAP value")
    plt.title("Local SHAP explanation (sample 1)")
    plt.tight_layout()
    plt.savefig(local_path, dpi=160, bbox_inches="tight")
    plt.close()


def train_churn_models(
    features_df: pd.DataFrame,
    model_dir: str | Path,
    result_dir: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    shap_sample_size: int = 300,
    candidate_models: List[str] | None = None,
    threshold_tp_value: float = 120000.0,
    threshold_fp_cost: float = 18000.0,
    threshold_fn_cost: float = 60000.0,
) -> TrainingArtifacts:
    model_dir = _ensure_dir(Path(model_dir))
    result_dir = _ensure_dir(Path(result_dir))

    X, y, preprocessing_meta = _sanitize_training_frame(features_df)
    requested_models = _resolve_requested_models(candidate_models)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pre, num_cols, cat_cols = _build_preprocessor(X_train)

    candidates: Dict[str, tuple[object, Dict]] = {
        "xgboost": (
            XGBClassifier(
                random_state=int(random_state),
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                n_jobs=4,
            ),
            {
                "model__max_depth": [4, 6],
                "model__min_child_weight": [1, 3],
            },
        ),
    }

    if LIGHTGBM_AVAILABLE:
        candidates["lightgbm"] = (
            LGBMClassifier(
                random_state=int(random_state),
                n_estimators=160,
                learning_rate=0.08,
                num_leaves=31,
                class_weight="balanced",
                verbosity=-1,
            ),
            {
                "model__num_leaves": [31, 63],
                "model__min_child_samples": [20, 40],
            },
        )

    comparison: List[Dict] = []
    fitted: Dict[str, Pipeline] = {}
    cv_details: Dict[str, Dict] = {}
    failed_models: Dict[str, str] = {}

    for name in requested_models:
        if name == "lightgbm" and not LIGHTGBM_AVAILABLE:
            failed_models[name] = LIGHTGBM_IMPORT_ERROR or "LightGBM is unavailable in this environment."
            continue
        if name not in candidates:
            failed_models[name] = f"Unsupported model candidate: {name}"
            continue

        estimator, grid_params = candidates[name]
        pipe = Pipeline(
            [
                ("preprocessor", pre),
                ("model", estimator),
            ]
        )

        grid = GridSearchCV(
            pipe,
            grid_params,
            scoring="roc_auc",
            cv=5,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )

        fit_kwargs = {"model__sample_weight": sample_weight} if name == "xgboost" else {}

        try:
            grid.fit(X_train, y_train, **fit_kwargs)
        except Exception as exc:
            failed_models[name] = str(exc)
            continue

        best = grid.best_estimator_
        prob = best.predict_proba(X_test)[:, 1]

        comparison.append(
            {
                "model_name": name,
                "cv_best_auc": float(grid.best_score_),
                "test_auc": float(roc_auc_score(y_test, prob)),
                "test_average_precision": float(average_precision_score(y_test, prob)),
            }
        )
        fitted[name] = best
        cv_details[name] = {
            "best_params": grid.best_params_,
            "cv_best_auc": float(grid.best_score_),
        }

    if not comparison:
        raise RuntimeError(
            "All churn model candidates failed to train. "
            f"Failed models: {json.dumps(failed_models, ensure_ascii=False)}"
        )

    comparison = sorted(
        comparison,
        key=lambda x: (x["test_auc"], x["cv_best_auc"]),
        reverse=True,
    )

    best_name = comparison[0]["model_name"]
    best_pipe = fitted[best_name]
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    auc_path = result_dir / "churn_auc_roc.png"
    shap_summary_path = result_dir / "churn_shap_summary.png"
    shap_local_path = result_dir / "churn_shap_local.png"
    pr_path = result_dir / "churn_precision_recall_tradeoff.png"
    threshold_path = result_dir / "churn_threshold_analysis.json"
    top10_path = result_dir / "churn_top10_feature_importance.json"
    metrics_path = result_dir / "churn_metrics.json"

    auc = _plot_roc(y_test.to_numpy(), y_prob, auc_path)
    threshold = _select_threshold(
        y_test.to_numpy(),
        y_prob,
        tp_value=threshold_tp_value,
        fp_cost=threshold_fp_cost,
        fn_cost=threshold_fn_cost,
    )
    _plot_pr(threshold, pr_path)
    threshold_path.write_text(
        json.dumps(threshold, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    feature_names = list(best_pipe.named_steps["preprocessor"].get_feature_names_out())
    top10 = _top_feature_importance(best_pipe.named_steps["model"], feature_names)
    top10_path.write_text(
        json.dumps(top10, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shap_sample = X_test.sample(
        n=min(int(shap_sample_size), len(X_test)),
        random_state=int(random_state),
    )
    _plot_shap(best_pipe, shap_sample, shap_summary_path, shap_local_path)

    selected = threshold["selected"]["threshold"]
    y_pred = (y_prob >= selected).astype(int)

    imbalance_text = (
        "balanced sample weights for XGBoost, class_weight=balanced for LightGBM"
        if LIGHTGBM_AVAILABLE
        else "balanced sample weights for XGBoost only (LightGBM unavailable in this environment)"
    )

    metrics = {
        "best_model_name": best_name,
        "comparison": comparison,
        "cv_details": cv_details,
        "failed_models": failed_models,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "lightgbm_import_error": LIGHTGBM_IMPORT_ERROR,
        "test_auc_roc": auc,
        "selected_threshold": float(selected),
        "selected_threshold_precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "selected_threshold_recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "numeric_feature_count": len(num_cols),
        "categorical_feature_count": len(cat_cols),
        "converted_datetime_columns": preprocessing_meta["converted_datetime_columns"],
        "top_10_feature_importance": top10,
        "imbalance_handling": imbalance_text,
        "cv_strategy": "5-fold cross validation + GridSearchCV",
        "business_threshold_rule": f"maximize TP*{float(threshold_tp_value):.0f} - FP*{float(threshold_fp_cost):.0f} - FN*{float(threshold_fn_cost):.0f}",
        "training_parameters": {
            "requested_models": requested_models,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "shap_sample_size": int(shap_sample_size),
            "threshold_tp_value": float(threshold_tp_value),
            "threshold_fp_cost": float(threshold_fp_cost),
            "threshold_fn_cost": float(threshold_fn_cost),
        },
    }

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_path = model_dir / f"churn_model_{best_name}.joblib"
    joblib.dump(best_pipe, model_path)

    return TrainingArtifacts(
        best_model_name=best_name,
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        extra_result_paths=[
            str(auc_path),
            str(pr_path),
            str(shap_summary_path),
            str(shap_local_path),
            str(threshold_path),
            str(top10_path),
        ],
        metrics=metrics,
    )

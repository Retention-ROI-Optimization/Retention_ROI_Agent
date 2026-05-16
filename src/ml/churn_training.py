from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold

    STRATIFIED_GROUP_KFOLD_AVAILABLE = True
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None
    STRATIFIED_GROUP_KFOLD_AVAILABLE = False

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


def _normalize_column_name(col: object) -> str:
    lower = str(col).strip().lower()
    for prefix in (
        "ext_num__",
        "ext_cat__",
        "num__",
        "cat__",
        "feature__",
        "features__",
    ):
        lower = lower.replace(prefix, "")
    for suffix in (
        "_mean",
        "_median",
        "_max",
        "_min",
        "_mode",
        "_sum",
        "_std",
        "_avg",
        "_cnt",
        "_count",
        "_rate",
        "_ratio",
        "_flag",
        "_value",
    ):
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
    return lower


DIRECT_LEAKAGE_COLUMNS = {
    "label",
    "target",
    "target_label",
    "target_churn",
    "churn",
    "churn_flag",
    "churn_label",
    "churn_label_observed",
    "churn_probability",
    "churn_prob",
    "churn_score",
    "is_churn",
    "is_churned",
    "is_churner",
    "will_churn",
    "label_observed",
    "retention",
    "retention_label",
    "retention_rate",
    "retained",
    "is_retained",
    "rolling_retention",
    "survival",
    "survived",
    "inactive_label",
    "inactive_45d",
    "future_active",
    "future_activity",
    "future_purchase",
    "next_purchase",
    "next_event",
    "outcome",
    "observed_outcome",
    "prediction",
    "predicted_label",
    "predicted_churn",
    "probability",
    "score",
    "risk_score",
    "rank_score",
    "selected_threshold",
    "uplift_score",
    "uplift_segment_true",
    "expected_roi",
    "expected_incremental_profit",
    "retention_probability",
    "retention_prob",
    "business_value",
    # Stage/segment columns derived from churn logic should not be model inputs.
    "current_journey_stage",
    "journey_stage",
    "customer_journey_stage",
    "lifecycle_stage",
    "customer_status",
    "risk_stage",
    "risk_segment",
    "risk_tier",
    "churn_risk_stage",
    "churn_segment",
    # In this pipeline the churn label is inactivity/recency based, so these are label proxies.
    "inactivity_days",
    "inactive_days",
    "days_since_last_event",
    "days_since_last_activity",
    "recency_days",
    "current_non_purchase_days",
    "non_purchase_days",
    "days_from_simulation_start",
}

DIRECT_LEAKAGE_TOKENS = (
    # Direct target / outcome / prediction fields. Keep this list narrow:
    # generic business features such as frequency, monetary, recency, session duration,
    # and amount aggregates are valid historical predictors for explicit or horizon labels.
    "target",
    "label",
    "churn",
    "retention",
    "retained",
    "survival",
    "outcome",
    "future_",
    "next_",
    "after_",
    "post_",
    "predicted",
    "prediction",
    "probability",
    "risk_score",
    "rank_score",
    "selected_threshold",
    "days_to_churn",
    "days_until_churn",
    "uplift_score",
    "expected_roi",
    "expected_incremental_profit",
)

# Features below are current-state proxies when the label itself is generated from
# the same current inactivity rule. For uploaded explicit labels or time-horizon
# future-activity labels they are allowed as historical predictors.
CURRENT_INACTIVITY_PROXY_TOKENS = (
    "inactivity_days",
    "inactive_days",
    "days_since_last_event",
    "days_since_last_activity",
    "recency_days",
    "current_non_purchase_days",
    "non_purchase_days",
)

CURRENT_INACTIVITY_LABEL_SOURCES = {
    "inactivity_rule",
    "current_inactivity_rule",
    "inactivity_rule_current_snapshot_fallback",
    "uploaded_inactivity_rule_observed",
}


# Final fail-safe blacklist. These are not ordinary behavioral features in this
# project; they are derived from the same inactivity/risk logic used to create
# the churn label or from simulator latent probabilities. They must never reach
# the model input.
STRICT_FORBIDDEN_FEATURE_TOKENS = (
    # Project-specific stage/status fields are derived from churn/risk logic.
    "current_journey_stage",
    "journey_stage",
    "customer_journey_stage",
    "lifecycle_stage",
    "customer_status",
    "churn_risk",
    "at_risk",
    "risk_stage",
    "risk_segment",
    "risk_tier",
    "churn_segment",
    "dormant",
    # Simulator latent probabilities / generated scores, not observed customer behavior.
    "base_visit_prob",
    "base_purchase_prob",
    "base_exposure_prob",
    "visit_probability",
    "purchase_probability",
    "exposure_probability",
    "visit_prob",
    "purchase_prob",
    "exposure_prob",
    "recent_visit_score",
    "recent_purchase_score",
    "recent_exposure_score",
    # Generated decision/business outputs must not be model inputs.
    "churn_probability",
    "uplift_score",
    "expected_roi",
    "expected_incremental_profit",
    "selection_score",
    "priority_score",
    "recommendation_score",
)
IDENTIFIER_TOKENS = (
    "customer_id",
    "user_id",
    "member_id",
    "account_id",
    "client_id",
    "order_id",
    "invoice",
    "session_id",
    "event_id",
    "transaction_id",
    "email",
    "phone",
    "uuid",
    "name",
)



def _is_current_inactivity_label_source(label_source: str | None) -> bool:
    normalized = str(label_source or "").strip().lower()
    return normalized in CURRENT_INACTIVITY_LABEL_SOURCES


def _is_current_inactivity_proxy(col: object) -> bool:
    lower = str(col).strip().lower()
    normalized = _normalize_column_name(col)
    return any(token in lower or token in normalized for token in CURRENT_INACTIVITY_PROXY_TOKENS)


def _identify_name_based_leakage_columns(columns, label_source: str | None = None) -> List[str]:
    """Drop target/outcome columns while preserving valid historical behavior.

    Earlier versions used a blanket token filter that also removed normal churn
    predictors such as frequency, monetary value, recency, and session behavior.
    That made the held-out AUC collapse toward 0.5 after leakage columns were
    removed.  The filter is now intentionally narrow; current-inactivity proxy
    columns are dropped only when the training label itself was produced by the
    same current inactivity rule.
    """
    leakage: List[str] = []
    current_inactivity_label = _is_current_inactivity_label_source(label_source)
    for col in columns:
        lower = str(col).strip().lower()
        normalized = _normalize_column_name(col)

        is_current_proxy = _is_current_inactivity_proxy(col)
        if current_inactivity_label and is_current_proxy:
            leakage.append(col)
            continue
        if (not current_inactivity_label) and is_current_proxy:
            # Recency/inactivity are valid historical predictors for explicit labels
            # and for labels measured in a future horizon.
            continue

        if lower in DIRECT_LEAKAGE_COLUMNS or normalized in DIRECT_LEAKAGE_COLUMNS:
            leakage.append(col)
            continue
        if any(token in lower for token in DIRECT_LEAKAGE_TOKENS):
            leakage.append(col)
            continue
        if any(token in normalized for token in DIRECT_LEAKAGE_TOKENS):
            leakage.append(col)
            continue
    return leakage


def _identify_strict_forbidden_columns(columns, label_source: str | None = None) -> List[str]:
    """Fail-safe removal for known project-specific generated risk fields."""
    forbidden: List[str] = []
    current_inactivity_label = _is_current_inactivity_label_source(label_source)
    for col in columns:
        lower = str(col).strip().lower()
        normalized = _normalize_column_name(col)
        if current_inactivity_label and _is_current_inactivity_proxy(col):
            forbidden.append(col)
            continue
        if any(token in lower for token in STRICT_FORBIDDEN_FEATURE_TOKENS):
            forbidden.append(col)
            continue
        if any(token in normalized for token in STRICT_FORBIDDEN_FEATURE_TOKENS):
            forbidden.append(col)
            continue
    return forbidden





def _identify_identifier_columns(X: pd.DataFrame, max_unique_ratio: float = 0.50) -> List[str]:
    """Remove IDs/high-cardinality categoricals that allow memorization rather than learning."""
    identifier_cols: List[str] = []
    n = max(len(X), 1)
    for col in X.columns:
        lower = str(col).strip().lower()
        if any(token in lower for token in IDENTIFIER_TOKENS):
            identifier_cols.append(col)
            continue

        if pd.api.types.is_object_dtype(X[col]) or str(X[col].dtype) == "category":
            nunique = X[col].nunique(dropna=True)
            # Event-sequence strings and near-unique categoricals explode the
            # one-hot matrix and tend to memorize individual histories.
            if ("sequence" in lower and nunique > 50) or (nunique > 50 and (nunique / n) >= max_unique_ratio):
                identifier_cols.append(col)
    return identifier_cols


def _identify_target_proxy_columns(
    X: pd.DataFrame,
    y: pd.Series,
    auc_cutoff: float = 0.995,
    purity_cutoff: float = 0.995,
) -> List[str]:
    """
    Data-driven guardrail: drop columns that almost exactly reproduce y.

    This intentionally uses the label only as a leakage detector, not as a feature
    selector. If a single column alone separates the label with AUC ~= 1.0 or
    category purity ~= 1.0, keeping it makes the test AUC meaningless.
    """
    proxy_cols: List[str] = []
    y_arr = pd.Series(y).astype(int).to_numpy()
    if len(np.unique(y_arr)) < 2:
        return proxy_cols

    for col in X.columns:
        s = X[col]
        if s.nunique(dropna=True) <= 1:
            continue

        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            mask = numeric.notna().to_numpy()
            if mask.sum() < 10 or len(np.unique(y_arr[mask])) < 2:
                continue
            try:
                auc = roc_auc_score(y_arr[mask], numeric.to_numpy()[mask])
                separability = max(float(auc), float(1.0 - auc))
            except Exception:
                continue
            if separability >= auc_cutoff:
                proxy_cols.append(col)
            continue

        # For categoricals, target leakage often appears as a category that maps
        # almost deterministically to the label.
        tmp = pd.DataFrame({"feature": s.astype("object").where(s.notna(), "__missing__"), "y": y_arr})
        stats = tmp.groupby("feature")["y"].agg(["mean", "count"])
        purity = np.maximum(stats["mean"], 1.0 - stats["mean"])
        weighted_purity = float((purity * stats["count"]).sum() / stats["count"].sum())
        if weighted_purity >= purity_cutoff:
            proxy_cols.append(col)

    return proxy_cols


def _looks_like_datetime_series(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or str(series.dtype) == "category"):
        return False
    sample = series.dropna().astype(str).head(80)
    if sample.empty:
        return False
    date_like = sample.str.contains(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        regex=True,
        na=False,
    ).mean()
    if date_like < 0.60:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.70)


def _extract_datetime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    converted_cols: List[str] = []

    datetime_cols = out.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    object_datetime_cols = [
        col for col in out.columns
        if col not in datetime_cols and _looks_like_datetime_series(out[col])
    ]
    datetime_cols.extend(object_datetime_cols)

    for col in datetime_cols:
        ts = pd.to_datetime(out[col], errors="coerce")
        # Absolute epoch time can leak the train/test period. Keep coarse calendar
        # features only. If recency features are needed, generate them upstream
        # using only information available at scoring time.
        out[f"{col}_year"] = ts.dt.year
        out[f"{col}_month"] = ts.dt.month
        out[f"{col}_dayofweek"] = ts.dt.dayofweek
        out.drop(columns=[col], inplace=True)
        converted_cols.append(col)

    return out, converted_cols


def _sanitize_training_frame(
    features_df: pd.DataFrame,
    *,
    label_source: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, Dict]:
    if "label" not in features_df.columns:
        raise ValueError("features_df must contain a binary 'label' column for churn training.")

    y = features_df["label"].astype(int)
    if y.nunique(dropna=True) < 2:
        raise ValueError("Churn training needs both positive and negative labels.")

    groups = None
    if "customer_id" in features_df.columns:
        groups = features_df["customer_id"].astype("object").where(
            features_df["customer_id"].notna(), "__missing_customer__"
        )

    X = features_df.drop(columns=["label", "customer_id"], errors="ignore").copy()

    name_leakage_columns = _identify_name_based_leakage_columns(X.columns, label_source=label_source)
    if name_leakage_columns:
        X = X.drop(columns=name_leakage_columns, errors="ignore")

    strict_forbidden_columns = _identify_strict_forbidden_columns(X.columns, label_source=label_source)
    if strict_forbidden_columns:
        X = X.drop(columns=strict_forbidden_columns, errors="ignore")

    identifier_columns = _identify_identifier_columns(X)
    if identifier_columns:
        X = X.drop(columns=identifier_columns, errors="ignore")

    target_proxy_columns = _identify_target_proxy_columns(X, y)
    if target_proxy_columns:
        X = X.drop(columns=target_proxy_columns, errors="ignore")

    X, converted_datetime_cols = _extract_datetime_features(X)

    post_datetime_forbidden_columns = _identify_strict_forbidden_columns(X.columns, label_source=label_source)
    if post_datetime_forbidden_columns:
        X = X.drop(columns=post_datetime_forbidden_columns, errors="ignore")

    constant_columns = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    if constant_columns:
        X = X.drop(columns=constant_columns, errors="ignore")

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

    if X.shape[1] == 0:
        raise ValueError(
            "No usable training features remain after leakage/identifier filtering. "
            "Check whether the input frame contains only labels, IDs, predictions, or future outcomes."
        )

    metadata = {
        "input_feature_count": int(features_df.shape[1] - 1),
        "training_feature_count": int(X.shape[1]),
        "converted_datetime_columns": converted_datetime_cols,
        "excluded_name_leakage_columns": [str(col) for col in name_leakage_columns],
        "excluded_strict_forbidden_columns": [str(col) for col in strict_forbidden_columns],
        "excluded_post_datetime_forbidden_columns": [str(col) for col in post_datetime_forbidden_columns],
        "excluded_identifier_columns": [str(col) for col in identifier_columns],
        "excluded_target_proxy_columns": [str(col) for col in target_proxy_columns],
        "excluded_constant_columns": [str(col) for col in constant_columns],
        "leakage_guardrail": (
            "dropped direct target/score/outcome columns, project-specific churn stage/inactivity proxies, "
            "ID-like columns, near-perfect single-column target proxies, and constant columns before splitting"
        ),
        "group_column": "customer_id" if groups is not None else None,
        "unique_groups": int(groups.nunique(dropna=False)) if groups is not None else None,
        "label_source": str(label_source or "unknown"),
        "current_inactivity_proxy_policy": (
            "dropped recency/inactivity proxy features because label_source is current-inactivity based"
            if _is_current_inactivity_label_source(label_source)
            else "kept historical recency/frequency/monetary behavior features because label_source is explicit or horizon-based"
        ),
    }
    return X, y, groups, metadata


def _make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None, pd.Series | None, Dict]:
    """Prefer a customer-level split so the same customer never appears in train and test."""
    split_meta: Dict = {}

    if groups is not None and groups.nunique(dropna=False) >= 2:
        groups_arr = groups.reset_index(drop=True)
        X_reset = X.reset_index(drop=True)
        y_reset = y.reset_index(drop=True)

        splitter = GroupShuffleSplit(
            n_splits=50,
            test_size=float(test_size),
            random_state=int(random_state),
        )
        best_split = None
        best_balance_gap = float("inf")
        overall_pos_rate = float(y_reset.mean())

        for train_idx, test_idx in splitter.split(X_reset, y_reset, groups_arr):
            y_train_candidate = y_reset.iloc[train_idx]
            y_test_candidate = y_reset.iloc[test_idx]
            if y_train_candidate.nunique() < 2 or y_test_candidate.nunique() < 2:
                continue
            gap = abs(float(y_test_candidate.mean()) - overall_pos_rate)
            if gap < best_balance_gap:
                best_split = (train_idx, test_idx)
                best_balance_gap = gap

        if best_split is not None:
            train_idx, test_idx = best_split
            train_groups = groups_arr.iloc[train_idx].reset_index(drop=True)
            test_groups = groups_arr.iloc[test_idx].reset_index(drop=True)
            overlap = set(train_groups.astype(str)).intersection(set(test_groups.astype(str)))
            split_meta = {
                "split_strategy": "customer-level GroupShuffleSplit",
                "customer_id_overlap_count": int(len(overlap)),
                "train_groups": int(train_groups.nunique(dropna=False)),
                "test_groups": int(test_groups.nunique(dropna=False)),
                "test_positive_rate_gap": float(best_balance_gap),
            }
            return (
                X_reset.iloc[train_idx].reset_index(drop=True),
                X_reset.iloc[test_idx].reset_index(drop=True),
                y_reset.iloc[train_idx].reset_index(drop=True),
                y_reset.iloc[test_idx].reset_index(drop=True),
                train_groups,
                test_groups,
                split_meta,
            )

    # Fallback for one-row-per-customer frames or missing customer_id.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )
    split_meta = {
        "split_strategy": "row-level stratified train_test_split fallback",
        "customer_id_overlap_count": None,
        "train_groups": None,
        "test_groups": None,
        "test_positive_rate_gap": float(abs(float(y_test.mean()) - float(y.mean()))),
    }
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        None,
        None,
        split_meta,
    )


def _build_cv(
    y_train: pd.Series,
    groups_train: pd.Series | None,
    random_state: int,
) -> Tuple[object, pd.Series | None, str]:
    min_class_count = int(y_train.value_counts().min())
    if min_class_count < 2:
        raise ValueError("Not enough samples in one of the classes for cross-validation.")

    if groups_train is not None and groups_train.nunique(dropna=False) >= 3 and STRATIFIED_GROUP_KFOLD_AVAILABLE:
        n_splits = min(5, int(groups_train.nunique(dropna=False)), min_class_count)
        if n_splits >= 2:
            return (
                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)),
                groups_train,
                f"{n_splits}-fold StratifiedGroupKFold grouped by customer_id",
            )

    n_splits = min(5, min_class_count)
    return (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)),
        None,
        f"{n_splits}-fold StratifiedKFold fallback",
    )


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
    label_source: str | None = None,
) -> TrainingArtifacts:
    model_dir = _ensure_dir(Path(model_dir))
    result_dir = _ensure_dir(Path(result_dir))

    X, y, groups, preprocessing_meta = _sanitize_training_frame(features_df, label_source=label_source)
    requested_models = _resolve_requested_models(candidate_models)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        split_meta,
    ) = _make_train_test_split(
        X,
        y,
        groups,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pre, num_cols, cat_cols = _build_preprocessor(X_train)
    cv, cv_groups, cv_strategy_text = _build_cv(y_train, groups_train, random_state)

    candidates: Dict[str, tuple[object, Dict]] = {
        "xgboost": (
            XGBClassifier(
                random_state=int(random_state),
                n_estimators=90,
                learning_rate=0.06,
                max_depth=3,
                min_child_weight=5,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_lambda=10.0,
                reg_alpha=1.0,
                eval_metric="logloss",
                n_jobs=4,
            ),
            {
                "model__max_depth": [2, 3],
                "model__min_child_weight": [5, 10],
            },
        ),
    }

    if LIGHTGBM_AVAILABLE:
        candidates["lightgbm"] = (
            LGBMClassifier(
                random_state=int(random_state),
                n_estimators=120,
                learning_rate=0.06,
                num_leaves=15,
                min_child_samples=40,
                class_weight="balanced",
                verbosity=-1,
            ),
            {
                "model__num_leaves": [15, 31],
                "model__min_child_samples": [40, 80],
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
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )

        fit_kwargs = {"model__sample_weight": sample_weight} if name == "xgboost" else {}
        if cv_groups is not None:
            fit_kwargs["groups"] = cv_groups

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

    excluded_columns = (
        preprocessing_meta.get("excluded_name_leakage_columns", [])
        + preprocessing_meta.get("excluded_strict_forbidden_columns", [])
        + preprocessing_meta.get("excluded_post_datetime_forbidden_columns", [])
        + preprocessing_meta.get("excluded_identifier_columns", [])
        + preprocessing_meta.get("excluded_target_proxy_columns", [])
        + preprocessing_meta.get("excluded_constant_columns", [])
    )

    forbidden_top_features = [
        item for item in top10
        if any(token in str(item.get("feature", "")).lower() for token in STRICT_FORBIDDEN_FEATURE_TOKENS)
    ]

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
        "label_source": preprocessing_meta.get("label_source"),
        "current_inactivity_proxy_policy": preprocessing_meta.get("current_inactivity_proxy_policy"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "numeric_feature_count": len(num_cols),
        "categorical_feature_count": len(cat_cols),
        "converted_datetime_columns": preprocessing_meta["converted_datetime_columns"],
        "excluded_leakage_columns": [str(col) for col in excluded_columns],
        "excluded_name_leakage_columns": preprocessing_meta.get("excluded_name_leakage_columns", []),
        "excluded_strict_forbidden_columns": preprocessing_meta.get("excluded_strict_forbidden_columns", []),
        "excluded_post_datetime_forbidden_columns": preprocessing_meta.get("excluded_post_datetime_forbidden_columns", []),
        "excluded_identifier_columns": preprocessing_meta.get("excluded_identifier_columns", []),
        "excluded_target_proxy_columns": preprocessing_meta.get("excluded_target_proxy_columns", []),
        "excluded_constant_columns": preprocessing_meta.get("excluded_constant_columns", []),
        "forbidden_top_features": forbidden_top_features,
        "leakage_guardrail": preprocessing_meta.get("leakage_guardrail"),
        "split_strategy": split_meta.get("split_strategy"),
        "split_diagnostics": split_meta,
        "cv_strategy": cv_strategy_text,
        "top_10_feature_importance": top10,
        "imbalance_handling": imbalance_text,
        "business_threshold_rule": f"maximize TP*{float(threshold_tp_value):.0f} - FP*{float(threshold_fp_cost):.0f} - FN*{float(threshold_fn_cost):.0f}",
        "training_parameters": {
            "requested_models": requested_models,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "shap_sample_size": int(shap_sample_size),
            "threshold_tp_value": float(threshold_tp_value),
            "threshold_fp_cost": float(threshold_fp_cost),
            "threshold_fn_cost": float(threshold_fn_cost),
            "label_source": str(label_source or "unknown"),
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

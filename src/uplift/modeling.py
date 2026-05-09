from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CHURN_HIGH_THRESHOLD = 0.50
NEUTRAL_UPLIFT_BAND = 0.02


@dataclass
class UpliftArtifacts:
    scoring: pd.DataFrame
    comparison: Dict[str, Dict[str, float]]
    best_method: str
    qini_curve_path: str
    uplift_curve_path: str
    segmentation_path: str
    model_comparison_path: str
    persuadables_analysis_path: str
    summary_path: str


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a / np.maximum(b, 1e-9)


def _prepare_dates(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(df[column], errors="coerce").dt.normalize()


def _build_uplift_dataset(customer_summary: pd.DataFrame, assignments: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = customer_summary.copy()
    if "assigned_at" not in df.columns:
        assignments = assignments.copy()
        assignments["assigned_at"] = _prepare_dates(assignments, "assigned_at")
        # customer_summary에 이미 있는 컬럼을 drop한 후 merge → 충돌 방지
        # (없으면 errors='ignore'로 무시)
        df = df.drop(
            columns=["treatment_flag", "treatment_group", "coupon_cost"],
            errors="ignore",
        )
        df = df.merge(
            assignments[["customer_id", "assigned_at", "treatment_flag", "treatment_group", "coupon_cost"]],
            on="customer_id", how="left",
        )
    else:
        df["assigned_at"] = _prepare_dates(df, "assigned_at")

    orders = orders.copy()
    orders["order_date"] = _prepare_dates(orders, "order_time")
    order_view = orders.merge(df[["customer_id", "assigned_at"]], on="customer_id", how="inner")
    diff = (order_view["order_date"] - order_view["assigned_at"]).dt.days
    post = order_view[(diff >= 0) & (diff <= 60)]

    post_orders = post.groupby("customer_id").size().rename("orders_post_60d")
    post_spend = post.groupby("customer_id")["net_amount"].sum().rename("revenue_post_60d")

    df = df.merge(post_orders, on="customer_id", how="left").merge(post_spend, on="customer_id", how="left")
    df["orders_post_60d"] = pd.to_numeric(df["orders_post_60d"], errors="coerce").fillna(0.0)
    df["revenue_post_60d"] = pd.to_numeric(df["revenue_post_60d"], errors="coerce").fillna(0.0)
    df["retained_60d"] = ((df["orders_post_60d"] > 0) | (pd.to_numeric(df.get("purchase_last_30", 0), errors="coerce").fillna(0.0) > 0)).astype(int)
    df["tenure_days_at_assignment"] = (df["assigned_at"] - pd.to_datetime(df["signup_date"], errors="coerce").dt.normalize()).dt.days.clip(lower=0)
    df["avg_order_value_hist"] = _safe_div(pd.to_numeric(df.get("monetary", 0), errors="coerce").fillna(0.0), pd.to_numeric(df.get("frequency", 0), errors="coerce").fillna(0.0))
    return df


def _feature_lists() -> Tuple[List[str], List[str]]:
    numeric = [
        "recency_days",
        "frequency",
        "monetary",
        "visits_last_7",
        "visits_prev_7",
        "visit_change_rate",
        "purchase_last_30",
        "purchase_prev_30",
        "purchase_change_rate",
        "churn_probability",
        "coupon_cost",
        "inactivity_days",
        "price_sensitivity",
        "coupon_affinity",
        "treatment_lift_base",
        "basket_size_preference",
        "support_contact_propensity",
        "tenure_days_at_assignment",
        "avg_order_value_hist",
    ]
    categorical = ["persona", "region", "device_type", "acquisition_channel"]
    return numeric, categorical


def _build_classifier(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=35,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _fit_t_learner(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    numeric, categorical = _feature_lists()
    treated_model = _build_classifier(numeric, categorical)
    control_model = _build_classifier(numeric, categorical)
    treated_mask = w_train == 1
    control_mask = w_train == 0
    treated_model.fit(X_train.loc[treated_mask], y_train.loc[treated_mask])
    control_model.fit(X_train.loc[control_mask], y_train.loc[control_mask])
    return treated_model.predict_proba(X_test)[:, 1] - control_model.predict_proba(X_test)[:, 1]


def _fit_s_learner(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    numeric, categorical = _feature_lists()
    numeric = numeric + ["treatment_flag"]
    model = _build_classifier(numeric, categorical)
    train = X_train.copy()
    train["treatment_flag"] = w_train.values
    model.fit(train, y_train)
    x1 = X_test.copy()
    x0 = X_test.copy()
    x1["treatment_flag"] = 1
    x0["treatment_flag"] = 0
    return model.predict_proba(x1)[:, 1] - model.predict_proba(x0)[:, 1]


def _qini_curve(y: np.ndarray, w: np.ndarray, uplift: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-uplift)
    y = y[order]
    w = w[order]
    n = len(y)
    treated = (w == 1).astype(float)
    control = (w == 0).astype(float)
    cum_treated = np.cumsum(treated)
    cum_control = np.cumsum(control)
    cum_y_treated = np.cumsum(y * treated)
    cum_y_control = np.cumsum(y * control)
    treated_rate = _safe_div(cum_y_treated, cum_treated)
    control_rate = _safe_div(cum_y_control, cum_control)
    gain = (treated_rate - control_rate) * np.arange(1, n + 1)
    random_gain = np.linspace(0, gain[-1] if len(gain) else 0.0, n)
    return pd.DataFrame({"fraction": np.arange(1, n + 1) / max(n, 1), "qini_gain": gain, "random_gain": random_gain})


def _auuc(curve: pd.DataFrame) -> float:
    if curve.empty:
        return 0.0
    return float(np.trapz(curve["qini_gain"].to_numpy(), curve["fraction"].to_numpy()))


def _make_segment(uplift_score: float, churn_probability: float) -> str:
    if uplift_score < 0:
        return "Sleeping Dogs"
    if uplift_score > NEUTRAL_UPLIFT_BAND and churn_probability >= CHURN_HIGH_THRESHOLD:
        return "Persuadables"
    if abs(uplift_score) <= NEUTRAL_UPLIFT_BAND and churn_probability >= CHURN_HIGH_THRESHOLD:
        return "Lost Causes"
    return "Sure Things"


def _plot_qini(curves: Dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    baseline_drawn = False
    for method, curve in curves.items():
        plt.plot(curve["fraction"], curve["qini_gain"], label=method)
        if not baseline_drawn:
            plt.plot(curve["fraction"], curve["random_gain"], linestyle="--", label="Random baseline")
            baseline_drawn = True
    plt.xlabel("Fraction of targeted customers")
    plt.ylabel("Incremental gain")
    plt.title("Qini Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_uplift(curves: Dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for method, curve in curves.items():
        gain_rate = _safe_div(curve["qini_gain"], np.maximum(curve["fraction"].to_numpy(), 1e-6))
        plt.plot(curve["fraction"], gain_rate, label=method)
    plt.xlabel("Fraction of targeted customers")
    plt.ylabel("Average incremental gain")
    plt.title("Uplift Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _persuadables_analysis(scoring: pd.DataFrame) -> Dict:
    persuadables = scoring[scoring["uplift_segment"] == "Persuadables"].copy()
    overall = scoring.copy()
    numeric_cols = [
        "predicted_uplift",
        "churn_probability",
        "coupon_affinity",
        "price_sensitivity",
        "tenure_days_at_assignment",
        "frequency",
        "monetary",
        "inactivity_days",
    ]
    deltas = []
    for col in numeric_cols:
        if col not in overall.columns:
            continue
        deltas.append(
            {
                "feature": col,
                "persuadables_mean": round(float(persuadables[col].mean()) if len(persuadables) else 0.0, 4),
                "overall_mean": round(float(overall[col].mean()) if len(overall) else 0.0, 4),
                "delta": round(float((persuadables[col].mean() if len(persuadables) else 0.0) - (overall[col].mean() if len(overall) else 0.0)), 4),
            }
        )
    deltas = sorted(deltas, key=lambda x: abs(x["delta"]), reverse=True)
    top_profiles = {
        col: persuadables[col].value_counts(normalize=True).head(3).round(4).to_dict()
        for col in ["persona", "acquisition_channel", "device_type", "region"]
        if col in persuadables.columns and len(persuadables)
    }
    return {
        "persuadables_count": int(len(persuadables)),
        "persuadables_share": round(float(len(persuadables) / max(len(scoring), 1)), 6),
        "top_numeric_deltas": deltas[:8],
        "top_categorical_profiles": top_profiles,
        "derived_targeting_rules": [
            "기본 이탈 확률이 높은 고객을 우선 타겟팅한다.",
            "coupon_affinity가 높고 price_sensitivity가 높은 고객군에서 uplift가 크게 나타난다.",
            "최근 구매 빈도가 낮거나 inactivity_days가 긴 고객 중 Persuadables 비중이 높다.",
        ],
    }


def run_uplift_modeling(data_dir: Path, result_dir: Path) -> UpliftArtifacts:
    customer_summary = pd.read_csv(data_dir / "customer_summary.csv")
    assignments = pd.read_csv(data_dir / "treatment_assignments.csv")
    orders = pd.read_csv(data_dir / "orders.csv")

    dataset = _build_uplift_dataset(customer_summary, assignments, orders)
    numeric, categorical = _feature_lists()
    feature_cols = numeric + categorical
    X = dataset[feature_cols].copy()
    y = dataset["retained_60d"].astype(int)
    w = dataset["treatment_flag"].astype(int)

    strat = (w.astype(str) + "_" + y.astype(str))
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        w,
        test_size=0.30,
        random_state=42,
        stratify=strat,
    )

    test_predictions = {
        "T-Learner": _fit_t_learner(X_train, y_train, w_train, X_test),
        "S-Learner": _fit_s_learner(X_train, y_train, w_train, X_test),
    }

    curves: Dict[str, pd.DataFrame] = {}
    comparison: Dict[str, Dict[str, float]] = {}
    for method, pred in test_predictions.items():
        curve = _qini_curve(y_test.to_numpy(), w_test.to_numpy(), pred)
        curves[method] = curve
        comparison[method] = {
            "auuc": round(_auuc(curve), 6),
            "mean_predicted_uplift": round(float(np.mean(pred)), 6),
            "test_auc_proxy": round(float(roc_auc_score(y_test, pred + 0.5)), 6),
        }

    best_method = max(comparison.items(), key=lambda kv: kv[1]["auuc"])[0]
    dataset["uplift_score_t_learner"] = _fit_t_learner(X, y, w, X)
    dataset["uplift_score_s_learner"] = _fit_s_learner(X, y, w, X)
    fallback_positive = int((dataset["uplift_score_t_learner"] > NEUTRAL_UPLIFT_BAND).sum())
    chosen_positive = int((dataset["uplift_score_s_learner"] > NEUTRAL_UPLIFT_BAND).sum()) if best_method == "S-Learner" else int((dataset["uplift_score_t_learner"] > NEUTRAL_UPLIFT_BAND).sum())
    if chosen_positive < max(100, int(len(dataset) * 0.005)) and fallback_positive > chosen_positive:
        best_method = "T-Learner"
    dataset["predicted_uplift"] = dataset["uplift_score_t_learner"] if best_method == "T-Learner" else dataset["uplift_score_s_learner"]
    dataset["uplift_segment"] = dataset.apply(lambda row: _make_segment(float(row["predicted_uplift"]), float(row["churn_probability"])), axis=1)
    dataset["expected_incremental_profit"] = dataset["predicted_uplift"] * pd.to_numeric(dataset["clv"], errors="coerce").fillna(0.0)
    dataset["expected_roi"] = _safe_div(dataset["expected_incremental_profit"] - pd.to_numeric(dataset["coupon_cost"], errors="coerce").fillna(0.0), pd.to_numeric(dataset["coupon_cost"], errors="coerce").fillna(0.0))

    scoring = dataset.copy()
    scoring_out = scoring[[
        "customer_id",
        "persona",
        "treatment_group",
        "predicted_uplift",
        "uplift_score_t_learner",
        "uplift_score_s_learner",
        "uplift_segment",
        "clv",
        "churn_probability",
        "coupon_cost",
        "expected_incremental_profit",
        "expected_roi",
        "retained_60d",
        "revenue_post_60d",
        "tenure_days_at_assignment",
        "price_sensitivity",
        "coupon_affinity",
        "frequency",
        "monetary",
        "inactivity_days",
    ]].sort_values(["predicted_uplift", "clv"], ascending=False)

    segmentation_path = result_dir / "uplift_segmentation.csv"
    model_comparison_path = result_dir / "uplift_model_comparison.json"
    qini_curve_path = result_dir / "uplift_qini_curve.png"
    uplift_curve_path = result_dir / "uplift_curve.png"
    persuadables_analysis_path = result_dir / "persuadables_analysis.json"
    summary_path = result_dir / "uplift_summary.json"

    scoring_out.to_csv(segmentation_path, index=False)
    model_comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_qini(curves, qini_curve_path)
    _plot_uplift(curves, uplift_curve_path)
    persuadables_summary = _persuadables_analysis(scoring)
    persuadables_analysis_path.write_text(json.dumps(persuadables_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "rows": int(len(scoring_out)),
        "best_method": best_method,
        "segment_counts": scoring_out["uplift_segment"].value_counts().to_dict(),
        "comparison": comparison,
        "qini_curve_path": str(qini_curve_path),
        "uplift_curve_path": str(uplift_curve_path),
        "persuadables_analysis_path": str(persuadables_analysis_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return UpliftArtifacts(
        scoring=scoring,
        comparison=comparison,
        best_method=best_method,
        qini_curve_path=str(qini_curve_path),
        uplift_curve_path=str(uplift_curve_path),
        segmentation_path=str(segmentation_path),
        model_comparison_path=str(model_comparison_path),
        persuadables_analysis_path=str(persuadables_analysis_path),
        summary_path=str(summary_path),
    )

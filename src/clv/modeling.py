from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class CLVArtifacts:
    predictions: pd.DataFrame
    predictions_path: str
    metrics_path: str
    distribution_report_path: str
    distribution_plot_path: str


def _prepare_dates(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(df[column], errors="coerce").dt.normalize()


def _compute_features_at_anchor(customers: pd.DataFrame, orders: pd.DataFrame, anchor_date: str) -> pd.DataFrame:
    anchor = pd.Timestamp(anchor_date).normalize()
    customers = customers.copy()
    customers["signup_date"] = _prepare_dates(customers, "signup_date")
    customers = customers[customers["signup_date"] <= anchor].copy()

    orders = orders.copy()
    orders["order_date"] = _prepare_dates(orders, "order_time")

    def agg_orders(window: int, prefix: str) -> pd.DataFrame:
        start = anchor - pd.Timedelta(days=window - 1)
        sliced = orders[(orders["order_date"] >= start) & (orders["order_date"] <= anchor)].copy()
        if sliced.empty:
            return pd.DataFrame(columns=["customer_id", f"{prefix}_orders", f"{prefix}_spend", f"{prefix}_avg_order"])
        out = (
            sliced.groupby("customer_id", as_index=False)
            .agg(**{
                f"{prefix}_orders": ("order_id", "count"),
                f"{prefix}_spend": ("net_amount", "sum"),
            })
        )
        out[f"{prefix}_avg_order"] = out[f"{prefix}_spend"] / out[f"{prefix}_orders"].where(out[f"{prefix}_orders"] > 0, 1.0)
        return out

    base = customers.copy()
    for window, prefix in [(30, "w30"), (90, "w90"), (180, "w180")]:
        base = base.merge(agg_orders(window, prefix), on="customer_id", how="left")

    for col in [c for c in base.columns if c.startswith("w")]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    base["tenure_days"] = (anchor - base["signup_date"]).dt.days.clip(lower=0)
    base["spend_velocity"] = base["w180_spend"] / np.maximum(base["tenure_days"], 1.0)
    base["order_velocity"] = base["w180_orders"] / np.maximum(base["tenure_days"], 1.0)
    base["anchor_date"] = anchor
    return base


def _future_spend_target(orders: pd.DataFrame, anchor_date: str, horizon_days: int, customer_ids: pd.Index) -> pd.Series:
    anchor = pd.Timestamp(anchor_date).normalize()
    orders = orders.copy()
    orders["order_date"] = _prepare_dates(orders, "order_time")
    end = anchor + pd.Timedelta(days=horizon_days)
    sliced = orders[(orders["order_date"] > anchor) & (orders["order_date"] <= end)]
    target = sliced.groupby("customer_id")["net_amount"].sum()
    return target.reindex(customer_ids, fill_value=0.0).astype(float)


def _build_regressor(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=260,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _plot_distribution(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(predictions["predicted_clv_12m"], bins=40)
    plt.xlabel("Predicted CLV (12 months)")
    plt.ylabel("Customers")
    plt.title("Predicted CLV Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_clv_pipeline(data_dir: Path, result_dir: Path) -> CLVArtifacts:
    customers = pd.read_csv(data_dir / "customers.csv")
    orders = pd.read_csv(data_dir / "orders.csv")

    calibration = _compute_features_at_anchor(customers, orders, "2025-06-30")
    calibration["future_spend_180d"] = _future_spend_target(orders, "2025-06-30", 180, calibration["customer_id"]) 

    numeric_features = [
        "price_sensitivity", "coupon_affinity", "treatment_lift_base", "support_contact_propensity",
        "basket_size_preference", "avg_order_value_mean", "avg_order_value_std", "days_from_simulation_start",
        "w30_orders", "w30_spend", "w30_avg_order", "w90_orders", "w90_spend", "w90_avg_order",
        "w180_orders", "w180_spend", "w180_avg_order", "tenure_days", "spend_velocity", "order_velocity",
    ]
    categorical_features = ["persona", "region", "device_type", "acquisition_channel"]

    X = calibration[numeric_features + categorical_features].copy()
    y = pd.to_numeric(calibration["future_spend_180d"], errors="coerce").fillna(0.0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = _build_regressor(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0.0, None)

    metrics = {
        "validation_window": "predict next 180 days from 2025-06-30 features",
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2": round(float(r2_score(y_test, preds)), 6),
        "actual_mean_180d": round(float(np.mean(y_test)), 4),
        "predicted_mean_180d": round(float(np.mean(preds)), 4),
    }

    current = _compute_features_at_anchor(customers, orders, "2025-12-31")
    current_X = current[numeric_features + categorical_features].copy()
    future_180_pred = np.clip(model.predict(current_X), 0.0, None)
    predicted_clv_12m = future_180_pred * 2.0

    predictions = current[["customer_id", "persona", "region", "device_type", "acquisition_channel", "tenure_days"]].copy()
    predictions["predicted_future_spend_180d"] = future_180_pred
    predictions["predicted_clv_12m"] = predicted_clv_12m
    high_value_threshold = float(np.quantile(predicted_clv_12m, 0.80)) if len(predicted_clv_12m) else 0.0
    predictions["high_value_threshold_80pct"] = high_value_threshold
    predictions["is_high_value_top20pct"] = predictions["predicted_clv_12m"] >= high_value_threshold
    predictions = predictions.sort_values(["predicted_clv_12m", "customer_id"], ascending=[False, True]).reset_index(drop=True)

    top_n = predictions.head(max(int(len(predictions) * 0.20), 1)).copy()
    distribution_report = {
        "prediction_horizon": "12 months (annualized from model's next-180-day prediction)",
        "high_value_threshold": round(high_value_threshold, 4),
        "high_value_customers": int(predictions["is_high_value_top20pct"].sum()),
        "top_20pct_share": round(float(predictions["is_high_value_top20pct"].mean()) if len(predictions) else 0.0, 6),
        "distribution": {
            "min": round(float(predictions["predicted_clv_12m"].min()) if len(predictions) else 0.0, 4),
            "median": round(float(predictions["predicted_clv_12m"].median()) if len(predictions) else 0.0, 4),
            "mean": round(float(predictions["predicted_clv_12m"].mean()) if len(predictions) else 0.0, 4),
            "p80": round(float(np.quantile(predictions["predicted_clv_12m"], 0.80)) if len(predictions) else 0.0, 4),
            "p95": round(float(np.quantile(predictions["predicted_clv_12m"], 0.95)) if len(predictions) else 0.0, 4),
            "max": round(float(predictions["predicted_clv_12m"].max()) if len(predictions) else 0.0, 4),
        },
        "top_n_preview": top_n[["customer_id", "predicted_clv_12m", "persona"]].head(50).to_dict(orient="records"),
    }

    predictions_path = result_dir / "clv_predictions.csv"
    metrics_path = result_dir / "clv_validation_metrics.json"
    distribution_report_path = result_dir / "clv_distribution_report.json"
    distribution_plot_path = result_dir / "clv_distribution.png"

    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    distribution_report_path.write_text(json.dumps(distribution_report, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_distribution(predictions, distribution_plot_path)

    return CLVArtifacts(
        predictions=predictions,
        predictions_path=str(predictions_path),
        metrics_path=str(metrics_path),
        distribution_report_path=str(distribution_report_path),
        distribution_plot_path=str(distribution_plot_path),
    )

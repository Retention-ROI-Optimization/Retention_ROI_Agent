from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calendar_multiplier(date: pd.Timestamp) -> float:
    """
    Small seasonality term so the data is not perfectly stationary.
    """
    weekday = date.weekday()
    month = date.month
    weekend_boost = 0.10 if weekday >= 5 else 0.0
    payday_boost = 0.05 if 25 <= date.day <= 28 else 0.0
    year_end_boost = 0.08 if month in (11, 12) else 0.0
    return 1.0 + weekend_boost + payday_boost + year_end_boost


def compute_visit_probability(
    customers: pd.DataFrame,
    tracker,
    active_mask: np.ndarray,
    date: pd.Timestamp,
) -> np.ndarray:
    logit = (
        -2.5
        + 4.2 * customers["base_visit_prob"].to_numpy()
        + 0.20 * tracker.recent_visit_score
        + 0.15 * tracker.recent_purchase_score
        + 0.20 * tracker.recent_exposure_score * customers["coupon_affinity"].to_numpy()
        + 0.15 * tracker.recent_exposure_score * customers["treatment_lift_base"].to_numpy()
        - 0.035 * tracker.inactivity_days * customers["churn_sensitivity_base"].to_numpy()
        + 0.02 * np.minimum(tracker.purchases_total, 8)
    )
    prob = sigmoid(logit) * calendar_multiplier(date)
    prob = np.clip(prob, 0.0, 0.92)
    prob[~active_mask] = 0.0
    return prob


def compute_browse_probability(customers: pd.DataFrame, visit_mask: np.ndarray, tracker) -> np.ndarray:
    base = customers["browse_prob_base"].to_numpy()
    prob = np.clip(base + 0.05 * np.tanh(tracker.recent_visit_score / 3.0), 0.05, 0.98)
    prob[~visit_mask] = 0.0
    return prob


def compute_search_probability(customers: pd.DataFrame, visit_mask: np.ndarray, tracker) -> np.ndarray:
    base = customers["search_prob_base"].to_numpy()
    prob = np.clip(
        base + 0.03 * customers["price_sensitivity"].to_numpy() + 0.03 * tracker.recent_exposure_score,
        0.02,
        0.95,
    )
    prob[~visit_mask] = 0.0
    return prob


def compute_add_to_cart_probability(
    customers: pd.DataFrame,
    browse_mask: np.ndarray,
    search_mask: np.ndarray,
    tracker,
) -> np.ndarray:
    base = customers["add_to_cart_prob_base"].to_numpy()
    prob = (
        base
        + 0.06 * search_mask.astype(float)
        + 0.05 * tracker.recent_exposure_score * customers["coupon_affinity"].to_numpy()
        - 0.04 * customers["price_sensitivity"].to_numpy()
    )
    prob = np.clip(prob, 0.02, 0.95)
    prob[~browse_mask] = 0.0
    return prob


def compute_purchase_probability(
    customers: pd.DataFrame,
    visit_mask: np.ndarray,
    add_to_cart_mask: np.ndarray,
    coupon_open_mask: np.ndarray,
    tracker,
) -> np.ndarray:
    visit_base = customers["purchase_given_visit_base"].to_numpy()
    cart_base = customers["purchase_given_cart_base"].to_numpy()

    prob = (
        visit_base
        + add_to_cart_mask.astype(float) * cart_base
        + 0.06 * coupon_open_mask.astype(float) * customers["coupon_redeem_prob_base"].to_numpy()
        + 0.07 * tracker.recent_exposure_score * customers["treatment_lift_base"].to_numpy()
        + 0.03 * tracker.recent_purchase_score
        - 0.03 * customers["price_sensitivity"].to_numpy()
        - 0.02 * np.tanh(tracker.recent_cart_abandon_score)
    )
    prob = np.clip(prob, 0.005, 0.92)
    prob[~visit_mask] = 0.0
    return prob


def compute_remove_cart_probability(
    customers: pd.DataFrame,
    add_to_cart_mask: np.ndarray,
    purchase_mask: np.ndarray,
    tracker,
) -> np.ndarray:
    base = customers["remove_from_cart_prob_base"].to_numpy()
    prob = np.clip(base + 0.03 * customers["price_sensitivity"].to_numpy() + 0.03 * tracker.recent_cart_abandon_score, 0.01, 0.90)
    prob[~add_to_cart_mask] = 0.0
    prob[purchase_mask] = 0.0
    return prob


def compute_coupon_open_probability(
    customers: pd.DataFrame,
    exposure_mask: np.ndarray,
    tracker,
) -> np.ndarray:
    base = customers["coupon_open_prob_base"].to_numpy()
    prob = np.clip(base + 0.08 * tracker.inactivity_days / np.maximum(tracker.inactivity_days + 10, 1), 0.01, 0.98)
    prob[~exposure_mask] = 0.0
    return prob


def compute_coupon_redeem_probability(
    customers: pd.DataFrame,
    coupon_open_mask: np.ndarray,
    purchase_mask: np.ndarray,
) -> np.ndarray:
    base = customers["coupon_redeem_prob_base"].to_numpy()
    prob = np.clip(base + 0.15 * purchase_mask.astype(float), 0.01, 0.99)
    prob[~coupon_open_mask] = 0.0
    return prob


def classify_customer_status(inactivity_days: np.ndarray, dormant_threshold: int, churn_threshold: int) -> np.ndarray:
    status = np.full(len(inactivity_days), "active", dtype=object)
    status[inactivity_days >= dormant_threshold] = "dormant"
    status[inactivity_days >= churn_threshold] = "churn_risk"
    return status

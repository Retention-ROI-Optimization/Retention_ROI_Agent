from __future__ import annotations

import calendar
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .personas import DEFAULT_PERSONAS, PersonaProfile


def _month_start(month_str: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month_str}-01")


def _random_signup_dates(
    n: int,
    rng: np.random.Generator,
    signup_months,
) -> pd.Series:
    month_choices = rng.choice(signup_months, size=n, replace=True)
    offsets = []

    for month_str in month_choices:
        year, month = map(int, month_str.split("-"))
        days_in_month = calendar.monthrange(year, month)[1]
        offsets.append(rng.integers(0, days_in_month))

    signup_dates = [
        _month_start(m) + pd.Timedelta(days=int(off))
        for m, off in zip(month_choices, offsets)
    ]
    return pd.Series(signup_dates), pd.Series(month_choices)


def generate_customers(
    config: SimulationConfig,
    personas: Optional[Dict[str, PersonaProfile]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Generate customer master data.

    Output is intentionally rich enough for both:
    - raw simulator use
    - later customer-level feature aggregation
    """
    personas = personas or DEFAULT_PERSONAS
    rng = rng or np.random.default_rng(config.random_seed)

    persona_names = list(personas.keys())
    persona_weights = np.array([personas[name].acquisition_weight for name in persona_names], dtype=float)
    persona_weights = persona_weights / persona_weights.sum()

    n = config.n_customers
    customer_ids = np.arange(1, n + 1, dtype=int)
    persona = rng.choice(persona_names, size=n, p=persona_weights)

    signup_date, acquisition_month = _random_signup_dates(
        n=n,
        rng=rng,
        signup_months=config.signup_months,
    )

    region = rng.choice(
        ["Seoul", "Busan", "Incheon", "Daejeon", "Daegu", "Gwangju"],
        size=n,
        p=[0.34, 0.18, 0.13, 0.10, 0.15, 0.10],
    )
    device_type = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=n,
        p=[0.62, 0.30, 0.08],
    )
    acquisition_channel = rng.choice(
        ["organic", "paid_ads", "referral", "email", "social"],
        size=n,
        p=[0.28, 0.24, 0.14, 0.17, 0.17],
    )

    base_visit_prob = np.zeros(n, dtype=float)
    browse_prob_base = np.zeros(n, dtype=float)
    search_prob_base = np.zeros(n, dtype=float)
    add_to_cart_prob_base = np.zeros(n, dtype=float)
    remove_from_cart_prob_base = np.zeros(n, dtype=float)
    purchase_given_cart_base = np.zeros(n, dtype=float)
    purchase_given_visit_base = np.zeros(n, dtype=float)
    coupon_open_prob_base = np.zeros(n, dtype=float)
    coupon_redeem_prob_base = np.zeros(n, dtype=float)
    avg_order_value_mean = np.zeros(n, dtype=float)
    avg_order_value_std = np.zeros(n, dtype=float)
    churn_sensitivity_base = np.zeros(n, dtype=float)
    price_sensitivity = np.zeros(n, dtype=float)
    recovery_prob_base = np.zeros(n, dtype=float)
    treatment_lift_base = np.zeros(n, dtype=float)

    for persona_name, profile in personas.items():
        mask = persona == persona_name
        count = int(mask.sum())
        if count == 0:
            continue

        base_visit_prob[mask] = np.clip(rng.normal(profile.visit_prob, 0.03, size=count), 0.05, 0.75)
        browse_prob_base[mask] = np.clip(rng.normal(profile.browse_prob, 0.05, size=count), 0.25, 0.95)
        search_prob_base[mask] = np.clip(rng.normal(profile.search_prob, 0.05, size=count), 0.05, 0.90)
        add_to_cart_prob_base[mask] = np.clip(rng.normal(profile.add_to_cart_prob, 0.04, size=count), 0.05, 0.80)
        remove_from_cart_prob_base[mask] = np.clip(rng.normal(profile.remove_from_cart_prob, 0.03, size=count), 0.01, 0.70)
        purchase_given_cart_base[mask] = np.clip(rng.normal(profile.purchase_given_cart_prob, 0.05, size=count), 0.05, 0.95)
        purchase_given_visit_base[mask] = np.clip(rng.normal(profile.purchase_given_visit_prob, 0.02, size=count), 0.01, 0.30)
        coupon_open_prob_base[mask] = np.clip(rng.normal(profile.coupon_open_prob, 0.05, size=count), 0.01, 0.95)
        coupon_redeem_prob_base[mask] = np.clip(rng.normal(profile.coupon_redeem_prob, 0.05, size=count), 0.01, 0.90)
        avg_order_value_mean[mask] = np.clip(rng.normal(profile.avg_order_mean, profile.avg_order_std * 0.25, size=count), 25000, None)
        avg_order_value_std[mask] = np.clip(rng.normal(profile.avg_order_std, profile.avg_order_std * 0.15, size=count), 6000, None)
        churn_sensitivity_base[mask] = np.clip(rng.normal(profile.churn_sensitivity, 0.10, size=count), 0.40, 1.80)
        price_sensitivity[mask] = np.clip(rng.normal(profile.price_sensitivity, 0.08, size=count), 0.05, 0.98)
        recovery_prob_base[mask] = np.clip(rng.normal(profile.recovery_prob, 0.05, size=count), 0.01, 0.80)
        treatment_lift_base[mask] = np.clip(rng.normal(profile.treatment_lift, 0.03, size=count), -0.15, 0.40)

    coupon_affinity = np.clip(
        0.55 * coupon_open_prob_base + 0.45 * coupon_redeem_prob_base + rng.normal(0, 0.04, size=n),
        0.02,
        0.98,
    )
    basket_size_preference = np.clip(
        rng.normal(1.4 + 2.0 * (avg_order_value_mean / avg_order_value_mean.max()), 0.35, size=n),
        1.0,
        5.0,
    )
    support_contact_propensity = np.clip(
        rng.normal(0.05 + 0.10 * price_sensitivity + 0.07 * churn_sensitivity_base, 0.03, size=n),
        0.01,
        0.45,
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "persona": persona,
            "signup_date": pd.to_datetime(signup_date),
            "acquisition_month": acquisition_month.astype(str),
            "region": region,
            "device_type": device_type,
            "acquisition_channel": acquisition_channel,
            "base_visit_prob": base_visit_prob,
            "browse_prob_base": browse_prob_base,
            "search_prob_base": search_prob_base,
            "add_to_cart_prob_base": add_to_cart_prob_base,
            "remove_from_cart_prob_base": remove_from_cart_prob_base,
            "purchase_given_cart_base": purchase_given_cart_base,
            "purchase_given_visit_base": purchase_given_visit_base,
            "coupon_open_prob_base": coupon_open_prob_base,
            "coupon_redeem_prob_base": coupon_redeem_prob_base,
            "avg_order_value_mean": avg_order_value_mean,
            "avg_order_value_std": avg_order_value_std,
            "churn_sensitivity_base": churn_sensitivity_base,
            "price_sensitivity": price_sensitivity,
            "coupon_affinity": coupon_affinity,
            "recovery_prob_base": recovery_prob_base,
            "treatment_lift_base": treatment_lift_base,
            "basket_size_preference": basket_size_preference,
            "support_contact_propensity": support_contact_propensity,
        }
    )

    customers["days_from_simulation_start"] = (
        customers["signup_date"] - pd.Timestamp(config.start_date)
    ).dt.days.astype(int)

    return customers.sort_values("customer_id").reset_index(drop=True)

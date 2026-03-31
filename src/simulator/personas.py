from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PersonaProfile:
    """Base behavioral priors for a customer segment."""

    visit_prob: float
    browse_prob: float
    search_prob: float
    add_to_cart_prob: float
    remove_from_cart_prob: float
    purchase_given_cart_prob: float
    purchase_given_visit_prob: float
    coupon_open_prob: float
    coupon_redeem_prob: float
    avg_order_mean: float
    avg_order_std: float
    churn_sensitivity: float
    price_sensitivity: float
    recovery_prob: float
    treatment_lift: float
    acquisition_weight: float


DEFAULT_PERSONAS: Dict[str, PersonaProfile] = {
    "vip": PersonaProfile(
        visit_prob=0.30,
        browse_prob=0.76,
        search_prob=0.32,
        add_to_cart_prob=0.34,
        remove_from_cart_prob=0.10,
        purchase_given_cart_prob=0.50,
        purchase_given_visit_prob=0.07,
        coupon_open_prob=0.26,
        coupon_redeem_prob=0.18,
        avg_order_mean=128000,
        avg_order_std=26000,
        churn_sensitivity=0.72,
        price_sensitivity=0.35,
        recovery_prob=0.46,
        treatment_lift=0.03,
        acquisition_weight=0.15,
    ),
    "coupon_sensitive": PersonaProfile(
        visit_prob=0.24,
        browse_prob=0.72,
        search_prob=0.38,
        add_to_cart_prob=0.30,
        remove_from_cart_prob=0.15,
        purchase_given_cart_prob=0.42,
        purchase_given_visit_prob=0.05,
        coupon_open_prob=0.55,
        coupon_redeem_prob=0.40,
        avg_order_mean=72000,
        avg_order_std=18000,
        churn_sensitivity=0.98,
        price_sensitivity=0.82,
        recovery_prob=0.38,
        treatment_lift=0.22,
        acquisition_weight=0.25,
    ),
    "churn_risk": PersonaProfile(
        visit_prob=0.18,
        browse_prob=0.62,
        search_prob=0.26,
        add_to_cart_prob=0.20,
        remove_from_cart_prob=0.22,
        purchase_given_cart_prob=0.28,
        purchase_given_visit_prob=0.03,
        coupon_open_prob=0.42,
        coupon_redeem_prob=0.22,
        avg_order_mean=56000,
        avg_order_std=15000,
        churn_sensitivity=1.28,
        price_sensitivity=0.67,
        recovery_prob=0.18,
        treatment_lift=0.14,
        acquisition_weight=0.25,
    ),
    "sure_thing": PersonaProfile(
        visit_prob=0.26,
        browse_prob=0.74,
        search_prob=0.31,
        add_to_cart_prob=0.29,
        remove_from_cart_prob=0.08,
        purchase_given_cart_prob=0.56,
        purchase_given_visit_prob=0.08,
        coupon_open_prob=0.16,
        coupon_redeem_prob=0.08,
        avg_order_mean=98000,
        avg_order_std=20000,
        churn_sensitivity=0.65,
        price_sensitivity=0.28,
        recovery_prob=0.44,
        treatment_lift=0.02,
        acquisition_weight=0.20,
    ),
    "lost_cause": PersonaProfile(
        visit_prob=0.12,
        browse_prob=0.52,
        search_prob=0.20,
        add_to_cart_prob=0.14,
        remove_from_cart_prob=0.25,
        purchase_given_cart_prob=0.18,
        purchase_given_visit_prob=0.02,
        coupon_open_prob=0.20,
        coupon_redeem_prob=0.05,
        avg_order_mean=48000,
        avg_order_std=12000,
        churn_sensitivity=1.42,
        price_sensitivity=0.72,
        recovery_prob=0.09,
        treatment_lift=-0.05,
        acquisition_weight=0.15,
    ),
}


def get_persona_names():
    return list(DEFAULT_PERSONAS.keys())

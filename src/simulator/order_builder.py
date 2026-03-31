from __future__ import annotations

import numpy as np
import pandas as pd


def _random_times_for_day(
    date: pd.Timestamp,
    n: int,
    rng: np.random.Generator,
    start_hour: int = 9,
    end_hour: int = 22,
) -> pd.Series:
    if n == 0:
        return pd.Series([], dtype="datetime64[ns]")
    total_minutes = (end_hour - start_hour) * 60
    offsets = rng.integers(0, total_minutes, size=n)
    return pd.Series(pd.Timestamp(date.normalize()) + pd.to_timedelta(start_hour * 60 + offsets, unit="m"))


def build_orders(
    customers: pd.DataFrame,
    purchase_mask: np.ndarray,
    date: pd.Timestamp,
    day_idx: int,
    order_sequence_start: int,
    coupon_open_mask: np.ndarray,
    coupon_cost_lookup: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create one order row per purchase event.
    """
    idx = np.flatnonzero(purchase_mask)
    if len(idx) == 0:
        return pd.DataFrame(
            columns=[
                "order_id",
                "customer_id",
                "order_time",
                "item_category",
                "quantity",
                "gross_amount",
                "discount_amount",
                "net_amount",
                "coupon_used",
            ]
        )

    customer_subset = customers.iloc[idx].reset_index(drop=True)
    quantity = np.clip(
        np.round(rng.normal(customer_subset["basket_size_preference"], 0.65)).astype(int),
        1,
        6,
    )
    gross = np.clip(
        rng.normal(customer_subset["avg_order_value_mean"], customer_subset["avg_order_value_std"]),
        15000,
        None,
    )
    coupon_used = coupon_open_mask[idx].astype(int)
    discount_amount = coupon_used * coupon_cost_lookup[idx]
    net_amount = np.maximum(gross - discount_amount, 5000)

    order_time = _random_times_for_day(date, len(idx), rng, start_hour=10, end_hour=22)
    categories = rng.choice(
        ["fashion", "beauty", "electronics", "home", "grocery", "sports"],
        size=len(idx),
        p=[0.20, 0.16, 0.18, 0.18, 0.16, 0.12],
    )

    order_ids = [f"ORD-{day_idx:03d}-{order_sequence_start + i:07d}" for i in range(len(idx))]

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_subset["customer_id"].astype(int),
            "order_time": order_time.to_numpy(),
            "item_category": categories,
            "quantity": quantity.astype(int),
            "gross_amount": gross.astype(float),
            "discount_amount": discount_amount.astype(float),
            "net_amount": net_amount.astype(float),
            "coupon_used": coupon_used.astype(int),
        }
    )

    return orders

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .event_rules import (
    compute_add_to_cart_probability,
    compute_browse_probability,
    compute_coupon_open_probability,
    compute_coupon_redeem_probability,
    compute_purchase_probability,
    compute_remove_cart_probability,
    compute_search_probability,
    compute_visit_probability,
)
from .order_builder import build_orders
from .state_tracker import StateTracker


def _event_frame(
    customer_ids: np.ndarray,
    date: pd.Timestamp,
    event_type: str,
    rng: np.random.Generator,
    item_category: Optional[np.ndarray] = None,
    quantity: Optional[np.ndarray] = None,
    session_id_prefix: str = "SES",
) -> pd.DataFrame:
    n = len(customer_ids)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "event_id",
                "customer_id",
                "timestamp",
                "event_type",
                "session_id",
                "item_category",
                "quantity",
            ]
        )

    minute_offsets = rng.integers(8 * 60, 22 * 60, size=n)
    timestamp = pd.Timestamp(date.normalize()) + pd.to_timedelta(minute_offsets, unit="m")
    session_seed = rng.integers(10_000_000, 99_999_999, size=n)

    return pd.DataFrame(
        {
            "event_id": [f"EVT-{event_type[:3].upper()}-{int(x)}" for x in session_seed],
            "customer_id": customer_ids.astype(int),
            "timestamp": timestamp.to_numpy(),
            "event_type": event_type,
            "session_id": [f"{session_id_prefix}-{int(x)}" for x in session_seed],
            "item_category": item_category if item_category is not None else None,
            "quantity": quantity if quantity is not None else None,
        }
    )


def simulate_events(
    customers: pd.DataFrame,
    assignments: pd.DataFrame,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate the full customer journey and return raw tables:
    - events
    - orders
    - campaign_exposures
    - state_snapshots
    - final_state_metrics
    """
    rng = rng or np.random.default_rng(config.random_seed)

    sim = customers.merge(assignments, on="customer_id", how="left").sort_values("customer_id").reset_index(drop=True)
    tracker = StateTracker(n_customers=len(sim))
    coupon_cost_lookup = sim["coupon_cost"].to_numpy()

    event_frames: List[pd.DataFrame] = []
    order_frames: List[pd.DataFrame] = []
    exposure_frames: List[pd.DataFrame] = []
    snapshot_frames: List[pd.DataFrame] = []

    order_seq = 1
    exposure_seq = 1

    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    signup_dates = pd.to_datetime(sim["signup_date"]).to_numpy()
    assigned_at_days = (pd.to_datetime(sim["assigned_at"]) - pd.Timestamp(config.start_date)).dt.days.to_numpy()

    for day_idx, date in enumerate(dates):
        tracker.start_day()
        active_mask = signup_dates <= np.datetime64(date)

        eligible_exposure = (
            active_mask
            & (sim["treatment_flag"].to_numpy() == 1)
            & (tracker.days_since_last_coupon >= config.coupon_cooldown_days)
            & (tracker.exposures_total < config.max_exposures_per_customer)
            & (
                (tracker.inactivity_days >= config.coupon_trigger_inactivity_days)
                | (assigned_at_days == day_idx)
            )
        )
        exposure_prob = np.clip(
            0.18 + 0.30 * sim["coupon_affinity"].to_numpy() + 0.10 * (tracker.inactivity_days >= config.coupon_trigger_inactivity_days),
            0.0,
            0.95,
        )
        exposure_mask = eligible_exposure & (rng.random(len(sim)) < exposure_prob)
        tracker.record_exposure(exposure_mask)

        if exposure_mask.any():
            exposure_ids = [f"EXP-{day_idx:03d}-{exposure_seq + i:07d}" for i in range(int(exposure_mask.sum()))]
            exposure_seq += int(exposure_mask.sum())
            exposure_times = pd.Timestamp(date.normalize()) + pd.to_timedelta(rng.integers(9 * 60, 17 * 60, size=int(exposure_mask.sum())), unit="m")
            exposure_frames.append(
                pd.DataFrame(
                    {
                        "exposure_id": exposure_ids,
                        "customer_id": sim.loc[exposure_mask, "customer_id"].to_numpy().astype(int),
                        "exposure_time": exposure_times.to_numpy(),
                        "campaign_type": sim.loc[exposure_mask, "campaign_type"].to_numpy(),
                        "coupon_cost": sim.loc[exposure_mask, "coupon_cost"].to_numpy().astype(int),
                    }
                )
            )

        coupon_open_prob = compute_coupon_open_probability(sim, exposure_mask, tracker)
        coupon_open_mask = exposure_mask & (rng.random(len(sim)) < coupon_open_prob)
        tracker.record_coupon_open(coupon_open_mask)

        visit_prob = compute_visit_probability(sim, tracker, active_mask, date)
        visit_mask = rng.random(len(sim)) < visit_prob
        tracker.record_visit(visit_mask, day_idx)

        browse_prob = compute_browse_probability(sim, visit_mask, tracker)
        browse_mask = visit_mask & (rng.random(len(sim)) < browse_prob)

        search_prob = compute_search_probability(sim, visit_mask, tracker)
        search_mask = visit_mask & (rng.random(len(sim)) < search_prob)

        add_cart_prob = compute_add_to_cart_probability(sim, browse_mask, search_mask, tracker)
        add_to_cart_mask = browse_mask & (rng.random(len(sim)) < add_cart_prob)
        tracker.record_cart_add(add_to_cart_mask)

        purchase_prob = compute_purchase_probability(sim, visit_mask, add_to_cart_mask, coupon_open_mask, tracker)
        purchase_mask = visit_mask & (rng.random(len(sim)) < purchase_prob)

        coupon_redeem_prob = compute_coupon_redeem_probability(sim, coupon_open_mask, purchase_mask)
        coupon_redeem_mask = coupon_open_mask & purchase_mask & (rng.random(len(sim)) < coupon_redeem_prob)
        tracker.record_coupon_redeem(coupon_redeem_mask)

        remove_cart_prob = compute_remove_cart_probability(sim, add_to_cart_mask, purchase_mask, tracker)
        remove_cart_mask = add_to_cart_mask & (rng.random(len(sim)) < remove_cart_prob)
        tracker.record_cart_remove(remove_cart_mask)

        support_prob = np.clip(
            sim["support_contact_propensity"].to_numpy() + 0.05 * remove_cart_mask.astype(float) + 0.03 * (tracker.inactivity_days > 20),
            0.0,
            0.55,
        )
        support_mask = visit_mask & (rng.random(len(sim)) < support_prob)

        event_frames.append(_event_frame(sim.loc[visit_mask, "customer_id"].to_numpy(), date, "visit", rng))
        event_frames.append(_event_frame(sim.loc[browse_mask, "customer_id"].to_numpy(), date, "page_view", rng))
        event_frames.append(_event_frame(sim.loc[search_mask, "customer_id"].to_numpy(), date, "search", rng))
        event_frames.append(_event_frame(sim.loc[add_to_cart_mask, "customer_id"].to_numpy(), date, "add_to_cart", rng))
        event_frames.append(_event_frame(sim.loc[remove_cart_mask, "customer_id"].to_numpy(), date, "remove_from_cart", rng))
        event_frames.append(_event_frame(sim.loc[coupon_open_mask, "customer_id"].to_numpy(), date, "coupon_open", rng))
        event_frames.append(_event_frame(sim.loc[coupon_redeem_mask, "customer_id"].to_numpy(), date, "coupon_redeem", rng))
        event_frames.append(_event_frame(sim.loc[support_mask, "customer_id"].to_numpy(), date, "support_contact", rng))
        event_frames.append(_event_frame(sim.loc[purchase_mask, "customer_id"].to_numpy(), date, "purchase", rng))

        orders = build_orders(
            customers=sim,
            purchase_mask=purchase_mask,
            date=date,
            day_idx=day_idx,
            order_sequence_start=order_seq,
            coupon_open_mask=coupon_redeem_mask,
            coupon_cost_lookup=coupon_cost_lookup,
            rng=rng,
        )
        if not orders.empty:
            order_seq += len(orders)
            order_frames.append(orders)
            amount_lookup = orders.set_index("customer_id")["net_amount"].to_dict()
            order_amounts = np.zeros(len(sim), dtype=float)
            purchase_idx = np.flatnonzero(purchase_mask)
            for idx in purchase_idx:
                cid = int(sim.iloc[idx]["customer_id"])
                order_amounts[idx] = amount_lookup.get(cid, 0.0)
            tracker.record_purchase(purchase_mask, order_amounts, day_idx)

        if (day_idx % config.snapshot_frequency_days == 0) or (day_idx == len(dates) - 1):
            snapshot_frames.append(
                tracker.to_snapshot(
                    customers=sim[["customer_id"]],
                    snapshot_date=date,
                    day_idx=day_idx,
                    dormant_threshold=config.dormant_inactivity_days,
                    churn_threshold=config.churn_inactivity_days,
                )
            )

    events = pd.concat([df for df in event_frames if not df.empty], ignore_index=True) if event_frames else pd.DataFrame()
    orders = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()
    exposures = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    state_snapshots = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    final_state = tracker.final_metrics(simulation_days=config.simulation_days)

    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"])
    if not orders.empty:
        orders["order_time"] = pd.to_datetime(orders["order_time"])
    if not exposures.empty:
        exposures["exposure_time"] = pd.to_datetime(exposures["exposure_time"])
    if not state_snapshots.empty:
        state_snapshots["snapshot_date"] = pd.to_datetime(state_snapshots["snapshot_date"])
        state_snapshots["last_visit_date"] = pd.to_datetime(state_snapshots["last_visit_date"])
        state_snapshots["last_purchase_date"] = pd.to_datetime(state_snapshots["last_purchase_date"])

    return events, orders, exposures, state_snapshots, final_state

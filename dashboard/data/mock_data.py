import numpy as np
import pandas as pd


def generate_mock_customers(n_customers: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)

    customer_ids = np.arange(1, n_customers + 1)

    persona = rng.choice(
        ["vip", "coupon_sensitive", "churn_risk", "sure_thing", "lost_cause"],
        size=n_customers,
        p=[0.15, 0.25, 0.25, 0.20, 0.15]
    )

    acquisition_month = rng.choice(
        pd.date_range("2025-01-01", "2025-06-01", freq="MS").strftime("%Y-%m").tolist(),
        size=n_customers
    )

    recency_days = rng.integers(1, 90, size=n_customers)
    frequency = rng.integers(1, 30, size=n_customers)
    monetary = rng.integers(30000, 1200000, size=n_customers)

    visits_last_7 = rng.integers(0, 15, size=n_customers)
    visits_prev_7 = rng.integers(0, 15, size=n_customers)
    purchase_last_30 = rng.integers(0, 8, size=n_customers)
    purchase_prev_30 = rng.integers(0, 8, size=n_customers)

    visit_change_rate = (visits_last_7 - visits_prev_7) / np.maximum(visits_prev_7, 1)
    purchase_change_rate = (purchase_last_30 - purchase_prev_30) / np.maximum(purchase_prev_30, 1)

    base_churn = (
        0.35 * (recency_days / 90)
        + 0.20 * (1 - np.minimum(frequency / 30, 1))
        + 0.15 * (1 - np.minimum(monetary / 1200000, 1))
        + 0.15 * (visit_change_rate < 0).astype(int)
        + 0.15 * (purchase_change_rate < 0).astype(int)
    )

    persona_boost = []
    for p in persona:
        if p == "vip":
            persona_boost.append(-0.12)
        elif p == "coupon_sensitive":
            persona_boost.append(0.04)
        elif p == "churn_risk":
            persona_boost.append(0.15)
        elif p == "sure_thing":
            persona_boost.append(-0.08)
        else:
            persona_boost.append(0.18)

    churn_probability = np.clip(base_churn + np.array(persona_boost), 0.01, 0.99)

    uplift_score = []
    for p in persona:
        if p == "vip":
            uplift_score.append(rng.uniform(0.02, 0.08))
        elif p == "coupon_sensitive":
            uplift_score.append(rng.uniform(0.15, 0.35))
        elif p == "churn_risk":
            uplift_score.append(rng.uniform(0.08, 0.22))
        elif p == "sure_thing":
            uplift_score.append(rng.uniform(0.00, 0.05))
        else:
            uplift_score.append(rng.uniform(-0.08, 0.03))

    uplift_score = np.array(uplift_score)

    clv = (
        monetary * rng.uniform(1.2, 2.8, size=n_customers)
        + frequency * rng.uniform(5000, 30000, size=n_customers)
    )

    coupon_cost = rng.integers(3000, 15000, size=n_customers)
    expected_incremental_profit = np.maximum(clv * uplift_score, -50000)
    expected_roi = (expected_incremental_profit - coupon_cost) / np.maximum(coupon_cost, 1)

    segment = []
    for u in uplift_score:
        if u >= 0.15:
            segment.append("Persuadables")
        elif 0.05 <= u < 0.15:
            segment.append("Sure Things")
        elif -0.02 <= u < 0.05:
            segment.append("Lost Causes")
        else:
            segment.append("Sleeping Dogs")

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "persona": persona,
        "acquisition_month": acquisition_month,
        "recency_days": recency_days,
        "frequency": frequency,
        "monetary": monetary,
        "visits_last_7": visits_last_7,
        "visits_prev_7": visits_prev_7,
        "visit_change_rate": visit_change_rate,
        "purchase_last_30": purchase_last_30,
        "purchase_prev_30": purchase_prev_30,
        "purchase_change_rate": purchase_change_rate,
        "churn_probability": churn_probability,
        "uplift_score": uplift_score,
        "clv": clv,
        "coupon_cost": coupon_cost,
        "expected_incremental_profit": expected_incremental_profit,
        "expected_roi": expected_roi,
        "uplift_segment": segment
    })

    return customers


def generate_mock_cohort_retention(seed: int = 42):
    rng = np.random.default_rng(seed)

    cohorts = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    periods = list(range(0, 7))

    rows = []
    for cohort in cohorts:
        base = rng.uniform(0.82, 0.95)
        decay = rng.uniform(0.08, 0.16)
        for p in periods:
            retention = max(base - decay * p + rng.uniform(-0.03, 0.03), 0.12)
            rows.append({
                "cohort_month": cohort,
                "period": p,
                "retention_rate": retention
            })

    return pd.DataFrame(rows)


def allocate_budget(customers: pd.DataFrame, budget: int):
    ranked = customers.copy()
    ranked["target_score"] = ranked["clv"] * ranked["uplift_score"]
    ranked = ranked.sort_values(["target_score", "expected_roi"], ascending=False).reset_index(drop=True)

    selected_rows = []
    spent = 0

    for _, row in ranked.iterrows():
        if row["uplift_score"] <= 0:
            continue
        if spent + row["coupon_cost"] <= budget:
            spent += row["coupon_cost"]
            selected_rows.append(row)

    if len(selected_rows) == 0:
        selected = ranked.head(0).copy()
    else:
        selected = pd.DataFrame(selected_rows)

    total_profit = selected["expected_incremental_profit"].sum() if not selected.empty else 0
    overall_roi = ((total_profit - spent) / spent) if spent > 0 else 0

    summary = {
        "budget": budget,
        "spent": spent,
        "remaining": budget - spent,
        "num_targeted": len(selected),
        "expected_incremental_profit": total_profit,
        "overall_roi": overall_roi
    }

    return selected, summary


def budget_allocation_by_segment(selected: pd.DataFrame):
    if selected.empty:
        return pd.DataFrame(columns=["uplift_segment", "customer_count", "allocated_budget", "expected_profit"])

    grouped = (
        selected.groupby("uplift_segment", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            allocated_budget=("coupon_cost", "sum"),
            expected_profit=("expected_incremental_profit", "sum")
        )
        .sort_values("allocated_budget", ascending=False)
    )
    return grouped
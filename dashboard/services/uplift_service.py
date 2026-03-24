import pandas as pd


def get_top_high_value_customers(customers: pd.DataFrame, top_n: int = 20):
    df = customers.copy()
    df["value_score"] = df["clv"] * df["uplift_score"]
    df = df.sort_values(["value_score", "clv"], ascending=False)
    return df.head(top_n)


def get_retention_targets(customers: pd.DataFrame, threshold: float, top_n: int = 30):
    df = customers.copy()

    condition = (
        (df["churn_probability"] >= threshold) &
        (df["uplift_score"] > 0.08) &
        (df["clv"] > df["clv"].median()) &
        (df["uplift_segment"] != "Sleeping Dogs")
    )

    target = df[condition].copy()
    target["priority_score"] = (
        0.45 * target["churn_probability"] +
        0.25 * target["uplift_score"] +
        0.30 * (target["clv"] / target["clv"].max())
    )

    target = target.sort_values("priority_score", ascending=False)
    return target.head(top_n)
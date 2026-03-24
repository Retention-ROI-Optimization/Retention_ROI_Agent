import pandas as pd


def get_churn_status(customers: pd.DataFrame, threshold: float):
    df = customers.copy()
    df["is_churn_risk"] = df["churn_probability"] >= threshold

    summary = {
        "total_customers": len(df),
        "at_risk_customers": int(df["is_churn_risk"].sum()),
        "risk_rate": float(df["is_churn_risk"].mean()),
        "avg_churn_prob": float(df["churn_probability"].mean())
    }

    risk_customers = df[df["is_churn_risk"]].sort_values(
        ["churn_probability", "clv"], ascending=[False, False]
    )

    return summary, risk_customers
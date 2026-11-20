
# Counterfactual Retention Lab

## Overview

The existing retention dashboard provides insights into customer churn risk, recommended retention actions, and budget allocation targets.

The **Counterfactual Retention Lab** extends these capabilities by comparing the expected net profit of different intervention scenarios for the same customer.

The scenarios include:

- No intervention
- Offer a KRW 5,000 coupon
- Customer support call
- Push notification or email
- Wait for 7 days

The goal of this view is not only to recommend the most appropriate action but also to explain **why a particular action is economically preferable to taking no action or choosing an alternative intervention**.

## Calculation Logic

The core calculation engine is implemented in `src/optimization/counterfactual.py`.

The fundamental calculations are as follows:

```text
Expected Net Profit (No Intervention)
    = CLV × (1 - Current Churn Probability)

Expected Net Profit (With Intervention)
    = CLV × (1 - Post-Intervention Churn Probability)
      - Intervention Cost

Incremental Net Profit
    = Expected Net Profit (With Intervention)
      - Expected Net Profit (No Intervention)
```

The post-intervention churn probability is estimated using existing customer features and model outputs, including:

- `churn_probability`
- `uplift_score`
- `CLV`
- `survival/timing`
- `coupon_affinity`
- `price_sensitivity`
- `discount_pressure_score`

When survival analysis artifacts are available, the estimation incorporates the predicted time to churn and urgency-based weighting.

If survival artifacts are unavailable, the system falls back to conservative default values to ensure that calculations remain operational.

## Important Considerations

These values are **model-based counterfactual estimates**, not actual outcomes observed after executing retention actions.

Therefore, reliable operational decision-making requires an additional validation process:

1. Include customers with low or medium prediction confidence in A/B tests or holdout groups.
2. Record actual intervention logs, including notifications sent, support calls, and coupon redemptions.
3. Compare realized revenue and retention rates between treatment and control groups to calculate incremental ROI.
4. Incorporate validation results into the uplift models and action selection policies to continuously improve recommendation accuracy.

## Added Files

- `src/optimization/counterfactual.py`: Counterfactual profit estimation engine
- `dashboard/services/counterfactual_service.py`: Service wrapper for dashboard integration
- `dashboard/app.py`: New dashboard view, **"Customer-Level Retention Strategy Comparison"**

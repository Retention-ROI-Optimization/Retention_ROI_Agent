
# Decision Logic

Retention ROI Agent does not directly translate customer churn prediction scores into sales or marketing actions.

Instead, it combines multiple signals to determine **which customers to target, how intensively to intervene, when to intervene, the expected incremental net profit, and the estimated ROI**.

---

## 1. Core Philosophy

Customers with high churn probabilities are not necessarily the most valuable retention targets.

- If a customer is highly likely to churn but unlikely to respond to an intervention, the expected benefit may not justify the cost.
- If a customer has high churn risk but low Customer Lifetime Value (CLV), the potential financial benefit of retention may be limited.
- Even if a customer has high CLV, an expensive intervention may be unnecessary when the predicted churn date is still far away.
- Continuously offering coupons to customers already exposed to excessive discounts may reduce long-term profitability.

Therefore, the platform combines the following signals:

```text
Churn Probability
+ Uplift Score
+ Customer Lifetime Value
+ Intervention Cost
+ Survival Timing
+ Coupon Fatigue / Discount Pressure
+ Budget Constraint
```

The objective is not simply to identify customers who are likely to churn, but to identify customers for whom retention interventions are expected to generate positive incremental value.

---

## 2. Core Formulas

For readability, the README presents the underlying calculations in a simplified, business-oriented form rather than exposing every implementation detail.

```text
Expected Incremental Profit
    = Customer Value × Response Potential × Churn Risk
      - Intervention Cost

Expected ROI
    = Expected Incremental Profit ÷ Intervention Cost
```

In the actual candidate selection process, additional factors and constraints are incorporated, including:

- Survival timing
- Intervention intensity
- Segment-specific conditions
- Budget cap
- Maximum target customer cap

These factors ensure that candidate selection considers operational feasibility alongside predicted profitability.

---

## 3. Candidate Customer Filtering

Before entering the budget optimization process, customers must satisfy several eligibility conditions.

- Churn probability must be greater than or equal to the user-defined threshold.
- Uplift score must be positive.
- Expected incremental profit must be positive.
- Intervention cost must be positive.
- Customers belonging to potentially harmful treatment segments, such as **Sleeping Dogs**, are excluded.

This filtering process is designed to identify customers for whom spending money on retention is economically justified, rather than simply selecting customers with high churn risk.

---

## 4. Intervention Intensity Candidate Generation

Each customer can have multiple intervention candidates rather than a single predetermined action.

These candidates represent different intervention intensity levels, such as `low`, `mid`, and `high`.

For example, the same customer may have the following options:

- **Low:** Low-cost messages or push notifications
- **Mid:** Moderate incentives or CRM outreach
- **High:** Customer consultations, VIP support, or high-cost incentives

For each candidate, the system calculates:

- Intervention cost
- Expected incremental profit
- Expected ROI
- Timing urgency
- Intervention intensity fit

The budget optimization engine then selects the final combination from the available **customer × intervention intensity** candidates.

---

## 5. Incorporating Survival Timing

When survival analysis results are available, customer-specific churn timing information is incorporated into budget optimization.

The key derived variables are as follows:

| Variable | Description |
| --- | --- |
| `predicted_median_time_to_churn_days` | Estimated time until the survival probability falls to 50% or below |
| `short_term_churn_probability` | Estimated probability of churn within a short-term prediction window |
| `timing_urgency_score` | Urgency score combining risk percentile, short-term churn probability, and predicted time to churn |
| `recommended_intervention_window` | Human-readable intervention window, such as within 14 days, 15–30 days, or 31–60 days |
| `timing_priority_bucket` | Operational priority category: `immediate`, `near_term`, `planned`, or `monitor` |

Timing information contributes to both priority scoring and intervention comparison.

As a result, two customers with similar expected ROI may receive different priorities if one is significantly more likely to churn in the immediate future.

This enables the platform to optimize not only **who should receive an intervention**, but also **when that intervention should occur**.

---

## 6. Budget Optimization Strategy

The final selection process uses a greedy optimization algorithm subject to several operational constraints.

### Optimization Constraints

- Total intervention cost must not exceed the user-defined budget.
- Each customer can receive at most one final intervention.
- The total number of targeted customers must not exceed the maximum target cap.
- High-intensity interventions must not exceed a predefined proportion of the overall target cap.
- Only candidates with positive expected incremental profit are eligible.

### Candidate Ranking Signals

Candidate prioritization incorporates the following signals:

- Expected ROI rank
- Expected incremental profit rank
- Churn probability
- Uplift score
- CLV rank
- Timing urgency
- Recommended intervention window
- Intervention intensity fit

This approach provides a more balanced selection strategy than ranking customers by expected ROI alone.

For example, it helps mitigate situations where:

- A candidate has high ROI but contributes very little absolute profit.
- A customer has high expected profitability but does not require immediate intervention.
- Too many high-cost interventions are selected at the expense of broader customer coverage.

The resulting strategy balances **profitability, intervention urgency, resource allocation, and operational constraints**.

---

## 7. Budget Sensitivity Map

The budget allocation view does not evaluate only the currently selected budget.

Instead, it generates a budget grid using increments of KRW 1 million to show how optimization results change as the available budget increases or decreases.

### Calculated Metrics

The sensitivity table includes:

- `budget`
- `spent`
- `remaining`
- `target_count`
- `expected_incremental_profit`
- `average_roi`
- `added_budget`
- `added_spend`
- `added_target_count`
- `added_profit`
- `marginal_profit_per_1m`
- `marginal_roi`
- `budget_status`
- `operator_message`

### Budget Efficiency Interpretation

| Budget Status | Interpretation |
| --- | --- |
| Expansion Worth Considering | The efficiency of additional spending is comparable to or higher than the current average |
| Gradual Expansion Possible | Additional spending remains beneficial, but its efficiency is lower than the current average |
| ROI Decline Begins | The number of targeted customers increases, but average ROI begins to decline |
| Low Efficiency | Additional budget generates relatively little incremental profit |
| Saturation or Potential Waste | Increasing the budget produces few or no additional targets or profits |

### Decision-Making Value

This analysis allows operators to evaluate:

- Whether increasing the budget is economically justified
- How much additional profit can be expected from incremental spending
- Where diminishing returns begin
- Whether the current budget is approaching saturation
- When further budget expansion should be reconsidered

This feature is particularly valuable in competitions, demonstrations, and business presentations.

Rather than merely demonstrating that a predictive model has been developed, it shows that the platform can support **practical budget allocation decisions based on expected business outcomes**.

---

## 8. Counterfactual Retention Lab

The Counterfactual Retention Lab compares multiple intervention scenarios for individual customers.

### Intervention Scenarios

- No intervention
- KRW 5,000 incentive
- Customer support call
- Push notification or email
- Wait for 7 days

### Core Calculation Logic

```text
Expected Net Profit (No Intervention)
    = CLV × (1 - Current Churn Probability)

Expected Net Profit (With Intervention)
    = CLV × (1 - Post-Intervention Churn Probability)
      - Intervention Cost

Incremental Net Profit vs. No Intervention
    = Expected Net Profit (With Intervention)
      - Expected Net Profit (No Intervention)
```

The post-intervention churn probability is estimated using signals such as:

- Churn probability
- Uplift score
- Timing urgency
- Coupon affinity
- Price sensitivity
- Discount pressure

### Interpretation and Limitations

These outputs represent **model-based counterfactual estimates**, not experimentally verified causal effects.

They provide decision-support estimates of how different actions might affect customer retention and profitability.

In production environments, the estimated effects should be validated through controlled experiments, such as A/B tests or holdout groups.

---

## 9. Personalized Recommendation Logic

Personalized recommendations are not generated for every customer in the dataset.

Instead, recommendations are generated only for customers selected as final retention targets under the current budget and churn threshold constraints.

### Recommendation Signals

The recommendation score combines:

- Individual purchase and transaction history
- Recent browsing and exploration signals
- Preferences of customers within the same persona or uplift segment
- Global popularity signals across the customer base
- Current targeting priority and expected ROI

### Domain-Specific Adaptation

In Financial Mode, general e-commerce categories are translated into financial product categories, such as:

- Deposits and savings
- Loans
- Credit cards
- Insurance
- Wealth management

Recommendation explanations are also adapted to financial-domain terminology.

This ensures that both recommendations and their supporting explanations remain relevant to the selected industry.

---

## 10. Live Action Queue Logic

In Live DB Mode, batch analysis artifacts are first seeded into PostgreSQL tables.

When new customer events arrive, the system executes the following processing workflow:

```text
POST /api/v1/user-live/events
    ↓
Store Incoming Event
    ↓
Update Customer Feature State
    ↓
Rescore Affected Customer
    ↓
Update Churn Score / CLV / Uplift / Expected ROI
    ↓
Evaluate Action Queue Eligibility Conditions
    ↓
Create or Update Queued Action
```

The dashboard displays the resulting records in the **Live Action Queue** table.

For each customer, operators can inspect:

- Recommended action
- Intervention intensity
- Expected profit
- Expected ROI
- Action status
- Trigger reason

This workflow connects batch-trained predictive models with event-driven operational decision-making.

As customer behavior changes, the system can update risk estimates and intervention priorities accordingly.

---

## 11. LLM Summarization and Question-Answering Logic

The LLM does not receive the entire raw dataset.

Instead, the system constructs a compact, view-specific summary payload containing the information relevant to the currently selected dashboard screen.

### Supported Capabilities

- Summarizing key metrics from the current view
- Explaining high-risk customer segments
- Suggesting adjustments to budget and churn thresholds
- Interpreting recommendation results and action queue entries
- Providing responses in Korean, English, and Japanese

### Context Management

The chatbot preserves the context of the dashboard view in which the conversation was initially opened.

Users can explicitly refresh the chatbot context to reflect the currently selected view when needed.

This design prevents the conversational context from changing unexpectedly during dashboard navigation.

It also allows users to maintain a consistent analytical discussion while moving between different operational views.

---

## 12. Validation and Limitations

Expected ROI, uplift estimates, and counterfactual outcomes are predictive indicators designed to support operational decision-making.

They should not be interpreted as definitive measurements of realized business impact.

To establish actual incremental business value, the following validation loop is required.

### Validation Workflow

1. **Create Holdout Groups**
   - Reserve a subset of final target customers as an untreated control group.

2. **Collect Intervention and Outcome Logs**
   - Record actual message delivery, customer consultations, coupon redemptions, purchases, and repeat visits.

3. **Measure Incremental Impact**
   - Compare observed retention rates and revenue between treatment and control groups.

4. **Update Models and Action Policies**
   - Incorporate validated incremental effects into the uplift model and retention action selection policy.

### Long-Term Objective

The platform is designed to support the following continuous improvement cycle:

```text
Predict Customer Churn and Value
    ↓
Estimate Intervention Outcomes
    ↓
Optimize Budget and Select Actions
    ↓
Execute Retention Interventions
    ↓
Measure Actual Incremental Impact
    ↓
Update Uplift Models and Action Policies
    ↓
Repeat
```

The strength of Retention ROI Agent lies not in claiming that predicted ROI or counterfactual estimates are guaranteed outcomes, but in explicitly recognizing their limitations and designing a workflow that connects **prediction, economic optimization, operational execution, and empirical validation**.

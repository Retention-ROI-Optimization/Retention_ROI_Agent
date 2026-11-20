
# Dashboard Decision Loop

This document explains how the Retention ROI Agent dashboard supports user decision-making through a structured workflow. Rather than simply displaying charts, the dashboard establishes an operational loop:

**Data Upload → Analysis → Budget Allocation → Action Comparison → Recommendations → Real-Time Action Queue**

---

## 1. Onboarding Wizard

### 1.1. Industry Mode Selection

The dashboard initially prompts users to select either Financial Mode or E-commerce Mode.

| Mode | Data Interpretation | Example Decisions |
| --- | --- | --- |
| Financial Mode | Deposits, loans, credit cards, transactions, account balances, delinquencies, and customer support history | Determine the priority of consultations, incentives, and product recommendations for customers at risk of account closure or inactivity |
| E-commerce Mode | Visits, searches, shopping carts, purchases, coupons, and category preferences | Determine how to deploy coupons, personalized recommendations, and push notifications to encourage repeat visits and purchases |

Column labels, event semantics, recommendation categories, and dashboard descriptions are adapted to the selected industry mode.

### 1.2. CSV/TSV Upload

The dashboard accepts datasets structured as customer snapshots, transaction tables, or event logs.

When sufficient event timestamps and `event_type` information are available, the system can perform more comprehensive behavioral time-series analysis, survival analysis, and live event replay.

### 1.3. Automatic Column Role Detection

Uploaded columns are automatically mapped to roles within the internal analytics schema.

Supported roles include:

- `customer_id`
- `timestamp`
- `event_type`
- `amount`
- `category`
- `churn_flag`
- `persona`
- `quantity`
- `region`

Users can review and modify the automatically detected mappings through an interactive table editor.

Rather than relying exclusively on full automation, the system follows a **human-in-the-loop approach that combines automatic suggestions with manual verification**.

### 1.4. Event Value Standardization

Raw event values may vary across datasets, including:

- `view`
- `visit`
- `product_click`
- `transaction`
- `purchase`
- `loan_repay`

These values are mapped to standardized internal event types.

This standardization ensures that downstream modeling, recommendation generation, and real-time event APIs operate using a consistent event taxonomy.

### 1.5. Churn Definition and Model Training

Users can define churn by adjusting the inactivity threshold, expressed as the number of consecutive days without customer activity.

This definition is used to establish churn labels and serves as the basis for:

- Churn prediction models
- Survival analysis
- Churn timing prediction

---

## 2. Seven Core Dashboard Views

The dashboard consists of seven primary analytical and operational views.

```text
1. Churn Overview
2. Churn Timing Prediction
3. Budget Allocation & Target Customers
4. Customer-Level Retention Strategy Comparison
5. Personalized Recommendations
6. Real-Time Operations Monitor
7. Weekly Action Performance Review
```

Together, these views support a continuous decision-making workflow, from identifying at-risk customers to evaluating intervention outcomes.

---

## 3. View 1: Churn Overview

### Objective

This view helps operators assess the overall scale of customer churn risk across the customer base.

### Key Information

- Total number of customers
- Number of at-risk customers
- Proportion of at-risk customers
- Average churn probability
- At-risk customer table
- Cohort retention analysis
- Overview of uplift and CLV segments

### Decision-Making Role

This view primarily serves a diagnostic purpose rather than directly determining retention actions.

It provides the foundation for selecting appropriate churn thresholds and identifying priority customer segments in subsequent budget allocation and recommendation views.

---

## 4. View 2: Churn Timing Prediction

### Objective

This view identifies not only **which customers are at risk of churning, but also when interventions should be initiated**.

### Key Outputs

- Predicted time to churn
- Estimated churn date
- Probability of churn within 30 days
- Expected financial loss within 30 days
- Survival probabilities by risk group
- List of customers with high operational priority

### Decision-Making Role

When the retention budget is limited, intervention timing can be as important as CLV and expected ROI.

For example, a customer with high CLV but an estimated churn date more than 60 days away may receive lower immediate priority than a customer expected to churn within the next seven days.

This enables operators to prioritize interventions based on both **economic value and temporal urgency**.

---

## 5. View 3: Budget Allocation & Target Customers

### Objective

This is one of the platform's core decision-making views.

The system dynamically selects retention targets based on the operator-defined budget, churn threshold, and maximum number of target customers.

### Key Outputs

- Total marketing budget
- Allocated budget
- Remaining budget
- Final number of targeted customers
- Expected net profit
- Overall ROI
- Budget allocation by customer segment
- Customer counts by intervention intensity
- Budget sensitivity analysis table
- Budget saturation range
- Low-efficiency budget range

### Decision-Making Role

This view enables operators to answer critical resource allocation questions:

- Does increasing the budget actually improve expected net profit?
- Is the current retention budget too small or excessively large?
- At what point does marginal ROI decline significantly?
- Which customer segments and intervention intensity levels receive the largest budget allocations?
- Does reducing the target customer cap retain only the most economically efficient customers?

By exposing these relationships, the dashboard supports **ROI-driven budget allocation rather than indiscriminate retention spending**.

---

## 6. View 4: Customer-Level Retention Strategy Comparison

### Objective

Instead of presenting a single recommended action, this view compares multiple intervention scenarios for the same customer.

### Intervention Scenarios

- No intervention
- KRW 5,000 incentive
- Customer support call
- Push notification or email
- Wait for 7 days

### Key Outputs

- Expected net profit for each action
- Incremental net profit compared with no intervention
- Intervention cost
- Estimated post-intervention churn probability
- Explanation of the recommended action
- Prediction confidence label

### Decision-Making Role

This view helps answer questions such as:

- Why should this customer receive a support call instead of a coupon?
- Why is waiting preferable to intervening immediately?
- How much incremental value is each intervention expected to generate?
- Is the expected benefit sufficient to justify the intervention cost?

By comparing alternative outcomes, this feature transforms model predictions into **interpretable, economically grounded business decisions**.

---

## 7. View 5: Personalized Recommendations

### Objective

This view provides specific recommendation categories or next-best actions for customers selected as final retention targets.

### Key Design Principle

The view does not simply display previously stored recommendation candidates.

Instead, recommendations are regenerated for customers selected under the current budget, churn threshold, and maximum target constraints.

This ensures that the recommendations remain consistent with the latest operational settings.

### Recommendation Signals

- Individual purchase and transaction history
- Recent interests and browsing behavior
- Preferences of customers within the same persona or uplift segment
- Global popularity signals across the customer base
- Current targeting priority and expected ROI

### Decision-Making Role

Operators can determine not only **which customers should be contacted**, but also **which product categories, products, or incentives should be offered**.

This connects customer targeting with actionable, personalized retention strategies.

---

## 8. View 6: Real-Time Operations Monitor

### Objective

This view demonstrates how batch analytics results can be integrated with an operational database and continuously updated as new customer events arrive.

### Key Features

- PostgreSQL Live DB connection status
- Total event count
- Total customer count
- Number of queued actions
- Timestamp of the latest score update
- Live Action Queue table
- Controls to start, stop, and reset the automated demo event stream
- Configurable ratio of new-customer events to existing-customer events

### Event Processing Workflow

```text
New Customer Event Received
    ↓
Update Customer Feature State
    ↓
Recalculate Churn / CLV / Uplift Scores
    ↓
Recalculate Expected ROI
    ↓
Enqueue Action if Eligibility Conditions Are Met
    ↓
Refresh Dashboard
```

### Decision-Making Role

Rather than relying solely on static CSV analysis, operators can observe how incoming customer behavior affects:

- Customer churn risk
- Predicted customer value
- Retention intervention priorities
- Expected ROI
- Action queue composition

This creates a bridge between offline model predictions and real-time retention operations.

---

## 9. View 7: Weekly Action Performance Review

### Objective

This view is designed to support performance evaluation after retention actions have been executed.

### Purpose

The platform's expected ROI and counterfactual estimates are prediction-based rather than directly observed outcomes.

In real-world operations, weekly performance reviews should assess:

- Proportion of queued actions that were actually executed
- Changes in repeat visits, repeat purchases, completed consultations, and incentive acceptance after interventions
- Performance differences across customer segments
- Potential over-allocation of high-intensity interventions
- Need for budget adjustments in the following week

### Decision-Making Role

This view provides the foundation for a **closed-loop retention optimization process**:

```text
Prediction
    ↓
Execution
    ↓
Performance Review
    ↓
Policy Improvement
    ↓
Updated Prediction
```

By incorporating observed intervention outcomes into future modeling and policy decisions, the platform is designed to support continuous improvement in retention efficiency.

---

## 10. Sidebar Analytics Controls

The sidebar provides shared analytical controls that influence multiple dashboard views simultaneously.

| Control | Description | Affected Views |
| --- | --- | --- |
| Churn Threshold | Defines the probability threshold for identifying at-risk customers | Churn Overview, Budget Allocation, Recommendations, Real-Time Queue |
| Total Marketing Budget | Specifies the budget available for retention interventions | Budget Allocation, Sensitivity Analysis, Target Selection |
| Maximum Target Customers | Limits the number of customers that can be targeted operationally | Budget Allocation, Target Selection, Recommendations |
| Display Customer Count | Controls the number of customer records displayed in preview tables | At-Risk Customers, Real-Time Monitoring, Explanation Tables |
| Recommendations per Customer | Sets the number of personalized recommendations generated for each customer | Personalized Recommendations |
| Language | Controls the language used in the UI, tables, charts, and LLM responses | Korean, English, Japanese |

### State Persistence

Changing the dashboard language does not reset analytical settings such as:

- Churn threshold
- Marketing budget
- Maximum target customer count

These values are preserved to maintain continuity across multilingual dashboard interactions.

---

## 11. Context-Aware LLM Chatbot

The dashboard includes an LLM-powered chatbot that generates responses based on the payload associated with the currently selected dashboard view.

### Supported Capabilities

- Summarizing metrics from the current dashboard view
- Explaining at-risk customers and customer segments
- Suggesting adjustments to budget and churn thresholds
- Interpreting action queue entries and recommendation results
- Providing responses in Korean, English, or Japanese

### Key Design Principle

The chatbot maintains its own conversational context independently of dashboard navigation.

The context associated with the initially opened view remains available even when the user navigates to another dashboard view.

When necessary, users can explicitly refresh the chatbot context to reflect the currently selected view.

This design helps maintain conversational continuity while allowing users to incorporate updated dashboard information when needed.

---

## 12. Summary

The Retention ROI Agent dashboard is more than a collection of customer retention charts.

It functions as a **retention decision-making console that transforms budget constraints into optimized customer targets and intervention strategies, while continuously updating operational action queues as new customer events arrive**.

Its overarching workflow is:

```text
Data Upload
    ↓
Churn & Value Analysis
    ↓
Budget Optimization
    ↓
Counterfactual Action Comparison
    ↓
Personalized Recommendations
    ↓
Real-Time Action Execution Queue
    ↓
Weekly Performance Review
    ↓
Continuous Policy Improvement
```

The ultimate goal is to connect predictive analytics with economically justified actions and operational feedback, enabling a structured, data-driven approach to customer retention management.

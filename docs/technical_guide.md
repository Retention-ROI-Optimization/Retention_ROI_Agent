
# Technical Guide

## 1. Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## 2. Running with Docker

Build and start the Docker services:

```bash
docker compose up --build
```

To run the services in detached mode (background execution):

```bash
docker compose up -d --build
```

## 3. Accessing the Dashboard

Once the services are running, open the Streamlit dashboard in your browser:

```text
http://localhost:8501
```

## 4. Budget, Profit, and ROI Calculation Logic

The platform estimates the economic value of retention interventions using the following simplified formulas:

```text
Expected Incremental Profit
    = Customer Value × Response Potential × Churn Risk
      - Intervention Cost

Expected ROI
    = Expected Incremental Profit / Intervention Cost
```

Customers with higher expected incremental profit and ROI are prioritized for retention interventions.

The optimization process also respects operational constraints, including:

- Total marketing budget
- Maximum number of target customers

This ensures that retention decisions consider both expected financial returns and available operational resources.

---

## 5. Live DB Mode

The platform supports operationalizing uploaded business data through PostgreSQL-based live tables.

After model training is completed, generated artifacts can be seeded into the live database.

When new customer events arrive, the corresponding records and prediction results can be updated.

### 5.1. Core Workflow

1. Start the Docker services.
2. Upload a financial or e-commerce CSV/TSV dataset.
3. Review the column mappings and run model training.
4. Seed the generated artifacts into PostgreSQL live tables.
5. Send customer events through the Live API.
6. Monitor changes in customer scores, recommendations, and action queues through the dashboard.

### 5.2. Health Check

Verify that the Live API is running correctly:

```bash
curl -s "http://localhost:8000/api/v1/user-live/health" | python3 -m json.tool
```

### 5.3. Seed Status Check

Check whether the generated artifacts have been successfully seeded into the live database:

```bash
curl -s "http://localhost:8000/api/v1/user-live/seed-status" | python3 -m json.tool
```

### 5.4. Example: Inserting a Customer Event

Send a new customer event through the Live API:

```bash
curl -X POST "http://localhost:8000/api/v1/user-live/events" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 1001,
    "event_type": "purchase",
    "event_time": "2026-05-10T03:30:00+09:00",
    "amount": 35000,
    "source_event_id": "event-1001-001",
    "channel": "web",
    "raw_payload": {"source": "demo"}
  }' | python3 -m json.tool
```

The event can trigger updates to the customer's feature state, prediction scores, and retention action queue.

### 5.5. Checking Customer Scores

Retrieve the latest prediction scores for a specific customer:

```bash
curl -s "http://localhost:8000/api/v1/user-live/scores?customer_id=1001" | python3 -m json.tool
```

### 5.6. Checking the Action Queue

Retrieve the retention actions associated with a specific customer:

```bash
curl -s "http://localhost:8000/api/v1/user-live/actions?customer_id=1001" | python3 -m json.tool
```

---

## 6. Repository Structure

```text
dashboard/
  app.py                    # Streamlit application entry point
  ui_labels.py              # User-friendly labels, table cell translations, and chart localization
  ui_llm_language.py        # Language instructions for LLM-generated responses
  ui_budget_formula.py      # UI components for budget, profit, and ROI formulas
  services/                 # API clients, data loading, insights, optimization, and LLM services
  utils/                    # Formatting and utility helpers

src/                        # Model training and data preprocessing pipelines

data/                       # Raw datasets and feature store data

results_*/                  # Mode-specific output artifacts

models_*/                   # Mode-specific trained model artifacts

scripts/                    # Validation scripts and live demo utilities
```

---

## 7. Validation Checklist

Before running a demonstration or submitting the project, verify the following requirements.

### Infrastructure and Application

- [ ] Docker services start successfully.
- [ ] Financial Mode and E-commerce Mode can be selected and switched correctly.
- [ ] Previously generated training results can be loaded from the initial screen.

### UI and Localization

- [ ] Switching languages does not reset analytical controls such as churn threshold, budget, or target cap.
- [ ] Table headers and cell values are clearly presented in Korean, English, and Japanese.
- [ ] Chart titles and axis labels are properly localized.
- [ ] Duplicate metric columns, such as `expected roi 2`, are not displayed.

### LLM Integration

- [ ] When an API key is provided, LLM-generated summaries are produced in the selected language.

### Real-Time Operations

- [ ] The real-time operations view displays live customer scores and action queues without unnecessary charts.

---

## 8. Financial Mode — Recommended Columns

For financial datasets, the following columns are recommended:

- `customer_id`
- `timestamp` or transaction date
- `event_type` or transaction type
- Deposit or account balance
- Outstanding loan balance or repayment status
- Credit card spending amount
- Number of delinquent days
- Customer support or consultation count
- Customer segment or membership tier

These features can help the platform analyze customer activity, financial engagement, service interactions, and potential churn risk.

---

## 9. E-commerce Mode — Recommended Columns

For e-commerce datasets, the following columns are recommended:

- `customer_id`
- Event timestamp
- Event type, such as page view, search, add-to-cart, purchase, or login
- Order amount
- Product category
- Coupon or discount usage indicator
- Customer segment or membership tier
- Recency of browsing or purchasing activity

These features provide the behavioral and transactional signals needed for churn prediction, personalized recommendations, and retention budget optimization.

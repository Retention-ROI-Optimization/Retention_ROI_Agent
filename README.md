# Retention ROI Agent

## Demo Video

[https://github.com/user-attachments/assets/video.mp4
](https://github.com/user-attachments/assets/a8b620c8-00bd-4ce2-9d33-da98e79b3fe2)

## Project Overview

Retention ROI Project is a data-driven decision system that covers the full retention workflow: **customer churn prediction, intervention strategy optimization, personalized recommendations, and real-time operations**.  
Rather than only predicting _who_ will churn, this system estimates **_when_ churn is likely to happen**, **_which_ offer should be given to _which_ customer for maximum ROI**, and **identifies the optimal execution priority under budget constraints**.

This project supports:

- Customer behavior analysis using simulated data
- Churn modeling and survival analysis for churn timing estimation
- Uplift, CLV, and segmentation-based targeting with budget optimization
- Customer-level action recommendations with operational explainability
- Strategy validation through A/B testing and simulation fidelity checks
- Pre-deployment validation through real-time replay pipelines

In short, this project is an end-to-end **Retention Decision Intelligence Pipeline** that helps marketing and CRM teams execute retention strategies based on data rather than intuition.


## Installment

```bash
pip install -r requirements.txt
```

## Docker Run

```bash
docker compose up --build
```

Detached mode:

```bash
docker compose up -d --build
```

Use detached mode when you want services to keep running in the background.

## Finance Mode

Finance Mode is designed for banks, card companies, fintech services, and other financial-service businesses. It can use customer-level snapshots, transaction logs, card usage, loan status, balance changes, delinquency indicators, and support history.

Recommended columns include:

- `customer_id`
- `timestamp` or transaction date
- `event_type` or transaction type
- deposit or account balance
- loan balance or repayment status
- card usage amount
- delinquency days
- support/contact count
- customer segment or membership tier

Typical use cases:

- Detect likely cancellation or account inactivity.
- Prioritize high-value customers at risk.
- Decide whether to offer retention benefits, service recovery, or financial-product guidance.
- Monitor live customer events in PostgreSQL-backed live mode.

## E-commerce Mode

E-commerce Mode is designed for online stores, subscription commerce, marketplace services, and retail CRM teams. It can use visit logs, search events, cart behavior, orders, coupon usage, category preferences, and purchase history.

Recommended columns include:

- `customer_id`
- event timestamp
- event type such as page view, search, cart, purchase, login
- order amount
- item category
- coupon or discount usage
- customer segment or membership tier
- browsing or purchase recency

Typical use cases:

- Detect likely churn or purchase inactivity.
- Prioritize retention targets under a marketing budget.
- Recommend categories, coupons, or CRM actions.
- Monitor real-time behavior changes and action queues.

## Dashboard Workflow

### 1. Start services

```bash
docker compose up -d --build
```

### 2. Open the dashboard

Open the Streamlit dashboard in your browser:

```text
http://localhost:8501
```

### 3. Choose a mode

Select either:

- **Finance Mode**
- **E-commerce Mode**

### 4. Upload CSV/TSV data

Upload the company dataset from the first screen. The dashboard analyzes the columns and proposes mappings for customer ID, timestamp, event type, amount, and feature columns.

### 5. Confirm mapping and train

After confirming the mapping, run the training pipeline from the dashboard. The pipeline creates feature stores, churn scores, target candidates, recommendations, explanations, and live-serving artifacts.

### 6. Open existing results

If previous training results exist, you can open the dashboard directly without uploading a new file.

## Budget, Profit, and ROI Logic

The budget optimization view uses the following business logic:

```text
Expected incremental profit = Customer value × Response potential × Churn risk - Intervention cost
Expected ROI = Expected incremental profit ÷ Intervention cost
```

Customers with higher expected incremental profit and ROI are selected first, while respecting the total marketing budget and maximum target-customer constraints.

## Live DB Mode

The platform can serve uploaded business data through PostgreSQL-backed live tables. After training, artifacts can be seeded into live tables and updated when new events arrive.

Core flow:

1. Start Docker services.
2. Upload a finance or e-commerce CSV/TSV file.
3. Confirm column mapping and run training.
4. Seed generated artifacts into PostgreSQL live tables.
5. Send customer events to the live API.
6. Confirm score, recommendation, and action-queue changes in the dashboard.

Health check:

```bash
curl -s "http://localhost:8000/api/v1/user-live/health" | python3 -m json.tool
```

Seed status:

```bash
curl -s "http://localhost:8000/api/v1/user-live/seed-status" | python3 -m json.tool
```

Example event insertion:

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

Check a customer score:

```bash
curl -s "http://localhost:8000/api/v1/user-live/scores?customer_id=1001" | python3 -m json.tool
```

Check action queue:

```bash
curl -s "http://localhost:8000/api/v1/user-live/actions?customer_id=1001" | python3 -m json.tool
```

## Validation Checklist

Before a demo or submission, verify:

- Docker services start successfully.
- Finance and E-commerce mode selection works.
- Existing trained results can be opened from the first screen.
- Analysis-control values do not reset when switching language.
- Table headers and table-cell values are understandable in Korean, English, and Japanese.
- Chart axes and titles are localized.
- Duplicate metric columns such as `expected roi 2` are not shown.
- LLM summaries are generated in the selected language when an API key is provided.
- Real-time operations view shows live scores and action queues without unnecessary charts.

## Repository Structure

```text
dashboard/
  app.py                    # Streamlit app entry point
  ui_labels.py              # Friendly labels, table-cell translation, chart localization
  ui_llm_language.py        # LLM output-language instructions
  ui_budget_formula.py      # Budget/profit/ROI formula UI block
  services/                 # API, data loading, insight, optimization, LLM clients
  utils/                    # Formatting helpers
src/                        # Training and preprocessing pipeline
data/                       # Raw and feature-store data folders
results_*/                  # Mode-specific output artifacts
models_*/                   # Mode-specific model artifacts
scripts/                    # Validation and live-demo helper scripts
```

## Notes for Hackathon Demos

For a concise demo, focus on:

1. Uploading or opening existing finance/e-commerce results.
2. Showing churn-risk customers.
3. Adjusting budget and churn threshold.
4. Showing final targets and recommended actions.
5. Demonstrating live event updates in the real-time operations view.

This keeps the presentation focused on business value instead of exposing unnecessary internal pipeline details.

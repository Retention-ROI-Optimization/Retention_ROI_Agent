# Technical Guide

## Installation

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

## Dashboard Access

```text
http://localhost:8501
```

## Budget, Profit, and ROI Logic

```text
Expected incremental profit = Customer value x Response potential x Churn risk - Intervention cost
Expected ROI = Expected incremental profit / Intervention cost
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

## Finance Mode — Recommended Columns

- `customer_id`
- `timestamp` or transaction date
- `event_type` or transaction type
- deposit or account balance
- loan balance or repayment status
- card usage amount
- delinquency days
- support/contact count
- customer segment or membership tier

## E-commerce Mode — Recommended Columns

- `customer_id`
- event timestamp
- event type such as page view, search, cart, purchase, login
- order amount
- item category
- coupon or discount usage
- customer segment or membership tier
- browsing or purchase recency


# Current Analysis Process

## 1. System Architecture

### Overall Architecture

```text
CSV Upload → Pipeline Training → Save Results → Dashboard Display
                                     │
                                     ▼ (seed)
                               PostgreSQL → Real-Time Operations Monitor
                                     ▲
                                     │ (events)
                               FastAPI Server ← External Systems
```

### Components

| Component | Role |
|-----------|------|
| **Streamlit** | User-facing dashboard interface |
| **FastAPI** | API server for exchanging data with external systems |
| **PostgreSQL** | Persistent storage for customer data, prediction scores, and action queues |
| **Redis** | Query result caching (20-second TTL) |
| **ML Models** | Churn prediction (LightGBM), survival analysis (Cox), CLV, and uplift modeling |

### PostgreSQL Integration

- A PostgreSQL container is automatically created using Docker Compose (for local development and demonstration).
- The `src/api/services/user_live_*.py` files contain the code responsible for reading from and writing to the PostgreSQL database `retention_db`.
- Database tables are automatically created in the application code using `CREATE TABLE IF NOT EXISTS`. Database migration tools such as Alembic are not used.
- In production, external systems send events to FastAPI endpoints, automatically triggering database updates, ML rescoring, and action queue updates.

---

## 2. Churn Probability Prediction

### 2.1 Churn Label Definition

- A customer is classified as churned (`churn=1`) after **30 or more days of inactivity** since their last activity.
- Users can adjust the inactivity threshold directly from the dashboard.
- Even if a company does not provide a `churn_flag` column, the system can automatically infer churn labels.

### 2.2 Model Training

A LightGBM classification model is trained using **88 features**.

Feature categories include:

- **Behavioral counters:** `visit_7d`, `purchase_30d`, `sessions_30d`, etc.
- **Time-based features:** `recency_days`, `days_since_last_event`, `inactivity_days`, `current_non_purchase_days`
- **Purchase patterns:** `avg_purchase_gap_days`, `cart_to_purchase_rate_30d`, etc.
- **Session and time-of-day patterns:** `evening_activity_ratio`, `weekend_purchase_ratio`, etc.

### 2.3 Real-Time Rescoring

- When a new event is received, the `score_changed_customers()` function performs a new prediction using the trained ML model.
- Updated churn probability scores are reflected in the operational data.

---

## 3. Churn Timing Prediction (Survival Analysis)

### 3.1 Data Preparation

- `landmark_date` = Last activity date in the dataset − 90 days (prediction horizon).
- **Before the landmark date:** Customer behavioral data is used to generate features.
- **After the landmark date:** Observed outcomes are used to determine how many days elapsed before churn occurred.

### 3.2 Cox Proportional Hazards Model Training

The system uses the Cox proportional hazards model implemented in the `lifelines` library.

- **Input:** Customer-level features, including visit frequency, purchase intervals, inactivity duration, and other behavioral indicators.
- **Training objective:** Learn how different feature combinations are associated with the timing and risk of customer churn.
- **Output:** Estimated hazard coefficients and corresponding hazard ratios for each feature.

### 3.3 Individual Survival Curves and Median Survival Time

- The model generates an individual survival curve for each customer over a 1–90-day prediction horizon.
- The point at which the survival probability falls to or below 50% is used as the predicted median time to churn.
- This value is stored as `predicted_median_time_to_churn_days`.

**Example:**

If customer 1894's estimated survival probability falls below 50% on day 8, the system reports:

> "Customer 1894 is predicted to reach the median churn time in 8 days."

### 3.4 Dashboard Display

The expected churn date is calculated as:

```text
Expected Churn Date = landmark_date + predicted_days
```

### 3.5 Configuring `horizon_days` (90-Day Observation Period)

- The default value is **90 days**.
- The parameter can be customized:

```python
run_survival_pipeline(..., horizon_days=60)
```

- The prediction horizon is separate from the churn inactivity threshold (30 days).
- It defines the future observation period within which the model can evaluate the actual churn outcome.

### 3.6 Recommended Historical Data Requirements

| Industry | Typical Customer Activity Frequency | Recommended Data History |
|----------|-------------------------------------|--------------------------|
| E-commerce (Fashion/Food) | 1–2 visits per week | 6 months – 1 year |
| Subscription Services (SaaS) | Daily | 3–6 months |
| Finance/Insurance | 1–2 interactions per month | 1–2 years |
| Automotive/Home Appliances | Once per quarter | 2–3 years |

**Key requirement:**

The historical dataset must contain enough information to calculate the longest feature aggregation window (90 days), plus an additional 90-day observation period.

Therefore, the recommended minimum historical data duration is **approximately 6 months**.

### 3.7 Automatic Prediction Horizon Adjustment

| Uploaded Data Duration | Automatic Horizon Adjustment | Prediction Capability |
|------------------------|------------------------------|-----------------------|
| 1 year or more | 90 days (default) | Predictions up to 90 days ahead |
| 6 months – 1 year | 90 days | Good prediction coverage |
| 90 days – 6 months | Reduced to half of the available data duration | Short- to medium-term predictions |
| 60–90 days | Reduced to 30 days | Short-term predictions only |
| Less than 60 days | **Survival analysis unavailable** | Churn probability only; churn timing prediction disabled |

---

## 4. Data Flow for Production Deployment

### 4.1 Current Architecture (Development/Demo)

```text
CSV Upload → Pipeline → Result Files → Seed → PostgreSQL → Dashboard
  (Manual)   (One-Time)                                  (File-Based + Partial DB Integration)
```

**Current characteristics:**

- Customer data is manually uploaded through CSV files.
- The ML pipeline is executed as a one-time batch process.
- Prediction results are saved to output files.
- The results are seeded into PostgreSQL.
- The dashboard displays information using a combination of file-based results and database records.

### 4.2 Required Architecture for Production Deployment

```text
Company System
      │
      ▼
POST /api/v1/user-live/events
      │
      ▼
FastAPI Server
      │
      ├── Store Event in PostgreSQL
      │
      ├── Update Customer Features
      │
      ├── Perform ML Rescoring
      │
      ├── Update Action Queue
      │
      └── Refresh Dashboard Data
```

**Example API request:**

```http
POST /api/v1/user-live/events
Content-Type: application/json
```

```json
{
  "customer_id": 123,
  "event_type": "purchase",
  "amount": 89000
}
```

**Processing workflow:**

1. An external system sends a customer event to the FastAPI endpoint.
2. The event is stored in PostgreSQL.
3. Relevant customer behavioral features are updated.
4. The trained ML model recalculates the customer's churn probability.
5. The action queue is updated based on the new prediction.
6. The dashboard reflects the updated operational data.

### 4.3 Data Integration Methods

| Integration Method | Real-Time Capability | Suitable Use Cases |
|--------------------|----------------------|--------------------|
| CSV Upload | Manual | Proof of Concept (PoC) / Demo |
| Event API Integration | Immediate updates | Production systems (recommended) |
| Direct Database Connection (ETL) | Periodic updates | Legacy systems |


# Feature Dictionary

This document defines the key features used in feature engineering for customer churn prediction in Retention ROI Agent.

The features capture customer recency, purchase behavior, monetary value, engagement patterns, session quality, marketing interactions, behavioral changes, and customer journey characteristics.

## 1. Feature Definitions

| Feature | Category | Definition |
|---|---|---|
| customer_age_days | base | Number of days since customer registration up to the reference date |
| days_since_last_event | recency | Number of days elapsed since the customer's most recent event |
| recency_days | RFM | Number of days since the customer's most recent purchase |
| frequency_30d | RFM | Number of purchases in the last 30 days |
| frequency_90d | RFM | Number of purchases in the last 90 days |
| monetary_30d | RFM | Total net revenue generated in the last 30 days |
| monetary_90d | RFM | Total net revenue generated in the last 90 days |
| avg_order_value_90d | RFM | Average order value over the last 90 days |
| monetary_per_visit_90d | value | Average revenue per visit over the last 90 days |
| visits_14d | behavior | Number of visits in the last 14 days |
| visits_prev_14d | behavior | Number of visits during the preceding 14-day period |
| visit_change_rate_14d | behavior_change | Rate of change in visits between the current and previous 14-day periods |
| purchases_14d | behavior | Number of purchases in the last 14 days |
| purchases_prev_14d | behavior | Number of purchases during the preceding 14-day period |
| purchase_change_rate_14d | behavior_change | Rate of change in purchases between the current and previous 14-day periods |
| searches_30d | behavior | Number of search events in the last 30 days |
| searches_prev_30d | behavior | Number of search events during the preceding 30-day period |
| add_to_cart_30d | behavior | Number of add-to-cart events in the last 30 days |
| add_to_cart_prev_30d | behavior | Number of add-to-cart events during the preceding 30-day period |
| coupon_open_30d | coupon | Number of coupon-open events in the last 30 days |
| coupon_open_prev_30d | coupon | Number of coupon-open events during the preceding 30-day period |
| coupon_open_rate_30d | coupon | Coupon open rate over the last 30 days |
| coupon_response_change_rate | behavior_change | Change in coupon response rate |
| avg_purchase_gap_days | cycle | Average number of days between consecutive purchases |
| median_purchase_gap_days | cycle | Median number of days between consecutive purchases |
| current_non_purchase_days | cycle | Number of days elapsed since the most recent purchase |
| purchase_cycle_anomaly | cycle | Ratio of current days without a purchase to the average purchase interval |
| avg_session_duration_sec_30d | session_quality | Average session duration in seconds over the last 30 days |
| avg_session_duration_sec_prev_30d | session_quality | Average session duration in seconds during the preceding 30-day period |
| session_duration_change_rate | behavior_change | Rate of change in average session duration |
| pageviews_per_session_30d | session_quality | Average number of page views per session over the last 30 days |
| pageviews_per_session_prev_30d | session_quality | Average number of page views per session during the preceding 30-day period |
| pageviews_change_rate | behavior_change | Rate of change in page views per session |
| search_to_purchase_conversion_30d | session_quality | Search-to-purchase conversion rate over the last 30 days |
| search_to_purchase_conversion_prev_30d | session_quality | Search-to-purchase conversion rate during the preceding 30-day period |
| search_purchase_conv_change_rate | behavior_change | Change in search-to-purchase conversion rate |
| cart_to_purchase_rate_30d | session_quality | Cart-to-purchase conversion rate over the last 30 days |
| cart_to_purchase_rate_prev_30d | session_quality | Cart-to-purchase conversion rate during the preceding 30-day period |
| cart_conversion_change_rate | behavior_change | Change in cart-to-purchase conversion rate |
| support_contact_30d | service | Number of customer support inquiry events in the last 30 days |
| support_contact_rate_30d | service | Number of customer support inquiries per session over the last 30 days |
| sessions_30d | session | Number of sessions in the last 30 days |
| sessions_prev_30d | session | Number of sessions during the preceding 30-day period |
| active_days_30d | engagement | Number of unique days with customer activity in the last 30 days |
| orders_with_coupon_ratio_90d | coupon | Proportion of orders using coupons over the last 90 days |
| coupon_redeem_rate_90d | coupon | Proportion of orders using coupons over the last 90 days |
| exposure_count_30d | marketing | Number of marketing campaign exposures in the last 30 days |
| coupon_cost_30d | marketing | Total coupon cost incurred over the last 30 days |
| weekend_purchase_ratio | temporal | Proportion of purchases made on weekends |
| weekday_purchase_ratio | temporal | Proportion of purchases made on weekdays |
| evening_activity_ratio | temporal | Proportion of customer activities occurring during evening hours |
| night_activity_ratio | temporal | Proportion of customer activities occurring during late-night hours |
| workhour_activity_ratio | temporal | Proportion of customer activities occurring during working hours |
| weekend_activity_ratio | temporal | Proportion of all customer activities occurring on weekends |
| event_diversity_90d | temporal | Diversity of event types observed over the last 90 days |
| recent_event_sequence | sequence | Sequence of the customer's most recent N events |
| behavior_cluster_id | sequence | Identifier of the behavioral cluster assigned based on activity over the last 90 days |
| dominant_event_type_90d | sequence | Most frequently occurring event type over the last 90 days |
| current_journey_stage | journey | Customer's current stage in the customer journey |
| journey_stage_days | journey | Number of days spent in the current customer journey stage |
| inactivity_days | journey | Number of days without customer activity |
| recent_visit_score | snapshot | Score representing recent visit activity |
| recent_purchase_score | snapshot | Score representing recent purchase activity |
| recent_exposure_score | snapshot | Score representing recent marketing exposure |

---

## 2. Missing Value and Outlier Handling

The feature engineering pipeline applies the following preprocessing rules to improve data consistency and model robustness.

| Data Type / Condition | Preprocessing Strategy |
|---|---|
| Numeric features | Replace missing values with the median |
| Categorical features | Replace missing values with `unknown` |
| Outliers | Apply winsorization at the 1st and 99th percentiles |
| Infinite values | Convert infinite values to `NaN`, then apply the corresponding missing-value handling strategy |

### Preprocessing Details

- **Numeric features:** Missing values are imputed using the median to reduce sensitivity to extreme observations.
- **Categorical features:** Missing or undefined categories are assigned the placeholder value `unknown`.
- **Outliers:** Extreme numerical values are capped at the 1st and 99th percentile boundaries through winsorization.
- **Infinite values:** Positive and negative infinity values are converted to `NaN` before applying the standard imputation procedure.

---

## 3. Churn Label Definition

The churn prediction model uses a forward-looking labeling strategy based on customer activity and future customer status.

### Training Population

Only customers who registered before the reference date are included in the training population.

### Churn Labeling Criteria

A customer is labeled as churned (`churn = 1`) if **all** of the following conditions are satisfied:

1. The customer has zero visits during the 45-day observation window following the reference date.
2. The customer has zero purchases during the same 45-day observation window.
3. The customer's final recorded future status is `churn_risk`.

### Labeling Rule

```text id="4r5n2m"
IF
    customer_registration_date < reference_date
AND
    visits_during_next_45_days == 0
AND
    purchases_during_next_45_days == 0
AND
    final_future_status == "churn_risk"
THEN
    churn_label = 1
ELSE
    churn_label = 0
```

This definition identifies customers who exhibit prolonged inactivity during the observation window and whose final future status indicates churn risk.

The resulting labels are used as the target variable for supervised churn prediction.

# Feature Dictionary

아래는 이탈 예측용 피처 엔지니어링에서 사용하는 주요 피처 정의다.

| Feature | Category | Definition |
|---|---|---|
| customer_age_days | base | 기준일까지의 고객 생존 일수 |
| days_since_last_event | recency | 마지막 이벤트 이후 경과 일수 |
| recency_days | RFM | 마지막 구매 이후 경과 일수 |
| frequency_30d | RFM | 최근 30일 구매 횟수 |
| frequency_90d | RFM | 최근 90일 구매 횟수 |
| monetary_30d | RFM | 최근 30일 순매출 합계 |
| monetary_90d | RFM | 최근 90일 순매출 합계 |
| avg_order_value_90d | RFM | 최근 90일 평균 주문 금액 |
| monetary_per_visit_90d | value | 최근 90일 방문 1회당 평균 매출 |
| visits_14d | behavior | 최근 14일 방문 수 |
| visits_prev_14d | behavior | 직전 14일 방문 수 |
| visit_change_rate_14d | behavior_change | 최근 14일 방문 변화율 |
| purchases_14d | behavior | 최근 14일 구매 수 |
| purchases_prev_14d | behavior | 직전 14일 구매 수 |
| purchase_change_rate_14d | behavior_change | 최근 14일 구매 변화율 |
| searches_30d | behavior | 최근 30일 검색 수 |
| searches_prev_30d | behavior | 직전 30일 검색 수 |
| add_to_cart_30d | behavior | 최근 30일 장바구니 추가 수 |
| add_to_cart_prev_30d | behavior | 직전 30일 장바구니 추가 수 |
| coupon_open_30d | coupon | 최근 30일 쿠폰 오픈 수 |
| coupon_open_prev_30d | coupon | 직전 30일 쿠폰 오픈 수 |
| coupon_open_rate_30d | coupon | 최근 30일 쿠폰 오픈율 |
| coupon_response_change_rate | behavior_change | 쿠폰 반응률 변화 |
| avg_purchase_gap_days | cycle | 평균 구매 주기 |
| median_purchase_gap_days | cycle | 중앙 구매 주기 |
| current_non_purchase_days | cycle | 현재 미구매 일수 |
| purchase_cycle_anomaly | cycle | 현재 미구매 일수 / 평균 구매 주기 |
| avg_session_duration_sec_30d | session_quality | 최근 30일 평균 세션 시간 |
| avg_session_duration_sec_prev_30d | session_quality | 직전 30일 평균 세션 시간 |
| session_duration_change_rate | behavior_change | 세션 시간 변화율 |
| pageviews_per_session_30d | session_quality | 최근 30일 세션당 페이지뷰 |
| pageviews_per_session_prev_30d | session_quality | 직전 30일 세션당 페이지뷰 |
| pageviews_change_rate | behavior_change | 세션당 페이지뷰 변화율 |
| search_to_purchase_conversion_30d | session_quality | 최근 30일 검색 후 구매 전환율 |
| search_to_purchase_conversion_prev_30d | session_quality | 직전 30일 검색 후 구매 전환율 |
| search_purchase_conv_change_rate | behavior_change | 검색 후 구매 전환율 변화 |
| cart_to_purchase_rate_30d | session_quality | 최근 30일 장바구니→구매 전환율 |
| cart_to_purchase_rate_prev_30d | session_quality | 직전 30일 장바구니→구매 전환율 |
| cart_conversion_change_rate | behavior_change | 장바구니 전환율 변화 |
| support_contact_30d | service | 최근 30일 문의 이벤트 수 |
| support_contact_rate_30d | service | 최근 30일 세션당 문의 비율 |
| sessions_30d | session | 최근 30일 세션 수 |
| sessions_prev_30d | session | 직전 30일 세션 수 |
| active_days_30d | engagement | 최근 30일 활동한 고유 일수 |
| orders_with_coupon_ratio_90d | coupon | 최근 90일 쿠폰 사용 주문 비율 |
| coupon_redeem_rate_90d | coupon | 최근 90일 쿠폰 사용 주문 비율 |
| exposure_count_30d | marketing | 최근 30일 캠페인 노출 수 |
| coupon_cost_30d | marketing | 최근 30일 쿠폰 비용 |
| weekend_purchase_ratio | temporal | 주말 구매 비율 |
| weekday_purchase_ratio | temporal | 평일 구매 비율 |
| evening_activity_ratio | temporal | 저녁 시간 활동 비율 |
| night_activity_ratio | temporal | 심야 활동 비율 |
| workhour_activity_ratio | temporal | 근무시간 활동 비율 |
| weekend_activity_ratio | temporal | 주말 전체 활동 비율 |
| event_diversity_90d | temporal | 최근 90일 이벤트 다양성 |
| recent_event_sequence | sequence | 최근 N개 이벤트 시퀀스 |
| behavior_cluster_id | sequence | 최근 90일 행동 군집 ID |
| dominant_event_type_90d | sequence | 최근 90일 최빈 이벤트 |
| current_journey_stage | journey | 현재 고객 여정 단계 |
| journey_stage_days | journey | 현재 단계 체류 일수 |
| inactivity_days | journey | 비활성 일수 |
| recent_visit_score | snapshot | 최근 방문 점수 |
| recent_purchase_score | snapshot | 최근 구매 점수 |
| recent_exposure_score | snapshot | 최근 노출 점수 |

## 결측치 및 이상치 처리

- numeric: 중앙값 대체
- categorical: `unknown` 대체
- outlier: 1st/99th percentile winsorization
- infinite 값: `NaN` 처리 후 동일 로직 적용

## 라벨 정의

- 기준일 이전 가입 고객만 학습 대상으로 사용
- 기준일 이후 45일 동안 방문 0회 + 구매 0회 + 마지막 미래 상태가 `churn_risk`이면 이탈(1)로 정의

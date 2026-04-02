# Retention Strategy

이 문서는 세그먼트별 리텐션 액션, 예상 비용, 예상 효과를 정의한다.

## 세그먼트별 전략

| Segment | Strategy | Estimated Cost | Expected Effect |
|---|---|---:|---:|
| High Value-Persuadables | VIP concierge + personalized offer | 30,000 | uplift × 1.15 |
| High Value-Sure Things | Loyalty touchpoint | 8,000 | uplift × 0.15 |
| High Value-Lost Causes | Deep-dive outreach | 12,000 | uplift × 0.10 |
| Low Value-Persuadables | Coupon campaign | 7,000 | uplift × 0.85 |
| Low Value-Lost Causes | No Action | 0 | 0 |
| Low Value-Sure Things | Light reminder | 3,000 | uplift × 0.05 |
| New Customers | Onboarding sequence | 5,000 | uplift × 0.20 |

## 최적화 기준

베이스라인 최적화는 다음 목적함수를 그리디 방식으로 근사한다.

- Objective: Maximize Σ(Uplift_i × CLV_i × Action_i)
- Constraint: Σ(Cost_i × Action_i) ≤ Budget

우선순위는 `expected_revenue / cost`가 높은 고객부터 선택한다.

## 확장 과제

- 세그먼트별 최소/최대 예산 제약
- 채널별 재고 또는 인력 제약
- 다중 목적 최적화(ROI와 churn-risk coverage 동시 고려)
- 선형계획 또는 정수계획 기반 최적화

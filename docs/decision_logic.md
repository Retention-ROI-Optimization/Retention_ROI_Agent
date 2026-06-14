# Decision Logic

Retention ROI Agent는 고객 이탈 예측 점수를 그대로 영업/마케팅 액션으로 사용하지 않는다. 여러 신호를 결합해 **개입할 고객, 개입 강도, 개입 시점, 기대 순이익, 예상 ROI**를 계산한다.

---

## 1. 기본 철학

이탈 확률이 높은 고객이 항상 좋은 타겟은 아니다.

- 이미 떠날 가능성이 너무 높고 반응 가능성이 낮으면 비용을 써도 회수하기 어렵다.
- 이탈 위험은 높지만 CLV가 낮으면 기대 이익이 작을 수 있다.
- CLV가 높아도 이탈 시점이 멀면 지금 고비용 개입을 할 필요가 없을 수 있다.
- 할인에 너무 많이 노출된 고객에게 쿠폰을 계속 주면 장기적으로 손해일 수 있다.

따라서 플랫폼은 다음 신호를 함께 사용한다.

```text
Churn Probability
+ Uplift Score
+ Customer Lifetime Value
+ Intervention Cost
+ Survival Timing
+ Coupon Fatigue / Discount Pressure
+ Budget Constraint
```

---

## 2. 핵심 수식

README에서는 복잡한 구현을 숨기고, 운영자가 이해할 수 있는 형태로 다음 개념을 사용한다.

```text
Expected incremental profit
= Customer value × Response potential × Churn risk - Intervention cost

Expected ROI
= Expected incremental profit ÷ Intervention cost
```

실제 후보 선정에서는 여기에 survival timing, intervention intensity, 세그먼트 조건, budget cap, target cap이 추가된다.

---

## 3. 후보 고객 필터링

예산 최적화에 들어가기 전, 기본 후보는 다음 조건을 만족해야 한다.

- 이탈 확률이 사용자가 설정한 threshold 이상
- uplift score가 양수
- expected incremental profit이 양수
- intervention cost가 양수
- Sleeping Dogs처럼 개입하면 역효과가 날 수 있는 세그먼트는 제외

이 필터는 “위험한 고객”이 아니라 “비용을 써서 개입할 의미가 있는 고객”을 찾기 위한 장치다.

---

## 4. 개입 강도 후보 생성

각 고객은 하나의 액션만 갖는 것이 아니라, low/mid/high 같은 개입 강도 후보를 가진다.

예를 들어 같은 고객에게도 다음 선택지가 생길 수 있다.

- low: 낮은 비용의 메시지/푸시
- mid: 적당한 혜택 또는 CRM 접촉
- high: 상담, VIP 케어, 고비용 혜택

후보별로 비용, 기대 이익, ROI, timing urgency, intensity fit을 계산한다. 이후 예산 최적화는 고객×개입강도 후보 중에서 최종 조합을 선택한다.

---

## 5. Survival Timing 반영

Survival 분석 결과가 있으면 고객별 이탈 시점 정보를 예산 최적화에 반영한다.

주요 파생값은 다음과 같다.

| 값 | 의미 |
| --- | --- |
| predicted_median_time_to_churn_days | 생존확률이 50% 아래로 내려가는 예상 시점 |
| short_term_churn_probability | 단기 이탈 가능성 |
| timing_urgency_score | 위험 percentile, 단기 이탈 가능성, 예상 이탈 시점을 결합한 긴급도 |
| recommended_intervention_window | 14일 이내, 15~30일, 31~60일 등 운영 문구 |
| timing_priority_bucket | immediate, near_term, planned, monitor |

타이밍 정보는 priority score와 action 비교에 들어간다. 즉, 같은 ROI라도 지금 당장 떠날 가능성이 큰 고객이 더 높은 우선순위를 받을 수 있다.

---

## 6. 예산 최적화 방식

최종 선택은 다음 제약을 가진 greedy optimization으로 수행된다.

- 총 비용은 사용자가 입력한 budget을 넘지 않음
- 고객 1명당 최종 action은 하나만 선택
- 최대 타겟 고객 수 cap을 넘지 않음
- high-intensity action은 전체 cap의 일정 비율 이상으로 과도하게 몰리지 않음
- 기대 이익이 양수인 후보만 선택

후보 정렬에는 다음 신호가 들어간다.

- expected ROI rank
- expected incremental profit rank
- churn probability
- uplift score
- CLV rank
- timing urgency
- intervention window
- intensity fit

이 방식은 단순 ROI 순 정렬보다 안정적이다. ROI가 높아도 절대 이익이 너무 작거나, 타이밍이 급하지 않거나, 고비용 액션이 과도하게 몰리는 상황을 완화한다.

---

## 7. Budget Sensitivity Map

예산 화면은 현재 예산 하나만 계산하지 않는다. 100만 원 단위 예산 grid를 만들어 예산이 바뀔 때 결과가 어떻게 달라지는지 보여준다.

표에서 계산하는 항목은 다음과 같다.

- budget
- spent
- remaining
- target_count
- expected_incremental_profit
- average_roi
- added_budget
- added_spend
- added_target_count
- added_profit
- marginal_profit_per_1m
- marginal_roi
- budget_status
- operator_message

이를 통해 다음 판단을 할 수 있다.

| 상태 | 해석 |
| --- | --- |
| 확대 검토 가능 | 추가 예산의 효율이 현재 평균과 비슷하거나 더 높음 |
| 점진 확대 가능 | 추가 예산 효율은 양호하지만 평균 효율보다 낮음 |
| ROI 하락 시작 | 타겟은 늘지만 평균 ROI가 낮아지기 시작 |
| 효율 낮음 | 추가 예산 대비 기대 순이익이 낮음 |
| 포화 또는 낭비 주의 | 예산을 더 늘려도 추가 타겟이나 이익이 거의 없음 |

경진대회나 데모에서는 이 표가 특히 강하다. “모델을 만들었다”가 아니라 “예산 증감 의사결정을 할 수 있다”는 점을 보여주기 때문이다.

---

## 8. Counterfactual Retention Lab

고객별 액션 비교는 다음 시나리오를 대상으로 한다.

- 무개입
- 5,000원 혜택
- 상담 전화
- 푸시/이메일
- 7일 대기

기본 아이디어는 다음과 같다.

```text
무개입 예상 순이익 = CLV × (1 - 현재 이탈 확률)
액션 예상 순이익 = CLV × (1 - 액션 후 이탈 확률) - 액션 비용
무개입 대비 개선액 = 액션 예상 순이익 - 무개입 예상 순이익
```

액션 후 이탈 확률은 churn probability, uplift score, timing urgency, coupon affinity, price sensitivity, discount pressure 등을 활용해 추정한다.

이 값은 실제 causal effect를 확정하는 값이 아니라, 의사결정을 돕는 반사실 추정치다. 실제 운영에서는 A/B test나 holdout group으로 검증해야 한다.

---

## 9. 개인화 추천 로직

개인화 추천은 모든 고객에게 생성하지 않는다. 현재 예산과 threshold 조건을 통과한 최종 타겟 고객에게만 생성한다.

추천 점수는 다음 신호를 섞는다.

- 고객 본인의 과거 구매/거래 이력
- 최근 탐색/조회 신호
- 같은 persona/uplift segment의 선호
- 전체 고객의 인기 신호
- 현재 타겟 priority와 expected ROI

금융 모드에서는 일반 이커머스 카테고리를 예금, 대출, 카드, 보험, 자산관리 등 금융상품 언어로 바꾸고, 추천 이유도 금융 도메인에 맞게 표현한다.

---

## 10. Live Action Queue 로직

Live DB 모드에서는 배치 분석 산출물을 PostgreSQL 테이블에 seed한 뒤, 신규 이벤트가 들어올 때 다음 흐름을 수행한다.

```text
POST /api/v1/user-live/events
  → event 저장
  → customer feature state 업데이트
  → changed customer 재채점
  → churn score / CLV / uplift / expected ROI 갱신
  → action queue 조건 확인
  → queued action 생성 또는 업데이트
```

대시보드는 이 결과를 Live Action Queue 테이블로 보여준다. 운영자는 고객별 recommended action, intervention intensity, expected profit, expected ROI, action status, trigger reason을 확인할 수 있다.

---

## 11. LLM 요약/질문 로직

LLM은 전체 원본 데이터를 그대로 보내지 않는다. 화면별로 필요한 요약 payload를 구성해 다음 작업을 한다.

- 현재 화면의 핵심 지표 요약
- 위험 세그먼트 설명
- 예산/threshold 변경 방향 제안
- 추천 결과와 action queue 해석
- 한국어/영어/일본어 답변

챗봇은 처음 열었던 화면의 컨텍스트를 유지하고, 사용자가 원하면 현재 화면으로 갱신할 수 있다. 화면 이동 중에도 질문 맥락이 갑자기 바뀌지 않도록 설계되어 있다.

---

## 12. 검증과 한계

expected ROI, uplift, counterfactual 결과는 운영 의사결정을 돕는 예측값이다. 실제 비즈니스 효과를 확정하려면 다음 루프가 필요하다.

1. 최종 타겟 중 일부를 holdout으로 남긴다.
2. 실제 발송, 상담, 쿠폰 사용, 구매/재방문 로그를 남긴다.
3. treatment와 control의 유지율·매출 차이를 비교한다.
4. 증분 효과를 다시 uplift model과 action policy에 반영한다.

이 프로젝트의 강점은 이 한계를 숨기지 않고, 예측에서 운영 검증까지 이어지는 구조를 고려했다는 점이다.

# Counterfactual Retention Lab

## 목적

기존 리텐션 화면은 고객의 이탈 위험도, 추천 액션, 예산 타겟을 보여준다. `Counterfactual Retention Lab`은 여기서 한 단계 더 나아가 같은 고객에 대해 다음 시나리오의 기대 순이익을 비교한다.

- 아무것도 하지 않음
- 5,000원 쿠폰 제공
- 상담 전화
- 푸시/이메일
- 7일 대기

이 뷰의 목표는 “무엇을 추천할지”뿐 아니라 “왜 그 액션이 무개입이나 다른 액션보다 경제적으로 나은지”를 설명하는 것이다.

## 계산 로직

핵심 계산은 `src/optimization/counterfactual.py`에 있다.

기본 아이디어는 다음과 같다.

```text
무개입 예상 순이익 = CLV × (1 - 현재 이탈 확률)
액션 예상 순이익 = CLV × (1 - 액션 후 이탈 확률) - 액션 비용
무개입 대비 개선액 = 액션 예상 순이익 - 무개입 예상 순이익
```

액션 후 이탈 확률은 기존 `churn_probability`, `uplift_score`, `CLV`, `survival/timing`, `coupon_affinity`, `price_sensitivity`, `discount_pressure_score` 등을 활용해 추정한다. Survival 산출물이 있으면 예상 이탈 시점과 긴급도 가중치가 반영된다. 없으면 안전한 기본값으로 계산한다.

## 주의사항

이 값은 실제 집행 결과가 아니라 예측 기반 반사실 추정치다. 따라서 운영 의사결정에서는 다음 절차가 필요하다.

1. 신뢰도 낮음/중간 고객은 A/B 테스트 또는 holdout에 포함한다.
2. 실제 발송/상담/쿠폰 사용 로그를 남긴다.
3. treatment와 control의 실제 매출·유지율 차이를 비교해 증분 ROI를 계산한다.
4. 검증 결과를 uplift 및 action policy에 다시 반영한다.

## 추가된 파일

- `src/optimization/counterfactual.py`: 반사실 손익 계산 엔진
- `dashboard/services/counterfactual_service.py`: 대시보드용 서비스 wrapper
- `dashboard/app.py`: 신규 뷰 `고객별 대응 전략 비교` 추가

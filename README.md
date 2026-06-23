<div align="center">

# Retention ROI Agent

**이탈 예측을 넘어, 리텐션 예산을 어디에·언제·어떤 방식으로 써야 하는지 결정하는 운영형 Retention Intelligence Copilot**

CSV/TSV 파일 하나를 올리면 고객 이탈 위험, 예상 이탈 시점, 개입 효과, 고객 가치, 예산 제약, 개인화 액션, 실시간 액션 큐까지 하나의 의사결정 흐름으로 연결합니다.

[데모 영상](https://drive.google.com/file/d/1WRAoRtl88iwrsZRCmMKu1qhgs2dDfbE5/view?usp=sharing) · [차별화 전략](docs/product_differentiation.md) · [대시보드 의사결정 루프](docs/dashboard_decision_loop.md) · [의사결정 로직](docs/decision_logic.md) · [기술 문서](docs/technical_guide.md) · [발표 자료](docs/presentation.pdf)

</div>

---

## 이 프로젝트의 핵심 차별점

대부분의 이탈 분석은 “누가 떠날 가능성이 높은가?”에서 멈춥니다. Retention ROI Agent는 거기서 한 단계 더 나아가 **그 고객을 잡는 것이 경제적으로 맞는지**, **언제 개입해야 하는지**, **쿠폰·상담·푸시·대기 중 무엇이 더 나은지**, **예산을 100만 원 더 쓰면 이익이 늘어나는지**까지 계산합니다.

> **이탈할 것 같은 고객**과 **잡았을 때 실제로 이익이 남는 고객**은 다릅니다.  
> 이 플랫폼은 churn probability가 아니라 **Retention ROI decision**을 만듭니다.

| 기존 솔루션 | 한계 | Retention ROI Agent의 차별점 |
| --- | --- | --- |
| 일반 이탈 예측 모델 | 위험도 Top N만 보여주고 예산·액션·타이밍은 사람이 따로 판단 | Churn × Uplift × CLV × Cost × Timing을 결합해 **예산 안에서 실행할 고객과 액션**을 선별 |
| GA4 / Amplitude / Mixpanel류 행동 분석 | 퍼널·이벤트 분석은 강하지만 고객별 개입 경제성 계산은 별도 작업 필요 | 이벤트 로그를 고객 단위 의사결정 테이블로 바꾸고 **리텐션 타겟·예상 ROI·개입 시점**까지 연결 |
| Salesforce Marketing Cloud / Braze / HubSpot류 캠페인 도구 | 캠페인 발송과 자동화는 강하지만 “누구에게 얼마를 써야 이익인가”는 별도 분석 필요 | 캠페인 실행 전에 **증분 효과와 비용 대비 기대 수익**으로 대상자를 정렬 |
| Tableau / Power BI 대시보드 | 현황 리포팅 중심. 조건을 바꿔도 의사결정 후보가 자동 재계산되지는 않음 | threshold·예산·타겟 상한 변경 시 **타겟 고객, 세그먼트 예산, 추천 액션, ROI가 함께 재계산** |
| Kaggle식 churn notebook / AutoML PoC | 모델 정확도와 feature importance 중심 | 대시보드에서 바로 운영 가능한 **Live DB, action queue, counterfactual lab, LLM 질의응답**까지 구현 |

더 자세한 비교는 [차별화 전략](docs/product_differentiation.md)에 정리했습니다.

---

## 핵심 기능

### 1. 산업별 데이터 온보딩: 금융/이커머스 모드

처음부터 시뮬레이터 데이터에 고정하지 않고, **금융 모드**와 **이커머스 모드**를 선택해 업로드 데이터를 해석합니다.

- 금융: 예금, 대출, 카드, 거래, 잔고, 연체, 상담 이력 기반 해지/휴면 위험 분석
- 이커머스: 방문, 검색, 장바구니, 구매, 쿠폰, 카테고리 선호 기반 재방문/구매 이탈 분석

<img src="assets/dash1.png" width="720" />

### 2. CSV/TSV 자동 매핑과 이벤트 표준화

업로드된 컬럼명이 제각각이어도 고객 ID, 이벤트 시각, 이벤트 유형, 금액, 카테고리, 이탈 라벨 후보를 자동 감지합니다. 이벤트 값도 내부 표준 타입으로 매핑해 이후 모델링과 실시간 이벤트 처리가 같은 스키마를 사용하도록 만듭니다.

<img src="assets/dash2.png" width="720" />

### 3. 이탈 현황: 위험도만이 아니라 운영 시작점 제공

전체 고객 수, 위험 고객 수, 위험 고객 비율, 평균 이탈 확률을 보여주고, 코호트 리텐션·세그먼트·Uplift/CLV 분석으로 넘어가기 전 “어떤 고객군이 문제인지”를 빠르게 파악합니다.

<img src="assets/dash3.png" width="720" />

### 4. 이탈 시점 예측: 언제 개입해야 하는가

Survival Analysis 기반으로 고객별 예상 이탈 시점, 30일 내 이탈 가능성, 예상 손실액을 계산합니다. 단순히 “위험함”이 아니라 **14일 이내 즉시 연락**, **15~30일 안에 연락**, **31~60일 안에 계획적 연락**처럼 운영 가능한 타이밍으로 바꿉니다.

<img src="assets/dash5.png" width="720" />

### 5. 예산 배분·타겟 고객: 리텐션 예산의 marginal ROI 계산

입력한 총 예산, 이탈 임계값, 최대 타겟 고객 수에 따라 고객별 개입 후보를 재계산합니다. 이 화면은 경진대회에서 가장 차별적으로 보이는 부분입니다.

- 예산 내 최종 타겟 고객 자동 선정
- 세그먼트별 예산 배분과 기대 순이익 계산
- 예산을 100만 원 더 썼을 때의 기대 순이익 증가분 계산
- 포화 예산 구간과 저효율 예산 구간 표시
- 고강도 개입 비중 cap으로 과도한 비용 집중 방지

<img src="assets/dash6.png" width="720" />

<img src="assets/dash7.png" width="720" />

<img src="assets/dash8.png" width="720" />

### 6. 고객별 대응 전략 비교: Counterfactual Retention Lab

같은 고객에 대해 **무개입, 5,000원 혜택, 상담 전화, 푸시/이메일, 7일 대기** 시나리오의 기대 순이익을 비교합니다. 추천 액션이 “모델이 골랐다”로 끝나지 않고, 왜 그 액션이 무개입이나 다른 액션보다 나은지 설명합니다.

자세한 로직은 [Counterfactual Lab 문서](docs/counterfactual_retention_lab.md)를 참고하세요.

<img src="assets/dash9.png" width="720" />

<img src="assets/dash10.png" width="720" />


### 7. 최종 타겟 고객 대상 개인화 추천

저장된 과거 추천 후보를 그대로 보여주지 않습니다. 현재 화면에서 선택한 예산·이탈 임계값·최대 타겟 조건으로 선별된 **최종 리텐션 타겟 고객에게만** 추천을 새로 생성합니다.

추천 점수는 고객 본인의 과거 구매/거래 이력, 최근 관심 신호, 유사 세그먼트 선호, 전체 인기 신호를 혼합합니다. 금융 모드에서는 카테고리와 사유 문구를 금융상품·금융행동 언어로 변환합니다.

<img src="assets/dash11.png" width="720" />

### 8. 실시간 운영 모니터: 점수가 아니라 액션 큐까지 갱신

PostgreSQL Live DB에 이벤트가 들어오면 고객 상태, 이탈 점수, 추천 후보, 액션 큐가 함께 갱신됩니다. 대시보드에서는 이벤트 수, 전체 고객 수, queued action 수, 최신 점수 갱신 시각, action queue 상세를 확인할 수 있습니다.

- FastAPI 이벤트 수신
- 고객 feature state 업데이트
- churn/CLV/uplift 재채점
- expected ROI 기반 action queue 적재
- 데모 스트림으로 신규/기존 고객 이벤트 자동 생성

<img src="assets/dash12.png" width="720" />

### 9. 화면 기반 AI 챗봇

LLM은 전체 데이터를 무작정 읽는 것이 아니라, 현재 대시보드 화면의 요약 payload를 바탕으로 답합니다. 사용자는 “왜 이 세그먼트가 위험한가?”, “예산을 늘리면 무엇이 달라지는가?”, “어떤 고객부터 연락해야 하는가?”를 화면 맥락 그대로 질문할 수 있습니다.

<img src="assets/dash4.png" width="260" />

---

## 의사결정 흐름

```text
CSV/TSV 업로드
  → 컬럼 역할 자동 감지
  → 이벤트 값 표준화
  → 이탈 기준 설정
  → Churn / Survival / Uplift / CLV 산출
  → 예산 제약 기반 타겟·액션 최적화
  → 개인화 추천 생성
  → PostgreSQL Live DB seed
  → 신규 이벤트 수신 시 점수·액션 큐 갱신
```

핵심 로직은 “예측 점수”가 아니라 아래 질문에 답하도록 설계되어 있습니다.

| 질문 | 플랫폼이 계산하는 것 |
| --- | --- |
| 누가 위험한가? | churn probability, risk segment |
| 언제 개입해야 하는가? | predicted time to churn, timing urgency, recommended intervention window |
| 잡을 가치가 있는가? | CLV, expected loss, expected incremental profit |
| 개입하면 반응할 가능성이 있는가? | uplift score, persuadable segment |
| 얼마를 써야 하는가? | coupon/action cost, budget allocation, marginal ROI |
| 무엇을 해야 하는가? | recommended action, intervention intensity, next best recommendation |
| 지금 운영 큐에 올릴 것인가? | live score, expected ROI, action queue status |

계산식과 제약 조건은 [의사결정 로직](docs/decision_logic.md)에 정리했습니다.

---

## 지원 도메인

| 모드 | 대상 산업 | 데이터 예시 | 주요 의사결정 |
| --- | --- | --- | --- |
| **금융 모드** | 은행, 카드사, 핀테크, 보험/자산관리 | 입출금, 대출 상환, 카드 결제, 잔액 변동, 연체, 상담 이력 | 해지/휴면 위험 고객 선별, 상담·혜택·상품 안내 우선순위 결정 |
| **이커머스 모드** | 온라인 쇼핑몰, 구독 커머스, 마켓플레이스 | 방문, 검색, 장바구니, 구매, 쿠폰 사용, 카테고리 선호 | 재방문/재구매 유도 대상 선별, 쿠폰·카테고리 추천, CRM 액션 큐 생성 |

업로드 데이터가 고객 스냅샷 형태여도 진행할 수 있고, 이벤트 로그가 충분하면 행동 시계열·이탈 시점·실시간 운영 분석이 더 풍부해집니다.

---

## 빠른 시작

```bash
# 1. 서비스 실행
docker compose up -d --build

# 2. 대시보드 접속
open http://localhost:8501
```

상세 설치, API 예시, 디렉토리 구조, 검증 체크리스트는 [기술 문서](docs/technical_guide.md)를 참고하세요.

---

## 문서

| 문서 | 설명 |
| --- | --- |
| [차별화 전략](docs/product_differentiation.md) | 기존 churn dashboard, CRM, CDP, BI 도구와의 차별점 |
| [대시보드 의사결정 루프](docs/dashboard_decision_loop.md) | 7개 핵심 화면이 어떤 의사결정으로 이어지는지 설명 |
| [의사결정 로직](docs/decision_logic.md) | 예산 최적화, counterfactual, 개인화 추천, live action queue 계산 방식 |
| [기술 문서](docs/technical_guide.md) | 설치, API, 디렉토리 구조, 검증 체크리스트 |
| [분석 과정](docs/analysis_process.md) | 모델링·생존분석·실시간 배포 흐름 설명 |
| [피처 사전](docs/feature_dictionary.md) | 생성되는 주요 피처와 의미 |
| [리텐션 전략](docs/retention_strategy.md) | 세그먼트별 리텐션 전략과 비용/효과 가정 |
| [Counterfactual Lab](docs/counterfactual_retention_lab.md) | 무개입 대비 액션별 기대 손익 비교 로직 |
| [발표 자료](docs/presentation.pdf) | 프로젝트 발표 슬라이드 |

---

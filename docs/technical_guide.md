# 기술 가이드

## 설치

```bash
pip install -r requirements.txt
```

## Docker 실행

```bash
docker compose up --build
```

백그라운드 실행 모드:

```bash
docker compose up -d --build
```

## 대시보드 접속

```text
http://localhost:8501
```

## 예산, 수익, ROI 산정 로직

```text
예상 증분 수익 = 고객 가치 x 반응 가능성 x 이탈 위험 - 개입 비용
예상 ROI = 예상 증분 수익 / 개입 비용
```

예상 증분 수익과 ROI가 높은 고객을 우선적으로 선정하되, 전체 마케팅 예산과 최대 타깃 고객 수 제약을 함께 준수합니다.

## Live DB 모드

플랫폼은 업로드된 비즈니스 데이터를 PostgreSQL 기반 라이브 테이블을 통해 운영 데이터로 제공할 수 있습니다. 학습이 완료된 후 생성된 아티팩트를 라이브 테이블에 시드할 수 있으며, 새로운 이벤트가 유입되면 해당 테이블이 업데이트됩니다.

핵심 흐름:

1. Docker 서비스를 시작합니다.
2. 금융 또는 이커머스 CSV/TSV 파일을 업로드합니다.
3. 컬럼 매핑을 확인하고 학습을 실행합니다.
4. 생성된 아티팩트를 PostgreSQL 라이브 테이블에 시드합니다.
5. 라이브 API로 고객 이벤트를 전송합니다.
6. 대시보드에서 점수, 추천, 액션 큐 변경 사항을 확인합니다.

헬스 체크:

```bash
curl -s "http://localhost:8000/api/v1/user-live/health" | python3 -m json.tool
```

시드 상태 확인:

```bash
curl -s "http://localhost:8000/api/v1/user-live/seed-status" | python3 -m json.tool
```

이벤트 삽입 예시:

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

고객 점수 확인:

```bash
curl -s "http://localhost:8000/api/v1/user-live/scores?customer_id=1001" | python3 -m json.tool
```

액션 큐 확인:

```bash
curl -s "http://localhost:8000/api/v1/user-live/actions?customer_id=1001" | python3 -m json.tool
```

## 저장소 구조

```text
dashboard/
  app.py                    # Streamlit 앱 진입점
  ui_labels.py              # 친화적인 라벨, 테이블 셀 번역, 차트 현지화
  ui_llm_language.py        # LLM 출력 언어 지시문
  ui_budget_formula.py      # 예산/수익/ROI 공식 UI 블록
  services/                 # API, 데이터 로딩, 인사이트, 최적화, LLM 클라이언트
  utils/                    # 포맷팅 헬퍼
src/                        # 학습 및 전처리 파이프라인
data/                       # 원본 데이터 및 피처 스토어 데이터 폴더
results_*/                  # 모드별 출력 아티팩트
models_*/                   # 모드별 모델 아티팩트
scripts/                    # 검증 및 라이브 데모 보조 스크립트
```

## 검증 체크리스트

데모 또는 제출 전에 다음 항목을 확인합니다.

- Docker 서비스가 정상적으로 시작되는지 확인합니다.
- 금융 모드와 이커머스 모드 선택이 정상 동작하는지 확인합니다.
- 첫 화면에서 기존 학습 결과를 열 수 있는지 확인합니다.
- 언어를 전환해도 분석 제어값이 초기화되지 않는지 확인합니다.
- 테이블 헤더와 테이블 셀 값이 한국어, 영어, 일본어에서 이해하기 쉽게 표시되는지 확인합니다.
- 차트 축과 제목이 현지화되어 표시되는지 확인합니다.
- `expected roi 2`와 같은 중복 지표 컬럼이 표시되지 않는지 확인합니다.
- API 키가 제공된 경우 LLM 요약이 선택된 언어로 생성되는지 확인합니다.
- 실시간 운영 화면에서 불필요한 차트 없이 라이브 점수와 액션 큐가 표시되는지 확인합니다.

## 금융 모드 — 권장 컬럼

- `customer_id`
- `timestamp` 또는 거래 일자
- `event_type` 또는 거래 유형
- 예금 또는 계좌 잔액
- 대출 잔액 또는 상환 상태
- 카드 사용 금액
- 연체 일수
- 고객 지원/상담 횟수
- 고객 세그먼트 또는 멤버십 등급

## 이커머스 모드 — 권장 컬럼

- `customer_id`
- 이벤트 타임스탬프
- 페이지 조회, 검색, 장바구니, 구매, 로그인 등의 이벤트 유형
- 주문 금액
- 상품 카테고리
- 쿠폰 또는 할인 사용 여부
- 고객 세그먼트 또는 멤버십 등급
- 탐색 또는 구매 최신성

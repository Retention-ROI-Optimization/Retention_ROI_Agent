# Retention ROI 프로젝트 협업 가이드 (CONTRIBUTING)

본 문서는 **AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 시스템** 프로젝트의 원활한 협업을 위한 GitHub 규칙을 정의합니다.

---

## 1. 🌿 브랜치 전략 (Branching Strategy)

우리는 단순화된 **GitHub Flow + dev 통합 브랜치 전략**을 사용합니다.

### 기본 브랜치

- `master` 브랜치
  - 언제나 배포 가능하거나 제출 가능한 안정 버전만 유지합니다.
  - 직접 Push를 금지합니다.
- `dev` 브랜치
  - 각 파트의 기능이 통합되고 테스트되는 개발 브랜치입니다.
  - 모든 기능 브랜치는 기본적으로 `dev`에서 분기하고 `dev`로 병합합니다.


### 기능 브랜치 규칙

기능 브랜치는 아래 형식을 따릅니다.

- `data/{기능명}`
- `ai/{기능명}`
- `be/{기능명}`
- `fe/{기능명}`
- `fix/{버그명}`
- `docs/{문서명}`
- `chore/{작업명}`

### 브랜치명 예시

- `data/customer-simulator`
- `data/cohort-analysis`
- `data/rfm-feature-engineering`
- `data/behavior-feature`
- `ai/churn-model-baseline`
- `ai/uplift-model`
- `ai/clv-estimator`
- `ai/budget-optimizer`
- `ai/ab-test-analysis`
- `be/predict-api`
- `be/postgres-schema`
- `be/docker-setup`
- `be/model-serving`
- `fe/dashboard-ui`
- `fe/api-integration`
- `fe/threshold-slider`
- `fe/roi-visualization`
- `fix/roi-calculation-bug`
- `docs/contributing-update`

### 브랜치 생성 원칙

1. 모든 작업은 반드시 `dev`의 최신 상태를 pull 받은 후 시작합니다.
2. 하나의 브랜치에서는 하나의 작업만 진행합니다.
3. 브랜치명은 소문자와 하이픈 사용을 권장합니다.

### 예시

```bash
git checkout dev
git pull origin dev
git checkout -b ai/churn-model-baseline
```
## 2. 🔀 머지 규칙 (Merge Rules)

### GitHub Branch Protection Rules 설정 (권장)
GitHub 레포지토리의 `Settings > Branches`에서 `main`과 `dev` 브랜치에 대해 다음 규칙을 켜두는 것을 강력히 권장합니다:
1. **Require pull request reviews before merging**: 최소 **1명 이상의 Approve**가 있어야 Merge 가능하도록 설정. 본인이 작성한 코드는 본인이 Merge할 수 없습니다.
2. **Require status checks to pass before merging**: CI(Test, Lint 등)가 성공해야만 Merge 가능. (추후 GitHub Actions 세팅 시)

### Merge 전략
- **`dev` 브랜치로 기능 브랜치 Merge 할 때**: `Squash and Merge`를 사용합니다. 
  - 이유: 자잘한 커밋 기록(예: "오타 수정", "print문 제거")을 하나로 깔끔하게 압축하여 `dev` 브랜치의 히스토리를 깨끗하게 유지하기 위함입니다.
- **`dev`에서 `main` 브랜치로 배포 준비 시**: 주기적으로 `Create Pull Request`를 띄워 파트 리더급 혹은 팀원 전체 리뷰 후 `Rebase and Merge` 혹은 `Merge Commit`을 생성하여 메인 브랜치로 보냅니다.

## 3. 👀 코드 리뷰 (Code Review) 포인트
PR을 올릴 때는 미리 만들어둔 **PR 템플릿** 양식에 맞춰 상세히 작성합니다.

- **Data / AI 팀**: 모델의 성능 하락(Degradation)이 없는지, Data Leakage(학습 데이터에 평가 데이터가 섞임)가 발생하지 않는 구조인지 중점 리뷰.
- **BE 팀**: API 응답 지연(Latency)이 심하지 않은지, Pydantic 모델의 에러 핸들링 로직 점검.
- **FE 팀**: UI/UX 깨짐 방지, 각 파트 서버가 죽었을 때 무한 로딩 등에 대한 예외 처리 점검.

## 4. 💬 커밋 메시지 컨벤션 (Commit Convention)

커밋 메시지는 작업 내용을 명확히 파악할 수 있도록 [Karma 스타일](https://karma-runner.github.io/6.0/dev/git-commit-msg.html)의 태그를 사용합시다.

- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 수정 (README, CONTRIBUTING 등)
- `style:` 코드 포맷팅, 세미콜론 누락 등 (코드 로직 변경 없음)
- `refactor:` 코드 리팩토링 (기능 변화 없음)
- `test:` 테스트 코드 작성
- `chore:` 빌드 업무 수정, 패키지 매니저 수정 (.gitignore, requirements.txt 등)

예시: `feat: data_loader.py에 CSV 파서 로직 추가`
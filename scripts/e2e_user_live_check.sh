#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
CUSTOMER_ID="${CUSTOMER_ID:-1001}"
EVENT_TIME="${EVENT_TIME:-2026-05-10T03:30:00+09:00}"
SOURCE_EVENT_ID="${SOURCE_EVENT_ID:-test-event-1001-001}"

json() {
  python3 -m json.tool
}

step() {
  printf '\n\033[1;34m[%s]\033[0m %s\n' "$1" "$2"
}

step 1 "live DB 상태 확인"
curl -fsS "${API_BASE_URL}/api/v1/user-live/health" | json

step 2 "기존 live table 초기화"
curl -fsS -X POST "${API_BASE_URL}/api/v1/user-live/reset?confirm=true" | json

step 3 "user 산출물을 PostgreSQL live table에 seed"
curl -fsS -X POST "${API_BASE_URL}/api/v1/user-live/seed-from-user-artifacts?reset=true" | json

step 4 "seed 결과 확인"
curl -fsS "${API_BASE_URL}/api/v1/user-live/seed-status" | json

step 5 "특정 고객 이벤트 발생"
curl -fsS -X POST "${API_BASE_URL}/api/v1/user-live/events" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": ${CUSTOMER_ID},
    \"event_type\": \"add_to_cart\",
    \"event_time\": \"${EVENT_TIME}\",
    \"amount\": 35000,
    \"source_event_id\": \"${SOURCE_EVENT_ID}\",
    \"item_category\": \"fashion\",
    \"channel\": \"web\",
    \"raw_payload\": {\"test\": true}
  }" | json

step 6 "해당 고객 feature_state 확인"
curl -fsS "${API_BASE_URL}/api/v1/user-live/feature-state?customer_id=${CUSTOMER_ID}" | json

step 7 "해당 고객 score 확인"
curl -fsS "${API_BASE_URL}/api/v1/user-live/scores?customer_id=${CUSTOMER_ID}" | json

step 8 "해당 고객 action_queue 확인"
curl -fsS "${API_BASE_URL}/api/v1/user-live/actions?customer_id=${CUSTOMER_ID}" | json

printf '\n\033[1;32mUser Live DB E2E check passed.\033[0m\n'

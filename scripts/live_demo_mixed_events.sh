#!/usr/bin/env bash
set -uo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
NEW_CUSTOMER_PROB="${NEW_CUSTOMER_PROB:-20}"   # 신규 고객 생성 확률 (%)
SLEEP_SEC="${SLEEP_SEC:-2}"
ACTION_THRESHOLD="${ACTION_THRESHOLD:-0.30}"
SCORES_LIMIT="${SCORES_LIMIT:-200}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1"
    exit 1
  }
}

require_cmd curl
require_cmd python3

echo "Live demo mixed event injection started"
echo "API_BASE_URL=${API_BASE_URL}"
echo "NEW_CUSTOMER_PROB=${NEW_CUSTOMER_PROB}%"
echo "SLEEP_SEC=${SLEEP_SEC}"
echo "ACTION_THRESHOLD=${ACTION_THRESHOLD}"
echo "Press Ctrl+C to stop"
echo

NEXT_CID=$(
  curl -s "${API_BASE_URL}/api/v1/user-live/scores" \
  | python3 -c 'import sys,json
try:
    rows=json.load(sys.stdin).get("records",[])
    ids=[int(r["customer_id"]) for r in rows if r.get("customer_id") is not None]
    print((max(ids) if ids else 1000) + 1)
except Exception:
    print(900001)'
)

while true; do
  ROLL=$((RANDOM % 100))

  if [ "$ROLL" -lt "$NEW_CUSTOMER_PROB" ]; then
    CID="$NEXT_CID"
    NEXT_CID=$((NEXT_CID + 1))

    PAYLOAD=$(
      CID="$CID" python3 - <<'PY'
import os, json, random, uuid
from datetime import datetime, timezone, timedelta

cid = int(os.environ["CID"])
now = datetime.now(timezone.utc)

category = random.choice(["fashion", "beauty", "electronics", "grocery", "sports", "home"])
channel = random.choice(["web", "app", "mobile_web"])
session_id = f"new-session-{cid}-{uuid.uuid4().hex[:8]}"

event_sequence = ["visit", random.choice(["page_view", "search"])]

if random.random() < 0.65:
    event_sequence.append("add_to_cart")

if random.random() < 0.35:
    event_sequence.append("purchase")

events = []

for i, event_type in enumerate(event_sequence):
    amount = 0

    if event_type == "add_to_cart":
        amount = random.choice([19000, 25000, 35000, 49000, 79000])
    elif event_type == "purchase":
        amount = random.choice([25000, 35000, 49000, 79000, 129000])

    events.append({
        "customer_id": cid,
        "event_type": event_type,
        "event_time": (now + timedelta(seconds=i)).isoformat(),
        "amount": amount,
        "source_event_id": f"new-{cid}-{event_type}-{int(now.timestamp())}-{i}-{uuid.uuid4().hex[:6]}",
        "item_category": category,
        "channel": channel,
        "session_id": session_id,
        "raw_payload": {
            "demo_stream": True,
            "customer_status": "new",
            "generated_by": "live_demo_mixed_events.sh"
        }
    })

print(json.dumps({"events": events}, ensure_ascii=False))
PY
    )

    curl -s -X POST \
      "${API_BASE_URL}/api/v1/user-live/events/batch?score_after_event=true&update_actions=true&action_threshold=${ACTION_THRESHOLD}" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD" \
      | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(
        "new_customer_batch:",
        "inserted=", d.get("inserted"),
        "changed=", d.get("changed_customer_ids"),
        "score_updated=", (d.get("scoring") or {}).get("updated_customers"),
        "queue_updated=", (d.get("actions") or {}).get("action_queue_updated")
    )
except Exception as e:
    print("new_customer_batch: failed", e)'

  else
    CID=$(
      curl -s "${API_BASE_URL}/api/v1/user-live/scores?limit=${SCORES_LIMIT}" \
      | python3 -c 'import sys,json,random
try:
    rows=json.load(sys.stdin).get("records",[])
    print(random.choice(rows)["customer_id"] if rows else 1001)
except Exception:
    print(1001)'
    )

    PAYLOAD=$(
      CID="$CID" python3 - <<'PY'
import os, json, random, uuid
from datetime import datetime, timezone

cid = int(os.environ["CID"])
now = datetime.now(timezone.utc)

event_type = random.choices(
    [
        "visit",
        "page_view",
        "search",
        "add_to_cart",
        "purchase",
        "support_contact",
        "refund",
        "coupon_open",
        "coupon_redeem"
    ],
    weights=[18, 24, 14, 16, 10, 7, 3, 5, 3],
    k=1
)[0]

amount = 0

if event_type == "add_to_cart":
    amount = random.choice([15000, 25000, 35000, 49000, 79000])
elif event_type == "purchase":
    amount = random.choice([25000, 35000, 59000, 89000, 149000])
elif event_type == "refund":
    amount = random.choice([15000, 25000, 35000, 59000])

payload = {
    "customer_id": cid,
    "event_type": event_type,
    "event_time": now.isoformat(),
    "amount": amount,
    "source_event_id": f"demo-existing-{cid}-{event_type}-{int(now.timestamp())}-{uuid.uuid4().hex[:6]}",
    "item_category": random.choice(["fashion", "beauty", "electronics", "grocery", "sports", "home"]),
    "channel": random.choice(["web", "app", "mobile_web", "crm"]),
    "session_id": f"session-{cid}-{uuid.uuid4().hex[:8]}",
    "raw_payload": {
        "demo_stream": True,
        "customer_status": "existing",
        "generated_by": "live_demo_mixed_events.sh"
    }
}

print(json.dumps(payload, ensure_ascii=False))
PY
    )

    curl -s -X POST \
      "${API_BASE_URL}/api/v1/user-live/events?score_after_event=true&update_actions=true&action_threshold=${ACTION_THRESHOLD}" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD" \
      | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    result = d.get("result", {})
    print(
        "existing_event:",
        "customer=", result.get("customer_id"),
        "event=", result.get("event_type"),
        "score_updated=", (d.get("scoring") or {}).get("updated_customers"),
        "queue_updated=", (d.get("actions") or {}).get("action_queue_updated")
    )
except Exception as e:
    print("existing_event: failed", e)'
  fi

  sleep "$SLEEP_SEC"
done
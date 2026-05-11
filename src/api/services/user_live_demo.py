from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.api.services.user_live_db import user_live_session


@dataclass
class DemoConfig:
    interval_seconds: float = 2.0
    new_customer_ratio: float = 0.3
    action_threshold: float = 0.30


@dataclass
class DemoState:
    running: bool = False
    task: asyncio.Task | None = None
    total_events_sent: int = 0
    new_customers_created: int = 0
    existing_customers_updated: int = 0
    latest_results: list[dict] = field(default_factory=list)
    started_at: datetime | None = None
    config: DemoConfig = field(default_factory=DemoConfig)


_demo_state = DemoState()

EVENT_TRANSITIONS = {
    "visit": {"page_view": 0.45, "search": 0.30, "add_to_cart": 0.05, "visit": 0.20},
    "page_view": {"add_to_cart": 0.30, "search": 0.20, "page_view": 0.25, "visit": 0.15, "purchase": 0.10},
    "search": {"page_view": 0.40, "add_to_cart": 0.25, "search": 0.15, "visit": 0.20},
    "add_to_cart": {"purchase": 0.45, "remove_from_cart": 0.15, "page_view": 0.20, "visit": 0.20},
    "purchase": {"visit": 0.35, "page_view": 0.30, "support_contact": 0.10, "search": 0.25},
    "support_contact": {"visit": 0.50, "page_view": 0.30, "search": 0.20},
    "remove_from_cart": {"page_view": 0.40, "search": 0.30, "visit": 0.30},
}

FIRST_EVENT_TYPES = ["visit", "page_view", "search"]
FIRST_EVENT_WEIGHTS = [0.55, 0.30, 0.15]

CATEGORIES = ["fashion", "beauty", "grocery", "sports", "health", "electronics"]


def _get_next_customer_id(db_url: str) -> int:
    with user_live_session(db_url) as conn:
        max_id = conn.execute(
            text("SELECT COALESCE(MAX(customer_id), 10000) FROM customer_feature_state")
        ).scalar_one()
    return int(max_id) + 1


def _pick_existing_customer(db_url: str) -> int | None:
    with user_live_session(db_url) as conn:
        row = conn.execute(
            text("SELECT customer_id FROM customer_scores ORDER BY RANDOM() LIMIT 1")
        ).scalar_one_or_none()
    return int(row) if row else None


def _pick_next_event_type(customer_id: int, db_url: str) -> str:
    with user_live_session(db_url) as conn:
        last = conn.execute(
            text("""
                SELECT event_type FROM customer_events
                WHERE customer_id = :cid
                ORDER BY event_time DESC LIMIT 1
            """),
            {"cid": customer_id},
        ).scalar_one_or_none()

    probs = EVENT_TRANSITIONS.get(last, EVENT_TRANSITIONS["visit"])
    types, weights = zip(*probs.items())
    return random.choices(types, weights=weights, k=1)[0]


def _random_amount(event_type: str) -> float:
    if event_type == "purchase":
        return round(max(random.gauss(45000, 15000), 10000), 0)
    if event_type == "add_to_cart":
        return round(max(random.gauss(35000, 10000), 5000), 0)
    return 0.0


def _generate_one_cycle(db_url: str, model_dir: Path, config: DemoConfig) -> dict[str, Any]:
    from src.api.routers.user_live import UserEventIn, _insert_event_and_update_feature_state
    from src.api.services.user_live_scoring import score_changed_customers
    from src.api.services.user_live_actions import update_live_actions_for_customers

    is_new = random.random() < config.new_customer_ratio
    existing_id = _pick_existing_customer(db_url) if not is_new else None

    if is_new or existing_id is None:
        customer_id = _get_next_customer_id(db_url)
        event_type = random.choices(FIRST_EVENT_TYPES, weights=FIRST_EVENT_WEIGHTS, k=1)[0]
        is_new = True
    else:
        customer_id = existing_id
        event_type = _pick_next_event_type(customer_id, db_url)

    event = UserEventIn(
        customer_id=customer_id,
        event_type=event_type,
        event_time=datetime.now(timezone.utc),
        amount=_random_amount(event_type),
        source_event_id=f"demo-{customer_id}-{int(time.time())}-{random.randint(0, 9999)}",
        item_category=random.choice(CATEGORIES),
        channel="demo_stream",
    )

    result = _insert_event_and_update_feature_state(db_url=db_url, event=event)

    scoring = None
    actions = None

    if result.get("inserted", True):
        try:
            scoring = score_changed_customers(
                db_url=db_url, model_dir=model_dir, customer_ids=[customer_id]
            )
        except Exception as exc:
            scoring = {"success": False, "error": str(exc)}

        if scoring and scoring.get("success"):
            try:
                actions = update_live_actions_for_customers(
                    db_url=db_url,
                    customer_ids=[customer_id],
                    threshold=config.action_threshold,
                )
            except Exception as exc:
                actions = {"success": False, "error": str(exc)}

    return {
        "customer_id": customer_id,
        "is_new": is_new,
        "event_type": event_type,
        "amount": event.amount,
        "score_updated": bool(scoring and scoring.get("success")),
        "churn_score": (scoring or {}).get("records", [{}])[0].get("churn_score") if scoring else None,
        "action_queued": bool(actions and actions.get("action_queue_updated", 0) > 0),
    }


async def _run_demo_loop(db_url: str, model_dir: Path, config: DemoConfig):
    _demo_state.running = True
    _demo_state.started_at = datetime.now(timezone.utc)

    while _demo_state.running:
        try:
            result = await asyncio.to_thread(_generate_one_cycle, db_url, model_dir, config)

            _demo_state.total_events_sent += 1
            if result["is_new"]:
                _demo_state.new_customers_created += 1
            else:
                _demo_state.existing_customers_updated += 1

            _demo_state.latest_results.append(result)
            if len(_demo_state.latest_results) > 500:
                _demo_state.latest_results = _demo_state.latest_results[-500:]

        except Exception:
            pass

        await asyncio.sleep(config.interval_seconds)


def start_demo(db_url: str, model_dir: Path, config: DemoConfig) -> dict[str, Any]:
    if _demo_state.running:
        return {"status": "already_running", **get_demo_status()}

    _demo_state.total_events_sent = 0
    _demo_state.new_customers_created = 0
    _demo_state.existing_customers_updated = 0
    _demo_state.latest_results = []
    _demo_state.config = config

    loop = asyncio.get_running_loop()
    _demo_state.task = loop.create_task(_run_demo_loop(db_url, model_dir, config))
    return {"status": "started", "config": {"interval_seconds": config.interval_seconds, "new_customer_ratio": config.new_customer_ratio}}


def stop_demo() -> dict[str, Any]:
    _demo_state.running = False
    if _demo_state.task and not _demo_state.task.done():
        _demo_state.task.cancel()
    _demo_state.task = None
    return {"status": "stopped", **get_demo_status()}


def reset_demo(db_url: str) -> dict[str, Any]:
    stop_demo()
    with user_live_session(db_url) as conn:
        deleted_events = conn.execute(
            text("DELETE FROM customer_events WHERE channel = 'demo_stream'")
        ).rowcount
        demo_cids = conn.execute(
            text("SELECT customer_id FROM customer_feature_state WHERE acquisition_channel = 'demo_stream'")
        ).scalars().all()
        deleted_scores = 0
        deleted_actions = 0
        deleted_recs = 0
        deleted_features = 0
        if demo_cids:
            cid_list = list(demo_cids)
            deleted_scores = conn.execute(
                text("DELETE FROM customer_scores WHERE customer_id = ANY(:ids)"),
                {"ids": cid_list},
            ).rowcount
            deleted_actions = conn.execute(
                text("DELETE FROM action_queue WHERE customer_id = ANY(:ids)"),
                {"ids": cid_list},
            ).rowcount
            deleted_recs = conn.execute(
                text("DELETE FROM recommendation_candidates WHERE customer_id = ANY(:ids)"),
                {"ids": cid_list},
            ).rowcount
            deleted_features = conn.execute(
                text("DELETE FROM customer_feature_state WHERE customer_id = ANY(:ids)"),
                {"ids": cid_list},
            ).rowcount
        conn.commit()
    _demo_state.total_events_sent = 0
    _demo_state.new_customers_created = 0
    _demo_state.existing_customers_updated = 0
    _demo_state.latest_results = []
    _demo_state.started_at = None
    return {
        "status": "reset",
        "deleted_events": deleted_events,
        "deleted_scores": deleted_scores,
        "deleted_actions": deleted_actions,
        "deleted_recommendations": deleted_recs,
        "deleted_feature_states": deleted_features,
    }


def get_demo_status() -> dict[str, Any]:
    return {
        "running": _demo_state.running,
        "total_events_sent": _demo_state.total_events_sent,
        "new_customers_created": _demo_state.new_customers_created,
        "existing_customers_updated": _demo_state.existing_customers_updated,
        "started_at": _demo_state.started_at.isoformat() if _demo_state.started_at else None,
        "latest_results": _demo_state.latest_results,
    }

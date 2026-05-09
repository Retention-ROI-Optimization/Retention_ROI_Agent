from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


_engine: Engine | None = None


def get_user_live_engine(db_url: str) -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(db_url, pool_pre_ping=True, future=True)
    return _engine


@contextmanager
def user_live_session(db_url: str):
    engine = get_user_live_engine(db_url)
    with engine.begin() as conn:
        yield conn


def init_user_live_tables(db_url: str) -> None:
    with user_live_session(db_url) as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS customer_events (
            event_id BIGSERIAL PRIMARY KEY,
            customer_id BIGINT NOT NULL,
            event_type TEXT NOT NULL,
            event_time TIMESTAMPTZ NOT NULL,
            amount NUMERIC DEFAULT 0,
            item_category TEXT,
            channel TEXT,
            session_id TEXT,
            raw_payload JSONB,
            processed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS customer_feature_state (
            customer_id BIGINT PRIMARY KEY,
            last_event_time TIMESTAMPTZ,
            visit_7d INT DEFAULT 0,
            browse_7d INT DEFAULT 0,
            purchase_30d INT DEFAULT 0,
            revenue_30d NUMERIC DEFAULT 0,
            support_30d INT DEFAULT 0,
            coupon_open_30d INT DEFAULT 0,
            coupon_redeem_30d INT DEFAULT 0,
            cart_remove_7d INT DEFAULT 0,
            inactivity_days FLOAT DEFAULT 0,
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS customer_scores (
            customer_id BIGINT PRIMARY KEY,
            churn_score FLOAT,
            clv FLOAT,
            uplift_score FLOAT,
            expected_roi FLOAT,
            expected_incremental_profit FLOAT,
            risk_segment TEXT,
            uplift_segment TEXT,
            model_version TEXT,
            scored_at TIMESTAMPTZ DEFAULT now()
        )
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS recommendation_candidates (
            id BIGSERIAL PRIMARY KEY,
            customer_id BIGINT NOT NULL,
            recommended_action TEXT,
            recommended_category TEXT,
            coupon_cost NUMERIC,
            expected_roi FLOAT,
            expected_incremental_profit FLOAT,
            priority_score FLOAT,
            reason_tags TEXT,
            generated_at TIMESTAMPTZ DEFAULT now()
        )
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS action_queue (
            id BIGSERIAL PRIMARY KEY,
            customer_id BIGINT NOT NULL,
            action_status TEXT DEFAULT 'queued',
            recommended_action TEXT,
            intervention_intensity TEXT,
            coupon_cost NUMERIC,
            expected_profit NUMERIC,
            expected_roi FLOAT,
            priority_score FLOAT,
            trigger_reason TEXT,
            queued_at TIMESTAMPTZ DEFAULT now(),
            dispatched_at TIMESTAMPTZ
        )
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_customer_events_customer_time
        ON customer_events (customer_id, event_time DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_customer_scores_churn
        ON customer_scores (churn_score DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_action_queue_status_priority
        ON action_queue (action_status, priority_score DESC)
        """))
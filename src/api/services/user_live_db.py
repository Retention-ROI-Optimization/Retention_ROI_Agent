from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine


_engine: Engine | None = None


def get_user_live_engine(db_url: str) -> Engine:
    """
    user mode 실시간 PostgreSQL 연결 엔진.
    FastAPI 프로세스 안에서 재사용한다.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            future=True,
        )
    return _engine


@contextmanager
def user_live_session(db_url: str) -> Iterator[Connection]:
    """
    트랜잭션 단위 DB 세션.
    with 블록이 정상 종료되면 commit, 예외가 나면 rollback된다.
    """
    engine = get_user_live_engine(db_url)
    with engine.begin() as conn:
        yield conn


def init_user_live_tables(db_url: str) -> None:
    """
    user mode 실시간 운영용 테이블 생성.
    2단계에서는 customer_events와 customer_feature_state가 핵심이다.
    customer_scores/recommendation/action_queue는 이후 단계에서 사용한다.
    """
    with user_live_session(db_url) as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS customer_events (
            event_id BIGSERIAL PRIMARY KEY,
            source_event_id TEXT,
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
            search_7d INT DEFAULT 0,
            add_to_cart_7d INT DEFAULT 0,
            cart_remove_7d INT DEFAULT 0,
            purchase_30d INT DEFAULT 0,
            revenue_30d NUMERIC DEFAULT 0,
            support_30d INT DEFAULT 0,
            refund_30d INT DEFAULT 0,
            coupon_open_30d INT DEFAULT 0,
            coupon_redeem_30d INT DEFAULT 0,
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

        # 1단계에서 이미 테이블을 만들었더라도 컬럼 보강 가능하게 처리
        conn.execute(text("""
        ALTER TABLE customer_events
        ADD COLUMN IF NOT EXISTS source_event_id TEXT
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS search_7d INT DEFAULT 0
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS add_to_cart_7d INT DEFAULT 0
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS refund_30d INT DEFAULT 0
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_customer_events_customer_time
        ON customer_events (customer_id, event_time DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_customer_events_type_time
        ON customer_events (event_type, event_time DESC)
        """))

        conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_customer_events_source_event_id
        ON customer_events (source_event_id)
        WHERE source_event_id IS NOT NULL
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS is_new_customer BOOLEAN DEFAULT FALSE
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMPTZ
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS event_count_total INT DEFAULT 0
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS persona TEXT
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS acquisition_channel TEXT
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_customer_feature_state_updated
        ON customer_feature_state (updated_at DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_customer_scores_churn
        ON customer_scores (churn_score DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_action_queue_status_priority
        ON action_queue (action_status, priority_score DESC)
        """))
def ensure_user_live_seed_columns(db_url: str) -> None:
    """
    3단계 seed 작업에 필요한 보강 컬럼을 추가한다.

    이미 1~2단계에서 테이블이 생성되어 있어도 안전하게 실행된다.
    feature_payload / score_payload / source_payload는 기존 CSV row 전체를 JSONB로 저장하기 위한 컬럼이다.
    이후 4단계 모델 재추론에서 원본 feature schema를 맞출 때 사용할 수 있다.
    """
    init_user_live_tables(db_url)

    with user_live_session(db_url) as conn:
        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS feature_payload JSONB DEFAULT '{}'::jsonb
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS seeded_at TIMESTAMPTZ
        """))

        conn.execute(text("""
        ALTER TABLE customer_feature_state
        ADD COLUMN IF NOT EXISTS source_updated_at TIMESTAMPTZ
        """))

        conn.execute(text("""
        ALTER TABLE customer_scores
        ADD COLUMN IF NOT EXISTS score_payload JSONB DEFAULT '{}'::jsonb
        """))

        conn.execute(text("""
        ALTER TABLE customer_scores
        ADD COLUMN IF NOT EXISTS seeded_at TIMESTAMPTZ
        """))

        conn.execute(text("""
        ALTER TABLE customer_scores
        ADD COLUMN IF NOT EXISTS persona TEXT
        """))

        conn.execute(text("""
        ALTER TABLE recommendation_candidates
        ADD COLUMN IF NOT EXISTS source_payload JSONB DEFAULT '{}'::jsonb
        """))

        conn.execute(text("""
        ALTER TABLE recommendation_candidates
        ADD COLUMN IF NOT EXISTS seeded_at TIMESTAMPTZ
        """))

        conn.execute(text("""
        ALTER TABLE action_queue
        ADD COLUMN IF NOT EXISTS source_payload JSONB DEFAULT '{}'::jsonb
        """))

        conn.execute(text("""
        ALTER TABLE action_queue
        ADD COLUMN IF NOT EXISTS seeded_at TIMESTAMPTZ
        """))
        conn.execute(text("""
        ALTER TABLE action_queue
        ADD COLUMN IF NOT EXISTS seeded_at TIMESTAMPTZ
        """))


def ensure_user_live_job_tables(db_url: str) -> None:
    """
    7단계 배치/주기 작업 실행 기록 테이블을 생성한다.

    user_live_job_runs는 이벤트 단위 실시간 처리와 분리된
    drift check, recent action refresh, 재학습/재계산 job의 실행 상태를 기록한다.
    """
    init_user_live_tables(db_url)

    with user_live_session(db_url) as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS user_live_job_runs (
            id BIGSERIAL PRIMARY KEY,
            job_name TEXT NOT NULL,
            job_status TEXT NOT NULL,
            started_at TIMESTAMPTZ DEFAULT now(),
            finished_at TIMESTAMPTZ,
            duration_sec FLOAT,
            message TEXT,
            result_payload JSONB DEFAULT '{}'::jsonb
        )
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_user_live_job_runs_started
        ON user_live_job_runs (started_at DESC)
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_user_live_job_runs_name_status
        ON user_live_job_runs (job_name, job_status)
        """))
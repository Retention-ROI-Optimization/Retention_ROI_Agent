"""
검증용 소형 CSV 생성기 (오늘 시점 기준 최근 1년).

특징:
  - 오늘(today)을 기준으로 최근 1년 내 이벤트 데이터 생성
  - 이탈 예측이 의미를 가지도록 고객을 4개 코호트로 분포:
      * 30% 최근 활성 (마지막 활동: 0~30일 전)
      * 30% 보통 활성  (30~90일 전)
      * 25% 위험      (90~180일 전)
      * 15% 이탈      (180~365일 전)
  - 다양한 회사 명명의 event_type 12종
  - 문자열 customer_id (factorize 검증)
  - purchase 계열에만 amount > 0 (orders 생성 검증)

사용:
  python3 scripts/make_test_csv.py
  → data/uploads/test_small.csv (~3,000행, 500명 고객, 최근 1년)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# 코호트 분포: (이름, 비율, 마지막 활동 일수 범위)
COHORTS = [
    ("active",  0.30, 0,   30),
    ("normal",  0.30, 30,  90),
    ("at_risk", 0.25, 90,  180),
    ("churned", 0.15, 180, 365),
]


def generate_test_csv(
    n_customers: int = 500,
    avg_events_per_customer: int = 6,
    out_path: str | Path = "data/uploads/test_small.csv",
    seed: int = 42,
    anchor_date: str | None = None,  # 기본: 오늘
) -> Path:
    rng = np.random.default_rng(seed)

    # 기준 날짜 (오늘 또는 사용자 지정)
    today = pd.Timestamp(anchor_date) if anchor_date else pd.Timestamp.now().normalize()

    # ── 코호트 배정 ──
    cohort_names = []
    for name, ratio, _, _ in COHORTS:
        cohort_names.extend([name] * int(round(n_customers * ratio)))
    # rounding 보정
    while len(cohort_names) < n_customers:
        cohort_names.append("normal")
    cohort_names = cohort_names[:n_customers]
    rng.shuffle(cohort_names)
    cohort_lookup = {c[0]: c for c in COHORTS}

    # ── 사용자별 프로파일 ──
    customer_ids = [f"USR_{i:05d}" for i in range(1, n_customers + 1)]

    # 가입일: 마지막 활동보다 무조건 이전. 최소 30일 이상 전 ~ 최대 2년 전.
    last_activity_dates = []
    signup_dates = []
    for cohort in cohort_names:
        _, _, lo_days, hi_days = cohort_lookup[cohort]
        days_ago = int(rng.integers(lo_days, hi_days + 1))
        last_act = today - pd.Timedelta(days=days_ago)
        last_activity_dates.append(last_act)

        # 가입일: 마지막 활동보다 30일~600일 전
        tenure_days = int(rng.integers(30, 600))
        signup_dates.append(last_act - pd.Timedelta(days=tenure_days))

    countries = rng.choice(["KR", "US", "JP", "VN", "BR"], size=n_customers, p=[0.4, 0.25, 0.15, 0.1, 0.1])
    devices = rng.choice(["mobile_app", "mobile_web", "desktop"], size=n_customers, p=[0.55, 0.3, 0.15])
    channels = rng.choice(["organic", "paid_ads", "email", "referral", "push"], size=n_customers)
    segments = rng.choice(["new", "active", "at_risk", "vip"], size=n_customers, p=[0.3, 0.45, 0.2, 0.05])

    # ── 회사 고유 event_type 값들 ──
    event_pool = [
        ("product_view", 0.25),
        ("page_view", 0.18),
        ("search_query", 0.10),
        ("add_to_cart", 0.08),
        ("checkout_start", 0.06),
        ("purchase", 0.05),
        ("payment_success", 0.04),
        ("support_chat", 0.03),
        ("refund_request", 0.02),
        ("login", 0.10),
        ("custom_promo_click", 0.05),
        ("scroll_to_bottom", 0.04),
    ]
    event_names = [e[0] for e in event_pool]
    event_probs = np.array([e[1] for e in event_pool])
    event_probs = event_probs / event_probs.sum()

    categories = ["fashion", "beauty", "grocery", "sports", "electronics"]

    # ── 이벤트 행 생성 ──
    rows = []
    for i, cid in enumerate(customer_ids):
        n_events = max(1, int(rng.poisson(avg_events_per_customer)))
        signup = signup_dates[i]
        last_act = last_activity_dates[i]
        cohort = cohort_names[i]

        # 이벤트 타임스탬프: signup ~ last_act 사이에 분포
        # 마지막 이벤트는 last_act에 가깝게, 나머지는 그 사이 random
        span_days = max((last_act - signup).days, 1)

        for ev_idx in range(n_events):
            ev = rng.choice(event_names, p=event_probs)
            # 마지막 이벤트는 last_act 근처(0~3일)
            if ev_idx == n_events - 1:
                offset_from_last = int(rng.integers(0, 4))
                ts = last_act - pd.Timedelta(days=offset_from_last, hours=int(rng.integers(0, 24)))
            else:
                rand_offset_days = int(rng.integers(0, span_days))
                ts = signup + pd.Timedelta(days=rand_offset_days, hours=int(rng.integers(0, 24)))

            # purchase 계열만 amount > 0
            if ev in {"purchase", "payment_success", "checkout_start"}:
                amount = round(float(rng.normal(45000, 15000)), 2)
                amount = max(amount, 5000)
            else:
                amount = 0.0

            rows.append({
                "user_id": cid,
                "event_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": ev,
                "country": countries[i],
                "device_type": devices[i],
                "acquisition_channel": channels[i],
                "customer_segment": segments[i],
                "product_category": rng.choice(categories),
                "amount": amount,
                "session_id": f"SES_{i}_{int(ts.timestamp()) % 10000}",
                "_cohort": cohort,  # 디버깅용 (저장 안 함)
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["user_id", "event_time"]).reset_index(drop=True)

    # 디버깅용 cohort 분포 (저장하기 전 추출)
    cohort_dist = df.drop_duplicates("user_id")["_cohort"].value_counts()
    df = df.drop(columns=["_cohort"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # 요약
    print(f"✅ 생성 완료: {out_path}")
    print(f"   기준일(today):     {today.strftime('%Y-%m-%d')}")
    print(f"   총 행 수:          {len(df):,}")
    print(f"   고유 고객 수:      {df['user_id'].nunique():,}")
    print(f"   이벤트 기간:       {df['event_time'].min()} ~ {df['event_time'].max()}")
    print(f"   고객 코호트 분포:")
    for name, _, lo, hi in COHORTS:
        cnt = int(cohort_dist.get(name, 0))
        print(f"     {name:8} ({lo:3}~{hi:3}일 전 마지막 활동): {cnt}명")
    print(f"   event_type 분포 (상위):")
    for ev, cnt in df["event_type"].value_counts().head(8).items():
        print(f"     {ev:25} {cnt:>6,}")
    print(f"   purchase amount 합계: {df.loc[df['amount'] > 0, 'amount'].sum():,.0f}원")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-customers", type=int, default=500)
    parser.add_argument("--avg-events", type=int, default=6)
    parser.add_argument("--out", type=str, default="data/uploads/test_small.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--anchor-date",
        type=str,
        default=None,
        help="기준일 (예: 2026-05-04). 미지정 시 오늘.",
    )
    args = parser.parse_args()

    generate_test_csv(
        n_customers=args.n_customers,
        avg_events_per_customer=args.avg_events,
        out_path=args.out,
        seed=args.seed,
        anchor_date=args.anchor_date,
    )

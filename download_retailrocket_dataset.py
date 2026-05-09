"""
download_retailrocket_dataset.py
=================================

Retailrocket Recommender System Dataset (2015-2016) 을 다운로드하여
Retention_ROI_Agent_customer-data 레포의 ingest 모드가 받을 수 있는
표준 CSV 형식으로 변환하는 스크립트.

원본은 270만 이벤트로 메모리 부담이 크므로 기본적으로 활동 고객 30,000명을
샘플링한다. --sample-size 로 조정 가능.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


GITHUB_MIRROR = (
    "https://raw.githubusercontent.com/"
    "Sapphirine/Real-time-Recommendation-System-Based-on-User-Behavior-Data/"
    "master/dataset/events.csv"
)
BACKUP_MIRROR = (
    "https://github.com/datasciencescoop/Datasets/"
    "raw/master/Retailrocket%20Events.csv"
)

EVENT_MAPPING = {
    "view": "page_view",
    "addtocart": "add_to_cart",
    "transaction": "purchase",
}

DEFAULT_SAMPLE_SIZE = 30_000
MIN_EVENTS_PER_USER = 5
RANDOM_SEED = 42


def download_file(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        print(f"[skip] 이미 존재함: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"[download] {url}")
    print(f"   -> {dest_path}")
    try:
        urlretrieve(url, dest_path)
        size_mb = dest_path.stat().st_size / 1e6
        if size_mb < 1.0:
            print(f"[warn] 다운로드된 파일이 너무 작음: {size_mb:.2f} MB.")
            return False
        print(f"[ok] 다운로드 완료: {size_mb:.1f} MB")
        return True
    except Exception as exc:
        print(f"[error] 다운로드 실패: {exc}")
        return False


def load_events(csv_path: Path) -> pd.DataFrame:
    print(f"[load] CSV 로딩 중...")
    df = pd.read_csv(csv_path)
    print(f"[ok] 원본 행 수: {len(df):,}")
    print(f"   - 컬럼: {list(df.columns)}")
    return df


def sample_active_users(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    print(f"[sample] 고객 샘플링 시작")
    initial_users = df["visitorid"].nunique()
    print(f"   - 원본 고유 고객 수: {initial_users:,}명")

    user_counts = df["visitorid"].value_counts()
    active_users = user_counts[user_counts >= MIN_EVENTS_PER_USER].index
    print(f"   - 이벤트 {MIN_EVENTS_PER_USER}건 이상 고객: {len(active_users):,}명")

    if len(active_users) <= sample_size:
        print(f"   - 전체 활동 고객 사용 ({len(active_users):,}명)")
        sampled_ids = active_users
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        sampled_ids = rng.choice(active_users, size=sample_size, replace=False)
        print(f"   - 무작위 추출: {sample_size:,}명")

    df_sampled = df[df["visitorid"].isin(sampled_ids)].copy()
    print(f"   - 샘플링 후 이벤트 수: {len(df_sampled):,}건 "
          f"(원본의 {len(df_sampled)/len(df)*100:.1f}%)")
    return df_sampled


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    print("[clean] 데이터 정제 시작")
    initial = len(df)

    df = df.dropna(subset=["visitorid", "event", "itemid", "timestamp"]).copy()
    print(f"   - 결측 제거: {initial - len(df):,}행 삭제")

    before = len(df)
    df = df[df["event"].isin(EVENT_MAPPING.keys())]
    print(f"   - 알 수 없는 이벤트 제거: {before - len(df):,}행 삭제")

    timestamps = pd.to_datetime(df["timestamp"], unit="ms")

    standardized = pd.DataFrame({
        "customer_id": df["visitorid"].astype(int).astype(str),
        "event_type": df["event"].map(EVENT_MAPPING),
        "timestamp": timestamps,
        "amount": 0.0,
        "quantity": 1,
        "product_id": df["itemid"].astype(int).astype(str),
    })

    standardized = standardized.sort_values(
        ["customer_id", "timestamp"]
    ).reset_index(drop=True)

    print(f"[ok] 정제 완료:")
    print(f"   - 최종 {len(standardized):,}행")
    print(f"   - 고유 고객: {standardized['customer_id'].nunique():,}명")
    print(f"   - 고유 상품: {standardized['product_id'].nunique():,}개")
    print(f"   - 기간: {standardized['timestamp'].min()} ~ {standardized['timestamp'].max()}")
    print(f"   - 이벤트 종류별 분포:")
    for evt, cnt in standardized["event_type"].value_counts().items():
        print(f"       {evt}: {cnt:,}")
    return standardized


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--no-sample", action="store_true")
    args = p.parse_args()

    base_dir = Path("data/user")
    raw_csv = base_dir / "retailrocket_events_raw.csv"
    output_csv = base_dir / "retailrocket_events.csv"

    success = download_file(GITHUB_MIRROR, raw_csv)
    if not success:
        print(f"[fallback] 백업 미러를 시도합니다...")
        success = download_file(BACKUP_MIRROR, raw_csv)

    if not success or not raw_csv.exists():
        print(f"\n[error] 자동 다운로드 실패. 수동 안내:")
        print(f"  https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset")
        print(f"  → events.csv 를 {raw_csv.resolve()} 위치로 복사")
        return 1

    df = load_events(raw_csv)

    if args.no_sample:
        print("[sample] --no-sample 옵션: 전체 데이터 사용")
    else:
        df = sample_active_users(df, args.sample_size)

    standardized = clean_and_standardize(df)

    standardized.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[done] 저장 완료: {output_csv}")
    print(f"\n다음 단계:")
    print(f"  py src\\main.py --mode ingest --csv-path {output_csv} --data-dir data\\raw_user")
    return 0


if __name__ == "__main__":
    sys.exit(main())

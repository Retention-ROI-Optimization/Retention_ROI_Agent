from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from dashboard.services.cohort_service import (
    get_activity_definition_label,
    get_available_activity_definitions,
    get_available_retention_modes,
    get_cohort_curve,
    get_cohort_summary,
    get_retention_mode_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cohort retention outputs.")
    parser.add_argument("--path", default="data/raw/cohort_retention.csv", help="Path to cohort_retention.csv")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    df = pd.read_csv(path)
    print(f"Loaded: {path} ({len(df):,} rows)")

    activity_options = get_available_activity_definitions(df)
    mode_options = get_available_retention_modes(df)
    print(f"Activity definitions: {activity_options}")
    print(f"Retention modes: {mode_options}")
    print()

    for activity_definition in activity_options:
        for retention_mode in mode_options:
            curve = get_cohort_curve(
                df,
                activity_definition=activity_definition,
                retention_mode=retention_mode,
            )
            if curve.empty:
                continue
            summary = get_cohort_summary(
                df,
                activity_definition=activity_definition,
                retention_mode=retention_mode,
            )
            print(
                f"[{get_activity_definition_label(activity_definition)} | {get_retention_mode_label(retention_mode)}]"
            )
            print(
                f"  cohorts={summary['cohort_count']}, observed_periods={summary['observed_periods']}, "
                f"month1={summary['month1_avg_retention']:.4f} "
                if pd.notna(summary['month1_avg_retention'])
                else f"  cohorts={summary['cohort_count']}, observed_periods={summary['observed_periods']}, month1=nan "
            )
            print(
                f"  comparable_period={summary['comparable_period']}, comparable_avg={summary['comparable_avg_retention']:.4f}"
                if pd.notna(summary['comparable_avg_retention'])
                else f"  comparable_period={summary['comparable_period']}, comparable_avg=nan"
            )
            if retention_mode == "point":
                print(f"  non_monotonic_cohorts={summary['non_monotonic_cohort_count']}")
            print()


if __name__ == "__main__":
    main()

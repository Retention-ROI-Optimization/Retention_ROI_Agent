from __future__ import annotations

import argparse
import json

from .abtest import run_campaign_result_analysis
from .pipeline import run_universal_onboarding_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal AutoOps CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    onboard = sub.add_parser("onboard", help="Map arbitrary CSV schema to canonical schema and refresh artifacts")
    onboard.add_argument("--csv", required=True)
    onboard.add_argument("--budget", type=int, default=50_000_000)
    onboard.add_argument("--threshold", type=float, default=0.5)
    onboard.add_argument("--max-customers", type=int, default=1000)
    onboard.add_argument("--mapping-json", default=None, help='Optional manual mapping JSON, e.g. {"customer_id":"회원번호"}')

    ab = sub.add_parser("campaign-results", help="Analyze post-campaign result CSV")
    ab.add_argument("--csv", required=True)
    ab.add_argument("--assignment-csv", default="results/campaign_assignment.csv")

    args = parser.parse_args()
    if args.command == "onboard":
        manual = json.loads(args.mapping_json) if args.mapping_json else None
        result = run_universal_onboarding_pipeline(
            args.csv,
            budget=args.budget,
            threshold=args.threshold,
            max_customers=args.max_customers,
            manual_mapping=manual,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "campaign-results":
        result = run_campaign_result_analysis(args.csv, assignment_csv=args.assignment_csv)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

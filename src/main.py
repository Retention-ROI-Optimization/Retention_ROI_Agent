from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflows.pipeline_runner import (
    ensure_simulation_outputs,
    run_churn_training_pipeline,
    run_feature_engineering_pipeline,
    run_optimize_pipeline,
    run_uplift_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retention ROI project entrypoint")
    parser.add_argument("--mode", required=True, choices=["features", "train", "uplift", "optimize", "simulate"])
    parser.add_argument("--budget", type=int, default=50000000)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--feature-store-dir", default="data/feature_store")
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 data/raw 결과가 있어도 무시하고 시뮬레이션 데이터를 다시 생성합니다.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="시뮬레이터 난수 시드. 지정하지 않으면 기본값(42)을 사용합니다.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="시드를 고정하지 않고 시스템 난수를 사용해 실행마다 다른 데이터를 생성합니다.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.seed is not None and args.randomize:
        raise SystemExit("--seed and --randomize cannot be used together.")

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    result_dir = Path(args.result_dir)
    feature_store_dir = Path(args.feature_store_dir)

    common_simulation_kwargs = {
        "force_simulation": args.force,
        "simulation_seed": args.seed,
        "randomize_simulation": args.randomize,
    }

    if args.mode == "simulate":
        ensure_simulation_outputs(
            data_dir,
            force=args.force,
            random_seed=args.seed,
            randomize=args.randomize,
        )
        print(f"Simulation outputs are ready in {data_dir}")
        if args.force:
            print("Simulation raw files were regenerated.")
        if args.randomize:
            print("Simulation used a randomized seed.")
        elif args.seed is not None:
            print(f"Simulation used seed={args.seed}.")
        else:
            print("Simulation used the default seed=42.")
        return 0

    if args.mode == "features":
        result = run_feature_engineering_pipeline(
            data_dir,
            result_dir,
            feature_store_dir=feature_store_dir,
            **common_simulation_kwargs,
        )
    elif args.mode == "train":
        result = run_churn_training_pipeline(
            data_dir,
            model_dir,
            result_dir,
            feature_store_dir=feature_store_dir,
            **common_simulation_kwargs,
        )
    elif args.mode == "uplift":
        result = run_uplift_pipeline(
            data_dir,
            result_dir,
            **common_simulation_kwargs,
        )
    else:
        result = run_optimize_pipeline(
            data_dir,
            result_dir,
            budget=args.budget,
            **common_simulation_kwargs,
        )

    print(f"Mode: {result['mode']}")
    if result.get("model_path"):
        print(f"Model saved to: {result['model_path']}")
    if result.get("metrics_path"):
        print(f"Metrics saved to: {result['metrics_path']}")
    if result.get("primary_result_path"):
        print(f"Primary result saved to: {result['primary_result_path']}")
    if result.get("extra_result_paths"):
        for path in result["extra_result_paths"]:
            print(f"Additional result: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

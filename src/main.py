from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime.scoring import (
    RealtimeStreamConfig,
    bootstrap_realtime_state,
    consume_stream_events,
    produce_events_to_stream,
)
from src.workflows.pipeline_runner import (
    ensure_simulation_outputs,
    run_ab_test_pipeline,
    run_churn_training_pipeline,
    run_cohort_journey_pipeline,
    run_clv_prediction_pipeline,
    run_explainability_pipeline,
    run_feature_engineering_pipeline,
    run_optimize_pipeline,
    run_recommendation_pipeline,
    run_segmentation_priority_pipeline,
    run_survival_pipeline,
    run_simulation_fidelity_pipeline,
    run_uplift_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retention ROI project entrypoint")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "features",
            "train",
            "uplift",
            "clv",
            "segment",
            "optimize",
            "abtest",
            "simulate",
            "recommend",
            "cohort",
            "survival",
            "explain",
            "fidelity",
            "realtime-bootstrap",
            "realtime-produce",
            "realtime-consume",
            "realtime-replay",
        ],
    )
    parser.add_argument("--budget", type=int, default=50000000)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--max-customers",
        dest="max_customers",
        type=int,
        default=1000,
        help="recommend 모드에서 최종 타겟팅 후보 상한을 지정합니다.",
    )
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
    parser.add_argument("--train-test-size", type=float, default=0.20, help="train 모드의 테스트셋 비율")
    parser.add_argument("--train-random-state", type=int, default=42, help="train 모드 난수 시드")
    parser.add_argument("--train-shap-sample-size", type=int, default=300, help="학습 아티팩트 SHAP 샘플 수")
    parser.add_argument(
        "--train-models",
        default="xgboost,lightgbm",
        help="train 모드 후보 모델 목록. 예: xgboost,lightgbm",
    )
    parser.add_argument("--threshold-tp-value", type=float, default=120000.0, help="threshold 선택 시 TP 보상")
    parser.add_argument("--threshold-fp-cost", type=float, default=18000.0, help="threshold 선택 시 FP 비용")
    parser.add_argument("--threshold-fn-cost", type=float, default=60000.0, help="threshold 선택 시 FN 비용")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="실시간 스코어링용 Redis URL")
    parser.add_argument("--stream-limit", type=int, default=10000, help="실시간 replay/produce에서 스트림에 적재할 이벤트 수")
    parser.add_argument("--stream-max-events", type=int, default=10000, help="실시간 consume/replay에서 처리할 최대 이벤트 수")
    parser.add_argument("--stream-sleep-ms", type=int, default=0, help="produce 모드에서 이벤트 간 대기(ms)")
    parser.add_argument("--survival-horizon-days", type=int, default=90, help="survival 모드 예측 horizon(day)")
    return parser


def _print_result(result: dict) -> int:
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
        return _print_result(result)

    if args.mode == "train":
        candidate_models = [item.strip() for item in str(args.train_models).split(",") if item.strip()]
        result = run_churn_training_pipeline(
            data_dir,
            model_dir,
            result_dir,
            feature_store_dir=feature_store_dir,
            test_size=args.train_test_size,
            random_state=args.train_random_state,
            shap_sample_size=args.train_shap_sample_size,
            candidate_models=candidate_models,
            threshold_tp_value=args.threshold_tp_value,
            threshold_fp_cost=args.threshold_fp_cost,
            threshold_fn_cost=args.threshold_fn_cost,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "uplift":
        result = run_uplift_pipeline(
            data_dir,
            result_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "clv":
        result = run_clv_prediction_pipeline(
            data_dir,
            result_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "segment":
        result = run_segmentation_priority_pipeline(
            data_dir,
            result_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "abtest":
        result = run_ab_test_pipeline(
            data_dir,
            result_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "recommend":
        result = run_recommendation_pipeline(
            data_dir,
            result_dir,
            budget=args.budget,
            threshold=args.threshold,
            max_customers=args.max_customers,
            model_dir=model_dir,
            feature_store_dir=feature_store_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "cohort":
        result = run_cohort_journey_pipeline(
            data_dir,
            result_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "survival":
        result = run_survival_pipeline(
            data_dir,
            model_dir,
            result_dir,
            feature_store_dir=feature_store_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "explain":
        result = run_explainability_pipeline(
            data_dir,
            result_dir,
            feature_store_dir=feature_store_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "fidelity":
        result = run_simulation_fidelity_pipeline(
            data_dir,
            result_dir,
            **common_simulation_kwargs,
        )
        return _print_result(result)

    if args.mode == "realtime-bootstrap":
        ensure_simulation_outputs(
            data_dir,
            force=args.force,
            random_seed=args.seed,
            randomize=args.randomize,
        )
        config = RealtimeStreamConfig(redis_url=args.redis_url)
        payload = bootstrap_realtime_state(data_dir, result_dir, config, reset_stream=True)
        print(json.dumps(payload.get('summary', {}), ensure_ascii=False, indent=2))
        return 0

    if args.mode == "realtime-produce":
        ensure_simulation_outputs(
            data_dir,
            force=args.force,
            random_seed=args.seed,
            randomize=args.randomize,
        )
        config = RealtimeStreamConfig(redis_url=args.redis_url)
        payload = produce_events_to_stream(
            data_dir,
            result_dir,
            config,
            limit=args.stream_limit,
            sleep_ms=args.stream_sleep_ms,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "realtime-consume":
        ensure_simulation_outputs(
            data_dir,
            force=args.force,
            random_seed=args.seed,
            randomize=args.randomize,
        )
        config = RealtimeStreamConfig(redis_url=args.redis_url)
        payload = consume_stream_events(
            data_dir,
            result_dir,
            config,
            max_events=args.stream_max_events,
        )
        print(json.dumps(payload.get('summary', {}), ensure_ascii=False, indent=2))
        return 0

    if args.mode == "realtime-replay":
        ensure_simulation_outputs(
            data_dir,
            force=args.force,
            random_seed=args.seed,
            randomize=args.randomize,
        )
        config = RealtimeStreamConfig(redis_url=args.redis_url)
        bootstrap_realtime_state(data_dir, result_dir, config, reset_stream=True)
        produce_events_to_stream(
            data_dir,
            result_dir,
            config,
            limit=args.stream_limit,
            reset_stream=True,
        )
        payload = consume_stream_events(
            data_dir,
            result_dir,
            config,
            max_events=args.stream_max_events,
        )
        print(json.dumps(payload.get('summary', {}), ensure_ascii=False, indent=2))
        return 0

    result = run_optimize_pipeline(
        data_dir,
        result_dir,
        budget=args.budget,
        model_dir=model_dir,
        feature_store_dir=feature_store_dir,
        **common_simulation_kwargs,
    )
    return _print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())

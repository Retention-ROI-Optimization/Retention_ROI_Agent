from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.api.services.analytics import allocate_budget, budget_allocation_by_segment, get_budget_result
from src.clv.modeling import run_clv_pipeline
from src.experiments.ab_testing import run_ab_test_analysis
from src.features.engineering import build_feature_dataset
from src.optimization.budgeting import run_budget_optimization
from src.segmentation.prioritization import run_segmentation_pipeline
from src.simulator.config import DEFAULT_CONFIG, SimulationConfig
from src.simulator.pipeline import run_simulation
from src.uplift.modeling import run_uplift_modeling
from src.recommendations.modeling import run_personalized_recommendation_pipeline
from src.analytics.cohort_journey import run_cohort_and_journey_analysis
from src.realtime.scoring import (
    RealtimeStreamConfig,
    bootstrap_realtime_state,
    consume_stream_events,
    produce_events_to_stream,
)
from src.survival.modeling import run_survival_pipeline as run_survival_modeling_pipeline


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_simulation_config(
    random_seed: Optional[int] = None,
    randomize: bool = False,
) -> SimulationConfig:
    if randomize:
        return DEFAULT_CONFIG.with_seed(None)
    if random_seed is None:
        return DEFAULT_CONFIG
    return DEFAULT_CONFIG.with_seed(random_seed)


def ensure_simulation_outputs(
    data_dir: Path,
    force: bool = False,
    random_seed: Optional[int] = None,
    randomize: bool = False,
) -> Dict[str, pd.DataFrame]:
    data_dir = ensure_directory(data_dir)
    required = [
        data_dir / "customers.csv",
        data_dir / "treatment_assignments.csv",
        data_dir / "campaign_exposures.csv",
        data_dir / "events.csv",
        data_dir / "orders.csv",
        data_dir / "state_snapshots.csv",
        data_dir / "customer_summary.csv",
        data_dir / "cohort_retention.csv",
    ]

    if not force and all(p.exists() for p in required):
        return {
            "customer_summary": pd.read_csv(data_dir / "customer_summary.csv"),
            "cohort_retention": pd.read_csv(data_dir / "cohort_retention.csv"),
        }

    config = _resolve_simulation_config(random_seed=random_seed, randomize=randomize)
    return run_simulation(
        config=config,
        export=True,
        output_dir=str(data_dir),
        file_format="csv",
    )




def _latest_mtime(paths: list[Path]) -> float:
    existing = [p.stat().st_mtime for p in paths if p.exists()]
    return max(existing) if existing else -1.0


def _needs_rebuild(targets: list[Path], dependencies: list[Path], force: bool = False) -> bool:
    if force:
        return True
    if not targets or any(not p.exists() for p in targets):
        return True
    return _latest_mtime(targets) < _latest_mtime(dependencies)

def load_customer_summary(
    data_dir: Path,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> pd.DataFrame:
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    return pd.read_csv(data_dir / "customer_summary.csv")


def run_feature_engineering_pipeline(
    data_dir: Path,
    result_dir: Path,
    feature_store_dir: Path | None = None,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    feature_store_dir = ensure_directory(feature_store_dir or Path("data/feature_store"))

    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )

    built = build_feature_dataset(data_dir=data_dir, feature_store_dir=feature_store_dir)

    summary_path = result_dir / "feature_engineering_summary.json"
    summary_path.write_text(
        json.dumps(built.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "mode": "features",
        "model_path": None,
        "metrics_path": str(summary_path),
        "primary_result_path": built.feature_store_csv_path,
        "extra_result_paths": [built.feature_store_metadata_path],
        "metadata": built.metadata,
    }


def run_churn_training_pipeline(
    data_dir: Path,
    model_dir: Path,
    result_dir: Path,
    feature_store_dir: Path | None = None,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    shap_sample_size: int = 300,
    candidate_models: list[str] | None = None,
    threshold_tp_value: float = 120000.0,
    threshold_fp_cost: float = 18000.0,
    threshold_fn_cost: float = 60000.0,
) -> Dict:
    from src.ml.churn_training import train_churn_models

    model_dir = ensure_directory(model_dir)
    result_dir = ensure_directory(result_dir)
    feature_store_dir = ensure_directory(feature_store_dir or Path("data/feature_store"))

    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )

    built = build_feature_dataset(data_dir=data_dir, feature_store_dir=feature_store_dir)
    artifacts = train_churn_models(
        built.features,
        model_dir=model_dir,
        result_dir=result_dir,
        test_size=test_size,
        random_state=random_state,
        shap_sample_size=shap_sample_size,
        candidate_models=candidate_models,
        threshold_tp_value=threshold_tp_value,
        threshold_fp_cost=threshold_fp_cost,
        threshold_fn_cost=threshold_fn_cost,
    )

    metrics = dict(artifacts.metrics)
    metrics["feature_store_csv_path"] = built.feature_store_csv_path
    metrics["feature_store_metadata_path"] = built.feature_store_metadata_path

    Path(artifacts.metrics_path).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "mode": "train",
        "model_path": artifacts.model_path,
        "metrics_path": artifacts.metrics_path,
        "primary_result_path": built.feature_store_csv_path,
        "extra_result_paths": artifacts.extra_result_paths + [built.feature_store_metadata_path],
        "metadata": metrics,
    }


def run_uplift_pipeline(
    data_dir: Path,
    result_dir: Path,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )

    artifacts = run_uplift_modeling(data_dir=data_dir, result_dir=result_dir)
    summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))

    return {
        "mode": "uplift",
        "model_path": None,
        "metrics_path": artifacts.summary_path,
        "primary_result_path": artifacts.segmentation_path,
        "extra_result_paths": [
            artifacts.model_comparison_path,
            artifacts.qini_curve_path,
            artifacts.uplift_curve_path,
            artifacts.persuadables_analysis_path,
        ],
        "metadata": summary,
    }


def run_clv_prediction_pipeline(
    data_dir: Path,
    result_dir: Path,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    artifacts = run_clv_pipeline(data_dir=data_dir, result_dir=result_dir)
    metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
    return {
        "mode": "clv",
        "model_path": None,
        "metrics_path": artifacts.metrics_path,
        "primary_result_path": artifacts.predictions_path,
        "extra_result_paths": [artifacts.distribution_report_path, artifacts.distribution_plot_path],
        "metadata": metrics,
    }


def run_segmentation_priority_pipeline(
    data_dir: Path,
    result_dir: Path,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    if _needs_rebuild([result_dir / 'uplift_segmentation.csv'], [data_dir / 'customer_summary.csv', data_dir / 'orders.csv'], force=force_simulation or randomize_simulation):
        run_uplift_pipeline(data_dir, result_dir)
    if _needs_rebuild([result_dir / 'clv_predictions.csv'], [data_dir / 'orders.csv', data_dir / 'customer_summary.csv'], force=force_simulation or randomize_simulation):
        run_clv_prediction_pipeline(data_dir, result_dir)
    artifacts = run_segmentation_pipeline(result_dir=result_dir, data_dir=data_dir)
    summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
    return {
        "mode": "segment",
        "model_path": None,
        "metrics_path": artifacts.summary_path,
        "primary_result_path": artifacts.customer_segments_path,
        "extra_result_paths": [artifacts.visualization_path],
        "metadata": summary,
    }


def run_optimize_pipeline(
    data_dir: Path,
    result_dir: Path,
    budget: int,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    if _needs_rebuild([result_dir / 'uplift_segmentation.csv'], [data_dir / 'customer_summary.csv', data_dir / 'orders.csv'], force=force_simulation or randomize_simulation):
        run_uplift_pipeline(data_dir, result_dir)
    if _needs_rebuild([result_dir / 'clv_predictions.csv'], [data_dir / 'orders.csv', data_dir / 'customer_summary.csv'], force=force_simulation or randomize_simulation):
        run_clv_prediction_pipeline(data_dir, result_dir)
    if _needs_rebuild([result_dir / 'customer_segments.csv'], [result_dir / 'uplift_segmentation.csv', result_dir / 'clv_predictions.csv', data_dir / 'customer_summary.csv'], force=force_simulation or randomize_simulation):
        run_segmentation_priority_pipeline(data_dir, result_dir, force_simulation=force_simulation, simulation_seed=simulation_seed, randomize_simulation=randomize_simulation)

    if _needs_rebuild([result_dir / 'optimization_selected_customers.csv', result_dir / 'optimization_summary.json'], [result_dir / 'customer_segments.csv'], force=force_simulation or randomize_simulation):
        artifacts = run_budget_optimization(result_dir=result_dir, budget=budget)
    else:
        artifacts = run_budget_optimization(result_dir=result_dir, budget=budget)
    return {
        "mode": "optimize",
        "model_path": None,
        "metrics_path": artifacts.summary_path,
        "primary_result_path": artifacts.segment_path,
        "extra_result_paths": [artifacts.selected_path, artifacts.scenario_path],
        "metadata": artifacts.summary,
    }


def run_ab_test_pipeline(
    data_dir: Path,
    result_dir: Path,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    if _needs_rebuild([result_dir / 'uplift_segmentation.csv'], [data_dir / 'customer_summary.csv', data_dir / 'orders.csv'], force=force_simulation or randomize_simulation):
        run_uplift_pipeline(data_dir, result_dir)
    artifacts = run_ab_test_analysis(result_dir=result_dir)
    metrics = json.loads(Path(artifacts.result_path).read_text(encoding="utf-8"))
    return {
        "mode": "abtest",
        "model_path": None,
        "metrics_path": artifacts.result_path,
        "primary_result_path": artifacts.report_path,
        "extra_result_paths": [],
        "metadata": metrics,
    }




def run_cohort_journey_pipeline(
    data_dir: Path,
    result_dir: Path,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    artifacts = run_cohort_and_journey_analysis(data_dir=data_dir, result_dir=result_dir)
    summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
    return {
        "mode": "cohort",
        "model_path": None,
        "metrics_path": artifacts.summary_path,
        "primary_result_path": artifacts.retention_curve_path,
        "extra_result_paths": [
            artifacts.churn_heatmap_path,
            artifacts.retention_milestone_csv_path,
            artifacts.sequence_csv_path,
            artifacts.sequence_plot_path,
            artifacts.pre_churn_event_csv_path,
            artifacts.pre_churn_event_plot_path,
            artifacts.funnel_csv_path,
            artifacts.funnel_plot_path,
            artifacts.churn_timing_csv_path,
            artifacts.churn_timing_plot_path,
            artifacts.report_path,
        ],
        "metadata": summary,
    }

def run_recommendation_pipeline(
    data_dir: Path,
    result_dir: Path,
    budget: int = 50000000,
    threshold: float = 0.50,
    max_customers: Optional[int] = 1000,
    per_customer: int = 3,
    candidate_limit: Optional[int] = None,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )

    customers = pd.read_csv(data_dir / "customer_summary.csv")
    selected_customers, budget_summary, _ = get_budget_result(
        customers=customers,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
    )

    if candidate_limit is not None and int(candidate_limit) > 0:
        resolved_candidate_limit = int(candidate_limit)
    elif not selected_customers.empty:
        if max_customers is not None and max_customers > 0:
            resolved_candidate_limit = min(int(max_customers), int(len(selected_customers)))
        else:
            resolved_candidate_limit = int(len(selected_customers))
    elif max_customers is not None and max_customers > 0:
        resolved_candidate_limit = int(max_customers)
    else:
        resolved_candidate_limit = 100

    artifacts = run_personalized_recommendation_pipeline(
        data_dir=data_dir,
        result_dir=result_dir,
        per_customer=per_customer,
        candidate_limit=resolved_candidate_limit,
        target_customers=selected_customers,
        target_source="optimized_targets",
    )
    summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
    budget_summary["threshold"] = float(threshold)
    summary["budget_context"] = budget_summary
    summary["candidate_limit"] = int(resolved_candidate_limit)
    summary["eligible_target_customers"] = int(len(selected_customers))
    Path(artifacts.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "mode": "recommend",
        "model_path": None,
        "metrics_path": artifacts.summary_path,
        "primary_result_path": artifacts.recommendations_path,
        "extra_result_paths": [],
        "metadata": summary,
    }

def run_survival_pipeline(
    data_dir: Path,
    model_dir: Path,
    result_dir: Path,
    feature_store_dir: Path | None = None,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    model_dir = ensure_directory(model_dir)
    result_dir = ensure_directory(result_dir)
    feature_store_dir = ensure_directory(feature_store_dir or Path("data/feature_store"))

    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )

    artifacts = run_survival_modeling_pipeline(
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
        feature_store_dir=feature_store_dir,
    )
    return {
        "mode": "survival",
        "model_path": artifacts.model_path,
        "metrics_path": artifacts.metrics_path,
        "primary_result_path": artifacts.predictions_path,
        "extra_result_paths": [artifacts.coefficients_path, artifacts.risk_plot_path],
        "metadata": artifacts.metrics,
    }


def run_realtime_bootstrap_pipeline(
    data_dir: Path,
    result_dir: Path,
    *,
    redis_url: str = 'redis://localhost:6379/0',
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    config = RealtimeStreamConfig(redis_url=redis_url)
    payload = bootstrap_realtime_state(data_dir, result_dir, config, reset_stream=True)
    return {
        "mode": "realtime-bootstrap",
        "model_path": None,
        "metrics_path": str(result_dir / 'realtime_scores_summary.json'),
        "primary_result_path": str(result_dir / 'realtime_scores_snapshot.csv'),
        "extra_result_paths": [],
        "metadata": payload.get('summary', {}),
    }


def run_realtime_replay_pipeline(
    data_dir: Path,
    result_dir: Path,
    *,
    redis_url: str = 'redis://localhost:6379/0',
    limit: Optional[int] = None,
    max_events: Optional[int] = None,
    force_simulation: bool = False,
    simulation_seed: Optional[int] = None,
    randomize_simulation: bool = False,
) -> Dict:
    result_dir = ensure_directory(result_dir)
    ensure_simulation_outputs(
        data_dir,
        force=force_simulation,
        random_seed=simulation_seed,
        randomize=randomize_simulation,
    )
    config = RealtimeStreamConfig(redis_url=redis_url)
    bootstrap_realtime_state(data_dir, result_dir, config, reset_stream=True)
    produce_events_to_stream(data_dir, result_dir, config, limit=limit, reset_stream=True)
    payload = consume_stream_events(data_dir, result_dir, config, max_events=max_events or limit)
    return {
        "mode": "realtime-replay",
        "model_path": None,
        "metrics_path": str(result_dir / 'realtime_scores_summary.json'),
        "primary_result_path": str(result_dir / 'realtime_scores_snapshot.csv'),
        "extra_result_paths": [],
        "metadata": payload.get('summary', {}),
    }

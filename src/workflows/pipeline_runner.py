from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.api.services.analytics import allocate_budget, budget_allocation_by_segment
from src.features.engineering import build_feature_dataset
from src.simulator.config import DEFAULT_CONFIG, SimulationConfig
from src.simulator.pipeline import run_simulation


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
) -> Dict:
    # lazy import:
    # features 모드에서 불필요하게 LightGBM/OpenMP 로딩이 일어나지 않게 한다.
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
    df = load_customer_summary(
        data_dir,
        force_simulation=force_simulation,
        simulation_seed=simulation_seed,
        randomize_simulation=randomize_simulation,
    ).copy()

    output = df[
        [
            "customer_id",
            "persona",
            "uplift_score",
            "uplift_segment",
            "clv",
            "churn_probability",
            "expected_incremental_profit",
            "expected_roi",
        ]
    ].sort_values(["uplift_score", "clv"], ascending=False)

    out_path = result_dir / "uplift_segmentation.csv"
    output.to_csv(out_path, index=False)

    meta = {
        "rows": int(len(output)),
        "segment_counts": output["uplift_segment"].value_counts().to_dict(),
    }
    meta_path = result_dir / "uplift_summary.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "mode": "uplift",
        "model_path": None,
        "metrics_path": str(meta_path),
        "primary_result_path": str(out_path),
        "extra_result_paths": [],
        "metadata": meta,
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
    df = load_customer_summary(
        data_dir,
        force_simulation=force_simulation,
        simulation_seed=simulation_seed,
        randomize_simulation=randomize_simulation,
    )

    selected, summary = allocate_budget(df, budget=budget)
    segment_allocation = budget_allocation_by_segment(selected)

    selected_path = result_dir / "optimization_selected_customers.csv"
    segment_path = result_dir / "optimization_segment_budget.csv"
    summary_path = result_dir / "optimization_summary.json"

    selected.to_csv(selected_path, index=False)
    segment_allocation.to_csv(segment_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "mode": "optimize",
        "model_path": None,
        "metrics_path": str(summary_path),
        "primary_result_path": str(segment_path),
        "extra_result_paths": [str(selected_path)],
        "metadata": summary,
    }

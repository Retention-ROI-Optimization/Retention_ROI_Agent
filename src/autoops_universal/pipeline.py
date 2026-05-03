from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_writer import write_all_artifacts
from .canonicalizer import canonicalize_dataframe
from .config import SETTINGS, UniversalAutoOpsConfig
from .io_utils import read_csv_safely, write_json
from .modeling import train_safe_churn_model
from .validation import validate_uploaded_dataset


def run_universal_onboarding_pipeline(
    csv_path: str | Path,
    *,
    config: UniversalAutoOpsConfig = SETTINGS,
    budget: int | None = None,
    threshold: float | None = None,
    max_customers: int | None = None,
    manual_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run schema mapping -> canonicalization -> retraining -> artifact refresh.

    This function is additive: it writes project-compatible artifacts but does not
    modify legacy source code.
    """
    config.ensure_dirs()
    budget = int(config.default_budget if budget is None else budget)
    threshold = float(config.default_threshold if threshold is None else threshold)
    max_customers = int(config.default_max_customers if max_customers is None else max_customers)

    raw = read_csv_safely(csv_path)
    validation = validate_uploaded_dataset(raw, manual_mapping=manual_mapping)
    canonical = canonicalize_dataframe(raw, manual_mapping=manual_mapping)
    canonical.diagnostics["dataset_validation"] = validation.as_dict()
    canonical.mapping_table.to_csv(config.result_dir / "schema_mapping_table.csv", index=False)
    write_json(config.result_dir / "schema_mapping_report.json", canonical.mapping.as_dict())
    write_json(config.result_dir / "onboarding_diagnostics.json", canonical.diagnostics)

    training = train_safe_churn_model(
        canonical.feature_table,
        model_dir=config.model_dir,
        result_dir=config.result_dir,
        random_state=config.random_state,
        max_train_rows=config.max_train_rows,
    )
    artifacts = write_all_artifacts(
        training.features_with_scores,
        config=config,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
    )

    metadata = {
        "mode": "universal_autoops",
        "status": "ready",
        "source_csv": str(Path(csv_path).resolve()),
        "budget": budget,
        "threshold": threshold,
        "max_customers": max_customers,
        "diagnostics": canonical.diagnostics,
        "training_metrics": training.metrics,
        "model_path": str(training.model_path),
        "artifacts": artifacts,
    }
    write_json(config.feature_store_dir / "customer_features_metadata.json", metadata)
    write_json(config.feature_store_dir / "survival" / "customer_features_metadata.json", metadata)
    write_json(config.result_dir / "platform_pipeline_status.json", metadata)
    return {
        "summary": {
            "status": "ready",
            "mode": "universal_autoops",
            "source_csv": Path(csv_path).name,
            "input_rows": canonical.diagnostics["profile"]["rows"],
            "canonical_customers": canonical.diagnostics["canonical_rows"],
            "grain": canonical.diagnostics["grain_detection"]["grain"],
            "label_source": canonical.diagnostics.get("label_source"),
            "selected_customers": artifacts["selected_customers"],
            "model": training.metrics.get("model"),
            "auc": training.metrics.get("auc"),
        },
        "metadata": metadata,
    }

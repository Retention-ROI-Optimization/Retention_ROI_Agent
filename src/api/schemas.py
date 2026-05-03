from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    api_version: str
    data_dir: str
    available_tables: List[str]


class TableListResponse(BaseModel):
    tables: Dict[str, bool]


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    returned: int


class CustomerListResponse(BaseModel):
    meta: PaginationMeta
    records: List[Dict[str, Any]]


class ChurnSummary(BaseModel):
    total_customers: int
    at_risk_customers: int
    risk_rate: float
    avg_churn_prob: float


class ChurnResponse(BaseModel):
    threshold: float
    summary: ChurnSummary
    top_at_risk: List[Dict[str, Any]]


class DistributionRow(BaseModel):
    name: str
    count: int
    share: float


class BudgetSummary(BaseModel):
    budget: int
    spent: int
    remaining: int
    num_targeted: int
    expected_incremental_profit: float
    overall_roi: float
    candidate_customers: int = 0
    max_customers_cap: int = 0
    candidate_segment_counts: Dict[str, int] = Field(default_factory=dict)


class BudgetResponse(BaseModel):
    budget: int
    summary: BudgetSummary
    selected_customers: List[Dict[str, Any]]
    segment_allocation: List[Dict[str, Any]]


class CohortRetentionRow(BaseModel):
    cohort_month: str
    period: int
    cohort_size: Optional[int] = None
    retained_customers: Optional[int] = None
    retention_rate: Optional[float] = None
    observed: Optional[bool] = None


class CohortRetentionResponse(BaseModel):
    periods: int
    records: List[CohortRetentionRow]


class DashboardSummaryResponse(BaseModel):
    customer_count: int
    cohort_row_count: int
    threshold: float
    budget: int
    churn_summary: ChurnSummary
    budget_summary: BudgetSummary
    persona_distribution: List[DistributionRow]
    uplift_segment_distribution: List[DistributionRow]


class SimulationConfigOverrides(BaseModel):
    n_customers: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    signup_months: Optional[List[str]] = None
    random_seed: Optional[int] = None
    treatment_share: Optional[float] = None
    min_customers_per_arm: Optional[int] = None
    stratify_treatment: Optional[bool] = None
    campaign_type: Optional[str] = None
    coupon_min_cost: Optional[int] = None
    coupon_max_cost: Optional[int] = None
    coupon_cooldown_days: Optional[int] = None
    coupon_trigger_inactivity_days: Optional[int] = None
    max_exposures_per_customer: Optional[int] = None
    snapshot_frequency_days: Optional[int] = None
    dormant_inactivity_days: Optional[int] = None
    churn_inactivity_days: Optional[int] = None
    default_export_dir: Optional[str] = None
    default_file_format: Optional[str] = None


class SimulationRunRequest(BaseModel):
    config: SimulationConfigOverrides = Field(default_factory=SimulationConfigOverrides)
    export: bool = True
    output_dir: Optional[str] = None
    file_format: Optional[str] = None
    persist_to_api_data_dir: bool = True


class SimulationRunResponse(BaseModel):
    exported: bool
    output_dir: Optional[str]
    file_format: Optional[str]
    tables: Dict[str, int]
    customer_count: int
    event_count: int
    order_count: int
    cohort_row_count: int


class PipelineRunRequest(BaseModel):
    budget: Optional[int] = None
    force_simulation: bool = False


class PipelineRunResponse(BaseModel):
    mode: str
    model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    primary_result_path: Optional[str] = None
    extra_result_paths: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    rows: int
    summary: Dict[str, Any] = Field(default_factory=dict)
    records: List[Dict[str, Any]] = Field(default_factory=list)


class TrainingArtifactsResponse(BaseModel):
    directories: Dict[str, str] = Field(default_factory=dict)
    feature_summary: Dict[str, Any] = Field(default_factory=dict)
    customer_features: List[Dict[str, Any]] = Field(default_factory=list)
    customer_features_metadata: Dict[str, Any] = Field(default_factory=dict)
    churn_metrics: Dict[str, Any] = Field(default_factory=dict)
    threshold_analysis: Dict[str, Any] = Field(default_factory=dict)
    top_feature_importance: List[Dict[str, Any]] = Field(default_factory=list)
    image_paths: Dict[str, Optional[str]] = Field(default_factory=dict)
    model_paths: Dict[str, Optional[str]] = Field(default_factory=dict)
    training_parameters: Dict[str, Any] = Field(default_factory=dict)


class SavedResultsArtifactsResponse(BaseModel):
    result_dir: str
    uplift_summary: Dict[str, Any] = Field(default_factory=dict)
    uplift_segmentation: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_summary: Dict[str, Any] = Field(default_factory=dict)
    optimization_segment_budget: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_selected_customers: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class RealtimeScoringResponse(BaseModel):
    summary: Dict[str, Any] = Field(default_factory=dict)
    records: List[Dict[str, Any]] = Field(default_factory=list)


class SurvivalArtifactsResponse(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    coefficients: List[Dict[str, Any]] = Field(default_factory=list)
    image_paths: Dict[str, Optional[str]] = Field(default_factory=dict)


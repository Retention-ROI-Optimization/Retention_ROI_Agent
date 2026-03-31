from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True)
class SimulationConfig:
    """
    Global configuration for the customer simulator.

    Defaults are chosen to satisfy two goals:
    1) Enough samples for treatment/control analysis.
    2) Outputs that remain small enough to run on a student machine.
    """

    n_customers: int = 20000
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    signup_months: Sequence[str] = (
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
    )
    random_seed: int = 42

    # Experiment design
    treatment_share: float = 0.50
    min_customers_per_arm: int = 5000
    stratify_treatment: bool = True

    # Marketing / coupon assumptions
    campaign_type: str = "retention_coupon"
    coupon_min_cost: int = 3000
    coupon_max_cost: int = 15000
    coupon_cooldown_days: int = 14
    coupon_trigger_inactivity_days: int = 14
    max_exposures_per_customer: int = 4

    # State snapshots
    snapshot_frequency_days: int = 7

    # Behavior thresholds
    dormant_inactivity_days: int = 21
    churn_inactivity_days: int = 45

    # Export
    default_export_dir: str = "data/raw"
    default_file_format: str = "csv"

    def __post_init__(self) -> None:
        start = datetime.fromisoformat(self.start_date)
        end = datetime.fromisoformat(self.end_date)

        if end <= start:
            raise ValueError("end_date must be after start_date.")

        if not (0 < self.treatment_share < 1):
            raise ValueError("treatment_share must be between 0 and 1.")

        treated = int(self.n_customers * self.treatment_share)
        control = self.n_customers - treated
        if min(treated, control) < self.min_customers_per_arm:
            raise ValueError(
                "n_customers is too small for the requested treatment/control minimum. "
                f"Need at least {self.min_customers_per_arm} customers per arm."
            )

        if self.snapshot_frequency_days <= 0:
            raise ValueError("snapshot_frequency_days must be positive.")

        if self.coupon_min_cost <= 0 or self.coupon_max_cost < self.coupon_min_cost:
            raise ValueError("Coupon cost bounds are invalid.")

    @property
    def start_ts(self):
        return datetime.fromisoformat(self.start_date)

    @property
    def end_ts(self):
        return datetime.fromisoformat(self.end_date)

    @property
    def simulation_days(self) -> int:
        return (self.end_ts - self.start_ts).days + 1


DEFAULT_CONFIG = SimulationConfig()

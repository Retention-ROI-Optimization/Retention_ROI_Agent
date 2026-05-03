from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd

TABLE_FILES: Dict[str, str] = {
    'customers': 'customers.csv',
    'treatment_assignments': 'treatment_assignments.csv',
    'campaign_exposures': 'campaign_exposures.csv',
    'events': 'events.csv',
    'orders': 'orders.csv',
    'state_snapshots': 'state_snapshots.csv',
    'customer_summary': 'customer_summary.csv',
    'cohort_retention': 'cohort_retention.csv',
}

DATE_COLUMNS = [
    'assigned_at', 'signup_date', 'snapshot_date', 'last_visit_date', 'last_purchase_date',
    'timestamp', 'order_time', 'exposure_time'
]


@dataclass
class DataRepository:
    data_dir: Path
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _cache_mtimes: Dict[str, int] = field(default_factory=dict)

    def resolve_path(self, table_name: str) -> Path:
        if table_name not in TABLE_FILES:
            raise KeyError(f'Unknown table: {table_name}')
        return self.data_dir / TABLE_FILES[table_name]

    def has_table(self, table_name: str) -> bool:
        return self.resolve_path(table_name).exists()

    def available_tables(self) -> Dict[str, bool]:
        return {name: self.has_table(name) for name in TABLE_FILES}

    def _mtime_ns(self, path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except FileNotFoundError:
            return -1

    def read_table(self, table_name: str, force_reload: bool = False) -> pd.DataFrame:
        path = self.resolve_path(table_name)
        if not path.exists():
            raise FileNotFoundError(f'Required table is missing: {path}')

        current_mtime = self._mtime_ns(path)
        cached_mtime = self._cache_mtimes.get(table_name)
        if (
            not force_reload
            and table_name in self._cache
            and cached_mtime == current_mtime
        ):
            return self._cache[table_name].copy()

        df = pd.read_csv(path)
        for column in DATE_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors='coerce')

        self._cache[table_name] = df
        self._cache_mtimes[table_name] = current_mtime
        return df.copy()

    def require_customer_summary(self, force_reload: bool = False) -> pd.DataFrame:
        return self.read_table('customer_summary', force_reload=force_reload)

    def require_cohort_retention(self, force_reload: bool = False) -> pd.DataFrame:
        return self.read_table('cohort_retention', force_reload=force_reload)

    def reload_all(self) -> None:
        self._cache.clear()
        self._cache_mtimes.clear()
        for table_name, present in self.available_tables().items():
            if present:
                self.read_table(table_name, force_reload=True)

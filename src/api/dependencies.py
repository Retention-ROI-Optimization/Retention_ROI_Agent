from __future__ import annotations

from functools import lru_cache

from .settings import SETTINGS, ApiSettings
from .services.repository import DataRepository


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return SETTINGS


@lru_cache(maxsize=1)
def get_repository() -> DataRepository:
    return DataRepository(data_dir=get_settings().resolved_data_dir)

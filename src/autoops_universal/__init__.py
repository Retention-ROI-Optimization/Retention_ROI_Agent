from __future__ import annotations

from typing import Any

"""Universal AutoOps add-on.

This package initializer is intentionally lightweight.
"""

__all__ = [
    "SETTINGS",
    "UniversalAutoOpsConfig",
    "run_universal_onboarding_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in {"SETTINGS", "UniversalAutoOpsConfig"}:
        from .config import SETTINGS, UniversalAutoOpsConfig

        value = {
            "SETTINGS": SETTINGS,
            "UniversalAutoOpsConfig": UniversalAutoOpsConfig,
        }[name]
        globals()[name] = value
        return value

    if name == "run_universal_onboarding_pipeline":
        from .pipeline import run_universal_onboarding_pipeline

        globals()[name] = run_universal_onboarding_pipeline
        return run_universal_onboarding_pipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

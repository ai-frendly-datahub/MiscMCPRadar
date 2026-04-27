from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "RadarStorage":
        from radar.storage import RadarStorage

        return RadarStorage
    if name == "collect_sources":
        from radar.collector import collect_sources

        return collect_sources
    if name == "apply_entity_rules":
        from radar.analyzer import apply_entity_rules

        return apply_entity_rules

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RadarStorage", "collect_sources", "apply_entity_rules"]

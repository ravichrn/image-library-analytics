"""Layer 1: load and cache the static scheduler profile."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from scheduler._types import ModelProfile

_log = logging.getLogger(__name__)

_PROFILE_PATH = Path("docs/scheduler_profile.json")
_CACHE: dict[str, ModelProfile] | None = None


def _load() -> dict[str, ModelProfile]:
    try:
        raw = json.loads(_PROFILE_PATH.read_text())
        result: dict[str, ModelProfile] = {}
        for name, m in raw.get("models", {}).items():
            result[name] = ModelProfile(
                safe_batch=int(m["safe_batch"]),
                best_batch=int(m["best_batch"]),
                max_batch=int(m["max_batch"]),
                valid_arms=[int(a) for a in m["valid_arms"]],
                profiled_imgs_per_s=float(m["profiled_imgs_per_s"]),
                profiled_p95_ms=float(m.get("profiled_p95_ms") or 0.0),
                profiled_system_available_mb=m.get("profiled_system_available_mb"),
                min_system_available_mb=m.get("min_system_available_mb"),
                profiled_cv_pct=m.get("profiled_cv_pct"),
            )
        return result
    except Exception as exc:
        _log.warning("scheduler: could not load %s: %s", _PROFILE_PATH, exc)
        return {}


def get_profile(model_name: str) -> ModelProfile | None:
    """Return the ModelProfile for model_name, or None if not found.

    The profile file is read once and cached for the process lifetime.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _load()
    return _CACHE.get(model_name)


def reload() -> None:
    """Force a cache reload — useful after regenerating scheduler_profile.json."""
    global _CACHE
    _CACHE = None

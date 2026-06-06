"""Batch-size scheduler for the image-library-analytics pipeline.

Public API
----------
pick_batch_size(model_name, ctx=None) -> int
    Returns the recommended batch size for the given model.
    Combines Layer 1 (static profile), Layer 3 (UCB bandit), and
    Layer 2 (runtime adaptation). Falls back to benchmark best_batch
    on any exception — the pipeline is never blocked.

record_outcome(model_name, batch_size, imgs_per_s, system_available_mb,
               p95_ms, failure_rate=0.0) -> None
    Call after each pass completes to update bandit state.
    Silently swallows all exceptions.
"""

from __future__ import annotations

import logging
import os
import platform

import psutil

from scheduler._types import ModelProfile, SchedulerContext
from scheduler.bandit import BatchBandit
from scheduler.profile import get_profile
from scheduler.runtime import adapt

_log = logging.getLogger(__name__)

# Set SCHEDULER_DETERMINISTIC=1 to skip the bandit and always return profile.best_batch
# adjusted only by Layer 2 runtime rules. Useful for reproducible benchmark reruns.
_DETERMINISTIC = os.environ.get("SCHEDULER_DETERMINISTIC", "").strip() == "1"

# Absolute last-resort defaults — only used when scheduler_profile.json is missing entirely.
# These are conservative values; the real source of truth is the benchmark profile.
_HARDCODED_DEFAULTS: dict[str, int] = {
    "siglip2-base": 16,
    "aesthetic-predictor-v2-5": 4,
    "arniqa": 8,
    "dinov3-b": 8,
    "RMBG-2.0": 8,
    "yolo26n-pose": 16,
}

_bandit = BatchBandit()


def _machine_id() -> str:
    return platform.node()


def _available_mb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024)


def _fallback(model_name: str) -> int:
    """Return a safe batch size when the scheduler stack fails.

    Prefers the benchmark-derived best_batch from the profile.
    Falls back to _HARDCODED_DEFAULTS only if the profile file is missing.
    Halves the value if current memory is below the benchmark minimum.
    """
    profile = get_profile(model_name)
    base = profile.best_batch if profile is not None else _HARDCODED_DEFAULTS.get(model_name, 4)
    try:
        avail = _available_mb()
        min_mb = (profile.min_system_available_mb if profile is not None else None) or 3000.0
        if avail < min_mb:
            return max(1, base // 2)
        return base
    except Exception:
        return base


def pick_batch_size(model_name: str, ctx: SchedulerContext | None = None) -> int:
    """Return the recommended batch size for model_name.

    Normal mode (default):
      1. Load static profile (Layer 1). Missing → fallback.
      2. UCB bandit selects arm (Layer 3).
      3. Runtime adaptation adjusts (Layer 2).

    Deterministic mode (SCHEDULER_DETERMINISTIC=1):
      1. Load static profile.
      2. Start from profile.best_batch (no bandit).
      3. Runtime adaptation adjusts (Layer 2 memory/latency rules still apply).

    On ANY exception → return _fallback(model_name).
    """
    try:
        profile: ModelProfile | None = get_profile(model_name)
        if profile is None:
            _log.debug("scheduler: no profile for %r, using fallback", model_name)
            return _fallback(model_name)

        if ctx is None:
            ctx = SchedulerContext()
        if not ctx.machine_id:
            ctx.machine_id = _machine_id()
        if ctx.available_mb is None:
            ctx.available_mb = _available_mb()

        if _DETERMINISTIC:
            proposed = profile.best_batch
            _log.debug("scheduler: %s  deterministic  best=%d", model_name, proposed)
        else:
            proposed = _bandit.select(model_name, ctx.machine_id, profile)

        final = adapt(proposed, profile, ctx)

        _log.debug(
            "scheduler: %s  proposed=%d  adapted=%d  avail=%.0fMB  det=%s",
            model_name,
            proposed,
            final,
            ctx.available_mb,
            _DETERMINISTIC,
        )
        return final

    except Exception as exc:
        _log.warning("scheduler: pick_batch_size failed for %r: %s — using fallback", model_name, exc)
        return _fallback(model_name)


def record_outcome(
    model_name: str,
    batch_size: int,
    imgs_per_s: float,
    system_available_mb: float,
    p95_ms: float,
    failure_rate: float = 0.0,
) -> None:
    """Update bandit state after a pass completes.

    Call from main.py after _pass_stats.append() and before _flush_pass_profile().
    Silently swallows all exceptions — never breaks the pipeline.

    Note on p95_ms: callers pass total_pass_time / n_batches * 1000 — a
    pass-level mean batch time proxy, not a true p95. The latency penalty
    weight in the bandit reward is 0.2; throughput dominates the signal.
    """
    try:
        profile = get_profile(model_name)
        if profile is None:
            return
        machine = _machine_id()
        _bandit.update(
            model_name=model_name,
            machine_id=machine,
            batch_size=batch_size,
            imgs_per_s=imgs_per_s,
            available_mb=system_available_mb,
            p95_ms=p95_ms,
            profiled_p95_ms=profile.profiled_p95_ms,
            failure_rate=failure_rate,
        )
        _log.debug(
            "scheduler: recorded %s bs=%d  %.1f img/s  %.0fMB  %.1fms  fail=%.2f",
            model_name,
            batch_size,
            imgs_per_s,
            system_available_mb,
            p95_ms,
            failure_rate,
        )
    except Exception as exc:
        _log.warning("scheduler: record_outcome failed for %r: %s", model_name, exc)

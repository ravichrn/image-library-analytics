"""Shared dataclasses for the scheduler package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelProfile:
    """Static profile for one model, loaded from scheduler_profile.json."""

    safe_batch: int
    best_batch: int
    max_batch: int
    valid_arms: list[int]
    profiled_imgs_per_s: float
    profiled_p95_ms: float
    # System memory observed when best_batch was benchmarked.
    # Layer 2 uses this as the "safe" threshold: if current memory is below
    # this, the benchmark was never run under these conditions, so fall back.
    profiled_system_available_mb: float | None = None
    # Minimum system memory observed across ALL benchmarked batch sizes.
    # Layer 2 uses this as the "danger" threshold: below this, definitely halve.
    min_system_available_mb: float | None = None
    profiled_cv_pct: float | None = None


@dataclass
class SchedulerContext:
    """Runtime context passed to pick_batch_size.

    All fields are optional — missing fields degrade gracefully.
    The scheduler builds a minimal context from psutil when the caller
    passes None.
    """

    available_mb: float | None = None  # psutil.virtual_memory().available / 1e6
    recent_p95_ms: float | None = None  # p95 latency of last N batches in this pass
    recent_throughput: float | None = None  # imgs/s over last N batches
    failure_rate: float = 0.0  # fraction of None results in recent batches
    machine_id: str = ""  # platform.node(); filled by __init__.py if empty

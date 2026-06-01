"""Layer 2 — runtime adaptation heuristics.

adapt() takes a proposed batch size (from the bandit), a model profile,
and the current SchedulerContext, and returns an adjusted integer.

Decision tree (first match wins):
  1. p95 latency > 2x profiled AND throughput < 90% of profiled -> halve
  2. otherwise → clamp to max_batch

Ambient system RAM thresholds (min_system_available_mb, profiled_system_available_mb)
are NOT used here. Those values reflect system state during benchmarking, not actual
model memory requirements. SigLIP2-base allocates ~88 MB GPU at bs=16 yet had
6 GB system RAM free during benchmarking — using that as a threshold incorrectly
halves the batch whenever other apps are open. The bandit reward already penalises
memory pressure via its ALPHA term; let it handle adaptation.
"""

from __future__ import annotations

import logging

from scheduler._types import ModelProfile, SchedulerContext

_log = logging.getLogger(__name__)

_LATENCY_SPIKE_FACTOR = 2.0
_THROUGHPUT_FLOOR = 0.90  # 90% of profiled → latency rule fires


def _nearest_valid_arm(value: int, valid_arms: list[int]) -> int:
    """Largest arm <= value, or min(valid_arms) if all arms exceed value."""
    candidates = [a for a in valid_arms if a <= value]
    return max(candidates) if candidates else min(valid_arms)


def _step_up(current: int, valid_arms: list[int]) -> int:
    """Next arm above current, or current if already at top."""
    above = [a for a in valid_arms if a > current]
    return min(above) if above else current


def _step_down(current: int, valid_arms: list[int]) -> int:
    """Next arm below current, or current if already at bottom."""
    below = [a for a in valid_arms if a < current]
    return max(below) if below else current


def adapt(
    proposed_batch: int,
    profile: ModelProfile,
    ctx: SchedulerContext,
) -> int:
    """Return an adjusted batch size based on runtime conditions.

    Always returns a value in profile.valid_arms and <= profile.max_batch.
    """
    arms = profile.valid_arms

    # Rule 1: latency spike with no throughput gain — actual evidence of pressure
    p95 = ctx.recent_p95_ms
    tput = ctx.recent_throughput
    if (
        p95 is not None
        and p95 > _LATENCY_SPIKE_FACTOR * profile.profiled_p95_ms
        and tput is not None
        and tput < _THROUGHPUT_FLOOR * profile.profiled_imgs_per_s
    ):
        adjusted = _nearest_valid_arm(max(1, proposed_batch // 2), arms)
        _log.debug("scheduler/runtime: rule1 (latency spike) %d→%d", proposed_batch, adjusted)
        return adjusted

    # Rule 2: clamp to max_batch
    return min(proposed_batch, profile.max_batch)

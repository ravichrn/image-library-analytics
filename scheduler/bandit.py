"""Layer 3 — UCB multi-armed bandit for batch size selection.

State is persisted to docs/scheduler_state.json across pipeline runs.
Each (model, machine) pair has independent arm statistics so a bandit
calibrated on one machine does not pollute results on another.

UCB score:
    mean_reward[arm] + sqrt(UCB_C * ln(total_pulls + 1) / n_pulls[arm])

Arms with n_pulls == 0 get score = +inf (must be explored before UCB applies).
Pre-seeding inserts PRESEED_PULLS synthetic pulls for best_batch so the
bandit starts near-optimal and does not waste the first few pipeline runs
on random exploration.

Reward function:
    imgs_per_s
    - ALPHA * max(0, 1 - available_mb / 8000)    # memory pressure penalty
    - BETA  * max(0, p95_ms/profiled_p95_ms - 1)  # latency penalty
    - GAMMA * failure_rate                         # failure penalty
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from scheduler._types import ModelProfile

_log = logging.getLogger(__name__)

_STATE_PATH = Path("docs/scheduler_state.json")

PRESEED_PULLS = 5
UCB_C = 2.0
ALPHA = 0.3
BETA = 0.2
GAMMA = 0.5


class BatchBandit:
    def __init__(
        self,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
        state_path: Path = _STATE_PATH,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._state_path = state_path
        self._state: dict = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._state_path.exists():
            return
        try:
            self._state = json.loads(self._state_path.read_text())
        except Exception as exc:
            _log.warning("scheduler/bandit: could not load state: %s", exc)
            self._state = {}

    def save_state(self) -> None:
        try:
            self._state["_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self._state, indent=2))
            tmp.replace(self._state_path)
        except Exception as exc:
            _log.warning("scheduler/bandit: could not save state: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, model_name: str, machine_id: str) -> str:
        return f"{model_name}::{machine_id}"

    def _arm_str(self, arm: int) -> str:
        return str(arm)

    def _preseed(self, key: str, profile: ModelProfile) -> None:
        """Insert synthetic pulls for all valid arms so no arm starts at UCB=+inf.

        best_batch gets PRESEED_PULLS at the profiled reward.
        All other arms get 1 pull with a proportionally scaled reward
        (based on linear throughput scaling) — enough to bound their UCB
        below the best arm's UCB while still allowing genuine exploration.
        """
        best_reward = self._compute_reward(
            imgs_per_s=profile.profiled_imgs_per_s,
            available_mb=profile.profiled_system_available_mb or 8000.0,
            p95_ms=profile.profiled_p95_ms,
            profiled_p95_ms=profile.profiled_p95_ms,
            failure_rate=0.0,
        )
        entry: dict = {"n_pulls": {}, "sum_reward": {}, "total_pulls": 0}
        total = 0
        for arm in profile.valid_arms:
            arm_s = self._arm_str(arm)
            if arm == profile.best_batch:
                n = PRESEED_PULLS
                r = best_reward * n
            else:
                # Estimate reward as proportional to (arm / best_batch)^0.7 — diminishing returns
                scale = (arm / profile.best_batch) ** 0.7
                n = 1
                r = best_reward * scale
            entry["n_pulls"][arm_s] = n
            entry["sum_reward"][arm_s] = r
            total += n
        entry["total_pulls"] = total
        self._state[key] = entry
        _log.debug("scheduler/bandit: pre-seeded %s best_reward=%.3f total_pulls=%d", key, best_reward, total)

    def _compute_reward(
        self,
        imgs_per_s: float,
        available_mb: float,
        p95_ms: float,
        profiled_p95_ms: float,
        failure_rate: float,
    ) -> float:
        memory_penalty = max(0.0, 1.0 - available_mb / 8000.0)
        latency_penalty = max(0.0, (p95_ms / profiled_p95_ms) - 1.0) if profiled_p95_ms > 0 else 0.0
        return imgs_per_s - self._alpha * memory_penalty - self._beta * latency_penalty - self._gamma * failure_rate

    def _ucb_score(self, mean: float, n: int, total: int) -> float:
        if n == 0:
            return float("inf")
        return mean + math.sqrt(UCB_C * math.log(total + 1) / n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, model_name: str, machine_id: str, profile: ModelProfile) -> int:
        """Return the UCB-optimal arm (batch size) for this (model, machine).

        Hard constraint: returned value <= profile.max_batch.
        On first call for this key, pre-seeds the best_batch arm.
        """
        self.load_state()
        key = self._key(model_name, machine_id)

        if key not in self._state:
            self._preseed(key, profile)

        entry = self._state[key]
        total = entry.get("total_pulls", 0)

        best_arm = profile.best_batch
        best_score = float("-inf")

        for arm in profile.valid_arms:
            if arm > profile.max_batch:
                continue
            arm_s = self._arm_str(arm)
            n = entry.get("n_pulls", {}).get(arm_s, 0)
            sum_r = entry.get("sum_reward", {}).get(arm_s, 0.0)
            mean = sum_r / n if n > 0 else 0.0
            score = self._ucb_score(mean, n, total)
            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def update(
        self,
        model_name: str,
        machine_id: str,
        batch_size: int,
        imgs_per_s: float,
        available_mb: float,
        p95_ms: float,
        profiled_p95_ms: float,
        failure_rate: float,
    ) -> None:
        """Record outcome for a completed pass and persist state."""
        self.load_state()
        key = self._key(model_name, machine_id)
        arm_s = self._arm_str(batch_size)
        reward = self._compute_reward(imgs_per_s, available_mb, p95_ms, profiled_p95_ms, failure_rate)

        entry = self._state.setdefault(key, {"n_pulls": {}, "sum_reward": {}, "total_pulls": 0})
        entry["n_pulls"][arm_s] = entry["n_pulls"].get(arm_s, 0) + 1
        entry["sum_reward"][arm_s] = entry["sum_reward"].get(arm_s, 0.0) + reward
        entry["total_pulls"] = entry.get("total_pulls", 0) + 1

        _log.debug(
            "scheduler/bandit: updated %s arm=%s reward=%.3f total_pulls=%d",
            key,
            arm_s,
            reward,
            entry["total_pulls"],
        )
        self.save_state()

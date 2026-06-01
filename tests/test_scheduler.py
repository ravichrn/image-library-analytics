"""Unit tests for the scheduler package.

No GPU, no real pipeline, no file I/O to docs/ (uses tmp_path fixtures).
Tests each layer independently and the public API fallback behaviour.
"""

from __future__ import annotations

import json

import pytest

from scheduler._types import ModelProfile, SchedulerContext
from scheduler.bandit import BatchBandit
from scheduler.runtime import (
    _nearest_valid_arm,
    _step_down,
    _step_up,
    adapt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile() -> ModelProfile:
    return ModelProfile(
        safe_batch=8,
        best_batch=16,
        max_batch=16,
        valid_arms=[1, 2, 4, 8, 12, 16],
        profiled_imgs_per_s=63.8,
        profiled_p95_ms=250.9,
        profiled_system_available_mb=7000.0,
    )


@pytest.fixture()
def siglip_profile() -> ModelProfile:
    return ModelProfile(
        safe_batch=2,
        best_batch=4,
        max_batch=4,
        valid_arms=[1, 2, 4],
        profiled_imgs_per_s=4.1,
        profiled_p95_ms=1014.1,
        profiled_system_available_mb=6000.0,
    )


@pytest.fixture()
def bandit(tmp_path) -> BatchBandit:
    return BatchBandit(state_path=tmp_path / "state.json")


# ---------------------------------------------------------------------------
# _types.py
# ---------------------------------------------------------------------------


class TestTypes:
    def test_model_profile_fields(self, profile):
        assert profile.safe_batch == 8
        assert profile.best_batch == 16
        assert profile.valid_arms == [1, 2, 4, 8, 12, 16]

    def test_scheduler_context_defaults(self):
        ctx = SchedulerContext()
        assert ctx.available_mb is None
        assert ctx.failure_rate == 0.0


# ---------------------------------------------------------------------------
# runtime.py helpers
# ---------------------------------------------------------------------------


class TestRuntimeHelpers:
    def test_nearest_valid_arm_exact(self):
        assert _nearest_valid_arm(8, [1, 2, 4, 8, 12, 16]) == 8

    def test_nearest_valid_arm_below(self):
        assert _nearest_valid_arm(10, [1, 2, 4, 8, 12, 16]) == 8

    def test_nearest_valid_arm_below_min(self):
        assert _nearest_valid_arm(0, [1, 2, 4]) == 1

    def test_step_up(self):
        assert _step_up(8, [1, 2, 4, 8, 12, 16]) == 12

    def test_step_up_at_max(self):
        assert _step_up(16, [1, 2, 4, 8, 12, 16]) == 16

    def test_step_down(self):
        assert _step_down(8, [1, 2, 4, 8, 12, 16]) == 4

    def test_step_down_at_min(self):
        assert _step_down(1, [1, 2, 4, 8, 12, 16]) == 1


# ---------------------------------------------------------------------------
# runtime.py adapt()
# ---------------------------------------------------------------------------


class TestAdapt:
    def test_rule1_very_low_memory_halves(self, profile):
        # below min_system_available_mb → halve
        threshold = (profile.min_system_available_mb or 3000) - 500
        ctx = SchedulerContext(available_mb=threshold)
        result = adapt(16, profile, ctx)
        assert result <= 8
        assert result in profile.valid_arms

    def test_rule2_moderate_memory_uses_safe(self, profile):
        # below profiled_system_available_mb but above min → safe_batch
        safe_thresh = profile.profiled_system_available_mb or 5000
        min_thresh = profile.min_system_available_mb or 3000
        ctx = SchedulerContext(available_mb=(safe_thresh + min_thresh) / 2)
        result = adapt(16, profile, ctx)
        assert result == profile.safe_batch

    def test_rule3_latency_spike_halves(self, profile):
        ctx = SchedulerContext(
            available_mb=8000,
            recent_p95_ms=profile.profiled_p95_ms * 2.5,
            recent_throughput=profile.profiled_imgs_per_s * 0.8,
        )
        result = adapt(16, profile, ctx)
        assert result <= 8

    def test_rule3_no_fire_if_throughput_ok(self, profile):
        ctx = SchedulerContext(
            available_mb=8000,
            recent_p95_ms=profile.profiled_p95_ms * 2.5,
            recent_throughput=profile.profiled_imgs_per_s * 0.95,
        )
        result = adapt(16, profile, ctx)
        assert result == 16

    def test_rule4_step_up(self, profile):
        ctx = SchedulerContext(available_mb=12000)
        result = adapt(8, profile, ctx)
        assert result > 8

    def test_rule5_default_clamp(self, profile):
        ctx = SchedulerContext(available_mb=7500)
        result = adapt(16, profile, ctx)
        assert result == 16

    def test_result_always_in_valid_arms(self, profile):
        for avail in [1000, 4000, 6000, 8000, 12000]:
            ctx = SchedulerContext(available_mb=avail)
            result = adapt(16, profile, ctx)
            assert result in profile.valid_arms, f"avail={avail} result={result}"

    def test_siglip_safe_batch_on_pressure(self, siglip_profile):
        # between min and profiled → safe_batch
        safe_thresh = siglip_profile.profiled_system_available_mb or 5000
        min_thresh = siglip_profile.min_system_available_mb or 4000
        ctx = SchedulerContext(available_mb=(safe_thresh + min_thresh) / 2)
        result = adapt(4, siglip_profile, ctx)
        assert result == siglip_profile.safe_batch


# ---------------------------------------------------------------------------
# bandit.py
# ---------------------------------------------------------------------------


class TestBandit:
    def test_select_returns_valid_arm(self, bandit, profile):
        arm = bandit.select("dinov3-b", "test-machine", profile)
        assert arm in profile.valid_arms

    def test_select_returns_best_batch_on_first_call(self, bandit, profile):
        # After pre-seeding, best_batch should have the highest mean reward
        arm = bandit.select("dinov3-b", "test-machine", profile)
        assert arm == profile.best_batch

    def test_preseed_creates_all_arms(self, bandit, profile):
        bandit.select("dinov3-b", "test-machine", profile)
        key = "dinov3-b::test-machine"
        assert key in bandit._state
        entry = bandit._state[key]
        for arm in profile.valid_arms:
            assert str(arm) in entry["n_pulls"]

    def test_preseed_best_has_most_pulls(self, bandit, profile):
        bandit.select("dinov3-b", "test-machine", profile)
        entry = bandit._state["dinov3-b::test-machine"]
        best_n = entry["n_pulls"][str(profile.best_batch)]
        for arm in profile.valid_arms:
            if arm != profile.best_batch:
                assert entry["n_pulls"][str(arm)] < best_n

    def test_update_increments_pull_count(self, bandit, profile):
        bandit.select("dinov3-b", "test-machine", profile)
        before = bandit._state["dinov3-b::test-machine"]["n_pulls"].get("16", 0)
        bandit.update("dinov3-b", "test-machine", 16, 33.2, 7000, 260, 250.9, 0.0)
        after = bandit._state["dinov3-b::test-machine"]["n_pulls"]["16"]
        assert after == before + 1

    def test_update_persists_state(self, bandit, profile, tmp_path):
        bandit.select("dinov3-b", "test-machine", profile)
        bandit.update("dinov3-b", "test-machine", 16, 33.2, 7000, 260, 250.9, 0.0)
        assert bandit._state_path.exists()
        saved = json.loads(bandit._state_path.read_text())
        assert "dinov3-b::test-machine" in saved

    def test_atomic_save_no_half_writes(self, bandit, profile, tmp_path):
        # Verify the .tmp file is cleaned up after successful save
        bandit.select("dinov3-b", "test-machine", profile)
        bandit.update("dinov3-b", "test-machine", 16, 33.2, 7000, 260, 250.9, 0.0)
        tmp = bandit._state_path.with_suffix(".json.tmp")
        assert not tmp.exists()

    def test_two_models_independent(self, bandit, profile, siglip_profile):
        arm1 = bandit.select("dinov3-b", "m1", profile)
        arm2 = bandit.select("siglip2-so400m", "m1", siglip_profile)
        assert arm1 == profile.best_batch
        assert arm2 == siglip_profile.best_batch

    def test_reward_computation(self, bandit):
        # No penalty case: high ips, plenty of memory, p95 on target, no failures
        r = bandit._compute_reward(
            imgs_per_s=50.0,
            available_mb=8000.0,
            p95_ms=250.0,
            profiled_p95_ms=250.0,
            failure_rate=0.0,
        )
        assert r == pytest.approx(50.0, abs=0.01)

    def test_reward_memory_penalty(self, bandit):
        r_low = bandit._compute_reward(50.0, 2000.0, 250.0, 250.0, 0.0)
        r_high = bandit._compute_reward(50.0, 8000.0, 250.0, 250.0, 0.0)
        assert r_low < r_high

    def test_reward_failure_penalty(self, bandit):
        r_clean = bandit._compute_reward(50.0, 8000.0, 250.0, 250.0, 0.0)
        r_failed = bandit._compute_reward(50.0, 8000.0, 250.0, 250.0, 0.5)
        assert r_failed < r_clean

    def test_load_missing_state_is_noop(self, bandit):
        bandit.load_state()
        assert bandit._state == {}


# ---------------------------------------------------------------------------
# Public API (scheduler/__init__.py) — with mocked profile path
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_pick_batch_size_unknown_model_uses_fallback(self):
        from scheduler import pick_batch_size

        result = pick_batch_size("completely-unknown-model-xyz")
        assert isinstance(result, int)
        assert result >= 1

    def test_record_outcome_unknown_model_silent(self):
        from scheduler import record_outcome

        # Must not raise
        record_outcome("unknown-model", 4, 10.0, 7000.0, 100.0, 0.0)

    def test_pick_batch_size_returns_int(self):
        from scheduler import pick_batch_size

        result = pick_batch_size("dinov3-b")
        assert isinstance(result, int)
        assert result >= 1

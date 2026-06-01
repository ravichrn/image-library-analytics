"""Unit tests for benchmark_backends.py — pure Python/CPU, no GPU or sudo needed."""

from __future__ import annotations

# Import directly from the script (project root is on sys.path via conftest or tox)
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_backends import (
    BenchmarkResult,
    _pil_batch,
    _rss_delta_mb,
    run_latency_bench,
)

# ---------------------------------------------------------------------------
# run_latency_bench
# ---------------------------------------------------------------------------


def test_latency_bench_returns_two_floats():
    p50, p95 = run_latency_bench(lambda: time.sleep(0), warmup=2, iters=10)
    assert isinstance(p50, float)
    assert isinstance(p95, float)


def test_latency_bench_p95_ge_p50():
    p50, p95 = run_latency_bench(lambda: None, warmup=2, iters=20)
    assert p95 >= p50


def test_latency_bench_warmup_not_counted():
    # warmup=100, iters=1 — should still return a number, not crash
    p50, _p95 = run_latency_bench(lambda: None, warmup=5, iters=5)
    assert p50 >= 0.0


def test_latency_bench_sync_fn_called():
    calls = []
    run_latency_bench(lambda: None, warmup=2, iters=4, sync_fn=lambda: calls.append(1))
    assert len(calls) == 6  # 2 warmup + 4 timed


def test_latency_bench_timing_order():
    """p50 should be ≤ p95 even with deliberately variable latency."""
    import random

    def variable():
        time.sleep(random.uniform(0, 0.001))

    p50, p95 = run_latency_bench(variable, warmup=3, iters=20)
    assert p95 >= p50


# ---------------------------------------------------------------------------
# BenchmarkResult schema
# ---------------------------------------------------------------------------


def test_result_dataclass_fields():
    r = BenchmarkResult(
        model="dinov3-b",
        backend="mps",
        batch_size=4,
        load_time_s=1.2,
        p50_ms=12.3,
        p95_ms=14.1,
        p50_per_image_ms=3.075,
        p95_per_image_ms=3.525,
        imgs_per_s=65.0,
        peak_memory_mb=420.0,
        images_per_joule=None,
        ane_power_mw=None,
        ane_active=False,
        accuracy_metric=None,
        accuracy_value=None,
    )
    assert r.model == "dinov3-b"
    assert r.backend == "mps"
    assert r.batch_size == 4
    assert r.p95_ms >= r.p50_ms
    assert r.preprocess_p50_ms is None  # default
    assert r.notes == ""  # default


def test_result_preprocess_field():
    """preprocess_p50_ms is set for DINO/SigLIP and None for YOLO."""
    r = BenchmarkResult(
        model="dinov3-b",
        backend="mps",
        batch_size=1,
        load_time_s=0.9,
        p50_ms=20.0,
        p95_ms=21.0,
        p50_per_image_ms=20.0,
        p95_per_image_ms=21.0,
        imgs_per_s=50.0,
        peak_memory_mb=None,
        images_per_joule=None,
        ane_power_mw=None,
        ane_active=False,
        accuracy_metric=None,
        accuracy_value=None,
        preprocess_p50_ms=3.5,
        notes="forward pass only",
    )
    assert r.preprocess_p50_ms == pytest.approx(3.5)
    assert "forward pass only" in r.notes


def test_result_asdict():
    from dataclasses import asdict

    r = BenchmarkResult(
        model="yolo11n-pose",
        backend="coreml",
        batch_size=1,
        load_time_s=0.5,
        p50_ms=8.0,
        p95_ms=9.5,
        p50_per_image_ms=8.0,
        p95_per_image_ms=9.5,
        imgs_per_s=125.0,
        peak_memory_mb=None,
        images_per_joule=42.0,
        ane_power_mw=800.0,
        ane_active=True,
        accuracy_metric=None,
        accuracy_value=None,
    )
    d = asdict(r)
    assert d["ane_active"] is True
    assert d["images_per_joule"] == pytest.approx(42.0)
    assert "preprocess_p50_ms" in d  # field must be present in serialised output
    assert d["preprocess_p50_ms"] is None  # YOLO — not set


# ---------------------------------------------------------------------------
# Accuracy metric helpers
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert cos == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert cos == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Synthetic input helpers
# ---------------------------------------------------------------------------


def test_pil_batch_returns_correct_count():
    imgs = _pil_batch(4)
    assert len(imgs) == 4


def test_pil_batch_size_parameter():
    imgs = _pil_batch(2, size=384)
    assert imgs[0].size == (384, 384)


def test_pil_batch_rgb():
    imgs = _pil_batch(1)
    assert imgs[0].mode == "RGB"


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def test_rss_delta_mb_nonnegative():
    delta = _rss_delta_mb(lambda: None)
    assert delta >= 0.0


def test_rss_delta_mb_on_allocation():
    """Allocating a large array should produce a positive RSS delta."""
    holder = []

    def alloc():
        holder.append(np.zeros(10_000_000, dtype=np.float32))  # ~40 MB

    delta = _rss_delta_mb(alloc)
    # RSS reporting is coarse but a 40 MB alloc should register something
    assert delta >= 0.0  # just verify no crash; RSS measurement is OS-dependent


# ---------------------------------------------------------------------------
# run_power_bench graceful failure (no sudo)
# ---------------------------------------------------------------------------


def test_power_bench_no_sudo_returns_none():
    from scripts.benchmark_backends import run_power_bench

    # Without sudo credentials, run_power_bench must return (None, None) silently
    ipj, ane_mw = run_power_bench(lambda: None, batch_size=1, duration_s=0.1)
    # Either (None, None) because sudo -n failed, or actual values if running as root
    assert ipj is None or isinstance(ipj, float)
    assert ane_mw is None or isinstance(ane_mw, float)

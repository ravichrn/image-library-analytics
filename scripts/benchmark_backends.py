#!/usr/bin/env python3
"""Apple Silicon backend benchmark: PyTorch-MPS vs MLX vs CoreML.

Two benchmark modes:

  Microbenchmark (default):
    Measures model forward pass latency and throughput at various batch sizes
    using synthetic PIL images already on-device. Preprocessing is timed
    separately. Multiple runs per cell produce a confidence score (CV%).

  End-to-end (--e2e --photos-dir PATH):
    Calls the actual extractor functions with real photo paths, exactly as the
    pipeline does: file decode, processor, forward pass, post-processing.
    Cache writes are excluded. Gives real scheduler-trustworthy throughput.

Metrics: p50/p95 latency, imgs/s, CV% across runs, process RSS delta,
system available RAM at benchmark time, optional perf-per-watt via powermetrics.

Run from project root:
    uv run python scripts/benchmark_backends.py --model dinov3 siglip yolo rmbg rmbg14 birefnet-lite aesthetic clipiqa
    uv run python scripts/benchmark_backends.py --model dinov3 --runs 3 --batch-sizes 1 4 8 16
    uv run python scripts/benchmark_backends.py --e2e --photos-dir /path/to/photos
    sudo uv run python scripts/benchmark_backends.py --model dinov3 --batch-sizes 1 4 8 16 --energy
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors.embedding import _MODEL_ID as _DINOV3_MODEL_ID
from extractors.pose import _CACHE_PATH as _YOLO_CACHE_PATH
from extractors.pose import _MODEL_NAME as _YOLO_MODEL_NAME
from extractors.saliency import _INPUT_SIZE as _RMBG_INPUT_SIZE
from extractors.scene import _IQ_INPUT_SIZE

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    model: str
    backend: str
    batch_size: int
    load_time_s: float
    p50_ms: float  # median-of-runs p50 per-batch latency
    p95_ms: float  # median-of-runs p95 per-batch latency
    p50_per_image_ms: float
    p95_per_image_ms: float
    imgs_per_s: float  # median imgs/s across runs
    peak_memory_mb: float | None  # process RSS delta during forward pass (approx)
    images_per_joule: float | None  # None if --energy not used or powermetrics unavailable
    ane_power_mw: float | None
    ane_active: bool
    accuracy_metric: str | None
    accuracy_value: float | None
    # Multi-run confidence (set when --runs > 1)
    runs: int = field(default=1)
    cv_pct: float | None = field(default=None)  # coefficient of variation of imgs/s across runs
    # Memory context
    system_available_mb: float | None = field(default=None)  # system free RAM when bench ran
    mps_alloc_mb_debug: float | None = field(default=None)  # torch.mps.current_allocated_memory() — debug only
    # Preprocessing-only timing (processor call, PIL→tensor). Set for micro mode; None for YOLO/e2e.
    preprocess_p50_ms: float | None = field(default=None)
    # "microbench" or "e2e" — e2e uses real paths and real extractor functions
    benchmark_mode: str = field(default="microbench")
    notes: str = field(default="")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def run_latency_bench(fn, warmup: int = 10, iters: int = 100, sync_fn=None, desc: str = "") -> tuple[float, float]:
    """Warmup `warmup` iters (discarded), then time `iters`. Returns (p50_ms, p95_ms) per call."""
    prefix = f"  {desc} " if desc else "  "
    total = warmup + iters
    for i in range(warmup):
        sys.stdout.write(f"\r{prefix}warmup {i + 1}/{warmup}")
        sys.stdout.flush()
        fn()
        if sync_fn:
            sync_fn()
    latencies: list[float] = []
    for i in range(iters):
        pct = int((warmup + i + 1) / total * 100)
        sys.stdout.write(f"\r{prefix}{pct:3d}%  [{i + 1}/{iters}]")
        sys.stdout.flush()
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    sys.stdout.write("\r" + " " * (len(prefix) + 20) + "\r")  # clear the line
    sys.stdout.flush()
    latencies.sort()
    return latencies[int(0.50 * len(latencies))], latencies[int(0.95 * len(latencies))]


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _rss_delta_mb(fn) -> float:
    """Process RSS delta while running fn(). Approx — coarse but honest."""
    proc = psutil.Process()
    before = proc.memory_info().rss
    fn()
    after = proc.memory_info().rss
    return max(0, after - before) / (1024 * 1024)


def _system_available_mb() -> float:
    """System-wide free unified memory (reflects browser/editor pressure too)."""
    return psutil.virtual_memory().available / (1024 * 1024)


def _mps_alloc_mb() -> float | None:
    """torch.mps.current_allocated_memory() — debug signal only, not reliable for activation peak."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return None


# ---------------------------------------------------------------------------
# Multi-run cell helper
# ---------------------------------------------------------------------------


@dataclass
class _CellStats:
    p50_ms: float
    p95_ms: float
    imgs_per_s: float
    cv_pct: float | None
    peak_memory_mb: float | None
    preprocess_p50_ms: float | None
    system_available_mb: float
    mps_alloc_mb_debug: float | None
    runs: int


def _bench_cell(
    fn,
    preprocess_fn,
    batch_size: int,
    warmup: int,
    iters: int,
    runs: int,
    sync_fn=None,
    desc: str = "",
) -> _CellStats:
    """Run a benchmark cell `runs` times; report median stats and CV% as confidence score.

    preprocess_fn: timed separately (PIL→tensor); pass None if not applicable.
    Each run re-calls run_latency_bench from scratch — no cross-run state.
    """
    system_mb = _system_available_mb()
    mps_debug = _mps_alloc_mb()

    # Preprocessing timing (quick: 3 warmup + 20 iters, not repeated across runs)
    pre_p50: float | None = None
    if preprocess_fn is not None:
        pre_p50, _ = run_latency_bench(
            preprocess_fn,
            warmup=min(warmup, 3),
            iters=min(iters, 20),
            desc=f"{desc}/preprocess",
        )

    # Memory delta — RSS around one forward pass before the main timing loop
    mem_mb = _rss_delta_mb(fn)

    # Multi-run timing
    p50_samples: list[float] = []
    p95_samples: list[float] = []
    for r in range(runs):
        run_desc = f"{desc} run{r + 1}/{runs}" if runs > 1 else desc
        p50, p95 = run_latency_bench(fn, warmup=warmup, iters=iters, sync_fn=sync_fn, desc=run_desc)
        p50_samples.append(p50)
        p95_samples.append(p95)

    med_p50 = statistics.median(p50_samples)
    med_p95 = statistics.median(p95_samples)
    ips_samples = [batch_size / (p / 1000) for p in p50_samples]
    med_ips = statistics.median(ips_samples)
    cv: float | None = None
    if runs > 1 and statistics.mean(ips_samples) > 0:
        cv = round(statistics.stdev(ips_samples) / statistics.mean(ips_samples) * 100, 1)

    return _CellStats(
        p50_ms=round(med_p50, 2),
        p95_ms=round(med_p95, 2),
        imgs_per_s=round(med_ips, 1),
        cv_pct=cv,
        peak_memory_mb=round(mem_mb, 1) if mem_mb else None,
        preprocess_p50_ms=round(pre_p50, 2) if pre_p50 is not None else None,
        system_available_mb=round(system_mb, 0),
        mps_alloc_mb_debug=round(mps_debug, 1) if mps_debug is not None else None,
        runs=runs,
    )


# ---------------------------------------------------------------------------
# Energy (powermetrics — requires sudo)
# ---------------------------------------------------------------------------


def run_power_bench(fn, batch_size: int, duration_s: float = 30.0) -> tuple[float | None, float | None]:
    """
    Runs inference loop for `duration_s`, captures powermetrics alongside.
    Returns (images_per_joule, ane_power_mw_mean).
    Returns (None, None) when powermetrics is unavailable or sudo not granted.
    Uses `sudo -n` so it fails immediately rather than prompting for a password.
    """
    try:
        proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power", "-i", "200", "--json-output"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return None, None
    time.sleep(0.3)
    if proc.poll() is not None:
        return None, None  # sudo -n failed (no cached credentials)

    t0 = time.perf_counter()
    n_images = 0
    while time.perf_counter() - t0 < duration_s:
        fn()
        n_images += batch_size
    wall = time.perf_counter() - t0
    proc.terminate()
    stdout, _ = proc.communicate(timeout=5)

    # powermetrics --json-output separates samples with null bytes
    gpu_mw, ane_mw, cpu_mw = [], [], []
    for chunk in stdout.decode(errors="replace").split("\x00"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            d = json.loads(chunk)
            if "gpu_power" in d:
                gpu_mw.append(float(d["gpu_power"]))
            if "ane_power" in d:
                ane_mw.append(float(d["ane_power"]))
            if "cpu_power" in d:
                cpu_mw.append(float(d["cpu_power"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    if not gpu_mw:
        return None, None

    total_w = (np.mean(gpu_mw) + np.mean(ane_mw or [0]) + np.mean(cpu_mw or [0])) / 1000.0
    joules = total_w * wall
    return (n_images / joules if joules > 0 else None), (float(np.mean(ane_mw)) if ane_mw else None)


# ---------------------------------------------------------------------------
# Synthetic inputs — latency benchmarks don't need real photos
# ---------------------------------------------------------------------------


def _pil_batch(n: int, size: int = 224) -> list[Image.Image]:
    rng = np.random.default_rng(42)
    return [Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Configuration — pulled from extractors so pipeline changes flow through
# automatically without manual sync here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MPS: DINOv3-B
# ---------------------------------------------------------------------------


def bench_dinov3_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    import os

    from transformers import AutoImageProcessor, AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None
    token = os.environ.get("HF_TOKEN") or None

    print("\n  [dinov3-b / mps] loading …")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(_DINOV3_MODEL_ID, token=token)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModel.from_pretrained(_DINOV3_MODEL_ID, torch_dtype=dtype, token=token).eval().to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [dinov3-b / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs)
        pre_imgs = _pil_batch(bs)
        inp = processor(images=imgs, return_tensors="pt")
        inp = {k: v.to(device).to(dtype) if v.is_floating_point() else v.to(device) for k, v in inp.items()}

        def fn(_m=model, _inp=inp):
            with torch.no_grad():
                _m(**_inp)

        def preprocess_fn(_proc=processor, _pimgs=pre_imgs):
            _proc(images=_pimgs, return_tensors="pt")

        cell = _bench_cell(fn, preprocess_fn, bs, warmup, iters, runs, sync_fn=sync, desc=f"dinov3-b/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [dinov3-b / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  pre={cell.preprocess_p50_ms:.1f}ms"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="dinov3-b",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                preprocess_p50_ms=cell.preprocess_p50_ms,
                notes="forward pass only; preprocess_p50_ms = PIL→tensor (HF processor); file decode excluded",
            )
        )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# MPS: SigLIP2-base (teacher model)
# ---------------------------------------------------------------------------


def bench_siglip_base_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    from transformers import AutoModel, AutoProcessor

    MODEL_ID = "google/siglip2-base-patch16-224"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None

    print("\n  [siglip2-base / mps] loading …")
    t0 = time.perf_counter()
    dtype = torch.float16 if device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).eval().to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [siglip2-base / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=224)
        pre_imgs = _pil_batch(bs, size=224)
        inp = processor(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.to(device).to(dtype) if v.is_floating_point() else v.to(device) for k, v in inp.items()}

        def fn(_m=model, _inp=inp):
            with torch.no_grad():
                _m.get_image_features(**_inp)

        def preprocess_fn(_proc=processor, _pimgs=pre_imgs):
            _proc(images=_pimgs, return_tensors="pt", padding=True)

        cell = _bench_cell(fn, preprocess_fn, bs, warmup, iters, runs, sync_fn=sync, desc=f"siglip-base/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [siglip2-base / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  pre={cell.preprocess_p50_ms:.1f}ms"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="siglip2-base",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                preprocess_p50_ms=cell.preprocess_p50_ms,
                notes="teacher model; sigmoid multi-label; 224px input; forward pass only",
            )
        )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# MPS: YOLO11n-pose
# ---------------------------------------------------------------------------


def bench_yolo_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    from ultralytics import YOLO

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\n  [{_YOLO_MODEL_NAME} / mps] loading …")
    t0 = time.perf_counter()
    model = YOLO(str(_YOLO_CACHE_PATH))
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [{_YOLO_MODEL_NAME} / mps] load={load_s}s")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=640)

        def fn(_m=model, _imgs=imgs, _dev=_device):
            _m(_imgs, verbose=False, device=_dev)

        cell = _bench_cell(fn, None, bs, warmup, iters, runs, sync_fn=None, desc=f"yolo/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [yolo26n-pose / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="yolo26n-pose",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                notes="input is synthetic PIL images; real pipeline passes file paths (different Ultralytics code path)",
            )
        )

    del model
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# MPS: RMBG-2.0 (saliency / background removal)
# ---------------------------------------------------------------------------


def bench_rmbg_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    import torchvision.transforms as _TV
    from transformers import AutoModelForImageSegmentation

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None
    dtype = torch.float16 if device != "cpu" else torch.float32

    transform = _TV.Compose(
        [
            _TV.Resize((_RMBG_INPUT_SIZE, _RMBG_INPUT_SIZE)),
            _TV.ToTensor(),
            _TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    print("\n  [RMBG-2.0 / mps] loading …")
    t0 = time.perf_counter()
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
    model.eval()
    model = model.half().to(device) if device != "cpu" else model.to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [RMBG-2.0 / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=_RMBG_INPUT_SIZE)
        tensors = torch.stack([transform(img) for img in imgs]).to(device=device, dtype=dtype)

        def fn(_m=model, _t=tensors):
            with torch.no_grad():
                _m(_t)

        def preprocess_fn(_t=transform, _imgs=imgs):
            torch.stack([_t(img) for img in _imgs])

        cell = _bench_cell(fn, preprocess_fn, bs, warmup, iters, runs, sync_fn=sync, desc=f"rmbg/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [RMBG-2.0 / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  pre={cell.preprocess_p50_ms:.1f}ms"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="RMBG-2.0",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                preprocess_p50_ms=cell.preprocess_p50_ms,
                notes=(
                    "forward pass only (torchvision normalised tensor, 256x256 matching real pipeline); real pipeline adds palette KMeans in background thread"
                ),
            )
        )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# MPS: RMBG-1.4 (IS-Net — lighter alternative to RMBG-2.0)
# ---------------------------------------------------------------------------


def bench_rmbg14_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    import torchvision.transforms as _TV
    from transformers import AutoModelForImageSegmentation

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None
    dtype = torch.float16 if device != "cpu" else torch.float32

    transform = _TV.Compose(
        [
            _TV.Resize((_RMBG_INPUT_SIZE, _RMBG_INPUT_SIZE)),
            _TV.ToTensor(),
            _TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    print("\n  [RMBG-1.4 / mps] loading …")
    t0 = time.perf_counter()
    try:
        model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    except Exception as exc:
        print(f"  [RMBG-1.4] skipped — incompatible with current transformers: {exc}")
        return []
    model.eval()
    model = model.half().to(device) if device != "cpu" else model.to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [RMBG-1.4 / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=_RMBG_INPUT_SIZE)
        tensors = torch.stack([transform(img) for img in imgs]).to(device=device, dtype=dtype)

        def fn(_m=model, _t=tensors):
            with torch.no_grad():
                _m(_t)

        def preprocess_fn(_t=transform, _imgs=imgs):
            torch.stack([_t(img) for img in _imgs])

        cell = _bench_cell(fn, preprocess_fn, bs, warmup, iters, runs, sync_fn=sync, desc=f"rmbg14/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [RMBG-1.4 / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  pre={cell.preprocess_p50_ms:.1f}ms"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="RMBG-1.4",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                preprocess_p50_ms=cell.preprocess_p50_ms,
                notes=(
                    f"IS-Net architecture; forward pass only"
                    f" (torchvision normalised tensor, {_RMBG_INPUT_SIZE}x{_RMBG_INPUT_SIZE}"
                    " matching RMBG-2.0 pipeline input)"
                ),
            )
        )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# MPS: BiRefNet-lite (Swin-T backbone — intermediate between RMBG-1.4 and 2.0)
# ---------------------------------------------------------------------------


def bench_birefnet_lite_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    import torchvision.transforms as _TV
    from transformers import AutoModelForImageSegmentation

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None
    dtype = torch.float16 if device != "cpu" else torch.float32

    transform = _TV.Compose(
        [
            _TV.Resize((_RMBG_INPUT_SIZE, _RMBG_INPUT_SIZE)),
            _TV.ToTensor(),
            _TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    print("\n  [BiRefNet-lite / mps] loading …")
    t0 = time.perf_counter()
    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
    model.eval()
    model = model.half().to(device) if device != "cpu" else model.to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [BiRefNet-lite / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=_RMBG_INPUT_SIZE)
        tensors = torch.stack([transform(img) for img in imgs]).to(device=device, dtype=dtype)

        def fn(_m=model, _t=tensors):
            with torch.no_grad():
                out = _m(_t)
                # BiRefNet returns a list of tensors; take the last one (finest scale)
                pred = out[-1] if isinstance(out, list | tuple) else out
                pred.sigmoid()

        def preprocess_fn(_t=transform, _imgs=imgs):
            torch.stack([_t(img) for img in _imgs])

        cell = _bench_cell(fn, preprocess_fn, bs, warmup, iters, runs, sync_fn=sync, desc=f"birefnet-lite/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [BiRefNet-lite / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  pre={cell.preprocess_p50_ms:.1f}ms"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="BiRefNet-lite",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                preprocess_p50_ms=cell.preprocess_p50_ms,
                notes=(
                    f"BiRefNet Swin-T backbone (44M params); forward pass only"
                    f" (torchvision normalised tensor, {_RMBG_INPUT_SIZE}x{_RMBG_INPUT_SIZE}"
                    " matching RMBG-2.0 pipeline input)"
                ),
            )
        )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# MPS: aesthetic-predictor-v2-5
# ---------------------------------------------------------------------------


def bench_aesthetic_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None

    print("\n  [aesthetic-predictor-v2-5 / mps] loading …")
    t0 = time.perf_counter()
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model, preprocessor = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(dtype).to(device).eval()
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [aesthetic-predictor-v2-5 / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=384)
        pre_imgs = _pil_batch(bs, size=384)
        pixel_values = preprocessor(images=imgs, return_tensors="pt").pixel_values.to(dtype).to(device)

        def fn(_m=model, _pv=pixel_values):
            with torch.no_grad():
                _ = _m(_pv).logits

        def preprocess_fn(_proc=preprocessor, _pimgs=pre_imgs):
            _proc(images=_pimgs, return_tensors="pt")

        cell = _bench_cell(fn, preprocess_fn, bs, warmup, iters, runs, sync_fn=sync, desc=f"aesthetic/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [aesthetic-predictor-v2-5 / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  pre={cell.preprocess_p50_ms:.1f}ms"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="aesthetic-predictor-v2-5",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                preprocess_p50_ms=cell.preprocess_p50_ms,
                notes="SigLIP SO400M backbone + MLP head; forward pass only; preprocess_p50_ms = PIL→tensor (SigLIP preprocessor)",
            )
        )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# MPS: CLIP-IQA+
# ---------------------------------------------------------------------------


def bench_clipiqa_mps(batch_sizes, warmup, iters, runs, run_energy, energy_duration) -> list[BenchmarkResult]:
    import torchvision.transforms as _TV

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None

    # _IQ_INPUT_SIZE imported from extractors.scene — matches the real pipeline transform
    transform = _TV.Compose([_TV.Resize((_IQ_INPUT_SIZE, _IQ_INPUT_SIZE)), _TV.ToTensor()])

    print("\n  [clipiqa+ / mps] loading …")
    t0 = time.perf_counter()
    try:
        import pyiqa

        metric = pyiqa.create_metric("clipiqa+", device=device)
    except Exception as exc:
        print(f"  [clipiqa+] skipped — pyiqa unavailable: {exc}")
        return []
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [clipiqa+ / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=_IQ_INPUT_SIZE)
        pre_imgs = _pil_batch(bs, size=_IQ_INPUT_SIZE)
        tensors = torch.stack([transform(img) for img in imgs]).to(device)

        def fn(_m=metric, _t=tensors):
            with torch.no_grad():
                _m(_t)

        def preprocess_fn(_t=transform, _pimgs=pre_imgs):
            torch.stack([_t(img) for img in _pimgs])

        cell = _bench_cell(fn, preprocess_fn, bs, warmup, iters, runs, sync_fn=sync, desc=f"clipiqa/mps bs={bs}")
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)
        cv_str = f"  cv={cell.cv_pct:.1f}%" if cell.cv_pct is not None else ""
        mem = cell.peak_memory_mb or 0
        print(
            f"  [clipiqa+ / mps] bs={bs:2d}"
            f"  p50={cell.p50_ms:.1f}ms  p95={cell.p95_ms:.1f}ms"
            f"  {cell.imgs_per_s} img/s{cv_str}"
            f"  pre={cell.preprocess_p50_ms:.1f}ms"
            f"  mem≈{mem:.0f}MB RSS  sys={cell.system_available_mb:.0f}MB free"
        )
        results.append(
            BenchmarkResult(
                model="clipiqa+",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=cell.p50_ms,
                p95_ms=cell.p95_ms,
                p50_per_image_ms=round(cell.p50_ms / bs, 3),
                p95_per_image_ms=round(cell.p95_ms / bs, 3),
                imgs_per_s=cell.imgs_per_s,
                peak_memory_mb=cell.peak_memory_mb,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                runs=cell.runs,
                cv_pct=cell.cv_pct,
                system_available_mb=cell.system_available_mb,
                mps_alloc_mb_debug=cell.mps_alloc_mb_debug,
                preprocess_p50_ms=cell.preprocess_p50_ms,
                notes="forward pass only (torchvision Resize+ToTensor 512×512); real pipeline decodes from file path",
            )
        )

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results


# ---------------------------------------------------------------------------
# End-to-end benchmark — calls real extractor functions with real photo paths
# ---------------------------------------------------------------------------

_E2E_SAMPLE_DEFAULT = 200  # use --e2e-samples 50 for a quick calibration run


def _sample_photos(photos_dir: Path, n: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".heic", ".tiff", ".webp"}
    paths = [p for p in sorted(photos_dir.rglob("*")) if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No image files found under {photos_dir}")
    rng = __import__("random").Random(42)
    return rng.sample(paths, min(n, len(paths)))


def _e2e_result(model_name: str, paths: list[Path], batch_size: int, wall_s_samples: list[float], load_s: float) -> BenchmarkResult:
    n = len(paths)
    ips_samples = [round(n / w, 1) for w in wall_s_samples if w > 0]
    med_ips = round(statistics.median(ips_samples), 1) if ips_samples else 0.0
    med_wall = statistics.median(wall_s_samples) if wall_s_samples else 0.0
    cv: float | None = None
    if len(ips_samples) > 1 and statistics.mean(ips_samples) > 0:
        cv = round(statistics.stdev(ips_samples) / statistics.mean(ips_samples) * 100, 1)
    sys_mb = _system_available_mb()
    return BenchmarkResult(
        model=model_name,
        backend="mps",
        batch_size=batch_size,
        load_time_s=load_s,
        p50_ms=0.0,
        p95_ms=0.0,  # not meaningful for e2e wall-time measurement
        p50_per_image_ms=round(med_wall / n * 1000, 1) if n else 0.0,
        p95_per_image_ms=0.0,
        imgs_per_s=med_ips,
        peak_memory_mb=None,
        images_per_joule=None,
        ane_power_mw=None,
        ane_active=False,
        accuracy_metric=None,
        accuracy_value=None,
        runs=len(wall_s_samples),
        cv_pct=cv,
        system_available_mb=round(sys_mb, 0),
        benchmark_mode="e2e",
        notes=(
            f"median wall-time over {n} real photos x {len(wall_s_samples)} run(s);"
            " includes file decode + processor + forward + post-processing; cache writes excluded"
        ),
    )


def bench_e2e(photos_dir: Path, batch_size: int, n_samples: int = _E2E_SAMPLE_DEFAULT, runs: int = 1) -> list[BenchmarkResult]:
    """Run every extractor against real photo paths. One result per extractor.

    Each extractor is timed `runs` times on the same photo set; median imgs/s
    and CV% are reported.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    paths = _sample_photos(photos_dir, n_samples)
    str_paths = [str(p) for p in paths]
    n_batches = (len(paths) + batch_size - 1) // batch_size
    run_tag = f"{runs} run(s)" if runs > 1 else "1 run"
    print(f"\n  [e2e] {len(paths)} photos × {run_tag}  {n_batches} batches  device={device}  bs={batch_size}")

    results: list[BenchmarkResult] = []

    def _run_with_progress(label: str, batch_fn) -> list[float]:
        """Call batch_fn(batch_str_paths) for each batch, `runs` times.

        Shows `label  run R/N  batch B/N  Xs elapsed` on a single overwritten line.
        batch_fn receives a list of str paths and returns anything (result discarded).
        """
        samples: list[float] = []
        for r in range(runs):
            run_pfx = f"run {r + 1}/{runs}  " if runs > 1 else ""
            t0 = time.perf_counter()
            for b, start in enumerate(range(0, len(paths), batch_size)):
                elapsed = time.perf_counter() - t0
                sys.stdout.write(f"\r  {label}  {run_pfx}batch {b + 1}/{n_batches}  {elapsed:.1f}s")
                sys.stdout.flush()
                batch_fn(str_paths[start : start + batch_size])
            samples.append(time.perf_counter() - t0)
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
        return samples

    # ── DINOv3-B ──────────────────────────────────────────────────────────────
    from extractors.embedding import extract_embedding_batch, load_dino_model

    t0 = time.perf_counter()
    model, processor, dev = load_dino_model()
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [e2e] DINOv3-B  loaded {load_s}s")
    wall = _run_with_progress("DINOv3-B", lambda b: extract_embedding_batch(b, model, processor, dev, batch_size=batch_size))
    results.append(_e2e_result("dinov3-b", paths, batch_size, wall, load_s))
    model = None
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    r = results[-1]
    cv_str = f"  cv={r.cv_pct:.1f}%" if r.cv_pct is not None else ""
    print(f"  [e2e] DINOv3-B  {r.imgs_per_s} img/s{cv_str}  sys={r.system_available_mb:.0f}MB free")

    # ── SigLIP2-base (teacher) ───────────────────────────────────────────────
    from transformers import AutoModel, AutoProcessor

    dtype = torch.float16 if device != "cpu" else torch.float32
    t0 = time.perf_counter()
    proc_b = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    mdl_b = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=dtype).eval().to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [e2e] SigLIP base  loaded {load_s}s")

    def _siglip_base_batch(batch):
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inp = proc_b(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.to(device).to(dtype) if v.is_floating_point() else v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            mdl_b.get_image_features(**inp)

    wall = _run_with_progress("SigLIP base", _siglip_base_batch)
    results.append(_e2e_result("siglip2-base", paths, batch_size, wall, load_s))
    mdl_b = None
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    r = results[-1]
    cv_str = f"  cv={r.cv_pct:.1f}%" if r.cv_pct is not None else ""
    print(f"  [e2e] SigLIP base  {r.imgs_per_s} img/s{cv_str}  sys={r.system_available_mb:.0f}MB free")

    # ── YOLO ────────────────────────────────────────────────────────────────
    from ultralytics import YOLO

    from extractors.pose import extract_pose_batch

    t0 = time.perf_counter()
    yolo = YOLO(str(_YOLO_CACHE_PATH))
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [e2e] YOLO  loaded {load_s}s")
    # Use extract_pose_batch — matches the real pipeline: iter_prefetched decodes
    # with 4-thread pool + PIL draft mode, then passes PIL images to YOLO directly
    # (bypassing YOLO's internal C++ OpenCV decode, same as main.py Pass 6).
    wall = _run_with_progress("YOLO    ", lambda b: extract_pose_batch(b, yolo, device, batch_size=batch_size))
    results.append(_e2e_result("yolo26n-pose", paths, batch_size, wall, load_s))
    yolo = None
    gc.collect()
    r = results[-1]
    cv_str = f"  cv={r.cv_pct:.1f}%" if r.cv_pct is not None else ""
    print(f"  [e2e] YOLO  {r.imgs_per_s} img/s{cv_str}  sys={r.system_available_mb:.0f}MB free")

    # ── RMBG-2.0 ────────────────────────────────────────────────────────────
    from extractors.saliency import extract_saliency_batch, load_saliency_model

    t0 = time.perf_counter()
    sal_model = load_saliency_model(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [e2e] RMBG-2.0  loaded {load_s}s")
    wall = _run_with_progress("RMBG-2.0", lambda b: extract_saliency_batch(b, sal_model, batch_size))
    results.append(_e2e_result("RMBG-2.0", paths, batch_size, wall, load_s))
    sal_model = None
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    r = results[-1]
    cv_str = f"  cv={r.cv_pct:.1f}%" if r.cv_pct is not None else ""
    print(f"  [e2e] RMBG-2.0  {r.imgs_per_s} img/s{cv_str}  sys={r.system_available_mb:.0f}MB free")

    # ── aesthetic-predictor-v2-5 ────────────────────────────────────────────
    from extractors.scene import extract_aesthetic_batch, load_aesthetic_model

    t0 = time.perf_counter()
    aes = load_aesthetic_model(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [e2e] aesthetic  loaded {load_s}s")
    wall = _run_with_progress("aesthetic", lambda b: extract_aesthetic_batch(b, aes, batch_size))
    results.append(_e2e_result("aesthetic-predictor-v2-5", paths, batch_size, wall, load_s))
    aes = None
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    r = results[-1]
    cv_str = f"  cv={r.cv_pct:.1f}%" if r.cv_pct is not None else ""
    print(f"  [e2e] aesthetic  {r.imgs_per_s} img/s{cv_str}  sys={r.system_available_mb:.0f}MB free")

    # ── CLIP-IQA+ ───────────────────────────────────────────────────────────
    from extractors.scene import extract_iq_batch, load_clipiqa_metric

    t0 = time.perf_counter()
    iq = load_clipiqa_metric(device)
    load_s = round(time.perf_counter() - t0, 2)
    if isinstance(iq, Exception):
        print(f"  [e2e] CLIP-IQA+ skipped: {iq}")
    else:
        print(f"  [e2e] CLIP-IQA+  loaded {load_s}s")
        wall = _run_with_progress("CLIP-IQA+", lambda b: extract_iq_batch(b, iq, batch_size))
        results.append(_e2e_result("clipiqa+", paths, batch_size, wall, load_s))
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        r = results[-1]
        cv_str = f"  cv={r.cv_pct:.1f}%" if r.cv_pct is not None else ""
        print(f"  [e2e] CLIP-IQA+  {r.imgs_per_s} img/s{cv_str}  sys={r.system_available_mb:.0f}MB free")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_ALL_MODELS = ["dinov3", "siglip-base", "yolo", "rmbg", "rmbg14", "birefnet-lite", "aesthetic", "clipiqa"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apple Silicon backend benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/benchmark_backends.py                   # all models, sensible defaults
  uv run python scripts/benchmark_backends.py --model dinov3    # one model only
  uv run python scripts/benchmark_backends.py --e2e             # add end-to-end pass (needs PHOTO_DIR)
  uv run python scripts/benchmark_backends.py --full            # micro + e2e + regenerate scheduler profile
  sudo uv run python scripts/benchmark_backends.py --energy     # add energy measurement (powermetrics)
""",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=_ALL_MODELS,
        choices=_ALL_MODELS,
        metavar="MODEL",
        help=f"Models to benchmark (default: all — {' '.join(_ALL_MODELS)})",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16],
        metavar="N",
        help="Batch sizes to sweep (default: 1 4 8 16)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per cell; median + CV%% reported (default: 3)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per run (default: 10)")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations per run (default: 100)")
    parser.add_argument("--e2e", action="store_true", help="Also run end-to-end pass with real photos (reads PHOTO_DIR from .env)")
    parser.add_argument("--full", action="store_true", help="Run all microbench + e2e + regenerate scheduler profile in one shot")
    parser.add_argument("--energy", action="store_true", help="Measure perf-per-watt via powermetrics (requires sudo)")
    parser.add_argument("--energy-duration", type=float, default=30.0, help="Energy measurement duration in seconds (default: 30)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/backend_comparison.json"),
        help="Output JSON (default: docs/backend_comparison.json)",
    )
    args = parser.parse_args()

    if args.full:
        args.e2e = True

    # Load .env so HF_TOKEN and PHOTO_DIR are available without shell export
    import os

    _dotenv = Path(__file__).parent.parent / ".env"
    if _dotenv.exists():
        for line in _dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:  # don't override shell exports
                os.environ[key] = val

    # Resolve PHOTO_DIR for e2e mode
    photos_dir: Path | None = None
    _env_photo_dir = os.environ.get("PHOTO_DIR", "")
    if _env_photo_dir:
        photos_dir = Path(_env_photo_dir).expanduser()

    all_results: list[BenchmarkResult] = []
    _models = args.model
    _step = 0

    def _step_print(model: str) -> None:
        nonlocal _step
        _step += 1
        print(f"\n[{_step}/{len(_models)}] {model} / mps")

    e2e_results: list[BenchmarkResult] = []

    if args.e2e:
        if not photos_dir:
            parser.error("--e2e / --full requires PHOTO_DIR set in .env")
        print(f"\n[e2e] {photos_dir}  bs=8  samples={_E2E_SAMPLE_DEFAULT}  runs={args.runs}")
        e2e_results.extend(bench_e2e(photos_dir, batch_size=8, n_samples=_E2E_SAMPLE_DEFAULT, runs=args.runs))

    _kw = dict(warmup=args.warmup, iters=args.iters, runs=args.runs, run_energy=args.energy, energy_duration=args.energy_duration)

    if "dinov3" in _models:
        _step_print("dinov3-b")
        all_results.extend(bench_dinov3_mps(args.batch_sizes, **_kw))
    if "siglip-base" in _models:
        _step_print("siglip2-base")
        all_results.extend(bench_siglip_base_mps(args.batch_sizes, **_kw))
    if "yolo" in _models:
        _step_print("yolo26n-pose")
        all_results.extend(bench_yolo_mps(args.batch_sizes, **_kw))
    if "rmbg" in _models:
        _step_print("RMBG-2.0")
        all_results.extend(bench_rmbg_mps(args.batch_sizes, **_kw))
    if "rmbg14" in _models:
        _step_print("RMBG-1.4")
        all_results.extend(bench_rmbg14_mps(args.batch_sizes, **_kw))
    if "birefnet-lite" in _models:
        _step_print("BiRefNet-lite")
        all_results.extend(bench_birefnet_lite_mps(args.batch_sizes, **_kw))
    if "aesthetic" in _models:
        _step_print("aesthetic-predictor-v2-5")
        all_results.extend(bench_aesthetic_mps(args.batch_sizes, **_kw))
    if "clipiqa" in _models:
        _step_print("clipiqa+")
        all_results.extend(bench_clipiqa_mps(args.batch_sizes, **_kw))

    # ------ Output — merge with existing file (deduplicate by model+backend+batch_size) ------
    # Both microbench and e2e results are merged independently so a partial re-run
    # (e.g. --model siglip --runs 3) only updates those rows and preserves everything else,
    # including existing e2e_results when --e2e is not passed this run.
    existing_data: dict = {}
    if args.output.exists():
        try:
            existing_data = json.loads(args.output.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    # Merge microbench results
    new_by_key = {(r.model, r.backend, r.batch_size): asdict(r) for r in all_results}
    merged: dict[tuple, dict] = {}
    for r in existing_data.get("results", []):
        merged[(r["model"], r["backend"], r["batch_size"])] = r
    merged.update(new_by_key)

    # Merge e2e results — preserve existing when --e2e not passed this run
    new_e2e_by_key = {(r.model, r.backend, r.batch_size): asdict(r) for r in e2e_results}
    merged_e2e: dict[tuple, dict] = {}
    for r in existing_data.get("e2e_results", []):
        merged_e2e[(r["model"], r["backend"], r["batch_size"])] = r
    merged_e2e.update(new_e2e_by_key)

    output = {
        "run_date": __import__("datetime").date.today().isoformat(),
        "hardware": f"Apple {__import__('platform').processor()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "runs_per_cell": args.runs,
        "energy_measured": args.energy,
        "system_available_mb_at_start": round(_system_available_mb(), 0),
        "benchmark_caveats": {
            "scope": (
                "DINO/SigLIP: times model forward pass only on pre-processed tensors already on device. "
                "Preprocessing (PIL→tensor via HF processor) is measured separately in preprocess_p50_ms "
                "but file decode is not included. "
                "YOLO: times inference on synthetic PIL images; real pipeline passes file paths, "
                "which may trigger a different Ultralytics preprocessing code path."
            ),
            "peak_memory_mb": (
                "DINO/SigLIP values are RSS delta (approx) — "
                "torch.mps.current_allocated_memory() delta returns 0 after activations are freed. "
                "YOLO values are also RSS delta. Neither reflects peak GPU allocation during forward pass. "
                "Use pipeline_profile.json for real peak memory under production conditions."
            ),
            "batch_scaling_signal": (
                "DINOv3-B and YOLO batch-scaling curves are valid. "
                "SigLIP flat scaling is valid. "
                "For absolute throughput numbers use pipeline_profile.json (end-to-end pass timing)."
            ),
            "scheduler_guidance": (
                "Use batch-scaling shape from this file and pass-level throughput from pipeline_profile.json. "
                "Do not use these microbenchmarks as the sole input to a runtime scheduler — "
                "prefer measured rules or bandits calibrated against real pipeline runs."
            ),
        },
        "results": list(merged.values()),
        "e2e_results": list(merged_e2e.values()),
    }

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    print(f"\n{'─' * 72}")
    print(f"Results ({len(all_results)} micro + {len(e2e_results)} e2e) → {args.output}")
    print(f"{'─' * 72}")
    print(f"{'model':<25} {'bs':>3}  {'p50':>7}  {'p95':>7}  {'img/s':>7}  {'cv%':>5}  {'sys MB':>7}")
    print(f"{'─' * 72}")
    for r in all_results:
        cv = f"{r.cv_pct:.1f}" if r.cv_pct is not None else "  —"
        sys_mb = f"{r.system_available_mb:.0f}" if r.system_available_mb is not None else "  —"
        print(f"{r.model:<25} {r.batch_size:>3}  {r.p50_ms:>6.1f}ms  {r.p95_ms:>6.1f}ms  {r.imgs_per_s:>6.1f}  {cv:>5}  {sys_mb:>7}")
    if e2e_results:
        print("\n  E2E (real photos, bs=8):")
        for r in e2e_results:
            sys_mb = f"{r.system_available_mb:.0f}" if r.system_available_mb is not None else "  —"
            print(f"  {r.model:<23} {r.imgs_per_s:>6.1f} img/s  avg {r.p50_per_image_ms:.0f}ms/img  sys={sys_mb}MB free")

    # --full: regenerate scheduler profile from updated benchmark data
    if args.full:
        import runpy

        _gen = Path(__file__).parent / "generate_scheduler_profile.py"
        print("\n[full] Regenerating scheduler profile …")
        runpy.run_path(str(_gen), run_name="__main__")
        print("[full] Done — docs/scheduler_profile.json updated")


if __name__ == "__main__":
    main()

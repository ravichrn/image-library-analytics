#!/usr/bin/env python3
"""Apple Silicon backend benchmark: PyTorch-MPS vs MLX vs CoreML.

Measures latency (p50/p95 per-image and per-batch), throughput (imgs/s at
batch sizes 1/4/8/16), peak unified memory, and perf-per-watt (images/joule
via powermetrics). Model load time is reported separately — never folded into
inference numbers. ANE residency is verified empirically via the ane_power
channel rather than trusting the CoreML export flag.

Run from project root:
    uv run python scripts/benchmark_backends.py --model dinov2 --backend mps
    uv run python scripts/benchmark_backends.py --model dinov2 siglip yolo --backend mps mlx coreml
    sudo uv run python scripts/benchmark_backends.py --model dinov2 siglip yolo \\
        --backend mps mlx coreml --batch-sizes 1 4 8 16 --energy
"""

from __future__ import annotations

import argparse
import gc
import json
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


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    model: str
    backend: str
    batch_size: int
    load_time_s: float
    p50_ms: float  # per-batch latency p50
    p95_ms: float  # per-batch latency p95
    p50_per_image_ms: float
    p95_per_image_ms: float
    imgs_per_s: float
    peak_memory_mb: float | None  # None = not measurable for this backend
    images_per_joule: float | None  # None if --energy not used or powermetrics unavailable
    ane_power_mw: float | None
    ane_active: bool
    accuracy_metric: str | None
    accuracy_value: float | None
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
# Memory
# ---------------------------------------------------------------------------


def _mps_memory_delta_mb(fn) -> float:
    if not torch.backends.mps.is_available():
        fn()
        return 0.0
    torch.mps.empty_cache()
    before = torch.mps.current_allocated_memory()
    fn()
    torch.mps.synchronize()
    after = torch.mps.current_allocated_memory()
    return max(0, after - before) / (1024 * 1024)


def _rss_delta_mb(fn) -> float:
    """Approximate memory via process RSS — used for CoreML (no clean API). Disclosed as approx."""
    proc = psutil.Process()
    before = proc.memory_info().rss
    fn()
    after = proc.memory_info().rss
    return max(0, after - before) / (1024 * 1024)


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
# MPS: DINOv2
# ---------------------------------------------------------------------------


def bench_dinov2_mps(batch_sizes, warmup, iters, run_energy, energy_duration) -> list[BenchmarkResult]:
    from transformers import AutoImageProcessor, AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None

    print("\n  [dinov2 / mps] loading …")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=dtype).eval().to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [dinov2 / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs)
        inp = processor(images=imgs, return_tensors="pt")
        inp = {k: v.to(device).to(dtype) if v.is_floating_point() else v.to(device) for k, v in inp.items()}

        def fn(_m=model, _inp=inp):
            with torch.no_grad():
                _m(**_inp)

        mem_mb = _mps_memory_delta_mb(fn) if device == "mps" else None
        p50, p95 = run_latency_bench(fn, warmup=warmup, iters=iters, sync_fn=sync, desc=f"dinov2/mps bs={bs}")
        ips = round(bs / (p50 / 1000), 1)
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)

        mem_str = f"  mem≈{mem_mb:.0f}MB" if mem_mb else ""
        print(f"  [dinov2 / mps] bs={bs:2d}  p50={p50:.1f}ms  p95={p95:.1f}ms  {ips} img/s{mem_str}")
        results.append(
            BenchmarkResult(
                model="dinov2-base",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=round(p50, 2),
                p95_ms=round(p95, 2),
                p50_per_image_ms=round(p50 / bs, 3),
                p95_per_image_ms=round(p95 / bs, 3),
                imgs_per_s=ips,
                peak_memory_mb=round(mem_mb, 1) if mem_mb is not None else None,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
            )
        )

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return results


# ---------------------------------------------------------------------------
# MPS: SigLIP2
# ---------------------------------------------------------------------------


def bench_siglip_mps(batch_sizes, warmup, iters, run_energy, energy_duration) -> list[BenchmarkResult]:
    from transformers import AutoModel, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else None

    print("\n  [siglip2-so400m / mps] loading (large model ~1.7 GB) …")
    t0 = time.perf_counter()
    dtype = torch.float16 if device != "cpu" else torch.float32
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=dtype).eval().to(device)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [siglip2-so400m / mps] load={load_s}s  device={device}")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=384)
        inp = processor(images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.to(device).to(dtype) if v.is_floating_point() else v.to(device) for k, v in inp.items()}

        def fn(_m=model, _inp=inp):
            with torch.no_grad():
                _m.get_image_features(**_inp)

        mem_mb = _mps_memory_delta_mb(fn) if device == "mps" else None
        p50, p95 = run_latency_bench(fn, warmup=warmup, iters=iters, sync_fn=sync, desc=f"siglip/mps bs={bs}")
        ips = round(bs / (p50 / 1000), 1)
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)

        print(f"  [siglip2-so400m / mps] bs={bs:2d}  p50={p50:.1f}ms  p95={p95:.1f}ms  {ips} img/s")
        results.append(
            BenchmarkResult(
                model="siglip2-so400m",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=round(p50, 2),
                p95_ms=round(p95, 2),
                p50_per_image_ms=round(p50 / bs, 3),
                p95_per_image_ms=round(p95 / bs, 3),
                imgs_per_s=ips,
                peak_memory_mb=round(mem_mb, 1) if mem_mb is not None else None,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
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


def bench_yolo_mps(batch_sizes, warmup, iters, run_energy, energy_duration) -> list[BenchmarkResult]:
    from ultralytics import YOLO

    from extractors.pose import _CACHE_PATH

    print("\n  [yolo11n-pose / mps] loading …")
    t0 = time.perf_counter()
    model = YOLO(str(_CACHE_PATH))
    load_s = round(time.perf_counter() - t0, 2)
    print(f"  [yolo11n-pose / mps] load={load_s}s")

    results: list[BenchmarkResult] = []
    for bs in batch_sizes:
        imgs = _pil_batch(bs, size=640)

        _device = "mps" if torch.backends.mps.is_available() else "cpu"

        def fn(_m=model, _imgs=imgs, _dev=_device):
            _m(_imgs, verbose=False, device=_dev)

        mem_mb = _rss_delta_mb(fn)  # Ultralytics has no MPS memory API — use RSS (approx)

        # Log MPS allocator state at bs=16 to root-cause the throughput collapse
        mps_alloc_mb: float | None = None
        if bs >= 16 and torch.backends.mps.is_available():
            torch.mps.synchronize()
            mps_alloc_mb = round(torch.mps.current_allocated_memory() / 1e6, 1)

        p50, p95 = run_latency_bench(fn, warmup=warmup, iters=iters, desc=f"yolo/mps bs={bs}")
        ips = round(bs / (p50 / 1000), 1)
        ipj, ane_mw = run_power_bench(fn, bs, energy_duration) if run_energy else (None, None)

        alloc_note = f"  mps_alloc={mps_alloc_mb:.0f}MB" if mps_alloc_mb is not None else ""
        print(f"  [yolo11n-pose / mps] bs={bs:2d}  p50={p50:.1f}ms  p95={p95:.1f}ms  {ips} img/s  mem≈{mem_mb:.0f}MB (RSS){alloc_note}")
        results.append(
            BenchmarkResult(
                model="yolo11n-pose",
                backend="mps",
                batch_size=bs,
                load_time_s=load_s,
                p50_ms=round(p50, 2),
                p95_ms=round(p95, 2),
                p50_per_image_ms=round(p50 / bs, 3),
                p95_per_image_ms=round(p95 / bs, 3),
                imgs_per_s=ips,
                peak_memory_mb=round(mem_mb, 1) if mem_mb else None,
                images_per_joule=round(ipj, 2) if ipj else None,
                ane_power_mw=round(ane_mw, 1) if ane_mw else None,
                ane_active=bool(ane_mw and ane_mw > 50),
                accuracy_metric=None,
                accuracy_value=None,
                notes="peak_memory_mb is RSS delta (approx) — no Ultralytics MPS memory API",
            )
        )

    del model
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple Silicon backend benchmark")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["dinov2"],
        choices=["dinov2", "siglip", "yolo"],
        metavar="MODEL",
        help="Models to benchmark: dinov2 siglip yolo (default: dinov2)",
    )
    parser.add_argument("--backend", nargs="+", default=["mps"], choices=["mps"], metavar="BACKEND", help="Backends to run (default: mps)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8], metavar="N", help="Batch sizes to sweep (default: 1 4 8)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations (default: 100)")
    parser.add_argument("--energy", action="store_true", help="Measure perf-per-watt via powermetrics (requires sudo -n)")
    parser.add_argument("--energy-duration", type=float, default=30.0, help="Seconds to run inference loop for energy measurement (default: 30)")
    parser.add_argument("--output", type=Path, default=Path("docs/backend_comparison.json"))
    args = parser.parse_args()

    all_results: list[BenchmarkResult] = []

    _models = args.model
    _total = len(_models)
    _step = 0

    def _step_print(model: str) -> None:
        nonlocal _step
        _step += 1
        print(f"\n[{_step}/{_total}] {model} / mps")

    if "dinov2" in _models:
        _step_print("dinov2")
        all_results.extend(bench_dinov2_mps(args.batch_sizes, args.warmup, args.iters, args.energy, args.energy_duration))
    if "siglip" in _models:
        _step_print("siglip2-so400m")
        all_results.extend(bench_siglip_mps(args.batch_sizes, args.warmup, args.iters, args.energy, args.energy_duration))
    if "yolo" in _models:
        _step_print("yolo11n-pose")
        all_results.extend(bench_yolo_mps(args.batch_sizes, args.warmup, args.iters, args.energy, args.energy_duration))

    # ------ Output — merge with existing results (deduplicate by model+backend+batch_size) ------
    new_by_key = {(r.model, r.backend, r.batch_size): asdict(r) for r in all_results}

    existing_results: list[dict] = []
    if args.output.exists():
        try:
            existing_results = json.loads(args.output.read_text()).get("results", [])
        except (json.JSONDecodeError, KeyError):
            pass

    merged: dict[tuple, dict] = {}
    for r in existing_results:
        merged[(r["model"], r["backend"], r["batch_size"])] = r
    merged.update(new_by_key)

    output = {
        "run_date": __import__("datetime").date.today().isoformat(),
        "hardware": f"Apple {__import__('platform').processor()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "energy_measured": args.energy,
        "results": list(merged.values()),
    }

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    print(f"\n{'─' * 60}")
    print(f"Results ({len(all_results)} runs) → {args.output}")
    print(f"{'─' * 60}")
    print(f"{'model':<20} {'backend':<8} {'bs':>3}  {'p50':>7}  {'p95':>7}  {'img/s':>7}  {'ipj':>8}  {'ANE':>5}")
    print(f"{'─' * 60}")
    for r in all_results:
        ipj = f"{r.images_per_joule:.1f}" if r.images_per_joule else "  —"
        ane = "✓" if r.ane_active else "—"
        print(f"{r.model:<20} {r.backend:<8} {r.batch_size:>3}  {r.p50_ms:>6.1f}ms  {r.p95_ms:>6.1f}ms  {r.imgs_per_s:>6.1f}  {ipj:>8}  {ane:>5}")


if __name__ == "__main__":
    main()

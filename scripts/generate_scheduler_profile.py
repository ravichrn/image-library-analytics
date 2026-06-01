#!/usr/bin/env python3
"""Generate docs/scheduler_profile.json from docs/backend_comparison.json.

Algorithm per model:
  1. Collect all microbench results, sort by batch_size.
  2. best_batch: smallest batch_size that reaches >= 99% of maximum throughput
     (point of diminishing returns — minimises memory use for equivalent throughput).
  3. safe_batch: best_batch // 2, snapped down to the nearest valid arm.
     Conservative fallback; Layer 2 runtime handles actual memory adaptation.
  4. max_batch: same as best_batch.
  5. valid_arms: [1, 2, 4, 8, 12, 16] filtered to <= max_batch.
  6. profiled_* values: taken from the best_batch row.

Run from project root:
    uv run python scripts/generate_scheduler_profile.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_BENCH_PATH = Path("docs/backend_comparison.json")
_OUT_PATH = Path("docs/scheduler_profile.json")
_ALL_ARMS = [1, 2, 4, 8, 12, 16, 32]
_BEST_THRESHOLD = 0.99  # smallest bs reaching this fraction of max throughput


def _nearest_arm_at_or_below(value: int, arms: list[int]) -> int:
    candidates = [a for a in arms if a <= value]
    return max(candidates) if candidates else arms[0]


def _derive_profile(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["batch_size"])
    max_ips = max(r["imgs_per_s"] for r in rows)
    threshold = _BEST_THRESHOLD * max_ips

    # best_batch: smallest bs reaching >= 99% of max throughput
    best_row = next(r for r in rows if r["imgs_per_s"] >= threshold)
    best_bs = best_row["batch_size"]
    valid_arms = [a for a in _ALL_ARMS if a <= best_bs]

    # safe_batch: half of best, snapped to nearest valid arm
    safe_bs = _nearest_arm_at_or_below(best_bs // 2, valid_arms) if best_bs > 1 else 1

    all_sys_mb = [r["system_available_mb"] for r in rows if r.get("system_available_mb")]
    return {
        "safe_batch": safe_bs,
        "best_batch": best_bs,
        "max_batch": best_bs,
        "valid_arms": valid_arms,
        "profiled_imgs_per_s": best_row["imgs_per_s"],
        "profiled_p95_ms": best_row.get("p95_ms", 0.0),
        # Memory at best_batch — Layer 2 "safe" threshold
        "profiled_system_available_mb": best_row.get("system_available_mb"),
        # Minimum across all batch sizes — Layer 2 "danger" threshold
        "min_system_available_mb": min(all_sys_mb) if all_sys_mb else None,
        "profiled_cv_pct": best_row.get("cv_pct"),
    }


def main() -> None:
    if not _BENCH_PATH.exists():
        print(f"ERROR: {_BENCH_PATH} not found. Run benchmark first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(_BENCH_PATH.read_text())
    microbench = [r for r in data.get("results", []) if r.get("benchmark_mode", "microbench") == "microbench"]

    if not microbench:
        print("ERROR: No microbench results found in backend_comparison.json.", file=sys.stderr)
        sys.exit(1)

    by_model: dict[str, list[dict]] = {}
    for r in microbench:
        by_model.setdefault(r["model"], []).append(r)

    models: dict[str, dict] = {}
    for model_name, rows in sorted(by_model.items()):
        models[model_name] = _derive_profile(rows)
        p = models[model_name]
        print(
            f"  {model_name:<28}  safe={p['safe_batch']:>2}  best={p['best_batch']:>2}  "
            f"max={p['max_batch']:>2}  {p['profiled_imgs_per_s']:>6} img/s  "
            f"cv={p['profiled_cv_pct']}%"
        )

    profile = {
        "generated": date.today().isoformat(),
        "source": str(_BENCH_PATH),
        "models": models,
    }

    _OUT_PATH.parent.mkdir(exist_ok=True)
    _OUT_PATH.write_text(json.dumps(profile, indent=2))
    print(f"\nWrote {_OUT_PATH}  ({len(models)} models)")


if __name__ == "__main__":
    main()

"""Generate a systems/benchmark report from pipeline profiling data.

Reads three JSON files written during pipeline runs and renders a static HTML
page that surfaces the engineering: per-model throughput, memory footprint,
cascade filtering rates, and the scheduler's bandit state.
"""

import json
from pathlib import Path

import jinja2
from markupsafe import Markup

OUTPUT_DIR = Path("docs")
_TEMPLATE_DIR = Path(__file__).parent / "templates"

_PROFILE_PATH = OUTPUT_DIR / "pipeline_profile.json"
_SCHED_PROFILE_PATH = OUTPUT_DIR / "scheduler_profile.json"
_SCHED_STATE_PATH = OUTPUT_DIR / "scheduler_state.json"
_BENCH_COMPARISON_PATH = OUTPUT_DIR / "backend_comparison.json"

# Colours assigned round-robin to models in the batch-scaling charts.
_PALETTE = ["#7c6af7", "#4ade80", "#fb923c", "#f472b6", "#38bdf8", "#facc15", "#a78bfa"]


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


_SO400M_MB = 3174  # SigLIP SO400M loaded for all photos without the aesthetic regressor


def _naive_peak_mb(passes: list[dict]) -> float:
    """Sum of all per-pass peak memory plus SO400M cost without the aesthetic regressor."""
    return sum(p.get("model_memory_mb") or 0.0 for p in passes) + _SO400M_MB


def _dominant_arm(arms: dict[str, int], rewards: dict[str, float]) -> str | None:
    """Return the batch-size arm with the highest mean reward."""
    best, best_mean = None, -1.0
    for arm_str, n in arms.items():
        if n == 0:
            continue
        mean = rewards.get(arm_str, 0.0) / n
        if mean > best_mean:
            best_mean = mean
            best = arm_str
    return best


def _prepare_batch_curves(bench: dict) -> dict:
    """Build per-model throughput and latency curves from backend_comparison.json."""
    by_model: dict[str, list] = {}
    for r in bench.get("results", []):
        by_model.setdefault(r["model"], []).append(r)

    # Sort each model's rows by batch_size; collect the union of all batch sizes.
    all_bs: set[int] = set()
    for rows in by_model.values():
        rows.sort(key=lambda r: r["batch_size"])
        all_bs.update(r["batch_size"] for r in rows)
    batch_sizes = sorted(all_bs)

    throughput_datasets = []
    latency_datasets = []
    for i, (model, rows) in enumerate(sorted(by_model.items())):
        colour = _PALETTE[i % len(_PALETTE)]
        bs_to_row = {r["batch_size"]: r for r in rows}
        throughput_datasets.append(
            {
                "label": model,
                "data": [bs_to_row[bs]["imgs_per_s"] if bs in bs_to_row else None for bs in batch_sizes],
                "borderColor": colour,
                "backgroundColor": colour + "22",
                "tension": 0.3,
                "spanGaps": True,
            }
        )
        latency_datasets.append(
            {
                "label": model,
                "data": [bs_to_row[bs].get("p50_per_image_ms") if bs in bs_to_row else None for bs in batch_sizes],
                "borderColor": colour,
                "backgroundColor": colour + "22",
                "tension": 0.3,
                "spanGaps": True,
            }
        )

    return {
        "batch_sizes": batch_sizes,
        "throughput_datasets": throughput_datasets,
        "latency_datasets": latency_datasets,
        "hardware": bench.get("hardware", ""),
        "run_date": bench.get("run_date", ""),
    }


def _prepare_bandit(state: dict) -> list[dict]:
    """Summarise bandit state into per-model rows for the template."""
    rows = []
    seen: set[str] = set()
    for key, entry in state.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        model_name = key.split("::")[0]
        if model_name in seen:
            continue
        seen.add(model_name)
        n_pulls: dict = entry.get("n_pulls", {})
        sum_reward: dict = entry.get("sum_reward", {})
        total = entry.get("total_pulls", 0)
        dominant = _dominant_arm(n_pulls, sum_reward)
        arms_detail = sorted(
            [
                {
                    "batch": int(arm),
                    "pulls": n,
                    "mean_reward": round(sum_reward.get(arm, 0.0) / n, 2) if n else 0.0,
                }
                for arm, n in n_pulls.items()
            ],
            key=lambda x: x["batch"],
        )
        rows.append(
            {
                "model": model_name,
                "total_pulls": total,
                "dominant_batch": int(dominant) if dominant else None,
                "arms": arms_detail,
            }
        )
    return sorted(rows, key=lambda r: r["model"])


def generate_performance_report(
    pass_stats: list[dict],
    profile_path: Path = _PROFILE_PATH,
    sched_profile_path: Path = _SCHED_PROFILE_PATH,
    sched_state_path: Path = _SCHED_STATE_PATH,
    bench_comparison_path: Path = _BENCH_COMPARISON_PATH,
    out_path: Path = OUTPUT_DIR / "performance_report.html",
) -> None:
    pipeline = _load_json(profile_path)
    sched_profile = _load_json(sched_profile_path)
    sched_state = _load_json(sched_state_path)
    bench_comparison = _load_json(bench_comparison_path)

    # Use live pass_stats when this run had active (non-skipped) passes.
    # Fall back to the stored pipeline_profile.json for warm/report-only runs
    # where all passes were skipped or nothing ran — so the report always shows
    # the last real run's data rather than an empty page.
    active_stats = [p for p in pass_stats if not p.get("skipped")]
    passes = active_stats if active_stats else [p for p in pipeline.get("passes", []) if not p.get("skipped")]

    naive_mb = _naive_peak_mb(passes)
    actual_peak_mb = max((p.get("model_memory_mb") or 0.0) for p in passes) if passes else 0.0

    cascade_passes = [p for p in passes if p.get("photos_eligible") and p["photos_eligible"] > p["photos"]]

    bandit_rows = _prepare_bandit(sched_state)

    sched_models = [
        {
            "model": name,
            "safe_batch": data["safe_batch"],
            "best_batch": data["best_batch"],
            "max_batch": data["max_batch"],
            "profiled_imgs_per_s": data["profiled_imgs_per_s"],
            "profiled_p95_ms": data["profiled_p95_ms"],
        }
        for name, data in sched_profile.get("models", {}).items()
    ]

    # Chart data — throughput (all passes with a rate)
    throughput_labels = [p["name"].split(":")[-1].strip() for p in passes if p.get("throughput_img_s")]
    throughput_values = [p["throughput_img_s"] for p in passes if p.get("throughput_img_s")]

    # Chart data — memory timeline: every pass in pipeline order, 0 for passes with no model load
    # This visualises the sequential load/unload lifecycle (the memory "pulse" pattern).
    memory_timeline_labels = [p["name"].split(":")[-1].strip() for p in passes]
    memory_timeline_values = [p.get("model_memory_mb") or 0.0 for p in passes]

    # Chart data — memory (only passes that loaded a model, for the bar chart)
    memory_labels = [p["name"].split(":")[-1].strip() for p in passes if p.get("model_memory_mb")]
    memory_values = [p["model_memory_mb"] for p in passes if p.get("model_memory_mb")]

    # Chart data — cascade filtering
    cascade_labels = [p["name"].split(":")[-1].strip() for p in cascade_passes]
    cascade_processed = [p["photos"] for p in cascade_passes]
    cascade_gated = [p["photos_eligible"] - p["photos"] for p in cascade_passes]

    # Microbenchmark batch-scaling curves (from backend_comparison.json)
    batch_curves = _prepare_batch_curves(bench_comparison) if bench_comparison else {}

    elapsed_s = pipeline.get("total_pipeline_elapsed_s", 0)
    total_for_pct = elapsed_s or 1
    data = {
        "pipeline": pipeline,
        "passes": passes,
        "run_type": pipeline.get("run_type", ""),
        "naive_peak_mb": round(naive_mb, 0),
        "actual_peak_mb": round(actual_peak_mb, 0),
        "actual_peak_gb": round(actual_peak_mb / 1024, 2),
        "naive_peak_gb": round(naive_mb / 1024, 2),
        "memory_savings_pct": round(100 * (1 - actual_peak_mb / naive_mb), 0) if naive_mb else 0,
        "total_elapsed_min": round(elapsed_s / 60, 1),
        "cascade_passes": cascade_passes,
        "sched_models": sched_models,
        "bandit_rows": bandit_rows,
        # Pass time breakdown — share of total elapsed per pass
        "time_rows": [
            {
                "label": p["name"].split(":")[-1].strip(),
                "elapsed_s": p["elapsed_s"],
                "elapsed_min": round(p["elapsed_s"] / 60, 1),
                "pct": round(p["elapsed_s"] / total_for_pct * 100, 1),
            }
            for p in passes
            if p.get("elapsed_s")
        ],
        # Horizontal bar data for throughput and memory sections
        "throughput_rows": [
            {"label": lbl, "value": v, "pct": round(v / max(throughput_values or [1]) * 100, 1)}
            for lbl, v in zip(throughput_labels, throughput_values, strict=False)
        ],
        "memory_rows": [
            {
                "label": lbl,
                "value_gb": round(v / 1024, 2),
                "pct": round(v / max(memory_values or [1]) * 100, 1),
            }
            for lbl, v in zip(memory_labels, memory_values, strict=False)
        ],
        # Memory timeline pre-converted to GB for Chart.js
        "chart_memory_timeline_labels": Markup(json.dumps(memory_timeline_labels)),
        "chart_memory_timeline_values_gb": Markup(json.dumps([round(v / 1024, 3) for v in memory_timeline_values])),
        "chart_cascade_labels": Markup(json.dumps(cascade_labels)),
        "chart_cascade_processed": Markup(json.dumps(cascade_processed)),
        "chart_cascade_gated": Markup(json.dumps(cascade_gated)),
        "batch_curves": batch_curves,
        "chart_batch_sizes": Markup(json.dumps(batch_curves.get("batch_sizes", []))),
        "chart_throughput_curve_datasets": Markup(json.dumps(batch_curves.get("throughput_datasets", []))),
        "chart_latency_curve_datasets": Markup(json.dumps(batch_curves.get("latency_datasets", []))),
    }

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    env.filters["tojson"] = lambda v: Markup(json.dumps(v))
    tmpl = env.get_template("performance_report.html.j2")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(tmpl.render(**data))

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from analysis import aggregate
from cache import load_cache, save_cache
from coach_client import aggregate_flags, calibrate_thresholds
from extractors import (
    _seed_count,
    aesthetic_regressor_available,
    auto_train_aesthetic_regressor,
    check_ood,
    classify_scene_batch,
    compute_coverage_threshold,
    empty_cache,
    encode_text_features_siglip,
    extract_aesthetic_batch,
    extract_color,
    extract_composition,
    extract_embedding_batch,
    extract_exif,
    extract_iq_batch,
    extract_jpeg_quality,
    extract_pose_batch,
    extract_saliency_batch,
    extract_vqa_batch,
    get_device,
    load_aesthetic_model,
    load_aesthetic_regressor,
    load_dino_model,
    load_iq_metric,
    load_pose_model,
    load_saliency_model,
    load_siglip_model,
    predict_aesthetic_scores,
    select_aesthetic_seed,
    train_and_save_aesthetic_regressor,
    unload_model,
    unload_pose_model,
)
from report import generate_html, generate_json
from report_performance import generate_performance_report
from scheduler import pick_batch_size, record_outcome
from scheduler.profile import get_profile
from sources import load_sources

console = Console()

CHEAP_KEYS = {"exif", "color", "composition", "jpeg_quality"}
BATCH_SIZE = 16


def _auto_batch(default: int) -> int:
    """Return a safe batch size based on available device memory."""
    try:
        import psutil

        dev = get_device()
        if dev == "cuda":
            free, _total = torch.cuda.mem_get_info()
            free_gb = free / 1e9
        elif dev == "mps":
            free_gb = psutil.virtual_memory().available / 1e9
        else:
            return default
        if free_gb >= 12:
            return min(default * 2, 32)
        if free_gb >= 6:
            return default
        return max(1, default // 2)
    except Exception:
        return default


def _download_renditions() -> bool:
    return os.environ.get("LIGHTROOM_DOWNLOAD_RENDITIONS", "false").lower() == "true"


_lr_token: str = ""
_lr_catalog_id: str = ""
_pass_stats: list[dict] = []
_cache_hits: int = 0
_cache_misses: int = 0
_PROFILE_PATH = Path("docs/pipeline_profile.json")


def _flush_pass_profile() -> None:
    """Merge the latest entry in _pass_stats into pipeline_profile.json and print timing."""
    if not _pass_stats or _pass_stats[-1].get("skipped"):
        return
    p = _pass_stats[-1]
    elapsed = p.get("elapsed_s", 0)
    tput = p.get("throughput_img_s")
    tput_str = f" · {tput:.0f} img/s" if tput else ""
    console.print(f"  [dim]Done in {elapsed:.1f}s{tput_str}[/dim]\n")

    existing: dict = {}
    if _PROFILE_PATH.exists():
        try:
            existing = json.loads(_PROFILE_PATH.read_text())
        except Exception:
            pass
    by_name = {p["name"]: p for p in existing.get("passes", [])}
    by_name[_pass_stats[-1]["name"]] = _pass_stats[-1]
    existing["passes"] = list(by_name.values())
    existing["run_date"] = datetime.now().strftime("%Y-%m-%d")
    _PROFILE_PATH.parent.mkdir(exist_ok=True)
    _PROFILE_PATH.write_text(json.dumps(existing, indent=2))


_PROFILE_FULL_PATH = Path("docs/pipeline_profile_full.json")
_PROFILE_INCREMENTAL_PATH = Path("docs/pipeline_profile_incremental.json")


def _flush_pipeline_summary(
    total_photos: int,
    cache_hits: int,
    cache_misses: int,
    total_elapsed_s: float,
    estimated_time_saved_s: float,
) -> None:
    """Write a merged end-of-run snapshot to pipeline_profile.json.

    Passes are merged by name across runs so partial runs (--only, --skip, or
    cache-update passes like --only exif) update only their passes and preserve
    timings from previous runs. The performance report therefore always shows the
    complete picture regardless of which passes ran this time.

    pipeline_profile_full.json is kept in sync with the same merged view.
    pipeline_profile_incremental.json records only what ran this time (for debugging).
    """
    active = [p for p in _pass_stats if not p.get("skipped")]
    if not active:
        return

    # Determine full vs incremental: use the heaviest ML pass (anything after Pass 1)
    # as the signal. If it processed ≥80% of total photos, this is a full run.
    ml_passes = [p for p in active if not p["name"].startswith("Pass 1")]
    ml_photos_max = max((p["photos"] for p in ml_passes), default=0)
    is_full_run = total_photos > 0 and (ml_photos_max / total_photos) >= 0.8

    # Merge current passes into the existing profile — preserves passes from prior runs
    existing_by_name: dict[str, dict] = {}
    if _PROFILE_PATH.exists():
        try:
            existing_by_name = {p["name"]: p for p in json.loads(_PROFILE_PATH.read_text()).get("passes", [])}
        except Exception:
            pass
    for p in active:
        existing_by_name[p["name"]] = p

    total = total_photos or 1
    merged_snapshot = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "run_type": "full" if is_full_run else "incremental",
        "total_photos": total_photos,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate_pct": round(100 * cache_hits / total, 1),
        "estimated_time_saved_s": round(estimated_time_saved_s, 1),
        "total_pipeline_elapsed_s": round(total_elapsed_s, 1),
        "passes": list(existing_by_name.values()),
    }
    _PROFILE_PATH.parent.mkdir(exist_ok=True)
    _PROFILE_PATH.write_text(json.dumps(merged_snapshot, indent=2))

    # Full-run reference always gets the merged view so --run-type full stays complete
    _PROFILE_FULL_PATH.write_text(json.dumps(merged_snapshot, indent=2))

    # Incremental reference records only what ran this time (debugging partial runs)
    if not is_full_run:
        partial_snapshot = {**merged_snapshot, "passes": active}
        _PROFILE_INCREMENTAL_PATH.write_text(json.dumps(partial_snapshot, indent=2))


def _memory_mb() -> float | None:
    """Current allocated device memory in MB, after syncing pending ops."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1e6
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            return torch.mps.current_allocated_memory() / 1e6
        except AttributeError:
            pass
    return None


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb() -> float | None:
    """Peak allocated device memory in MB since last _reset_peak()."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    # MPS: use current as proxy (called right after model load, before unload)
    return _memory_mb()


def _log_scheduler(model: str, batch: int) -> None:
    prof = get_profile(model)
    prof_str = f"profiled {prof.profiled_imgs_per_s:.0f} img/s · " if prof else ""
    avail_gb = psutil.virtual_memory().available / 1e9
    console.print(f"  [dim]Scheduler → {model}: batch={batch}  {prof_str}avail={avail_gb:.1f} GB[/dim]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Library Analytics")
    parser.add_argument("--sample", type=int, default=None, help="Analyze a random sample of N photos")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for YOLO pose pass (default: scheduler)")
    parser.add_argument("--prune", action="store_true", help="Remove cache entries for photos no longer in your library")
    parser.add_argument("--dry-run", action="store_true", help="With --prune: preview what would be removed without deleting.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip all ML passes — load everything from cache and regenerate the report only.",
    )
    parser.add_argument(
        "--run-type",
        choices=["latest", "full", "incremental"],
        default="latest",
        metavar="TYPE",
        help="With --report-only: which stored run to use for the performance report (latest/full/incremental). Default: latest.",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        metavar="PASSES",
        help="Comma-separated passes to skip: exif,scene,aesthetic,iq,dino,saliency,pose",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        metavar="PASSES",
        help="Run only these passes (comma-separated, same names as --skip)",
    )
    parser.add_argument(
        "--teacher",
        action="store_true",
        help=(
            "Force full aesthetic-predictor-v2-5 on all photos instead of warm-start regressor (generates fresh ground-truth labels for regressor retraining)."
        ),
    )
    parser.add_argument(
        "--clear-key",
        type=str,
        default=None,
        metavar="KEY(S)",
        help="Delete cached values for KEY (or comma-separated keys) from all records, then exit. "
        "Use 'ml' as a shorthand for all ML keys: scene,caption,aesthetic_score,iq_score,dinov3,saliency,pose_data",
    )
    parser.add_argument(
        "--lightroom-album",
        type=str,
        default=None,
        metavar="NAME",
        help="Filter Lightroom source to photos in this album name.",
    )
    parser.add_argument(
        "--lightroom-since",
        type=str,
        default=None,
        metavar="DATE",
        help="Only process Lightroom photos captured on or after DATE (ISO format: 2024-01-01).",
    )
    return parser.parse_args()


def _progress(*cols):
    return Progress(*cols, console=console)


def _prefetch_renditions(records: list[dict], max_workers: int = 16) -> None:
    """Download all missing renditions in parallel before ML passes begin."""
    if not _download_renditions():
        return
    todo = [r for r in records if not r.get("path") and r.get("source") in ("lightroom", "both")]
    if not todo:
        return

    global _lr_token, _lr_catalog_id
    if not _lr_token:
        from sources.lightroom import get_token_and_catalog

        _lr_token, _lr_catalog_id = get_token_and_catalog()

    from sources.lightroom import download_rendition

    console.print(f"[bold]Pre-fetching:[/bold] {len(todo)} renditions ([cyan]{max_workers}[/cyan] parallel workers)")

    with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
        task = p.add_task("Downloading renditions...", total=len(todo))

        def _fetch(r: dict) -> None:
            path = download_rendition(r["lightroom_id"], _lr_catalog_id, _lr_token, r["hash"])
            if path:
                r["path"] = str(path)
            p.advance(task)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_fetch, r) for r in todo]
            for _ in as_completed(futs):
                pass

    downloaded = sum(1 for r in todo if r.get("path"))
    console.print(f"  {downloaded}/{len(todo)} renditions ready.\n")


def _ensure_path(r: dict) -> Path | None:
    """Fallback single-photo download for anything missed by pre-fetch."""
    if r.get("path"):
        return Path(r["path"])
    if r.get("source") not in ("lightroom", "both"):
        return None
    if not _download_renditions():
        return None

    global _lr_token, _lr_catalog_id
    if not _lr_token:
        from sources.lightroom import get_token_and_catalog

        _lr_token, _lr_catalog_id = get_token_and_catalog()

    from sources.lightroom import download_rendition

    path = download_rendition(r["lightroom_id"], _lr_catalog_id, _lr_token, r["hash"])
    if path:
        r["path"] = str(path)
    return path


def _save_batch(batch: list[dict]) -> None:
    for r in batch:
        save_cache(r["hash"], r)


def _prune_stale(sources_str: str, raw: list[dict], dry_run: bool = False) -> None:
    from cache import prune_cache

    if dry_run:
        console.print("[bold]Prune (dry-run):[/bold] previewing — no changes will be made.")

    source_names = [s.strip().lower() for s in sources_str.split(",")]
    keep: set[str] = set()

    if "local" in source_names:
        # Local hashes come from SHA256 of actual files on disk — authoritative as-is
        keep |= {r["hash"] for r in raw if r.get("source") in ("local", "both")}

    if "lightroom" in source_names:
        # raw comes from cache, not the API — need a fresh full fetch to find deletions
        console.print("[bold]Prune:[/bold] fetching full Lightroom asset list to detect deletions...")
        global _lr_token, _lr_catalog_id
        if not _lr_token:
            from sources.lightroom import get_token_and_catalog

            _lr_token, _lr_catalog_id = get_token_and_catalog()
        from sources.lightroom import fetch_current_hashes

        keep |= fetch_current_hashes(_lr_token, _lr_catalog_id)

    if dry_run:
        from cache import load_all_cached

        stale_count = sum(1 for r in load_all_cached() if r.get("hash") not in keep)
        renditions_dir = Path("artifacts/cache/renditions")
        stale_renditions = sum(1 for f in renditions_dir.iterdir() if f.stem not in keep) if renditions_dir.exists() else 0
        console.print(
            f"  [dim]Would prune[/dim] [yellow]{stale_count}[/yellow] cache entr{'y' if stale_count == 1 else 'ies'}"
            + (f" and [yellow]{stale_renditions}[/yellow] rendition file{'s' if stale_renditions != 1 else ''}" if stale_renditions else "")
            + " [dim](dry-run — nothing deleted)[/dim]\n"
        )
        return

    removed = prune_cache(keep)

    # Clean up orphaned rendition files for pruned hashes
    renditions_dir = Path("artifacts/cache/renditions")
    removed_renditions = 0
    if renditions_dir.exists():
        for f in renditions_dir.iterdir():
            if f.stem not in keep:
                try:
                    f.unlink()
                    removed_renditions += 1
                except Exception:
                    pass

    if removed or removed_renditions:
        console.print(
            f"  Pruned [yellow]{removed}[/yellow] cache entr{'y' if removed == 1 else 'ies'}"
            + (f" and [yellow]{removed_renditions}[/yellow] rendition file{'s' if removed_renditions != 1 else ''}" if removed_renditions else "")
            + ".\n"
        )
    else:
        console.print("  Cache is up to date — nothing to prune.\n")


def main() -> None:
    load_dotenv()
    args = parse_args()
    batch_size = args.batch_size if args.batch_size is not None else pick_batch_size("yolo26n-pose")

    # ── report-only fast path: skip all source fetching ─────────────────────
    if args.report_only:
        from cache import load_all_cached

        _run_type_paths = {
            "latest": _PROFILE_PATH,
            "full": _PROFILE_FULL_PATH,
            "incremental": _PROFILE_INCREMENTAL_PATH,
        }
        perf_profile_path = _run_type_paths[args.run_type]
        if args.run_type != "latest" and not perf_profile_path.exists():
            console.print(f"[yellow]⚠ No {args.run_type} run profile found at {perf_profile_path} — falling back to latest.[/yellow]")
            perf_profile_path = _PROFILE_PATH
        run_label = f" (run-type: {args.run_type})" if args.run_type != "latest" else ""
        console.print(f"\n[bold]--report-only:[/bold] reading from local cache — no source scan{run_label}.")
        all_records = load_all_cached()
        aggregated = aggregate(all_records)
        calibrate_thresholds(all_records)
        coach_data = aggregate_flags(all_records)
        data = {"photos": all_records, "aggregated": aggregated, "coach": coach_data}
        generate_json(data)
        generate_html(data)
        generate_performance_report(_pass_stats, profile_path=perf_profile_path)
        console.print("\n[bold green]Done![/bold green]")
        console.print("  [dim]Analytics →[/dim] [cyan]docs/analytics_report.html[/cyan]")
        console.print("  [dim]Performance →[/dim] [cyan]docs/performance_report.html[/cyan]\n")
        return

    sources = os.environ.get("SOURCES", "local")
    _has_lightroom = "lightroom" in sources
    # Lightroom refreshes only on a plain run. Detect via sys.argv: any flag that
    # isn't a Lightroom-specific filter means the user wants a specific operation.
    # No need to enumerate operation flags — new flags are handled automatically.
    _LIGHTROOM_FLAGS = {"--lightroom-album", "--lightroom-since"}
    _skip_lightroom = any(a for a in sys.argv[1:] if a.startswith("-") and a not in _LIGHTROOM_FLAGS)

    def _build_records(raw: list[dict]) -> dict[str, dict]:
        global _cache_hits, _cache_misses
        result: dict[str, dict] = {}
        for r in raw:
            h = r["hash"]
            cached = load_cache(h) or {}
            if cached:
                _cache_hits += 1
            else:
                _cache_misses += 1
            merged = {**r, **{k: v for k, v in cached.items() if k not in ("path", "hash", "source")}}
            # Always use fresh lightroom_exif from the API (never from cache — it can change)
            if r.get("lightroom_exif"):
                merged["lightroom_exif"] = r["lightroom_exif"]
            # Apply XMP camera fields onto exif unconditionally — Lightroom renditions have EXIF stripped
            if merged.get("lightroom_exif") and merged.get("exif"):
                for field, val in merged["lightroom_exif"].items():
                    if val is not None:
                        merged["exif"][field] = val
            result[h] = merged
        return result

    # ── 1. First pass: local + cached Lightroom records — no API calls ────────
    console.print(f"\n[bold]Loading sources:[/bold] [cyan]{sources}[/cyan]")
    raw = load_sources(
        sample=args.sample,
        lightroom_album=args.lightroom_album,
        lightroom_since=args.lightroom_since,
        refresh_lightroom=False,
    )

    global _cache_hits, _cache_misses
    records = _build_records(raw) if raw else {}

    # ── 2. Decide whether to refresh Lightroom ────────────────────────────────
    # Always fetch if Phase 1 found nothing — cache is cold or source is Lightroom-only.
    # Otherwise only refresh on a plain run where all photos are fully cached (incremental
    # update check). Any operation flag → skip refresh, focus on the requested work.
    _ML_KEYS_CHECK = {"dinov3", "scene", "aesthetic_score", "iq_score"}
    _do_lightroom_refresh = _has_lightroom and (
        not records  # cold cache / Lightroom-only with no cached records
        or (not _skip_lightroom and _cache_misses == 0 and all(_ML_KEYS_CHECK.issubset(r.keys()) for r in records.values()))
    )

    if _do_lightroom_refresh:
        if not records:
            console.print("[dim]No cached records found — fetching from Lightroom…[/dim]")
        else:
            console.print("[dim]All photos cached — checking Lightroom for updates…[/dim]")
        _cache_hits = 0
        _cache_misses = 0
        raw = load_sources(
            sample=args.sample,
            lightroom_album=args.lightroom_album,
            lightroom_since=args.lightroom_since,
            refresh_lightroom=True,
        )
        records = _build_records(raw)
    elif _has_lightroom and _skip_lightroom:
        console.print("[dim]Skipping Lightroom refresh.[/dim]")

    if not records:
        console.print("[red]No photos found. Check SOURCES and PHOTO_DIR in .env[/red]")
        sys.exit(1)

    console.print(f"Found [green]{len(records)}[/green] photo(s).\n")

    if _has_lightroom and not _download_renditions():
        _no_path_count = sum(1 for r in records.values() if not r.get("path"))
        if _no_path_count > len(records) // 2:
            console.print(
                f"  [bold yellow]⚠ {_no_path_count}/{len(records)} Lightroom photos have no local file.[/bold yellow]\n"
                "  [dim]Set LIGHTROOM_DOWNLOAD_RENDITIONS=true in .env to download renditions for ML passes.[/dim]\n"
            )

    if args.prune:
        _prune_stale(sources, list(records.values()), dry_run=args.dry_run)

    for r in records.values():
        if r.get("source") in ("lightroom", "both") and r.get("lightroom_id"):
            save_cache(r["hash"], r)

    # ── --clear-key: wipe one or more cache keys from all records, then exit ────
    _ML_KEYS = {"scene", "caption", "aesthetic_score", "iq_score", "dinov3", "saliency", "pose_data"}
    if args.clear_key:
        raw_keys = "scene,caption,aesthetic_score,iq_score,dinov3,saliency,pose_data" if args.clear_key == "ml" else args.clear_key
        keys = {k.strip() for k in raw_keys.split(",") if k.strip()}
        cleared = 0
        for r in records.values():
            changed = False
            for key in keys:
                if key in r:
                    del r[key]
                    changed = True
            if changed:
                save_cache(r["hash"], r)
                cleared += 1
        # If aesthetic_score is cleared, delete the regressor so the next run
        # does a fresh warm-start instead of an incremental OOD pass against stale seeds.
        if "aesthetic_score" in keys:
            from extractors.heads import _AESTHETIC_REG_PATH

            if _AESTHETIC_REG_PATH.exists():
                _AESTHETIC_REG_PATH.unlink()
                console.print("  Deleted [yellow]aesthetic_regressor.joblib[/yellow] — will warm-start on next run.\n")

        label = "ml" if args.clear_key == "ml" else ", ".join(sorted(keys))
        console.print(f"Cleared [yellow]{label}[/yellow] from [green]{cleared}[/green] cache entr{'y' if cleared == 1 else 'ies'}.\n")
        return

    # ── selective-pass filtering ──────────────────────────────────────────────
    _PASS_NAMES = {"exif", "scene", "aesthetic", "iq", "dino", "saliency", "pose"}
    _skip_passes: set[str] = set()
    if args.only:
        _skip_passes = _PASS_NAMES - {p.strip() for p in args.only.split(",") if p.strip()}
    elif args.skip:
        _skip_passes = {p.strip() for p in args.skip.split(",") if p.strip()}

    # ── 1b. Pre-fetch all renditions in parallel ──────────────────────────────
    _prefetch_renditions(list(records.values()))

    def needs(key: str) -> list:
        return [r for r in records.values() if r.get(key) is None]

    def needs_path(key: str) -> list:
        todo = []
        for r in records.values():
            if r.get(key) is not None:
                continue
            if r.get("path") or (r.get("source") in ("lightroom", "both") and _download_renditions()):
                todo.append(r)
        return todo

    # ── Per-pass content gates ────────────────────────────────────────────────
    # Each gate decides, given a record, whether it needs the expensive pass.
    # Centralised here so adding a new gated pass doesn't require hunting for
    # the predicate buried inside the pass block.
    _RMBG_SUBJECT_SCENES = {"people and portraits", "animals", "food", "interior"}
    _RMBG_SCORE_THRESHOLD = 0.35  # raw cosine similarity; clear matches score 0.4–0.7

    def _gate_saliency(r: dict) -> bool:
        scores = (r.get("scene") or {}).get("scene_scores") or {}
        if any(scores.get(lbl, 0) >= _RMBG_SCORE_THRESHOLD for lbl in _RMBG_SUBJECT_SCENES):
            return True
        return (r.get("caption") or {}).get("has_person") == "yes"

    def _gate_pose(r: dict) -> bool:
        return r.get("caption", {}).get("has_person", "").startswith("yes")

    PASS_GATES: dict[str, object] = {
        "saliency": _gate_saliency,
        "pose": _gate_pose,
    }

    # ── Pass 1 — cheap extractors (parallel, single image open per photo) ─────
    todo = [
        r
        for r in records.values()
        if not CHEAP_KEYS.issubset(r.keys()) and (r.get("path") or (r.get("source") in ("lightroom", "both") and _download_renditions()))
    ]
    if "exif" in _skip_passes:
        console.print("[bold]Pass 1/6:[/bold] skipped (--skip exif)\n")
        _pass_stats.append({"name": "Pass 1: EXIF · Color · Composition · JPEG Quality", "skipped": True})
    elif todo:
        console.print(f"[bold]Pass 1/6:[/bold] EXIF · Color · Composition ({len(todo)} photos)")
        _p1_t0 = time.perf_counter()

        _failures: list[int] = []  # thread-safe via GIL; append is atomic
        _no_path: list[int] = []
        _succeeded: set[str] = set()  # hashes that completed extraction this run

        def _process_cheap(r: dict) -> None:
            img_path = _ensure_path(r)
            if not img_path:
                _no_path.append(1)
                return
            try:
                from PIL import Image

                with Image.open(img_path) as img:
                    # EXIF and JPEG quality live in the file header — read before any
                    # pixel decode so they don't pay the full-image load cost.
                    r["exif"] = extract_exif(img)
                    r["jpeg_quality"] = extract_jpeg_quality(img)
                    # Draft mode: libjpeg-turbo decodes at reduced resolution.
                    # Composition needs ≤300px, color ≤200px — capped at 300.
                    img.draft("RGB", (300, 300))
                    img.load()
                    r["color"] = extract_color(img)
                    r["composition"] = extract_composition(img)
                # Lightroom XMP fields take priority over PIL (renditions have EXIF stripped)
                if r.get("lightroom_exif"):
                    for field, val in r["lightroom_exif"].items():
                        if val is not None:
                            r["exif"][field] = val
                _succeeded.add(r["hash"])
            except Exception:
                _failures.append(1)

        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Extracting metadata...", total=len(todo))
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
                futs = {pool.submit(_process_cheap, r): r for r in todo}
                for fut in as_completed(futs):
                    r = futs[fut]
                    if r["hash"] in _succeeded:
                        save_cache(r["hash"], r)
                    p.advance(task)
        if _no_path:
            console.print(f"  [dim]{len(_no_path)} photo(s) skipped — no local file (set LIGHTROOM_DOWNLOAD_RENDITIONS=true to enable)[/dim]")
        if _failures:
            console.print(f"  [yellow]⚠ {len(_failures)} image(s) failed extraction in Pass 1[/yellow]")
        _p1_elapsed = time.perf_counter() - _p1_t0
        _pass_stats.append(
            {
                "name": "Pass 1: EXIF · Color · Composition · JPEG Quality",
                "photos": len(todo),
                "elapsed_s": round(_p1_elapsed, 2),
                "throughput_img_s": round(len(todo) / _p1_elapsed, 1) if _p1_elapsed > 0 else None,
                "load_s": None,
                "model_memory_mb": None,
            }
        )
        _flush_pass_profile()

    # ── Pass 2 — DINOv3-B embeddings (backbone for heads + clustering) ────────
    # Runs first so trained heads can be used immediately on new photos.
    dino_batch = pick_batch_size("dinov3-b")
    _log_scheduler("dinov3-b", dino_batch)
    _dino_todo = needs_path("dinov3")
    if "dino" in _skip_passes:
        console.print("[bold]Pass 2/7:[/bold] skipped (--skip dino)\n")
        _pass_stats.append({"name": "Pass 2: DINOv3-B", "skipped": True})
    elif _dino_todo:
        console.print(f"[bold]Pass 2/7:[/bold] DINOv3-B embeddings ({len(_dino_todo)} photos, batch={dino_batch})")
        _reset_peak()
        _p2_dino_t0 = time.perf_counter()
        with console.status("Loading DINOv3-B..."):
            dino_model, dino_processor, device = load_dino_model()
        _p2_dino_load_s = time.perf_counter() - _p2_dino_t0
        _p2_dino_model_mb = _peak_mb()
        if _p2_dino_model_mb:
            console.print(f"  [dim]Peak memory: {_p2_dino_model_mb:.0f} MB[/dim]")

        _dino_failed = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Extracting embeddings...", total=len(_dino_todo))
            all_dino_paths = [_ensure_path(r) for r in _dino_todo]
            embeddings = extract_embedding_batch(
                all_dino_paths,
                dino_model,
                dino_processor,
                device,
                batch_size=dino_batch,
                on_batch=lambda n: p.advance(task, n),
            )
            for r, emb in zip(_dino_todo, embeddings, strict=False):
                if emb is not None:
                    r["dinov3"] = emb
            _dino_failed = embeddings.count(None)

        if _dino_failed:
            console.print(f"  [yellow]⚠ {_dino_failed} image(s) failed DINOv3-B embedding[/yellow]")
        _save_batch(_dino_todo)
        unload_model(dino_model)
        gc.collect()
        _p2_dino_elapsed = time.perf_counter() - _p2_dino_t0
        _pass_stats.append(
            {
                "name": "Pass 2: DINOv3-B",
                "photos": len(_dino_todo),
                "elapsed_s": round(_p2_dino_elapsed, 2),
                "throughput_img_s": round(len(_dino_todo) / _p2_dino_elapsed, 1) if _p2_dino_elapsed > 0 else None,
                "load_s": round(_p2_dino_load_s, 2),
                "model_memory_mb": round(_p2_dino_model_mb, 1) if _p2_dino_model_mb else None,
            }
        )
        _n2_dino = max(1, len(_dino_todo) // dino_batch)
        record_outcome(
            "dinov3-b",
            dino_batch,
            imgs_per_s=len(_dino_todo) / _p2_dino_elapsed if _p2_dino_elapsed > 0 else 0.0,
            system_available_mb=psutil.virtual_memory().available / 1e6,
            p95_ms=_p2_dino_elapsed / _n2_dino * 1000,
            failure_rate=_dino_failed / len(_dino_todo) if _dino_todo else 0.0,
        )
        _flush_pass_profile()

    # ── Pass 3 — Scene classification + VQA ─────────────────────────────────
    todo = [r for r in records.values() if "scene" not in r or "caption" not in r]
    todo = [r for r in todo if r.get("path") or (r.get("source") in ("lightroom", "both") and _download_renditions())]
    if "scene" in _skip_passes:
        console.print("[bold]Pass 3/7:[/bold] skipped (--skip scene)\n")
        _pass_stats.append({"name": "Pass 3: scene + VQA", "skipped": True})
    elif todo:
        siglip_batch = pick_batch_size("siglip2-base")
        _log_scheduler("siglip2-base", siglip_batch)
        console.print(f"[bold]Pass 3/7:[/bold] SigLIP2-base scene + VQA ({len(todo)} photos, batch={siglip_batch})")
        device = get_device()
        _reset_peak()
        _p2_t0 = time.perf_counter()
        _siglip_failed = 0
        with console.status("Loading SigLIP2-base..."):
            siglip_model, siglip_processor = load_siglip_model(device)
            scene_feats, vqa_feats = encode_text_features_siglip(siglip_model, siglip_processor, device)
        _p2_load_s = time.perf_counter() - _p2_t0
        _p2_model_mb = _peak_mb()
        if _p2_model_mb:
            console.print(f"  [dim]Peak memory: {_p2_model_mb:.0f} MB[/dim]")

        from extractors.heads import _exif_season, _exif_time_of_day
        from extractors.prefetch import iter_prefetched

        _all_siglip_paths = [_ensure_path(r) for r in todo]
        _siglip_start = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Classifying scenes + VQA...", total=len(todo))
            for batch_paths, imgs, valid_idx in iter_prefetched(_all_siglip_paths, siglip_batch):
                batch = todo[_siglip_start : _siglip_start + len(batch_paths)]
                scene_results, img_feats, _ = classify_scene_batch(
                    batch_paths, siglip_model, siglip_processor, device, scene_feats, _preloaded=(imgs, valid_idx)
                )
                vqa_results = extract_vqa_batch(img_feats, vqa_feats) if img_feats is not None else [{} for _ in batch]
                for r, scene_res, vqa_res in zip(batch, scene_results, vqa_results, strict=False):
                    if scene_res["scene"].get("scene_scores"):
                        r["scene"] = scene_res["scene"]
                        r["caption"] = vqa_res
                        cap = r["caption"]
                        cap["time_of_day"] = _exif_time_of_day(r) or "afternoon"
                        cap["season"] = _exif_season(r) or "summer"
                    else:
                        _siglip_failed += 1
                p.advance(task, len(batch))
                _siglip_start += len(batch_paths)

        if _siglip_failed:
            console.print(f"  [yellow]⚠ {_siglip_failed} image(s) failed SigLIP[/yellow]")
        _save_batch(todo)
        del scene_feats, vqa_feats
        unload_model(siglip_model)
        gc.collect()

        _p2_elapsed = time.perf_counter() - _p2_t0
        _pass_stats.append(
            {
                "name": "Pass 3: SigLIP2-base",
                "photos": len(todo),
                "elapsed_s": round(_p2_elapsed, 2),
                "throughput_img_s": round(len(todo) / _p2_elapsed, 1) if _p2_elapsed > 0 else None,
                "load_s": round(_p2_load_s, 2),
                "model_memory_mb": round(_p2_model_mb, 1) if _p2_model_mb else None,
            }
        )
        _n2 = max(1, len(todo) // siglip_batch)
        record_outcome(
            "siglip2-base",
            siglip_batch,
            imgs_per_s=len(todo) / _p2_elapsed if _p2_elapsed > 0 else 0.0,
            system_available_mb=psutil.virtual_memory().available / 1e6,
            p95_ms=_p2_elapsed / _n2 * 1000,
            failure_rate=_siglip_failed / len(todo) if todo else 0.0,
        )
        _flush_pass_profile()

    # ── Pass 4a — aesthetic score (warm-start regressor or full SigLIP) ────────
    aes_batch = pick_batch_size("aesthetic-predictor-v2-5")
    _log_scheduler("aesthetic-predictor-v2-5", aes_batch)
    aes_todo_all = needs_path("aesthetic_score")
    # Only photos with a dinov3 embedding can use the regressor / warm-start.
    aes_todo = [r for r in aes_todo_all if r.get("dinov3")]

    if "aesthetic" in _skip_passes:
        console.print("[bold]Pass 4a/7:[/bold] skipped (--skip aesthetic)\n")
        _pass_stats.append({"name": "Pass 4a: aesthetic-predictor-v2-5", "skipped": True})

    elif aes_todo_all:
        _p3a_t0 = time.perf_counter()
        _p3a_load_s: float = 0.0
        _p3a_model_mb: float | None = None
        _p3a_n_siglip: int = 0
        _p3a_n_regressor: int = 0
        _p3a_pass_name = "Pass 4a: aesthetic-predictor-v2-5"
        _aes_ran_siglip = False

        # ── Inline helper: run aesthetic-predictor-v2-5 on a list of records ──
        def _run_siglip_aesthetic(todo: list[dict], label: str) -> int:
            """Score `todo` via the aesthetic predictor. Returns failure count."""
            nonlocal _p3a_load_s, _p3a_model_mb
            _reset_peak()
            _t_load = time.perf_counter()
            with console.status("Loading aesthetic-predictor-v2-5..."):
                aesthetic = load_aesthetic_model(get_device())
            _p3a_load_s += time.perf_counter() - _t_load
            _p3a_model_mb = _peak_mb()
            if _p3a_model_mb:
                console.print(f"  [dim]Peak memory: {_p3a_model_mb:.0f} MB[/dim]")
            failed = 0
            with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
                task = p.add_task(label, total=len(todo))
                for start in range(0, len(todo), aes_batch):
                    chunk = todo[start : start + aes_batch]
                    paths = [_ensure_path(r) for r in chunk]
                    scores = extract_aesthetic_batch(paths, aesthetic, aes_batch)
                    for r, score in zip(chunk, scores, strict=False):
                        if score is not None:
                            r["aesthetic_score"] = score
                            r["aesthetic_score_source"] = "siglip"
                    failed += scores.count(None)
                    p.advance(task, len(chunk))
            if failed:
                console.print(f"  [yellow]⚠ {failed} image(s) failed aesthetic scoring[/yellow]")
            del aesthetic
            gc.collect()
            empty_cache()
            return failed

        _use_teacher = getattr(args, "teacher", False)
        _reg_exists = aesthetic_regressor_available()
        _reg = load_aesthetic_regressor() if _reg_exists else None
        _reg_has_seed = _reg is not None and "X_seed" in _reg

        # ── Case 0: --teacher → full SigLIP on everything, no regressor ──────
        if _use_teacher:
            console.print(f"[bold]Pass 4a/7:[/bold] Aesthetic (teacher/full SigLIP) — {len(aes_todo_all)} photos")
            _failed = _run_siglip_aesthetic(aes_todo_all, "Scoring aesthetics (teacher)...")
            _save_batch(aes_todo_all)
            _p3a_n_siglip = len(aes_todo_all)
            _p3a_pass_name = "Pass 4a: aesthetic-predictor-v2-5 (teacher)"
            _aes_ran_siglip = True

        # ── Case 1: no regressor on disk → warm-start ─────────────────────────
        elif not _reg_exists and len(aes_todo) >= 2:
            K = _seed_count(len(aes_todo))
            console.print(
                f"[bold]Pass 4a/7:[/bold] Aesthetic warm-start — {len(aes_todo)} photos  [dim](no regressor — selecting {K} seed photos via k-means)[/dim]"
            )

            with console.status(f"Clustering {len(aes_todo)} embeddings → {K} seeds..."):
                seed_records, X_seed = select_aesthetic_seed(aes_todo)
            console.print(f"  Seed: [cyan]{len(seed_records)}[/cyan] photos (k-means K={K})")

            _failed = _run_siglip_aesthetic(seed_records, f"Scoring {len(seed_records)} seed photos (SigLIP)...")
            _p3a_n_siglip = len(seed_records)

            with console.status(f"Training regressor on {len(seed_records)} seed labels..."):
                X_all = np.array([r["dinov3"] for r in aes_todo], dtype=np.float32)
                threshold = compute_coverage_threshold(X_all, X_seed)
                seed_records_ok = [r for r in seed_records if r.get("aesthetic_score") is not None]
                X_seed_ok = np.array([r["dinov3"] for r in seed_records_ok], dtype=np.float32)
                y_seed_ok = np.array([r["aesthetic_score"] for r in seed_records_ok], dtype=np.float32)
                train_and_save_aesthetic_regressor(X_seed_ok, y_seed_ok, threshold)

            _reg_new = load_aesthetic_regressor()
            non_seed = [r for r in aes_todo if r.get("aesthetic_score") is None]
            if non_seed:
                console.print(f"  Predicting [cyan]{len(non_seed)}[/cyan] remaining photos via regressor...")
                preds = predict_aesthetic_scores(non_seed, _reg_new)
                for r, s in zip(non_seed, preds, strict=False):
                    r["aesthetic_score"] = s
                    r["aesthetic_score_source"] = "regressor"
            _p3a_n_regressor = len(non_seed)
            _save_batch(aes_todo)
            saved_pct = 100 * _p3a_n_regressor / len(aes_todo) if aes_todo else 0
            console.print(f"  [green]✓[/green] {_p3a_n_siglip} SigLIP + {_p3a_n_regressor} regressor ([green]{saved_pct:.0f}%[/green] of SigLIP calls saved)")
            _p3a_pass_name = f"Pass 4a: aesthetic-predictor-v2-5 (K={len(seed_records)})"
            _aes_ran_siglip = True

        # ── Case 2: regressor exists with X_seed → OOD-based incremental ──────
        elif _reg_has_seed and aes_todo:
            console.print(f"[bold]Pass 4a/7:[/bold] Aesthetic incremental — {len(aes_todo)} new photos")

            with console.status("Checking coverage (OOD detection)..."):
                X_query = np.array([r["dinov3"] for r in aes_todo], dtype=np.float32)
                ood_mask = check_ood(X_query, _reg["X_seed"], _reg["coverage_threshold"])
            n_ind = int((~ood_mask).sum())
            n_ood = int(ood_mask.sum())
            console.print(f"  In-distribution: [green]{n_ind}[/green] → regressor (0 SigLIP)  |  OOD: [yellow]{n_ood}[/yellow] → new seeds")

            # In-distribution: predict with existing regressor
            ind_records = [r for r, o in zip(aes_todo, ood_mask, strict=False) if not o]
            if ind_records:
                preds = predict_aesthetic_scores(ind_records, _reg)
                for r, s in zip(ind_records, preds, strict=False):
                    r["aesthetic_score"] = s
                    r["aesthetic_score_source"] = "regressor"
            _p3a_n_regressor += len(ind_records)

            if n_ood > 0:
                ood_records = [r for r, o in zip(aes_todo, ood_mask, strict=False) if o]
                K_new = _seed_count(n_ood)
                with console.status(f"Clustering {n_ood} OOD embeddings → {K_new} seeds..."):
                    seed_ood, _X_seed_new = select_aesthetic_seed(ood_records)
                console.print(f"  OOD seed: [cyan]{len(seed_ood)}[/cyan] photos (k-means K={K_new})")

                _failed = _run_siglip_aesthetic(seed_ood, f"Scoring {len(seed_ood)} OOD seeds (SigLIP)...")
                _p3a_n_siglip = len(seed_ood)
                _aes_ran_siglip = True

                seed_ood_ok = [r for r in seed_ood if r.get("aesthetic_score") is not None]
                X_seed_new_ok = np.array([r["dinov3"] for r in seed_ood_ok], dtype=np.float32)
                y_seed_new = np.array([r["aesthetic_score"] for r in seed_ood_ok], dtype=np.float32)

                X_seed_combined = np.vstack([_reg["X_seed"], X_seed_new_ok])
                y_seed_combined = np.concatenate([_reg["y_seed"], y_seed_new])

                with console.status(f"Retraining on {len(y_seed_combined)} seeds (old + new)..."):
                    X_full = np.array(
                        [r["dinov3"] for r in records.values() if r.get("dinov3")],
                        dtype=np.float32,
                    )
                    new_threshold = compute_coverage_threshold(X_full, X_seed_combined)
                    train_and_save_aesthetic_regressor(X_seed_combined, y_seed_combined, new_threshold)

                _reg_new = load_aesthetic_regressor()

                # Re-predict all previously regressor-scored photos with improved model
                old_reg_records = [r for r in records.values() if r.get("aesthetic_score_source") == "regressor"]
                if old_reg_records:
                    console.print(f"  Re-predicting [cyan]{len(old_reg_records)}[/cyan] existing regressor-scored photos with improved model...")
                    old_preds = predict_aesthetic_scores(old_reg_records, _reg_new)
                    for r, s in zip(old_reg_records, old_preds, strict=False):
                        r["aesthetic_score"] = s
                    _save_batch(old_reg_records)

                # Predict remaining OOD non-seed photos
                ood_non_seed = [r for r in ood_records if r.get("aesthetic_score") is None]
                if ood_non_seed:
                    preds_ood = predict_aesthetic_scores(ood_non_seed, _reg_new)
                    for r, s in zip(ood_non_seed, preds_ood, strict=False):
                        r["aesthetic_score"] = s
                        r["aesthetic_score_source"] = "regressor"
                _p3a_n_regressor += len(ood_non_seed)

            _save_batch(aes_todo)
            console.print(f"  [green]✓[/green] {n_ind} in-dist regressor + {_p3a_n_siglip} OOD SigLIP seeds + {_p3a_n_regressor - n_ind} OOD regressor")
            _p3a_pass_name = "Pass 4a: aesthetic-predictor-v2-5 (OOD)"

        # ── Case 3: regressor exists but old format (no X_seed) → full SigLIP ─
        elif _reg_exists and not _reg_has_seed and aes_todo_all:
            console.print(f"[bold]Pass 4a/7:[/bold] Aesthetic — {len(aes_todo_all)} photos [dim](regressor has no seed index — running full SigLIP)[/dim]")
            _failed = _run_siglip_aesthetic(aes_todo_all, "Scoring aesthetics...")
            _save_batch(aes_todo_all)
            _p3a_n_siglip = len(aes_todo_all)
            _p3a_pass_name = "Pass 4a: aesthetic-predictor-v2-5"
            _aes_ran_siglip = True

        # ── Case 4: no regressor and too few photos for warm-start → full SigLIP
        elif not _reg_exists:
            console.print(f"[bold]Pass 4a/7:[/bold] Aesthetic — {len(aes_todo_all)} photos [dim](too few for warm-start — running full SigLIP)[/dim]")
            _failed = _run_siglip_aesthetic(aes_todo_all, "Scoring aesthetics...")
            _save_batch(aes_todo_all)
            _p3a_n_siglip = len(aes_todo_all)
            _p3a_pass_name = "Pass 4a: aesthetic-predictor-v2-5"
            _aes_ran_siglip = True

        _p3a_elapsed = time.perf_counter() - _p3a_t0
        _p3a_total = _p3a_n_siglip + _p3a_n_regressor
        _pass_stats.append(
            {
                "name": _p3a_pass_name,
                "photos": _p3a_total or len(aes_todo_all),
                "photos_eligible": len(aes_todo_all),
                "photos_siglip": _p3a_n_siglip,
                "photos_regressor": _p3a_n_regressor,
                "elapsed_s": round(_p3a_elapsed, 2),
                "throughput_img_s": round(_p3a_total / _p3a_elapsed, 1) if _p3a_elapsed > 0 and _p3a_total else None,
                "load_s": round(_p3a_load_s, 2) if _p3a_load_s else None,
                "model_memory_mb": round(_p3a_model_mb, 1) if _p3a_model_mb else None,
            }
        )
        if _p3a_n_siglip > 0:
            _n3a = max(1, _p3a_n_siglip // aes_batch)
            record_outcome(
                "aesthetic-predictor-v2-5",
                aes_batch,
                imgs_per_s=_p3a_n_siglip / _p3a_elapsed if _p3a_elapsed > 0 else 0.0,
                system_available_mb=psutil.virtual_memory().available / 1e6,
                p95_ms=_p3a_elapsed / _n3a * 1000,
                failure_rate=0.0,
            )
        _flush_pass_profile()

        # Retrain on all accumulated SigLIP-labelled records to improve future runs
        if _aes_ran_siglip:
            with console.status("Retraining regressor on all accumulated SigLIP labels..."):
                _reg_trained = auto_train_aesthetic_regressor(list(records.values()))
            if _reg_trained:
                import json as _json

                from extractors.heads import _AESTHETIC_REG_PATH

                _reg_meta = _json.loads((_AESTHETIC_REG_PATH.parent / "aesthetic_regressor_meta.json").read_text())
                console.print(f"  [green]✓ Regressor updated[/green]  n={_reg_meta['n_samples']}  R²={_reg_meta['cv_r2']}  MAE={_reg_meta['cv_mae']}")
        console.print()

    # ── Pass 4b — ARNIQA: technical IQ score (MPS/CUDA/CPU) ─────────────────
    iq_batch = pick_batch_size("arniqa")
    _log_scheduler("arniqa", iq_batch)
    iq_todo = needs_path("iq_score")
    if "iq" in _skip_passes or os.environ.get("SKIP_IQ", "false").lower() == "true":
        console.print("[bold]Pass 4b/7:[/bold] ARNIQA skipped\n")
        _pass_stats.append({"name": "Pass 4b: ARNIQA", "skipped": True})
    elif iq_todo:
        device = get_device()
        console.print(f"[bold]Pass 4b/7:[/bold] ARNIQA technical quality ({len(iq_todo)} photos, batch={iq_batch}, {device})")
        _reset_peak()
        _p3b_t0 = time.perf_counter()
        with console.status("Loading ARNIQA (pyiqa)..."):
            iq_metric = load_iq_metric(device)
        _p3b_load_s = time.perf_counter() - _p3b_t0
        _p3b_model_mb = _peak_mb()
        if _p3b_model_mb:
            console.print(f"  [dim]Peak memory: {_p3b_model_mb:.0f} MB[/dim]")

        if isinstance(iq_metric, Exception):
            console.print(f"  [yellow]ARNIQA failed to load: {iq_metric}. Skipping.[/yellow]\n")
            iq_metric = None
        if iq_metric is not None:
            _iq_failed = 0
            with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
                task = p.add_task("Scoring technical quality...", total=len(iq_todo))
                all_iq_paths = [_ensure_path(r) for r in iq_todo]
                iq_scores = extract_iq_batch(
                    all_iq_paths,
                    iq_metric,
                    batch_size=iq_batch,
                    on_batch=lambda n: p.advance(task, n),
                )
                for r, iq in zip(iq_todo, iq_scores, strict=False):
                    r["iq_score"] = iq
                _iq_failed = iq_scores.count(None)
            if _iq_failed:
                console.print(f"  [yellow]⚠ {_iq_failed} image(s) failed IQ scoring[/yellow]")
            _save_batch(iq_todo)
            del iq_metric
            gc.collect()
            empty_cache()
            _p3b_elapsed = time.perf_counter() - _p3b_t0
            _pass_stats.append(
                {
                    "name": "Pass 4b: ARNIQA",
                    "photos": len(iq_todo),
                    "elapsed_s": round(_p3b_elapsed, 2),
                    "throughput_img_s": round(len(iq_todo) / _p3b_elapsed, 1) if _p3b_elapsed > 0 else None,
                    "load_s": round(_p3b_load_s, 2),
                    "model_memory_mb": round(_p3b_model_mb, 1) if _p3b_model_mb else None,
                }
            )
            _n3b = max(1, len(iq_todo) // iq_batch)
            record_outcome(
                "arniqa",
                iq_batch,
                imgs_per_s=len(iq_todo) / _p3b_elapsed if _p3b_elapsed > 0 else 0.0,
                system_available_mb=psutil.virtual_memory().available / 1e6,
                p95_ms=_p3b_elapsed / _n3b * 1000,
                failure_rate=_iq_failed / len(iq_todo) if iq_todo else 0.0,
            )
            _flush_pass_profile()

    # ── Pass 5 — RMBG 2.0 saliency (batched) ──────────────────────────────────
    sal_batch = pick_batch_size("RMBG-2.0")
    _log_scheduler("RMBG-2.0", sal_batch)
    _saliency_candidates = needs_path("saliency")
    todo = [r for r in _saliency_candidates if PASS_GATES["saliency"](r)]
    if "saliency" in _skip_passes:
        console.print("[bold]Pass 5/7:[/bold] skipped (--skip saliency)\n")
        _pass_stats.append({"name": "Pass 5: RMBG-2.0", "skipped": True})
    elif todo:
        console.print(f"[bold]Pass 5/7:[/bold] Saliency ({len(todo)}/{len(_saliency_candidates)} photos with subject signal, batch={sal_batch})")
        _reset_peak()
        _p5_t0 = time.perf_counter()
        with console.status("Loading RMBG-2.0 (briaai/RMBG-2.0)..."):
            saliency_pipe = load_saliency_model(get_device())
        _p5_load_s = time.perf_counter() - _p5_t0
        _p5_model_mb = _peak_mb()
        if _p5_model_mb:
            console.print(f"  [dim]Peak memory: {_p5_model_mb:.0f} MB[/dim]")

        _sal_failed = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Detecting subjects...", total=len(todo))
            for start in range(0, len(todo), sal_batch):
                batch = todo[start : start + sal_batch]
                paths = [_ensure_path(r) for r in batch]
                sal_results = extract_saliency_batch(paths, saliency_pipe, sal_batch)
                for r, sal in zip(batch, sal_results, strict=False):
                    if sal.get("subject_area_pct") is not None:
                        r["saliency"] = sal
                    else:
                        _sal_failed += 1
                p.advance(task, len(batch))

        if _sal_failed:
            console.print(f"  [yellow]⚠ {_sal_failed} image(s) failed saliency extraction[/yellow]")
        _save_batch(todo)
        del saliency_pipe
        gc.collect()
        empty_cache()
        _p5_elapsed = time.perf_counter() - _p5_t0
        _pass_stats.append(
            {
                "name": "Pass 5: RMBG-2.0",
                "photos": len(todo),
                "photos_eligible": len(_saliency_candidates),
                "elapsed_s": round(_p5_elapsed, 2),
                "throughput_img_s": round(len(todo) / _p5_elapsed, 1) if _p5_elapsed > 0 else None,
                "load_s": round(_p5_load_s, 2),
                "model_memory_mb": round(_p5_model_mb, 1) if _p5_model_mb else None,
            }
        )
        _n5 = max(1, len(todo) // sal_batch)
        record_outcome(
            "RMBG-2.0",
            sal_batch,
            imgs_per_s=len(todo) / _p5_elapsed if _p5_elapsed > 0 else 0.0,
            system_available_mb=psutil.virtual_memory().available / 1e6,
            p95_ms=_p5_elapsed / _n5 * 1000,
            failure_rate=_sal_failed / len(todo) if todo else 0.0,
        )
        _flush_pass_profile()

    # ── Pass 6 — YOLO11-Pose: object detection + pose (portrait photos) ────────
    _pose_eligible = needs_path("pose_data")
    todo = [r for r in _pose_eligible if PASS_GATES["pose"](r)]
    if "pose" in _skip_passes:
        console.print("[bold]Pass 6/7:[/bold] skipped (--skip pose)\n")
        _pass_stats.append({"name": "Pass 6: YOLO26n-pose", "skipped": True})
    elif todo:
        console.print(f"[bold]Pass 6/7:[/bold] YOLO11-Pose object detection ({len(todo)} portrait photos)")
        device = get_device()
        _reset_peak()
        _p6_t0 = time.perf_counter()
        with console.status("Loading YOLO11n-pose..."):
            pose_model = load_pose_model(device)
        _p6_load_s = time.perf_counter() - _p6_t0
        _p6_model_mb = _peak_mb()
        if _p6_model_mb:
            console.print(f"  [dim]Peak memory: {_p6_model_mb:.0f} MB[/dim]")

        _pose_failed = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Detecting objects + pose...", total=len(todo))
            all_pose_paths = [_ensure_path(r) for r in todo]

            def _pose_on_batch(n: int) -> None:
                nonlocal _p6_model_mb
                # YOLO loads on CPU at model-load time; weights move to device on first inference.
                # Capture the actual on-device footprint after the first batch completes.
                if not _p6_model_mb:
                    _p6_model_mb = _memory_mb()
                p.advance(task, n)

            pose_results = extract_pose_batch(
                all_pose_paths,
                pose_model,
                device,
                batch_size=batch_size,
                on_batch=_pose_on_batch,
            )
            for r, pr in zip(todo, pose_results, strict=False):
                if pr:
                    r["pose_data"] = pr
                else:
                    _pose_failed += 1

        if _pose_failed:
            console.print(f"  [yellow]⚠ {_pose_failed} image(s) failed pose extraction[/yellow]")
        _save_batch(todo)
        unload_pose_model(pose_model)
        gc.collect()
        _p6_elapsed = time.perf_counter() - _p6_t0
        _pass_stats.append(
            {
                "name": "Pass 6: YOLO26n-pose",
                "photos": len(todo),
                "photos_eligible": len(_pose_eligible),
                "elapsed_s": round(_p6_elapsed, 2),
                "throughput_img_s": round(len(todo) / _p6_elapsed, 1) if _p6_elapsed > 0 else None,
                "load_s": round(_p6_load_s, 2),
                "model_memory_mb": round(_p6_model_mb, 1) if _p6_model_mb else None,
            }
        )
        _n6 = max(1, len(todo) // batch_size)
        record_outcome(
            "yolo26n-pose",
            batch_size,
            imgs_per_s=len(todo) / _p6_elapsed if _p6_elapsed > 0 else 0.0,
            system_available_mb=psutil.virtual_memory().available / 1e6,
            p95_ms=_p6_elapsed / _n6 * 1000,
            failure_rate=_pose_failed / len(todo) if todo else 0.0,
        )
        _flush_pass_profile()

    all_records = list(records.values())
    _ml_complete = sum(1 for r in all_records if r.get("scene") is not None)
    _no_file = sum(1 for r in all_records if not r.get("path"))
    _summary = f"[green]{len(all_records)} photo(s)[/green] · ML complete: [green]{_ml_complete}[/green]"
    if _no_file:
        _summary += f" · No local file/rendition: [yellow]{_no_file}[/yellow]"
    console.print(_summary + "\n")

    console.print("[bold]Running quality issue detection...[/bold]")
    calibrate_thresholds(all_records)

    console.print("[bold]Computing aggregated statistics...[/bold]")
    aggregated = aggregate(all_records)
    coach_data = aggregate_flags(all_records)
    console.print()

    data = {"photos": all_records, "aggregated": aggregated, "coach": coach_data}
    generate_json(data)
    generate_html(data)
    # ── Pipeline profile summary ──────────────────────────────────────────────
    if _pass_stats:
        _total_photos = len(all_records)
        _hit_rate = 100 * _cache_hits / _total_photos if _total_photos else 0.0
        _active = [p for p in _pass_stats if not p.get("skipped")]
        _total_elapsed = sum(p["elapsed_s"] for p in _active)
        # Estimate time saved: avg per-photo time × cache hits
        _avg_s_per_photo = _total_elapsed / max(sum(p["photos"] for p in _active), 1)
        _saved_s = _avg_s_per_photo * _cache_hits

        console.print("\n[bold]── Pipeline Profile ─────────────────────────────────────────────────[/bold]")
        t = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        t.add_column("Pass", style="cyan")
        t.add_column("Photos", justify="right")
        t.add_column("Filtered", justify="right")
        t.add_column("Time", justify="right")
        t.add_column("img/s", justify="right")
        t.add_column("Load", justify="right")
        t.add_column("Peak mem", justify="right")
        for p in _pass_stats:
            if p.get("skipped"):
                t.add_row(p["name"], "–", "–", "skipped", "–", "–", "–")
            else:
                eligible = p.get("photos_eligible")
                processed = p["photos"]
                if eligible and eligible > processed:
                    filtered_str = f"{processed}/{eligible}"
                else:
                    filtered_str = "–"
                t.add_row(
                    p["name"],
                    str(processed),
                    filtered_str,
                    f"{p['elapsed_s']:.1f}s",
                    f"{p['throughput_img_s']:.1f}" if p.get("throughput_img_s") else "–",
                    f"{p['load_s']:.1f}s" if p.get("load_s") else "–",
                    f"{p['model_memory_mb']:.0f} MB" if p.get("model_memory_mb") else "–",
                )
        console.print(t)
        console.print(
            f"  Total: [green]{_total_elapsed:.1f}s[/green]  ·  "
            f"Cache: [green]{_cache_hits}[/green] hits / [yellow]{_cache_misses}[/yellow] misses "
            f"([green]{_hit_rate:.1f}%[/green])  ·  "
            f"~[green]{_saved_s:.0f}s[/green] saved by cache\n"
        )
        _flush_pipeline_summary(_total_photos, _cache_hits, _cache_misses, _total_elapsed, _saved_s)

    generate_performance_report(_pass_stats)

    console.print("\n[bold green]Done![/bold green]")
    console.print("  [dim]JSON →[/dim] [cyan]docs/results.json[/cyan]")
    console.print("  [dim]Analytics →[/dim] [cyan]docs/analytics_report.html[/cyan]")
    console.print("  [dim]Performance →[/dim] [cyan]docs/performance_report.html[/cyan]\n")


if __name__ == "__main__":
    main()

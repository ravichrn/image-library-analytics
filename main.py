import argparse
import gc
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from analysis import aggregate
from cache import load_cache, save_cache
from coach_client import aggregate_flags
from extractors import (
    classify_scene_batch,
    encode_scene_labels_siglip,
    encode_vqa_labels_siglip,
    extract_aesthetic_batch,
    extract_color,
    extract_composition,
    extract_ela,
    extract_embedding_batch,
    extract_exif,
    extract_iq_batch,
    extract_pose_batch,
    extract_saliency_batch,
    extract_vqa_batch,
    load_aesthetic_model,
    load_clipiqa_metric,
    load_dino_model,
    load_pose_model,
    load_saliency_model,
    load_siglip_model,
    unload_model,
    unload_pose_model,
)
from report import generate_html, generate_json
from sources import load_sources

console = Console()

CHEAP_KEYS = {"exif", "color", "composition", "ela"}
BATCH_SIZE = 16


def _auto_batch(default: int) -> int:
    """Return a safe batch size based on available device memory."""
    try:
        import psutil

        if torch.backends.mps.is_available():
            free_gb = psutil.virtual_memory().available / 1e9
        elif torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            free_gb = free / 1e9
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


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Library Analytics")
    parser.add_argument("--sample", type=int, default=None, help="Analyze a random sample of N photos")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for ML passes (default 16)")
    parser.add_argument("--prune", action="store_true", help="Remove cache entries for photos no longer in your library")
    parser.add_argument("--dry-run", action="store_true", help="With --prune: preview what would be removed without deleting.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip all ML passes — load everything from cache and regenerate the report only.",
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
        "--clear-key",
        type=str,
        default=None,
        metavar="KEY",
        help="Delete cached values for KEY (e.g. saliency, iq_score) from all records, then exit.",
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
        renditions_dir = Path("cache/renditions")
        stale_renditions = sum(1 for f in renditions_dir.iterdir() if f.stem not in keep) if renditions_dir.exists() else 0
        console.print(
            f"  [dim]Would prune[/dim] [yellow]{stale_count}[/yellow] cache entr{'y' if stale_count == 1 else 'ies'}"
            + (f" and [yellow]{stale_renditions}[/yellow] rendition file{'s' if stale_renditions != 1 else ''}" if stale_renditions else "")
            + " [dim](dry-run — nothing deleted)[/dim]\n"
        )
        return

    removed = prune_cache(keep)

    # Clean up orphaned rendition files for pruned hashes
    renditions_dir = Path("cache/renditions")
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
    batch_size = args.batch_size

    sources = os.environ.get("SOURCES", "local")
    console.print(f"\n[bold]Loading sources:[/bold] [cyan]{sources}[/cyan]")

    # ── 1. Fetch metadata from all sources ───────────────────────────────────
    raw = load_sources(sample=args.sample, lightroom_album=args.lightroom_album, lightroom_since=args.lightroom_since)
    if not raw:
        console.print("[red]No photos found. Check SOURCES and PHOTO_DIR in .env[/red]")
        sys.exit(1)
    console.print(f"Found [green]{len(raw)}[/green] photo(s).\n")

    if "lightroom" in sources and not _download_renditions():
        _no_path_count = sum(1 for r in raw if not r.get("path"))
        if _no_path_count > len(raw) // 2:
            console.print(
                f"  [bold yellow]⚠ {_no_path_count}/{len(raw)} Lightroom photos have no local file.[/bold yellow]\n"
                "  [dim]Set LIGHTROOM_DOWNLOAD_RENDITIONS=true in .env to download renditions for ML passes.[/dim]\n"
            )

    if args.prune:
        _prune_stale(sources, raw, dry_run=args.dry_run)

    records: dict[str, dict] = {}
    for r in raw:
        h = r["hash"]
        cached = load_cache(h) or {}
        merged = {**r, **{k: v for k, v in cached.items() if k not in ("path", "hash", "source")}}
        # Always use fresh lightroom_exif from the API (never from cache — it can change)
        if r.get("lightroom_exif"):
            merged["lightroom_exif"] = r["lightroom_exif"]
        # Apply XMP camera fields onto exif unconditionally — Lightroom renditions have EXIF stripped
        if merged.get("lightroom_exif") and merged.get("exif"):
            for field, val in merged["lightroom_exif"].items():
                if val is not None:
                    merged["exif"][field] = val
        records[h] = merged

    for r in records.values():
        if r.get("source") in ("lightroom", "both") and r.get("lightroom_id"):
            save_cache(r["hash"], r)

    # ── --clear-key: wipe one cache key from all records, then exit ───────────
    if args.clear_key:
        from cache import load_all_cached

        key = args.clear_key
        cleared = 0
        for r in load_all_cached():
            if key in r:
                del r[key]
                save_cache(r["hash"], r)
                cleared += 1
        console.print(f"Cleared [yellow]{key}[/yellow] from [green]{cleared}[/green] cache entr{'y' if cleared == 1 else 'ies'}.\n")
        return

    # ── selective-pass filtering ──────────────────────────────────────────────
    _PASS_NAMES = {"exif", "scene", "aesthetic", "iq", "dino", "saliency", "pose"}
    _skip_passes: set[str] = set()
    if args.only:
        _skip_passes = _PASS_NAMES - {p.strip() for p in args.only.split(",") if p.strip()}
    elif args.skip:
        _skip_passes = {p.strip() for p in args.skip.split(",") if p.strip()}

    # ── report-only shortcut ──────────────────────────────────────────────────
    if args.report_only:
        console.print("[bold]--report-only:[/bold] skipping all ML passes, regenerating report from cache.")
        all_records = list(records.values())
        aggregated = aggregate(all_records)
        coach_data = aggregate_flags(all_records)
        data = {"photos": all_records, "aggregated": aggregated, "coach": coach_data}
        generate_json(data)
        generate_html(data)
        console.print("\n[bold green]Done![/bold green]")
        console.print("  [dim]HTML →[/dim] [cyan]docs/report.html[/cyan]  ← open in your browser\n")
        return

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

    # ── Pass 1 — cheap extractors (parallel, single image open per photo) ─────
    todo = [
        r
        for r in records.values()
        if not CHEAP_KEYS.issubset(r.keys()) and (r.get("path") or (r.get("source") in ("lightroom", "both") and _download_renditions()))
    ]
    if "exif" in _skip_passes:
        console.print("[bold]Pass 1/6:[/bold] skipped (--skip exif)\n")
    elif todo:
        console.print(f"[bold]Pass 1/6:[/bold] EXIF · Color · Composition ({len(todo)} photos)")

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
                    img.load()
                    r["exif"] = extract_exif(img)
                    r["color"] = extract_color(img)
                    r["composition"] = extract_composition(img)
                    r["ela"] = extract_ela(img_path)
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
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 6)) as pool:
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
        console.print()

    # ── Pass 2 — SigLIP 2: scene classification + zero-shot VQA ─────────────────
    # Image features computed once per batch, reused for both scene and VQA.
    # SO400M is large — cap batch at 4 to avoid MPS memory pressure.
    siglip_batch = _auto_batch(4)
    todo = [r for r in records.values() if "scene" not in r or "caption" not in r]
    todo = [r for r in todo if r.get("path") or (r.get("source") in ("lightroom", "both") and _download_renditions())]
    if "scene" in _skip_passes:
        console.print("[bold]Pass 2/6:[/bold] skipped (--skip scene)\n")
    elif todo:
        console.print(f"[bold]Pass 2/6:[/bold] SigLIP 2 scene + VQA ({len(todo)} photos, batch={siglip_batch})")
        device = _device()
        with console.status("Loading SigLIP 2 SO400M..."):
            siglip_model, siglip_processor = load_siglip_model(device)
            scene_feats = encode_scene_labels_siglip(siglip_model, siglip_processor, device)
            vqa_feats = encode_vqa_labels_siglip(siglip_model, siglip_processor, device)

        _siglip_failed = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Classifying scenes + VQA...", total=len(todo))
            for start in range(0, len(todo), siglip_batch):
                batch = todo[start : start + siglip_batch]
                paths = [_ensure_path(r) for r in batch]
                scene_results, img_feats = classify_scene_batch(paths, siglip_model, siglip_processor, device, scene_feats)
                vqa_results = extract_vqa_batch(img_feats, vqa_feats) if img_feats is not None else [{} for _ in batch]
                for r, scene_res, vqa_res in zip(batch, scene_results, vqa_results, strict=False):
                    if scene_res["scene"].get("scene_scores"):
                        if "scene" not in r:
                            r["scene"] = scene_res["scene"]
                        if "caption" not in r and vqa_res:
                            r["caption"] = vqa_res
                    else:
                        _siglip_failed += 1
                p.advance(task, len(batch))

        if _siglip_failed:
            console.print(f"  [yellow]⚠ {_siglip_failed} image(s) failed SigLIP (no file or load error)[/yellow]")
        _save_batch(todo)
        unload_model(siglip_model)
        del scene_feats, vqa_feats
        gc.collect()
        console.print()

    # ── Pass 3a — aesthetic-predictor-v2-5: aesthetic score ──────────────────
    # SigLIP-based MLP — run after SigLIP is unloaded to avoid holding two encoders at once
    aes_batch = _auto_batch(4)
    aes_todo = needs_path("aesthetic_score")
    if "aesthetic" in _skip_passes:
        console.print("[bold]Pass 3a/6:[/bold] skipped (--skip aesthetic)\n")
    elif aes_todo:
        console.print(f"[bold]Pass 3a/6:[/bold] Aesthetic scoring ({len(aes_todo)} photos, batch={aes_batch})")
        with console.status("Loading aesthetic-predictor-v2-5..."):
            aesthetic = load_aesthetic_model(_device())

        _aes_failed = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Scoring aesthetics...", total=len(aes_todo))
            for start in range(0, len(aes_todo), aes_batch):
                batch = aes_todo[start : start + aes_batch]
                paths = [_ensure_path(r) for r in batch]
                scores = extract_aesthetic_batch(paths, aesthetic, aes_batch)
                for r, score in zip(batch, scores, strict=False):
                    r["aesthetic_score"] = score
                _aes_failed += scores.count(None)
                p.advance(task, len(batch))

        if _aes_failed:
            console.print(f"  [yellow]⚠ {_aes_failed} image(s) failed aesthetic scoring[/yellow]")
        _save_batch(aes_todo)
        del aesthetic
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        console.print()

    # ── Pass 3b — CLIP-IQA+: technical IQ score (MPS/CUDA/CPU) ──────────────
    iq_batch = _auto_batch(8)
    iq_todo = needs_path("iq_score")
    if "iq" in _skip_passes or os.environ.get("SKIP_IQ", "false").lower() == "true":
        console.print("[bold]Pass 3b/6:[/bold] CLIP-IQA+ skipped\n")
    elif iq_todo:
        device = _device()
        console.print(f"[bold]Pass 3b/6:[/bold] CLIP-IQA+ technical quality ({len(iq_todo)} photos, batch={iq_batch}, {device})")
        with console.status("Loading CLIP-IQA+ (pyiqa)..."):
            iq_metric = load_clipiqa_metric(device)

        if isinstance(iq_metric, Exception):
            console.print(f"  [yellow]CLIP-IQA+ failed to load: {iq_metric}. Skipping.[/yellow]\n")
            iq_metric = None
        if iq_metric is not None:
            save_every = 200
            _iq_failed = 0
            with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
                task = p.add_task("Scoring technical quality...", total=len(iq_todo))
                for start in range(0, len(iq_todo), iq_batch):
                    batch = iq_todo[start : start + iq_batch]
                    paths = [_ensure_path(r) for r in batch]
                    iq_scores = extract_iq_batch(paths, iq_metric, batch_size=iq_batch)
                    for r, iq in zip(batch, iq_scores, strict=False):
                        r["iq_score"] = iq
                    _iq_failed += iq_scores.count(None)
                    p.advance(task, len(batch))
                    if (start + iq_batch) % save_every == 0:
                        _save_batch(iq_todo[max(0, start + iq_batch - save_every) : start + iq_batch])
            if _iq_failed:
                console.print(f"  [yellow]⚠ {_iq_failed} image(s) failed IQ scoring[/yellow]")
            _save_batch(iq_todo)
            del iq_metric
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        console.print()

    # ── Pass 4 — DINOv2 embeddings (batched) ─────────────────────────────────
    dino_batch = _auto_batch(8)
    todo = needs_path("dinov2")
    if "dino" in _skip_passes:
        console.print("[bold]Pass 4/6:[/bold] skipped (--skip dino)\n")
    elif todo:
        console.print(f"[bold]Pass 4/6:[/bold] DINOv2 embeddings ({len(todo)} photos, batch={dino_batch})")
        with console.status("Loading DINOv2 (base)..."):
            dino_model, dino_processor, device = load_dino_model()

        _dino_failed = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Extracting embeddings...", total=len(todo))
            for start in range(0, len(todo), dino_batch):
                batch = todo[start : start + dino_batch]
                paths = [_ensure_path(r) for r in batch]
                embeddings = extract_embedding_batch(paths, dino_model, dino_processor, device)
                for r, emb in zip(batch, embeddings, strict=False):
                    if emb is not None:
                        r["dinov2"] = emb
                _dino_failed += embeddings.count(None)
                p.advance(task, len(batch))

        if _dino_failed:
            console.print(f"  [yellow]⚠ {_dino_failed} image(s) failed DINOv2 embedding[/yellow]")
        _save_batch(todo)
        unload_model(dino_model)
        gc.collect()
        console.print()

    # ── Pass 5 — RMBG 2.0 saliency (batched) ──────────────────────────────────
    sal_batch = _auto_batch(8)
    todo = needs_path("saliency")
    if "saliency" in _skip_passes:
        console.print("[bold]Pass 5/6:[/bold] skipped (--skip saliency)\n")
    elif todo:
        console.print(f"[bold]Pass 5/6:[/bold] Saliency ({len(todo)} photos, batch={sal_batch})")
        with console.status("Loading RMBG-2.0 (briaai/RMBG-2.0)..."):
            saliency_pipe = load_saliency_model(_device())

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
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        console.print()

    # ── Pass 6 — YOLO11-Pose: object detection + pose (portrait photos) ────────
    todo = [
        r
        for r in records.values()
        if "pose_data" not in r
        and r.get("caption", {}).get("has_person", "").startswith("yes")
        and (r.get("path") or (r.get("source") in ("lightroom", "both") and _download_renditions()))
    ]
    if "pose" in _skip_passes:
        console.print("[bold]Pass 6/6:[/bold] skipped (--skip pose)\n")
    elif todo:
        console.print(f"[bold]Pass 6/6:[/bold] YOLO11-Pose object detection ({len(todo)} portrait photos)")
        device = _device()
        with console.status("Loading YOLO11n-pose..."):
            pose_model = load_pose_model(device)

        _pose_failed = 0
        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Detecting objects + pose...", total=len(todo))
            for start in range(0, len(todo), batch_size):
                batch = todo[start : start + batch_size]
                paths = [_ensure_path(r) for r in batch]
                pose_results = extract_pose_batch(paths, pose_model, device)
                for r, pr in zip(batch, pose_results, strict=False):
                    if pr:
                        r["pose_data"] = pr
                    else:
                        _pose_failed += 1
                p.advance(task, len(batch))

        if _pose_failed:
            console.print(f"  [yellow]⚠ {_pose_failed} image(s) failed pose extraction[/yellow]")
        _save_batch(todo)
        unload_pose_model(pose_model)
        gc.collect()
        console.print()

    all_records = list(records.values())
    _ml_complete = sum(1 for r in all_records if r.get("scene") is not None)
    _no_file = sum(1 for r in all_records if not r.get("path"))
    _summary = f"[green]{len(all_records)} photo(s)[/green] · ML complete: [green]{_ml_complete}[/green]"
    if _no_file:
        _summary += f" · No local file/rendition: [yellow]{_no_file}[/yellow]"
    console.print(_summary + "\n")

    console.print("[bold]Computing aggregated statistics...[/bold]")
    aggregated = aggregate(all_records)

    console.print("[bold]Running quality issue detection...[/bold]")
    coach_data = aggregate_flags(all_records)
    console.print()

    data = {"photos": all_records, "aggregated": aggregated, "coach": coach_data}
    generate_json(data)
    generate_html(data)

    console.print("\n[bold green]Done![/bold green]")
    console.print("  [dim]JSON →[/dim] [cyan]docs/results.json[/cyan]")
    console.print("  [dim]HTML →[/dim] [cyan]docs/report.html[/cyan]  ← open in your browser\n")


if __name__ == "__main__":
    main()

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
    classify_scene_and_aesthetic_batch,
    encode_scene_labels,
    extract_caption_batch,
    extract_color,
    extract_composition,
    extract_depth_batch,
    extract_ela,
    extract_embedding_batch,
    extract_exif,
    extract_saliency_batch,
    load_aesthetic_predictor,
    load_caption_model,
    load_clip_models,
    load_depth_model,
    load_dino_model,
    load_saliency_model,
    unload_model,
)
from sources import load_sources
from report import generate_html, generate_json

console = Console()

CHEAP_KEYS = {"exif", "color", "composition", "ela"}
BATCH_SIZE = 16


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
    return parser.parse_args()


def _progress(*cols):
    return Progress(*cols, console=console)


def _prefetch_renditions(records: list[dict], max_workers: int = 16) -> None:
    """Download all missing renditions in parallel before ML passes begin."""
    if not _download_renditions():
        return
    todo = [r for r in records
            if not r.get("path") and r.get("source") in ("lightroom", "both")]
    if not todo:
        return

    global _lr_token, _lr_catalog_id
    if not _lr_token:
        from sources.lightroom import get_token_and_catalog
        _lr_token, _lr_catalog_id = get_token_and_catalog()

    from sources.lightroom import download_rendition
    console.print(f"[bold]Pre-fetching:[/bold] {len(todo)} renditions "
                  f"([cyan]{max_workers}[/cyan] parallel workers)")

    with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                   MofNCompleteColumn(), TaskProgressColumn()) as p:
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


def _prune_stale(sources_str: str, raw: list[dict]) -> None:
    from cache import prune_cache
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
    keep_renditions = os.environ.get("LIGHTROOM_KEEP_RENDITIONS", "true").lower() == "true"
    console.print(f"\n[bold]Loading sources:[/bold] [cyan]{sources}[/cyan]")

    # ── 1. Fetch metadata from all sources ───────────────────────────────────
    raw = load_sources(sample=args.sample)
    if not raw:
        console.print("[red]No photos found. Check SOURCES and PHOTO_DIR in .env[/red]")
        sys.exit(1)
    console.print(f"Found [green]{len(raw)}[/green] photo(s).\n")

    if args.prune:
        _prune_stale(sources, raw)

    records: dict[str, dict] = {}
    for r in raw:
        h = r["hash"]
        cached = load_cache(h) or {}
        merged = {**r, **{k: v for k, v in cached.items() if k not in ("path", "hash", "source")}}
        records[h] = merged

    for r in records.values():
        if r.get("source") in ("lightroom", "both") and r.get("lightroom_id"):
            save_cache(r["hash"], r)

    # ── 1b. Pre-fetch all renditions in parallel ──────────────────────────────
    _prefetch_renditions(list(records.values()))

    def needs(key: str) -> list:
        return [r for r in records.values() if key not in r]

    def needs_path(key: str) -> list:
        todo = []
        for r in records.values():
            if key in r:
                continue
            if r.get("path") or (
                r.get("source") in ("lightroom", "both")
                and _download_renditions()
            ):
                todo.append(r)
        return todo

    # ── 2. Pass 1 — cheap extractors (parallel, single image open per photo) ─
    todo = [r for r in records.values()
            if not CHEAP_KEYS.issubset(r.keys())
            and (r.get("path") or (r.get("source") in ("lightroom", "both")
                 and _download_renditions()))]
    if todo:
        console.print(f"[bold]Pass 1/5:[/bold] EXIF · Color · Composition ({len(todo)} photos)")

        def _process_cheap(r: dict) -> None:
            img_path = _ensure_path(r)
            if not img_path:
                return
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img.load()
                    r["exif"] = extract_exif(img)
                    r["color"] = extract_color(img)
                    r["composition"] = extract_composition(img)
                    r["ela"] = extract_ela(img_path)
            except Exception:
                pass

        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                       MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Extracting metadata...", total=len(todo))
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
                futs = {pool.submit(_process_cheap, r): r for r in todo}
                for fut in as_completed(futs):
                    p.advance(task)

        _save_batch(todo)
        console.print()

    # ── 3. Pass 2 — CLIP: scene + aesthetic (batched) ────────────────────────
    todo = needs_path("scene")
    if todo:
        console.print(f"[bold]Pass 2/5:[/bold] CLIP scene + aesthetic ({len(todo)} photos, batch={batch_size})")
        with console.status("Loading CLIP (ViT-L/14) + aesthetic predictor..."):
            clip_model, clip_preprocess, clip_tokenizer = load_clip_models()
            device = _device()
            clip_model.to(device)
            scene_feats = encode_scene_labels(clip_model, clip_tokenizer, device)
            aesthetic_mlp = load_aesthetic_predictor(device)

        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                       MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Classifying scenes...", total=len(todo))
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                paths = [_ensure_path(r) for r in batch]
                results = classify_scene_and_aesthetic_batch(
                    paths, clip_model, clip_preprocess, device, scene_feats, aesthetic_mlp
                )
                for r, res in zip(batch, results):
                    r["scene"] = res["scene"]
                    r["aesthetic_score"] = res["aesthetic_score"]
                p.advance(task, len(batch))

        _save_batch(todo)
        unload_model(clip_model)
        del scene_feats, aesthetic_mlp
        gc.collect()
        console.print()

    # ── 4. Pass 3 — DINOv2 embeddings (batched) ──────────────────────────────
    todo = needs_path("dinov2")
    if todo:
        console.print(f"[bold]Pass 3/5:[/bold] DINOv2 embeddings ({len(todo)} photos, batch={batch_size})")
        with console.status("Loading DINOv2 (small)..."):
            dino_model, dino_processor, device = load_dino_model()

        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                       MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Extracting embeddings...", total=len(todo))
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                paths = [_ensure_path(r) for r in batch]
                embeddings = extract_embedding_batch(paths, dino_model, dino_processor, device)
                for r, emb in zip(batch, embeddings):
                    r["dinov2"] = emb
                p.advance(task, len(batch))

        _save_batch(todo)
        unload_model(dino_model)
        gc.collect()
        console.print()

    # ── 5. Pass 4 — Depth Anything v2 (batched) ──────────────────────────────
    todo = needs_path("depth")
    if todo:
        console.print(f"[bold]Pass 4/5:[/bold] Depth Anything v2 ({len(todo)} photos, batch={batch_size})")
        with console.status("Loading Depth Anything v2 (Small)..."):
            depth_pipe = load_depth_model(_device())

        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                       MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Estimating depth...", total=len(todo))
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                paths = [_ensure_path(r) for r in batch]
                depth_results = extract_depth_batch(paths, depth_pipe, batch_size=batch_size)
                for r, depth in zip(batch, depth_results):
                    r["depth"] = depth
                    if not keep_renditions and r.get("source") in ("lightroom", "both"):
                        try:
                            p_obj = Path(r["path"]) if r.get("path") else None
                            if p_obj:
                                p_obj.unlink(missing_ok=True)
                                r.pop("path", None)
                        except Exception:
                            pass
                p.advance(task, len(batch))

        _save_batch(todo)
        del depth_pipe
        gc.collect()
        console.print()

    # ── 6. Pass 5 — BLIP captions (batched, all questions per photo in one call)
    todo = needs_path("caption")
    if todo:
        console.print(f"[bold]Pass 5/6:[/bold] BLIP captions ({len(todo)} photos, batch={batch_size})")
        with console.status("Loading BLIP VQA..."):
            caption_model, caption_tokenizer = load_caption_model(_device())

        paths = [_ensure_path(r) for r in todo]

        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                       MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Captioning photos...", total=len(todo))
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                batch_paths = paths[start:start + batch_size]
                caption_results = extract_caption_batch(batch_paths, caption_model, caption_tokenizer, batch_size)
                for r, cap in zip(batch, caption_results):
                    r["caption"] = cap
                    if not keep_renditions and r.get("source") in ("lightroom", "both"):
                        try:
                            p_obj = Path(r["path"]) if r.get("path") else None
                            if p_obj:
                                p_obj.unlink(missing_ok=True)
                                r.pop("path", None)
                        except Exception:
                            pass
                p.advance(task, len(batch))

        _save_batch(todo)
        del caption_model, caption_tokenizer
        gc.collect()
        console.print()

    # ── 7. Pass 6 — U²-Net saliency (batched) ────────────────────────────────
    todo = needs_path("saliency")
    if todo:
        console.print(f"[bold]Pass 6/6:[/bold] Saliency ({len(todo)} photos, batch={batch_size})")
        with console.status("Loading RMBG-1.4 saliency (briaai/RMBG-1.4)..."):
            saliency_pipe = load_saliency_model(_device())

        with _progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                       MofNCompleteColumn(), TaskProgressColumn()) as p:
            task = p.add_task("Detecting subjects...", total=len(todo))
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                paths = [_ensure_path(r) for r in batch]
                sal_results = extract_saliency_batch(paths, saliency_pipe, batch_size)
                for r, sal in zip(batch, sal_results):
                    r["saliency"] = sal
                p.advance(task, len(batch))

        _save_batch(todo)
        del saliency_pipe
        gc.collect()
        console.print()

    all_records = list(records.values())
    console.print(f"[green]All {len(all_records)} photo(s) processed.[/green]\n")

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

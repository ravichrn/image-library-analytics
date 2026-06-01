"""
Adobe Lightroom API source.

Fetches asset metadata (develop settings, EXIF, ratings, keywords) from Lightroom cloud.
Rendition downloads happen on-demand in main.py, interleaved with ML processing.
Delta sync via cache/.sync_state.json (only fetches assets updated since last run).
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://lr.adobe.io/v2"
RENDITIONS_DIR = Path("artifacts/cache/renditions")
SYNC_STATE_PATH = Path("artifacts/cache/.sync_state.json")


def _api_get(path: str, token: str, params: dict | None = None) -> dict:
    url = f"{API_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-API-Key", os.environ.get("LIGHTROOM_CLIENT_ID", ""))
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode()
        if text.startswith("while (1) {}"):
            text = text.split("\n", 1)[1]
        return json.loads(text)


def _get_catalog_id(token: str) -> str:
    return _api_get("/catalog", token)["id"]


def _fetch_all_assets(catalog_id: str, token: str, captured_after: str | None = None) -> list[dict]:
    """Fetch all assets (full catalog metadata scan).

    `captured_after` is an ISO date string and filters by *capture date* only —
    used for --lightroom-since scope limiting, not for delta detection.
    Delta detection is done per-asset by comparing `updated` timestamps in the caller.
    """
    assets = []
    params: dict = {"limit": 500, "subtype": "image"}
    if captured_after:
        params["captured_after"] = captured_after
    while True:
        data = _api_get(f"/catalogs/{catalog_id}/assets", token, params)
        assets.extend(data.get("resources", []))
        next_link = data.get("links", {}).get("next", {}).get("href")
        if not next_link:
            break
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(next_link).query)
        params = {k: v[0] for k, v in qs.items()}
    return assets


def _fetch_all_albums(catalog_id: str, token: str) -> list[dict]:
    """Return [{id, name, subtype}] for all album-like entries (excludes collection_set folders)."""
    albums: list[dict] = []
    params: dict = {"limit": 500}
    while True:
        data = _api_get(f"/catalogs/{catalog_id}/albums", token, params)
        for r in data.get("resources", []):
            subtype = r.get("payload", {}).get("subtype", "")
            if subtype != "collection_set":
                albums.append(
                    {
                        "id": r["id"],
                        "name": r["payload"].get("name", r["id"]),
                        "subtype": subtype,
                    }
                )
        nxt = data.get("links", {}).get("next", {}).get("href")
        if not nxt:
            break
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(nxt).query)
        params = {k: v[0] for k, v in qs.items()}
    return albums


def _fetch_album_asset_index(catalog_id: str, token: str, albums: list[dict]) -> dict[str, list[str]]:
    """Return {asset_id: [album_name, ...]} for every asset that belongs to at least one album.

    The album-assets endpoint can return the asset ID either at top-level ``id`` or nested
    inside ``asset.id`` depending on the API version.  We try both and store every variant so
    the lookup succeeds regardless of how ``lightroom_id`` was originally stored.
    """
    index: dict[str, list[str]] = {}
    for album in albums:
        params: dict = {"limit": 500}
        while True:
            data = _api_get(f"/catalogs/{catalog_id}/albums/{album['id']}/assets", token, params)
            for r in data.get("resources", []):
                # top-level id is sometimes a composite album-asset ID; the real asset id
                # may live in r["asset"]["id"] — store both so the lookup always hits
                ids_to_store = {r["id"]}
                nested = r.get("asset", {})
                if isinstance(nested, dict) and nested.get("id"):
                    ids_to_store.add(nested["id"])
                for aid in ids_to_store:
                    index.setdefault(aid, []).append(album["name"])
            nxt = data.get("links", {}).get("next", {}).get("href")
            if not nxt:
                break
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(nxt).query)
            params = {k: v[0] for k, v in qs.items()}
    return index


def _asset_sha256(asset: dict) -> str | None:
    return asset.get("payload", {}).get("importSource", {}).get("sha256")


def _parse_xmp_str(xmp_str: str) -> dict:
    """Parse a raw XMP string and return a dict of crs: key→value pairs.

    Handles two formats emitted by different Lightroom API versions:
      Attribute-style (older): <rdf:Description crs:Exposure2012="+0.50" .../>
      Element-style (newer):   <crs:Exposure2012>+0.50</crs:Exposure2012>
    """
    result = {}
    for key, raw_val in re.findall(r'crs:(\w+)="([^"]+)"', xmp_str):
        try:
            result[key] = float(raw_val)
        except ValueError:
            result[key] = raw_val
    for key, raw_val in re.findall(r"<crs:(\w+)>([^<]+)</crs:\w+>", xmp_str):
        if key not in result:
            try:
                result[key] = float(raw_val)
            except ValueError:
                result[key] = raw_val
    return result


def _fetch_xmp_from_href(href: str, token: str, catalog_id: str = "") -> str:
    """Fetch raw XMP from a relative link href returned by the Lightroom API.

    href is relative to the catalog resource (e.g. 'assets/{id}/xmp/develop').
    catalog_id is required to build the correct absolute URL.
    """
    if href.startswith("http"):
        url = href
    elif href.startswith("catalogs/"):
        url = f"{API_BASE}/{href}"
    else:
        url = f"{API_BASE}/catalogs/{catalog_id}/{href}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-API-Key", os.environ.get("LIGHTROOM_CLIENT_ID", ""))
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).warning("XMP fetch failed for %s: %s", url, exc)
        return ""


def _xmp_develop_href(links: dict) -> str:
    """Return the XMP develop link href if the API advertises one with actual data."""
    link = links.get("/rels/xmp/develop", {})
    if link.get("fileSize", 0) > 0:
        return link.get("href", "")
    return ""


def _parse_develop(
    payload: dict, links: dict | None = None, token: str = "", pre_fetched_xmp: str | None = None, asset_id: str = "", catalog_id: str = ""
) -> dict:
    """Parse Lightroom develop settings.

    Two mutually exclusive formats — one per photo:
      NEW (crsVersion ≥ 16.4, 2024+): develop data served via /xmp/develop link endpoint.
      OLD (crsVersion ≤ 15.x, pre-2024): develop data inline as xmpCameraRaw string.

    Resolution order:
      1. Pre-fetched XMP from the link endpoint (parallel batch, non-empty only).
         Empty string means the pre-fetch failed — fall through and retry serially.
      2. Serial fetch from the link endpoint (fallback when pre-fetch failed).
      3. Inline xmpCameraRaw string (old format, no HTTP call needed).
      4. Nothing found — return {}. Warn if the photo is marked as edited so the
         raw XMP can be inspected to detect any new format.

    Delta sync: lightroom_updated (stored per record) is compared on each run;
    only photos with a changed updated timestamp are re-processed.
    """
    import warnings as _warnings

    dev = payload.get("develop", {})

    # 1. New API — pre-fetched XMP (non-empty only; empty = pre-fetch failed, retry below)
    if pre_fetched_xmp:
        result = _parse_xmp_str(pre_fetched_xmp)
        if result:
            return result
        # Pre-fetch returned content but no crs: keys — may be new XML format.
        # Fall through to serial fetch for one more try; log the raw content.
        _warnings.warn(
            f"[lightroom] XMP pre-fetch for {asset_id!r} returned content but no crs: keys were found. Raw (first 500 chars): {pre_fetched_xmp[:500]!r}",
            stacklevel=3,
        )

    # 2. New API — serial fetch (covers pre-fetch failure and format-debug retry)
    href = _xmp_develop_href(links or {})
    if href and token:
        xmp_str = _fetch_xmp_from_href(href, token, catalog_id)
        if xmp_str:
            result = _parse_xmp_str(xmp_str)
            if result:
                return result
            _warnings.warn(
                f"[lightroom] Serial XMP fetch for {asset_id!r} returned content but no crs: keys. Raw (first 500 chars): {xmp_str[:500]!r}",
                stacklevel=3,
            )

    # 3. Old API — inline xmpCameraRaw string (pre-2024 format, no HTTP call)
    xmp_inline = dev.get("xmpCameraRaw", "")
    if isinstance(xmp_inline, str) and xmp_inline:
        return _parse_xmp_str(xmp_inline)

    # 4. Nothing found
    if dev.get("fromDefaults") is False:
        _warnings.warn(
            f"[lightroom] No develop data for edited asset {asset_id!r}. "
            f"develop keys: {list(dev.keys())}  "
            f"xmpCameraRaw type: {type(xmp_inline).__name__}  "
            f"links: {list((links or {}).keys())}",
            stacklevel=3,
        )
    return {}


# ── Reporting helpers (kept for sync summary output) ─────────────────────────


class _DevelopFormat:
    OLD_INLINE = "old_inline"  # inline xmpCameraRaw string (crsVersion ≤ 15.x)
    LINK_FETCHED = "link_fetched"  # fetched via /rels/xmp/develop link (crsVersion ≥ 16.4+)
    UNEDITED = "unedited"  # fromDefaults=True, no XMP expected
    UNKNOWN = "unknown"  # edited but nothing parseable found


def _classify_develop_format(dev: dict, links: dict | None = None) -> str:
    """Classify purely for sync summary reporting — not used in fetch logic."""
    if _xmp_develop_href(links or {}):
        return _DevelopFormat.LINK_FETCHED
    if isinstance(dev.get("xmpCameraRaw", ""), str) and dev.get("xmpCameraRaw", ""):
        return _DevelopFormat.OLD_INLINE
    if dev.get("fromDefaults") is False:
        return _DevelopFormat.UNKNOWN
    return _DevelopFormat.UNEDITED


def _rational(val) -> float | None:
    """Convert [numerator, denominator] or scalar to float."""
    try:
        if isinstance(val, list | tuple) and len(val) == 2:
            return val[0] / val[1] if val[1] else None
        return float(val)
    except Exception:
        return None


def _parse_xmp_exif(payload: dict) -> dict:
    """Extract camera EXIF from payload.xmp — available directly from Lightroom API."""
    xmp = payload.get("xmp", {})
    tiff = xmp.get("tiff", {})
    exif = xmp.get("exif", {})
    aux = xmp.get("aux", {})
    src = payload.get("importSource", {})

    make = tiff.get("Make")
    model = tiff.get("Model")
    lens = aux.get("Lens") or aux.get("LensModel")

    fl_raw = exif.get("FocalLengthIn35mmFilm") or exif.get("FocalLength")
    fl = _rational(fl_raw)
    focal_category = None
    if fl:
        focal_category = "wide" if fl < 35 else ("normal" if fl <= 70 else "telephoto")

    ap_raw = exif.get("FNumber") or exif.get("ApertureValue")
    aperture = _rational(ap_raw)
    dof_category = None
    if aperture:
        dof_category = "shallow" if aperture < 2.8 else ("mid" if aperture < 8 else "deep")

    iso = exif.get("ISOSpeedRatings")
    iso_int = int(iso) if iso is not None else None
    light_category = None
    if iso_int is not None:
        light_category = "low_light" if iso_int > 1600 else ("indoor" if iso_int > 400 else "bright")

    et = _rational(exif.get("ExposureTime"))
    shutter_category = None
    if et:
        shutter_category = "freeze" if et <= 1 / 500 else ("hand" if et <= 1 / 60 else ("slow" if et <= 1 / 15 else "bulb"))

    metering_map = {"pattern": "multi_segment", "center weighted average": "center_weighted", "spot": "spot", "average": "average", "multi-spot": "multi_spot"}
    metering_raw = exif.get("MeteringMode", "")
    metering = metering_map.get(str(metering_raw).lower()) if metering_raw else None

    flash_fired = exif.get("FlashFired")
    flash_fired = bool(flash_fired) if flash_fired is not None else None

    cap_date = payload.get("captureDate", "")
    hour = year_month = year = time_of_day = None
    if cap_date:
        try:
            from datetime import datetime

            from extractors.exif import hour_to_time_of_day

            dt = datetime.fromisoformat(cap_date.replace("Z", "+00:00"))
            hour = dt.hour
            year_month = dt.strftime("%Y-%m")
            year = dt.year
            time_of_day = hour_to_time_of_day(hour)
        except Exception:
            pass

    orig_h = src.get("originalHeight", 0)
    orig_w = src.get("originalWidth", 0)
    megapixels = round(orig_h * orig_w / 1_000_000, 2) if orig_h and orig_w else None

    result = {
        "focal_length_mm": round(fl, 2) if fl else None,
        "aperture_f": round(aperture, 2) if aperture else None,
        "iso": iso_int,
        "shutter_speed": round(et, 6) if et else None,
        "shutter_category": shutter_category,
        "focal_category": focal_category,
        "dof_category": dof_category,
        "light_category": light_category,
        "camera_make": str(make).strip() if make else None,
        "camera_model": str(model).strip() if model else None,
        "lens_model": str(lens).strip() if lens else None,
        "flash_fired": flash_fired,
        "metering_mode": metering,
        "time_of_day": time_of_day,
        "hour": hour,
        "year_month": year_month,
        "year": year,
        "megapixels": megapixels,
        "gps_lat": None,
        "gps_lon": None,
    }
    return result


def _parse_asset(asset: dict, token: str = "", pre_fetched_xmp: str | None = None, catalog_id: str = "") -> dict:
    payload = asset.get("payload", {})
    links = asset.get("links", {})
    raw_filename = payload.get("importSource", {}).get("fileName", "")
    asset_id = asset.get("id", "")
    return {
        "lightroom_id": asset_id,
        "lightroom_develop": _parse_develop(payload, links=links, token=token, pre_fetched_xmp=pre_fetched_xmp, asset_id=asset_id, catalog_id=catalog_id),
        "lightroom_rating": payload.get("rating", 0),
        "lightroom_label": payload.get("colorLabel", ""),
        "lightroom_pick": payload.get("pick", 0),
        "lightroom_keywords": payload.get("keywords", []),
        "lightroom_capture_date": payload.get("captureDate", ""),
        "lightroom_updated": asset.get("updated"),
        "lightroom_filename": raw_filename,
        "lightroom_exif": _parse_xmp_exif(payload),
    }


def download_rendition(lightroom_id: str, catalog_id: str, token: str, sha256: str) -> Path | None:
    """Download 2048px rendition for a single asset. Returns path or None on failure."""
    RENDITIONS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RENDITIONS_DIR / f"{sha256}.jpg"
    if dest.exists():
        return dest
    url = f"{API_BASE}/catalogs/{catalog_id}/assets/{lightroom_id}/renditions/2048"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-API-Key", os.environ.get("LIGHTROOM_CLIENT_ID", ""))
    try:
        with urllib.request.urlopen(req) as resp:
            dest.write_bytes(resp.read())
        return dest
    except Exception:
        return None


def fetch_current_hashes(token: str, catalog_id: str) -> set[str]:
    """Full API fetch — returns sha256 hashes of all current assets."""
    assets = _fetch_all_assets(catalog_id, token)
    return {sha for a in assets if (sha := _asset_sha256(a))}


def get_token_and_catalog() -> tuple[str, str]:
    """Return (access_token, catalog_id) — used by main.py for on-demand downloads."""
    from auth.lightroom import get_access_token

    token = get_access_token()
    return token, _get_catalog_id(token)


def _refresh_albums(catalog_id: str, token: str, records: list[dict], console, save_cache_fn) -> None:
    """Refresh album membership for all records — albums change independently of asset metadata."""
    try:
        albums = _fetch_all_albums(catalog_id, token)
        if albums:
            album_index = _fetch_album_asset_index(catalog_id, token, albums)
            tagged = 0
            for r in records:
                lid = r.get("lightroom_id")
                if lid is not None:
                    r["lightroom_album_names"] = album_index.get(lid, [])
                    if r["lightroom_album_names"]:
                        tagged += 1
                    save_cache_fn(r["hash"], r)
            console.print(f"  Albums: [green]{len(albums)}[/green] found, [green]{tagged}[/green] photos in at least one album.")
        else:
            console.print("  [yellow]No albums found — your Lightroom library may have no collections.[/yellow]")
            for r in records:
                if r.get("lightroom_id") and "lightroom_album_names" not in r:
                    r["lightroom_album_names"] = []
    except Exception as exc:
        import traceback

        console.print(f"  [red]Album fetch failed:[/red] {type(exc).__name__}: {exc}")
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")


def load_lightroom(
    sample: int | None = None,
    album_name: str | None = None,
    since_override: str | None = None,
) -> list[dict]:
    """
    Fetch asset metadata from Lightroom cloud with per-asset change detection.

    Every run fetches all asset metadata (lightweight JSON, ~11 API calls for 5k photos).
    Each asset's `updated` timestamp is compared against the cached `lightroom_updated`
    value to detect changes — this catches develop setting edits, rating changes, keyword
    updates, and any other Lightroom metadata change, not just newly imported photos.

    Assets are re-processed when:
      - They are new (not in the local cache)
      - Their `updated` timestamp differs from the cached value (any change in Lightroom)
      - Their cached `lightroom_develop` is empty (stale data recovery for photos that
        were synced before Lightroom finished processing their develop settings)

    If nothing has changed and all develop data is present, returns immediately from
    the local cache with only an album membership refresh.

    album_name: restrict results to photos in this Lightroom album.
    since_override: if set, only consider assets captured on or after this ISO date
                    (--lightroom-since flag; does not affect change detection logic).
    """
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    from auth.lightroom import get_access_token
    from cache import load_all_cached, load_cache, save_cache

    console = Console()
    if since_override:
        console.print(f"  [dim]--lightroom-since {since_override}: limiting scope to assets captured after this date.[/dim]")

    token = get_access_token()
    catalog_id = _get_catalog_id(token)

    # ── Fetch all asset metadata (no date filter — lightweight JSON scan) ─────
    console.print("  Scanning Lightroom catalog for changes...", end=" ")
    all_assets = _fetch_all_assets(catalog_id, token, captured_after=since_override)
    console.print(f"[dim]{len(all_assets)} assets in catalog[/dim]")

    # ── Per-asset change detection ────────────────────────────────────────────
    # Build a lookup of cached records by lightroom_id for O(1) comparison.
    cached_by_lid: dict[str, dict] = {}
    for r in load_all_cached():
        lid = r.get("lightroom_id")
        if lid:
            cached_by_lid[lid] = r

    assets_to_process: list[dict] = []
    for asset in all_assets:
        lid = asset.get("id")
        sha256 = _asset_sha256(asset)
        if not sha256 or not lid:
            continue

        cached = cached_by_lid.get(lid)
        if cached is None:
            # New photo — not in local cache at all
            assets_to_process.append(asset)
        elif asset.get("updated") != cached.get("lightroom_updated"):
            # Any Lightroom change: develop edit, rating, pick, keyword, etc.
            assets_to_process.append(asset)
        elif cached.get("lightroom_develop") is None:
            # lightroom_develop key is absent — XMP was never fetched for this photo.
            # {} means "fetched and confirmed empty/unedited" — do NOT re-fetch.
            # Only re-fetch if the API advertises a non-empty XMP endpoint.
            if _xmp_develop_href(asset.get("links", {})):
                assets_to_process.append(asset)

    new_count = sum(1 for a in assets_to_process if a.get("id") not in cached_by_lid)
    upd_count = len(assets_to_process) - new_count
    empty_dev_count = sum(1 for a in assets_to_process if cached_by_lid.get(a.get("id")) and not cached_by_lid[a.get("id")].get("lightroom_develop"))
    recovery_suffix = f"  [dim](including {empty_dev_count} empty-develop recoveries)[/dim]" if empty_dev_count else ""
    console.print(f"  Changes detected: [green]{new_count} new[/green]  [yellow]{upd_count} updated[/yellow]{recovery_suffix}")

    # ── Nothing to process — load from cache with album refresh ───────────────
    if not assets_to_process:
        console.print("  [dim]Up to date. Loading from local cache.[/dim]")
        cached_records = load_all_cached()
        _refresh_albums(catalog_id, token, cached_records, console, save_cache)
        if album_name:
            cached_records = [r for r in cached_records if album_name in (r.get("lightroom_album_names") or [])]
            console.print(f"  [dim]--lightroom-album '{album_name}':[/dim] {len(cached_records)} photos match.")
        if sample and sample < len(cached_records):
            import random

            cached_records = random.sample(cached_records, sample)
        return cached_records

    # ── Process changed / new assets ──────────────────────────────────────────
    if sample and sample < len(assets_to_process):
        import random

        assets_to_process = random.sample(assets_to_process, sample)

    # ── Parallel XMP pre-fetch for new-format photos ──────────────────────────
    # Photos with crsVersion ≥ 16.4 store develop settings at a separate
    # /xmp/develop endpoint.  Fetching them sequentially inside the main loop
    # is the bottleneck (one HTTP round-trip per photo).  Pre-fetch all of them
    # concurrently so the main loop just reads from a local dict.
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import as_completed as _as_completed

    # Assets that have an XMP develop link — link-driven, no format assumptions
    xmp_needs_fetch = [a for a in assets_to_process if _xmp_develop_href(a.get("links", {}))]
    _xmp_cache: dict[str, str] = {}  # asset_id → raw XMP string

    _XMP_WORKERS = 12  # concurrent HTTP connections; stays well under Adobe rate limits

    if xmp_needs_fetch:
        console.print(f"  Pre-fetching XMP for [cyan]{len(xmp_needs_fetch)}[/cyan] new-format photos ([dim]{_XMP_WORKERS} workers[/dim])...")
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("{task.description}"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            console=console,
        ) as xmp_progress:
            xmp_task = xmp_progress.add_task("Fetching XMP develop data...", total=len(xmp_needs_fetch))

            def _fetch_one_xmp(asset: dict) -> tuple[str, str]:
                aid = asset.get("id", "")
                href = _xmp_develop_href(asset.get("links", {}))
                return aid, (_fetch_xmp_from_href(href, token, catalog_id) if href else "")

            with ThreadPoolExecutor(max_workers=_XMP_WORKERS) as pool:
                futs = {pool.submit(_fetch_one_xmp, a): a for a in xmp_needs_fetch}
                for fut in _as_completed(futs):
                    asset_id, xmp_str = fut.result()
                    _xmp_cache[asset_id] = xmp_str
                    xmp_progress.advance(xmp_task)

    new_records = []
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("{task.description}"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing metadata", total=len(assets_to_process))
        fmt_counts: dict[str, int] = {
            _DevelopFormat.OLD_INLINE: 0,
            _DevelopFormat.LINK_FETCHED: 0,
            _DevelopFormat.UNEDITED: 0,
            _DevelopFormat.UNKNOWN: 0,
        }
        import warnings as _warnings_mod

        unknown_assets: list[str] = []

        for asset in assets_to_process:
            sha256 = _asset_sha256(asset)
            if not sha256:
                progress.advance(task)
                continue

            # Track develop format for sync summary (reporting only — not used in fetch logic)
            dev = asset.get("payload", {}).get("develop", {})
            links = asset.get("links", {})
            fmt = _classify_develop_format(dev, links)
            fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1
            if fmt == _DevelopFormat.UNKNOWN:
                unknown_assets.append(f"{asset.get('id', '')} crsVersion={dev.get('crsVersion', '?')} devkeys={list(dev.keys())} links={list(links.keys())}")

            # Pass pre-fetched XMP so _parse_develop doesn't make a serial HTTP call
            pre_xmp = _xmp_cache.get(asset.get("id", ""))
            with _warnings_mod.catch_warnings(record=True):
                _warnings_mod.simplefilter("always")
                record = _parse_asset(asset, token=token, pre_fetched_xmp=pre_xmp, catalog_id=catalog_id)

            record["hash"] = sha256
            record["source"] = "lightroom"

            cached_rendition = RENDITIONS_DIR / f"{sha256}.jpg"
            if cached_rendition.exists():
                record["path"] = str(cached_rendition)

            # Merge: Lightroom metadata overwrites stale fields; pixel-level ML data
            # (color, composition, embeddings, etc.) is preserved from existing cache.
            existing = load_cache(sha256) or {}
            merged = {**existing, **record}
            save_cache(sha256, merged)
            new_records.append(merged)
            progress.advance(task)

    # Report develop format distribution
    if new_records:
        parts = [
            f"[dim]inline[/dim]: {fmt_counts[_DevelopFormat.OLD_INLINE]}",
            f"[cyan]link[/cyan]: {fmt_counts[_DevelopFormat.LINK_FETCHED]}",
            f"[dim]unedited[/dim]: {fmt_counts[_DevelopFormat.UNEDITED]}",
        ]
        if fmt_counts[_DevelopFormat.UNKNOWN]:
            parts.append(f"[bold red]unknown: {fmt_counts[_DevelopFormat.UNKNOWN]} ← edited photo with no parseable XMP — check warning log[/bold red]")
        console.print("  Develop formats — " + "  ".join(parts))
        if unknown_assets:
            for ua in unknown_assets[:3]:
                console.print(f"  [red]  {ua}[/red]")
            if len(unknown_assets) > 3:
                console.print(f"  [red]  ...and {len(unknown_assets) - 3} more[/red]")

    _write_sync_state(total_assets=len(all_assets))

    # Merge processed records on top of full cached set
    all_by_hash = {r["hash"]: r for r in load_all_cached()}
    for r in new_records:
        all_by_hash[r["hash"]] = r

    _refresh_albums(catalog_id, token, list(all_by_hash.values()), console, save_cache)

    all_records = list(all_by_hash.values())

    if album_name:
        filtered = [r for r in all_records if album_name in (r.get("lightroom_album_names") or [])]
        console.print(f"  [dim]--lightroom-album '{album_name}':[/dim] {len(filtered)}/{len(all_records)} photos match.")
        return filtered

    return all_records


def _read_sync_state() -> dict:
    if SYNC_STATE_PATH.exists():
        try:
            return json.loads(SYNC_STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _write_sync_state(total_assets: int) -> None:
    SYNC_STATE_PATH.parent.mkdir(exist_ok=True)
    SYNC_STATE_PATH.write_text(
        json.dumps(
            {
                "last_synced": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_assets": total_assets,
            },
            indent=2,
        )
    )

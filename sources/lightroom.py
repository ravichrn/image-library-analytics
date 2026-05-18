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
RENDITIONS_DIR = Path("cache/renditions")
SYNC_STATE_PATH = Path("cache/.sync_state.json")


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


def _fetch_all_assets(catalog_id: str, token: str, since: str | None = None) -> list[dict]:
    assets = []
    params: dict = {"limit": 500, "subtype": "image"}
    if since:
        params["captured_after"] = since
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


def _parse_develop(payload: dict) -> dict:
    xmp_str = payload.get("develop", {}).get("xmpCameraRaw", "")
    if not xmp_str or not isinstance(xmp_str, str):
        return {}
    result = {}
    for key, raw_val in re.findall(r'crs:(\w+)="([^"]+)"', xmp_str):
        try:
            result[key] = float(raw_val)
        except ValueError:
            result[key] = raw_val
    return result


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


def _parse_asset(asset: dict) -> dict:
    payload = asset.get("payload", {})
    raw_filename = payload.get("importSource", {}).get("fileName", "")
    return {
        "lightroom_id": asset.get("id"),
        "lightroom_catalog_id": asset.get("catalog_id", ""),
        "lightroom_develop": _parse_develop(payload),
        "lightroom_rating": payload.get("rating", 0),
        "lightroom_label": payload.get("colorLabel", ""),
        "lightroom_pick": payload.get("pick", 0),
        "lightroom_keywords": payload.get("keywords", []),
        "lightroom_capture_date": payload.get("captureDate", ""),
        "lightroom_updated": asset.get("updated"),
        "lightroom_filename": raw_filename,
        "lightroom_filename_stem": Path(raw_filename).stem if raw_filename else "",
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
    """Full API fetch (no delta filter) — returns sha256 hashes of all current assets."""
    assets = _fetch_all_assets(catalog_id, token, since=None)
    return {sha for a in assets if (sha := _asset_sha256(a))}


def get_token_and_catalog() -> tuple[str, str]:
    """Return (access_token, catalog_id) — used by main.py for on-demand downloads."""
    from auth.lightroom import get_access_token

    token = get_access_token()
    return token, _get_catalog_id(token)


def load_lightroom(
    sample: int | None = None,
    album_name: str | None = None,
    since_override: str | None = None,
) -> list[dict]:
    """
    Fetch asset metadata from Lightroom cloud.

    On the first run fetches everything. On subsequent runs does a delta fetch
    (assets updated since last sync). If the delta is empty the API returns
    immediately and all records are loaded from the local SQLite cache —
    no further network calls needed.

    album_name: if set, restrict results to photos in the named album.
    since_override: if set, only fetch assets captured on or after this ISO date string.
    """
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    from auth.lightroom import get_access_token
    from cache import load_all_cached, load_cache, save_cache

    console = Console()
    sync_state = _read_sync_state()
    # since_override takes priority over delta-sync state
    since = since_override or sync_state.get("last_synced")

    if since_override:
        console.print(f"  [dim]--lightroom-since {since_override}: filtering assets captured after this date.[/dim]")

    token = get_access_token()
    catalog_id = _get_catalog_id(token)

    console.print("  Checking Lightroom for new/updated assets...", end=" ")
    assets = _fetch_all_assets(catalog_id, token, since=since)
    console.print(f"[green]{len(assets)} new/updated[/green]")

    # No delta — serve everything from local cache, but still refresh album membership
    if not assets and since:
        console.print("  [dim]Up to date. Loading from local cache.[/dim]")
        cached_records = load_all_cached()
        try:
            albums = _fetch_all_albums(catalog_id, token)
            if albums:
                album_index = _fetch_album_asset_index(catalog_id, token, albums)
                tagged = 0
                for r in cached_records:
                    lid = r.get("lightroom_id")
                    if lid is not None:
                        r["lightroom_album_names"] = album_index.get(lid, [])
                        if r["lightroom_album_names"]:
                            tagged += 1
                        save_cache(r["hash"], r)
                console.print(f"  Albums: [green]{len(albums)}[/green] found, [green]{tagged}[/green] photos in at least one album.")
            else:
                console.print("  [yellow]No albums found — your Lightroom library may have no collections.[/yellow]")
        except Exception as exc:
            import traceback

            console.print(f"  [red]Album fetch failed:[/red] {type(exc).__name__}: {exc}")
            console.print(f"  [dim]{traceback.format_exc()}[/dim]")
        if album_name:
            cached_records = [r for r in cached_records if album_name in (r.get("lightroom_album_names") or [])]
            console.print(f"  [dim]--lightroom-album '{album_name}':[/dim] {len(cached_records)} photos match.")
        if sample and sample < len(cached_records):
            import random

            cached_records = random.sample(cached_records, sample)
        return cached_records

    # First run or delta present — parse and persist new/updated assets
    if sample and sample < len(assets):
        import random

        assets = random.sample(assets, sample)

    new_records = []
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("{task.description}"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing metadata", total=len(assets))
        for asset in assets:
            sha256 = _asset_sha256(asset)
            if not sha256:
                progress.advance(task)
                continue

            record = _parse_asset(asset)
            record["hash"] = sha256
            record["source"] = "lightroom"
            record["lightroom_catalog_id"] = catalog_id

            cached_rendition = RENDITIONS_DIR / f"{sha256}.jpg"
            if cached_rendition.exists():
                record["path"] = str(cached_rendition)

            # Merge with existing cache to preserve pixel-level data (color, composition, etc.)
            existing = load_cache(sha256) or {}
            merged = {**existing, **record}
            save_cache(sha256, merged)
            record = merged
            new_records.append(record)
            progress.advance(task)

    _write_sync_state(total_assets=sync_state.get("total_assets", 0) + len(new_records))

    # Merge new records on top of full cached set so callers always get everything
    all_by_hash = {r["hash"]: r for r in load_all_cached()}
    for r in new_records:
        all_by_hash[r["hash"]] = r

    # Fetch album membership and tag every record (always fresh — albums change independently)
    try:
        albums = _fetch_all_albums(catalog_id, token)
        if albums:
            album_index = _fetch_album_asset_index(catalog_id, token, albums)
            tagged = 0
            for r in all_by_hash.values():
                lid = r.get("lightroom_id")
                if lid is not None:
                    r["lightroom_album_names"] = album_index.get(lid, [])
                    if r["lightroom_album_names"]:
                        tagged += 1
                    save_cache(r["hash"], r)
            console.print(f"  Albums: [green]{len(albums)}[/green] found, [green]{tagged}[/green] photos in at least one album.")
        else:
            console.print("  [yellow]No albums found — your Lightroom library may have no collections.[/yellow]")
            for r in all_by_hash.values():
                if r.get("lightroom_id") and "lightroom_album_names" not in r:
                    r["lightroom_album_names"] = []
    except Exception as exc:
        import traceback

        console.print(f"  [red]Album fetch failed:[/red] {type(exc).__name__}: {exc}")
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")

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

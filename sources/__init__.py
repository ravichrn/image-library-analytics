import os
from pathlib import Path

# ML result keys that are worth carrying over from a local record into a Lightroom record
_ML_KEYS = {"exif", "color", "composition", "scene", "aesthetic_score", "dinov2", "depth", "caption"}


def load_sources(sample: int | None = None) -> list[dict]:
    """
    Load photo records from all configured sources (SOURCES env var).

    Deduplication strategy (in order):
    1. Exact SHA-256 match — same bytes, same record.
    2. Filename stem match (e.g. DSC02136) — local export and Lightroom original
       are the same photo. ML results from the local record are merged into the
       Lightroom record so no analysis work is lost.
    """
    raw_sources = os.environ.get("SOURCES", "local")
    source_names = [s.strip().lower() for s in raw_sources.split(",") if s.strip()]

    by_hash: dict[str, dict] = {}

    for name in source_names:
        if name == "local":
            from sources.local import load_local
            records = load_local(sample=sample)
        elif name == "lightroom":
            from sources.lightroom import load_lightroom
            records = load_lightroom(sample=sample)
        else:
            raise ValueError(f"Unknown source: {name!r}. Valid values: local, lightroom")

        for r in records:
            h = r["hash"]
            if h in by_hash:
                existing = by_hash[h]
                for k, v in r.items():
                    if k.startswith("lightroom_") or k == "source":
                        existing[k] = v
                if existing.get("source") == "local" and r.get("source") == "lightroom":
                    existing["source"] = "both"
                continue

            # Filename-stem match: local export vs Lightroom original (different hash, same photo)
            stem = Path(r.get("path", "")).stem if r.get("path") else None
            lr_stem = r.get("lightroom_filename_stem", "")
            match_stem = stem or lr_stem

            if match_stem:
                matched_hash = _find_by_stem(by_hash, match_stem, r.get("source"))
                if matched_hash:
                    existing = by_hash[matched_hash]
                    if r.get("source") == "lightroom":
                        # Carry ML results from existing local record into the Lightroom record
                        for k in _ML_KEYS:
                            if k in existing and k not in r:
                                r[k] = existing[k]
                        r["source"] = "both"
                        # Replace the local entry with the richer Lightroom record
                        del by_hash[matched_hash]
                        by_hash[h] = r
                    else:
                        # r is local — merge its ML results into the existing Lightroom record
                        for k in _ML_KEYS:
                            if k in r and k not in existing:
                                existing[k] = r[k]
                        existing["source"] = "both"
                    continue

            by_hash[h] = r

    return list(by_hash.values())


def _find_by_stem(by_hash: dict, stem: str, incoming_source: str | None) -> str | None:
    """Return the hash of an existing record whose filename stem matches."""
    for h, rec in by_hash.items():
        existing_stem = (
            Path(rec.get("path", "")).stem if rec.get("path") else None
        ) or rec.get("lightroom_filename_stem", "")
        if existing_stem and existing_stem == stem:
            return h
    return None

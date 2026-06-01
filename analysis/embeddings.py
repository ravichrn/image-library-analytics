import hashlib
import json
import random as _random
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.exceptions import EfficiencyWarning
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values
from umap import UMAP

from ._helpers import _show_thumbnails, _thumb_b64

_CACHE_PATH = Path("docs/embeddings_cache.json")


def _fingerprint(records: list[dict]) -> str:
    """SHA-256 of all (photo_hash, dinov3_vec) pairs sorted by hash."""
    h = hashlib.sha256()
    for r in sorted((r for r in records if r.get("dinov3")), key=lambda r: r["hash"]):
        h.update(r["hash"].encode())
        h.update(np.asarray(r["dinov3"], dtype=np.float32).tobytes())
    return h.hexdigest()


# Cosine-distance threshold for near-duplicate clustering. eps=0.05 ≈ 0.95 cosine
# similarity — the point where near-identical burst frames cluster together without
# merging visually distinct shots. Validated in tests/test_near_dup.py (precision/recall).
NEAR_DUP_EPS = 0.05
NEAR_DUP_MIN_SAMPLES = 2


def near_duplicate_labels(vectors, eps: float = NEAR_DUP_EPS, min_samples: int = NEAR_DUP_MIN_SAMPLES) -> np.ndarray:
    """Cluster L2-normalized embeddings by cosine distance. Returns a DBSCAN label per
    vector; label -1 means "no near-duplicate". Pure function — testable in isolation.

    Uses a sparse radius graph (NearestNeighbors) instead of a dense n*n matrix so
    memory stays O(n * k) rather than O(n^2). Safe at 100k+ photos.
    """
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms

    nn = NearestNeighbors(radius=eps, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(mat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EfficiencyWarning)
        sparse_dist = nn.radius_neighbors_graph(mode="distance")
        sparse_dist = sort_graph_by_row_values(sparse_dist, warn_when_not_sorted=False)
        return DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(sparse_dist)


def analyze(records: list[dict]) -> dict:
    fp = _fingerprint(records)
    if _CACHE_PATH.exists():
        try:
            cached = json.loads(_CACHE_PATH.read_text())
            if cached.get("_fingerprint") == fp:
                return {k: v for k, v in cached.items() if k != "_fingerprint"}
        except Exception:
            pass

    # ── UMAP + KMeans ────────────────────────────────────────────────────────
    UMAP_MAX_POINTS = 800
    valid = [(i, r) for i, r in enumerate(records) if r.get("dinov3") is not None]
    clusters_out = {"n_clusters": 0, "labels": []}
    umap_out = {"points": [], "total": len(valid), "sampled": len(valid)}

    if len(valid) >= 2:
        if len(valid) > UMAP_MAX_POINTS:
            rng = _random.Random(42)
            umap_valid = rng.sample(valid, UMAP_MAX_POINTS)
        else:
            umap_valid = valid

        _, umap_records = zip(*umap_valid, strict=False)
        matrix = np.array([r["dinov3"] for r in umap_records], dtype=np.float32)

        n_clusters = max(2, min(10, len(umap_valid) // 30))
        km = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
        labels = km.fit_predict(matrix).tolist()

        n_neighbors = min(15, len(umap_valid) - 1)
        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, transform_seed=42, init="spectral")
        coords = reducer.fit_transform(matrix)

        points = []
        for (_orig_i, rec), lbl, (x, y) in zip(umap_valid, labels, coords, strict=False):
            points.append(
                {
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "cluster": lbl,
                    "hash": rec.get("hash", ""),
                    "path": Path(rec.get("path", "") or "").name,
                    "aesthetic_score": rec.get("aesthetic_score"),
                    "in_album": bool(rec.get("lightroom_album_names")),
                    "thumb": _thumb_b64(rec) if _show_thumbnails() else None,
                }
            )
        clusters_out = {
            "n_clusters": n_clusters,
            "labels": labels,
        }
        umap_out = {"points": points, "total": len(valid), "sampled": len(umap_valid)}

    # ── Burst / near-duplicate detection ─────────────────────────────────────
    burst_groups: dict = {}
    storage_tiers: dict = {}
    dino_records = [
        (r["hash"], r["dinov3"], r.get("aesthetic_score") or 0, r.get("path", ""), r.get("composition", {}).get("sharpness_score") or 0)
        for r in records
        if r.get("dinov3")
    ]
    if len(dino_records) >= 2:
        _hashes_d, vecs_d, scores_d, paths_d, sharp_d = zip(*dino_records, strict=False)
        labels_d = near_duplicate_labels(vecs_d)

        max_sharp = max(sharp_d) or 1.0

        def _hero_rank(i: int) -> float:
            return 0.65 * (scores_d[i] / 100) + 0.35 * (sharp_d[i] / max_sharp)

        cluster_ids = [ll for ll in set(labels_d) if ll != -1]
        groups_d = []
        for cid in cluster_ids:
            idxs = [i for i, ll in enumerate(labels_d) if ll == cid]
            best = max(idxs, key=_hero_rank)
            redundant = [paths_d[i] for i in idxs if i != best]
            groups_d.append(
                {
                    "size": len(idxs),
                    "best_path": paths_d[best],
                    "best_score": round(float(scores_d[best]), 2),
                    "redundant_paths": redundant[:3],
                }
            )
        groups_d.sort(key=lambda g: -g["size"])

        burst_groups = {
            "total_bursts": len(groups_d),
            "photos_in_bursts": sum(g["size"] for g in groups_d),
            "largest_burst": groups_d[0]["size"] if groups_d else 0,
            "groups": [{k: v for k, v in g.items() if k not in ("redundant_paths", "best_path")} for g in groups_d[:10]],
        }

        total_w_emb = len(dino_records)
        redundant_count = sum(g["size"] - 1 for g in groups_d)
        storage_tiers = {
            "cluster_count": len(groups_d),
            "hero_count": len(groups_d),
            "redundant_count": redundant_count,
            "unclustered_count": int(np.sum(labels_d == -1)),
            "redundancy_pct": round(redundant_count / total_w_emb * 100, 1) if total_w_emb else 0.0,
            "tiers": [{"size": g["size"], "best_score": g["best_score"]} for g in groups_d[:10]],
        }

    # ── Event grouping (time-gap > 4 h = new event) ──────────────────────────
    events: dict = {}
    timed = []
    for r in records:
        ts_str = r.get("lightroom_capture_date") or ""
        if not ts_str:
            exif_ym = r.get("exif", {}).get("year_month")
            exif_h = r.get("exif", {}).get("hour")
            if exif_ym and exif_h is not None:
                try:
                    ts_str = f"{exif_ym}-01T{exif_h:02d}:00:00"
                except Exception:
                    continue
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            timed.append((ts, r))
        except Exception:
            continue

    if len(timed) >= 2:
        timed.sort(key=lambda x: x[0])
        GAP_HOURS = 4
        event_groups: list[list] = []
        current: list = [timed[0]]
        for i in range(1, len(timed)):
            delta = (timed[i][0] - timed[i - 1][0]).total_seconds() / 3600
            if delta > GAP_HOURS:
                if len(current) >= 2:
                    event_groups.append(current)
                current = []
            current.append(timed[i])
        if len(current) >= 2:
            event_groups.append(current)

        def _event_narrative(ev_records):
            hero = max(ev_records, key=lambda r: r.get("aesthetic_score") or 0)
            detailed = hero.get("caption", {}).get("detailed_caption", "").strip()
            if detailed:
                return detailed
            captions = [r.get("caption", {}) for r in ev_records if r.get("caption")]
            if not captions:
                return ""
            dino_ev = [(i, r.get("dinov3")) for i, r in enumerate(ev_records) if r.get("dinov3")]
            if len(dino_ev) >= 3:
                vecs_ev = np.array([v for _, v in dino_ev], dtype=np.float32)
                nrm = np.linalg.norm(vecs_ev, axis=1, keepdims=True) + 1e-9
                vecs_ev = vecs_ev / nrm
                sim_ev = vecs_ev @ vecs_ev.T
                chosen = [0]
                for _ in range(2):
                    min_sims = [min(sim_ev[c, j] for c in chosen) for j in range(len(dino_ev))]
                    nxt = max(range(len(dino_ev)), key=lambda j: -min_sims[j] if j not in chosen else float("inf"))
                    chosen.append(nxt)
                rep_caps = [ev_records[dino_ev[c][0]].get("caption", {}) for c in chosen]
            else:
                sorted_ev = sorted(ev_records, key=lambda r: r.get("aesthetic_score") or 0, reverse=True)
                rep_caps = [r.get("caption", {}) for r in sorted_ev[:3] if r.get("caption")]
            settings = [c.get("setting") for c in rep_caps if c.get("setting")]
            tod = next((c.get("time_of_day") for c in rep_caps if c.get("time_of_day")), None)
            weather = next((c.get("weather") for c in rep_caps if c.get("weather")), None)
            setting_str = settings[0] if settings else "mixed"
            parts = [f"A {setting_str} shoot"]
            if tod:
                parts.append(f"during {tod}")
            if weather and weather.lower() not in ("clear", "unknown"):
                parts.append(f"in {weather} conditions")
            return ". ".join(parts) + "."

        from collections import Counter as _Counter2

        event_list = []
        for idx, group in enumerate(event_groups):
            ts_list = [t for t, _ in group]
            ev_records = [r for _, r in group]
            duration_mins = (ts_list[-1] - ts_list[0]).total_seconds() / 60
            scene_label_counts: _Counter2 = _Counter2()
            for r in ev_records:
                for sc in (r.get("scene") or {}).get("scene_types") or []:
                    scene_label_counts[sc] += 1
            top_scenes_ev = [sc for sc, _ in scene_label_counts.most_common(3)]
            hero_r = max(ev_records, key=lambda r: r.get("aesthetic_score") or 0)
            event_list.append(
                {
                    "event_id": idx + 1,
                    "date": ts_list[0].strftime("%Y-%m-%d"),
                    "photo_count": len(ev_records),
                    "duration_mins": round(duration_mins, 1),
                    "top_scenes": top_scenes_ev,
                    "hero_score": round(float(hero_r.get("aesthetic_score") or 0), 2),
                    "narrative": _event_narrative(ev_records),
                }
            )
        event_list.sort(key=lambda e: e["date"], reverse=True)
        cutoff_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        display_events = [e for e in event_list if e["photo_count"] >= 50 and e["date"] >= cutoff_date]
        events = {
            "total_events": len(event_list),
            "avg_photos_per_event": round(sum(e["photo_count"] for e in event_list) / len(event_list), 1) if event_list else 0,
            "largest_event": max((e["photo_count"] for e in event_list), default=0),
            "events": display_events,
        }

    result = {
        "clusters": clusters_out,
        "umap": umap_out,
        "burst_groups": burst_groups,
        "storage_tiers": storage_tiers,
        "events": events,
    }

    try:
        _CACHE_PATH.parent.mkdir(exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"_fingerprint": fp, **result}))
        tmp.replace(_CACHE_PATH)
    except Exception:
        pass

    return result

import random as _random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from umap import UMAP

from ._helpers import _show_thumbnails, _thumb_b64


def analyze(records: list[dict]) -> dict:
    # ── UMAP + KMeans ────────────────────────────────────────────────────────
    UMAP_MAX_POINTS = 800
    valid = [(i, r) for i, r in enumerate(records) if r.get("dinov2") is not None]
    clusters_out = {"n_clusters": 0, "labels": [], "centers": []}
    umap_out = {"points": [], "total": len(valid), "sampled": len(valid)}

    if len(valid) >= 2:
        if len(valid) > UMAP_MAX_POINTS:
            by_scene: dict = defaultdict(list)
            for item in valid:
                scene = item[1].get("scene", {}).get("scene_type", "unknown")
                by_scene[scene].append(item)
            sampled: list = []
            for scene_items in by_scene.values():
                quota = max(1, round(len(scene_items) / len(valid) * UMAP_MAX_POINTS))
                sampled.extend(_random.sample(scene_items, min(quota, len(scene_items))))
            _random.shuffle(sampled)
            sampled = sampled[:UMAP_MAX_POINTS]
            umap_valid = sampled
        else:
            umap_valid = valid

        _, umap_records = zip(*umap_valid, strict=False)
        matrix = np.array([r["dinov2"] for r in umap_records], dtype=np.float32)

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
                    "path": Path(rec.get("path", "") or "").name,
                    "scene_type": rec.get("scene", {}).get("scene_type", "unknown"),
                    "aesthetic_score": rec.get("aesthetic_score"),
                    "in_album": bool(rec.get("lightroom_album_names")),
                    "thumb": _thumb_b64(rec) if _show_thumbnails() else None,
                }
            )
        clusters_out = {
            "n_clusters": n_clusters,
            "labels": labels,
            "centers": [[round(float(v), 4) for v in c] for c in coords[[{lbl: j for j, lbl in enumerate(labels)}[i] for i in range(n_clusters)]]],
        }
        umap_out = {"points": points, "total": len(valid), "sampled": len(umap_valid)}

    # ── Burst / near-duplicate detection ─────────────────────────────────────
    burst_groups: dict = {}
    storage_tiers: dict = {}
    dino_records = [
        (r["hash"], r["dinov2"], r.get("aesthetic_score") or 0, r.get("scene", {}).get("scene_type", ""), r.get("path", "")) for r in records if r.get("dinov2")
    ]
    if len(dino_records) >= 2:
        _hashes_d, vecs_d, scores_d, scenes_d, paths_d = zip(*dino_records, strict=False)
        mat_d = np.array(vecs_d, dtype=np.float32)
        norms_d = np.linalg.norm(mat_d, axis=1, keepdims=True) + 1e-9
        mat_d = mat_d / norms_d
        sim_d = mat_d @ mat_d.T
        dist_d = np.clip(1.0 - sim_d, 0, 2).astype(np.float64)
        labels_d = DBSCAN(eps=0.05, min_samples=2, metric="precomputed").fit_predict(dist_d)

        cluster_ids = [ll for ll in set(labels_d) if ll != -1]
        groups_d = []
        for cid in cluster_ids:
            idxs = [i for i, ll in enumerate(labels_d) if ll == cid]
            best = max(idxs, key=lambda i: scores_d[i])
            redundant = [paths_d[i] for i in idxs if i != best]
            groups_d.append(
                {
                    "size": len(idxs),
                    "best_path": paths_d[best],
                    "best_score": round(float(scores_d[best]), 2),
                    "scene": scenes_d[best],
                    "redundant_paths": redundant[:3],
                }
            )
        groups_d.sort(key=lambda g: -g["size"])

        burst_groups = {
            "total_bursts": len(groups_d),
            "photos_in_bursts": sum(g["size"] for g in groups_d),
            "largest_burst": groups_d[0]["size"] if groups_d else 0,
            "groups": [{k: v for k, v in g.items() if k != "redundant_paths"} for g in groups_d[:10]],
        }

        total_w_emb = len(dino_records)
        redundant_count = sum(g["size"] - 1 for g in groups_d)
        _safe_keys = {"size", "best_score", "scene"}
        storage_tiers = {
            "cluster_count": len(groups_d),
            "hero_count": len(groups_d),
            "redundant_count": redundant_count,
            "unclustered_count": int(np.sum(labels_d == -1)),
            "redundancy_pct": round(redundant_count / total_w_emb * 100, 1) if total_w_emb else 0.0,
            "tiers": [{k: v for k, v in g.items() if k in _safe_keys} for g in groups_d[:10]],
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
            dino_ev = [(i, r.get("dinov2")) for i, r in enumerate(ev_records) if r.get("dinov2")]
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
            scenes_ev = [r.get("scene", {}).get("scene_type") for r in ev_records if r.get("scene")]
            top_scene_ev = _Counter2(s for s in scenes_ev if s).most_common(1)
            hero_r = max(ev_records, key=lambda r: r.get("aesthetic_score") or 0)
            event_list.append(
                {
                    "event_id": idx + 1,
                    "date": ts_list[0].strftime("%Y-%m-%d"),
                    "photo_count": len(ev_records),
                    "duration_mins": round(duration_mins, 1),
                    "top_scene": top_scene_ev[0][0] if top_scene_ev else "",
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

    return {
        "clusters": clusters_out,
        "umap": umap_out,
        "burst_groups": burst_groups,
        "storage_tiers": storage_tiers,
        "events": events,
    }

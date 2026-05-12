import colorsys
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from umap import UMAP

from extractors import hour_to_time_of_day


def _mean(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return round(float(np.mean(clean)), 4) if clean else None


def _std(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return round(float(np.std(clean)), 4) if clean else None


def _cv(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    m = np.mean(clean)
    if m == 0:
        return None
    return round(float(np.std(clean) / m), 4)


def _dist(values: list, known_keys: list) -> dict:
    c = Counter(v if v is not None else "unknown" for v in values)
    result = {k: c.get(k, 0) for k in known_keys}
    result["unknown"] = c.get("unknown", 0)
    return result


def aggregate(records: list[dict]) -> dict:
    def col(key, subkey=None):
        if subkey:
            return [r.get(key, {}).get(subkey) for r in records]
        return [r.get(key) for r in records]

    def ecol(key):
        return [r.get("exif", {}).get(key) for r in records]

    def ccol(key):
        return [r.get("composition", {}).get(key) for r in records]

    # ── color stats ─────────────────────────────────────────────────────────
    sats = [r.get("color", {}).get("avg_saturation") for r in records]
    brights = [r.get("color", {}).get("avg_brightness") for r in records]
    contrasts = [r.get("color", {}).get("contrast") for r in records]
    warm_ratios = [r.get("color", {}).get("warm_ratio") for r in records]
    cool_ratios = [r.get("color", {}).get("cool_ratio") for r in records]
    warmths = [r.get("color", {}).get("warmth") for r in records]
    warmth_dist = Counter(v if v is not None else "unknown" for v in warmths)

    color_stats = {
        "avg_saturation": _mean(sats),
        "avg_brightness": _mean(brights),
        "avg_contrast": _mean(contrasts),
        "avg_warmth_ratio": _mean(warm_ratios),
        "avg_cool_ratio": _mean(cool_ratios),
        "saturation_std": _std(sats),
        "contrast_std": _std(contrasts),
        "warmth_distribution": {
            "warm": warmth_dist.get("warm", 0),
            "cool": warmth_dist.get("cool", 0),
            "neutral": warmth_dist.get("neutral", 0),
        },
    }

    # ── composition stats ────────────────────────────────────────────────────
    thirds = ccol("thirds_score")
    syms = ccol("symmetry_score")
    neg_spaces = ccol("negative_space")
    horizons = ccol("horizon_position")

    def _horizon_bucket(v):
        if v is None:
            return "none"
        if v < 0.33:
            return "high"
        if v <= 0.66:
            return "mid"
        return "low"

    horizon_dist = Counter(_horizon_bucket(v) for v in horizons)
    subject_isolations = ccol("subject_isolation")
    foreground_clutters = ccol("foreground_clutter")

    composition_stats = {
        "avg_thirds_score": _mean(thirds),
        "avg_symmetry_score": _mean(syms),
        "avg_negative_space": _mean(neg_spaces),
        "avg_subject_isolation": _mean(subject_isolations),
        "avg_foreground_clutter": _mean(foreground_clutters),
        "horizon_position_distribution": {
            "high": horizon_dist.get("high", 0),
            "mid": horizon_dist.get("mid", 0),
            "low": horizon_dist.get("low", 0),
            "none": horizon_dist.get("none", 0),
        },
    }

    # ── exif stats ───────────────────────────────────────────────────────────
    # Derive time_of_day from lightroom_capture_date when EXIF hour is missing
    def _time_of_day(r: dict) -> str | None:
        val = r.get("exif", {}).get("time_of_day")
        if val:
            return val
        cap = r.get("lightroom_capture_date", "")
        if cap and "T" in cap:
            try:
                h = int(cap.split("T")[1].split(":")[0])
                return hour_to_time_of_day(h)
            except Exception:
                pass
        return None

    exif_stats = {
        "time_of_day_distribution": _dist(
            [_time_of_day(r) for r in records], ["golden_hour", "midday", "morning_evening", "night"],
        ),
        "focal_category_distribution": _dist(ecol("focal_category"), ["wide", "normal", "telephoto"]),
        "dof_category_distribution": _dist(ecol("dof_category"), ["shallow", "mid", "deep"]),
        "light_category_distribution": _dist(ecol("light_category"), ["bright", "indoor", "low_light"]),
    }

    # ── aesthetic stats ──────────────────────────────────────────────────────
    a_scores = [r.get("aesthetic_score") for r in records if r.get("aesthetic_score") is not None]

    def _score_bucket(v):
        if v < 33:
            return "low"
        if v <= 66:
            return "mid"
        return "high"

    score_dist = Counter(_score_bucket(v) for v in a_scores)

    # 10-bucket histogram: 0-10, 10-20, ..., 90-100
    buckets_10 = [0] * 10
    for s in a_scores:
        buckets_10[min(9, int(s / 10))] += 1

    aesthetic_stats = {
        "avg_score": round(float(np.mean(a_scores)), 3) if a_scores else None,
        "min_score": round(float(np.min(a_scores)), 3) if a_scores else None,
        "max_score": round(float(np.max(a_scores)), 3) if a_scores else None,
        "score_std": round(float(np.std(a_scores)), 3) if a_scores else None,
        "median_score": round(float(np.median(a_scores)), 3) if a_scores else None,
        "score_distribution": {
            "low": score_dist.get("low", 0),
            "mid": score_dist.get("mid", 0),
            "high": score_dist.get("high", 0),
        },
        "score_histogram_10": buckets_10,
    }

    # ── scene stats ──────────────────────────────────────────────────────────
    from extractors.scene import SCENE_LABELS
    scene_types = [r.get("scene", {}).get("scene_type") for r in records]
    scene_dist = _dist(scene_types, SCENE_LABELS + ["unknown"])
    dominant = max((k for k in scene_dist if k != "unknown"), key=scene_dist.get, default="unknown")
    scene_stats = {"scene_distribution": scene_dist, "dominant_scene": dominant}

    # ── editing consistency ──────────────────────────────────────────────────
    sat_cv = _cv(sats)
    con_cv = _cv(contrasts)
    bri_cv = _cv(brights)
    avg_cv = np.mean([v for v in [sat_cv, con_cv, bri_cv] if v is not None]) if any(
        v is not None for v in [sat_cv, con_cv, bri_cv]
    ) else None

    if avg_cv is None:
        interp = "unknown"
    elif avg_cv < 0.1:
        interp = "highly consistent"
    elif avg_cv < 0.25:
        interp = "moderate variation"
    else:
        interp = "eclectic"

    editing_consistency = {
        "saturation_cv": sat_cv,
        "contrast_cv": con_cv,
        "brightness_cv": bri_cv,
        "interpretation": interp,
    }

    # ── embeddings → UMAP + KMeans ───────────────────────────────────────────
    UMAP_MAX_POINTS = 800  # above this the scatter becomes unreadable
    valid = [(i, r) for i, r in enumerate(records) if r.get("dinov2") is not None]
    clusters_out = {"n_clusters": 0, "labels": [], "centers": []}
    umap_out = {"points": [], "total": len(valid), "sampled": len(valid)}

    if len(valid) >= 2:
        # Stratified sample by scene type to preserve distribution
        if len(valid) > UMAP_MAX_POINTS:
            import random as _random
            by_scene: dict = defaultdict(list)
            for item in valid:
                scene = item[1].get("scene", {}).get("scene_type", "unknown")
                by_scene[scene].append(item)
            sampled: list = []
            for scene_items in by_scene.values():
                quota = max(1, round(len(scene_items) / len(valid) * UMAP_MAX_POINTS))
                sampled.extend(_random.sample(scene_items, min(quota, len(scene_items))))
            # top up or trim to exactly UMAP_MAX_POINTS
            _random.shuffle(sampled)
            sampled = sampled[:UMAP_MAX_POINTS]
            umap_valid = sampled
        else:
            umap_valid = valid

        _, umap_records = zip(*umap_valid)
        matrix = np.array([r["dinov2"] for r in umap_records], dtype=np.float32)

        n_clusters = max(2, min(10, len(umap_valid) // 30))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(matrix).tolist()

        n_neighbors = min(15, len(umap_valid) - 1)
        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        coords = reducer.fit_transform(matrix)

        points = []
        for (orig_i, rec), lbl, (x, y) in zip(umap_valid, labels, coords):
            points.append({
                "x": round(float(x), 4), "y": round(float(y), 4),
                "cluster": lbl, "path": Path(rec.get("path", "")).name,
                "scene_type": rec.get("scene", {}).get("scene_type", "unknown"),
                "aesthetic_score": rec.get("aesthetic_score"),
            })
        clusters_out = {
            "n_clusters": n_clusters, "labels": labels,
            "centers": [[round(float(v), 4) for v in c] for c in coords[
                [labels.index(i) for i in range(n_clusters)]
            ]],
        }
        umap_out = {"points": points, "total": len(valid), "sampled": len(umap_valid)}

    # ════════════════════════════════════════════════════════════════════════
    # NEW AGGREGATIONS
    # ════════════════════════════════════════════════════════════════════════

    # ── shooting hours ───────────────────────────────────────────────────────
    # Prefer EXIF hour; fall back to lightroom_capture_date for cloud-only photos
    shooting_hours = {str(h): 0 for h in range(24)}
    for r in records:
        hour = r.get("exif", {}).get("hour")
        if hour is None:
            cap = r.get("lightroom_capture_date", "")
            if cap and "T" in cap:
                try:
                    hour = int(cap.split("T")[1].split(":")[0])
                except Exception:
                    pass
        if hour is not None:
            shooting_hours[str(hour)] = shooting_hours.get(str(hour), 0) + 1

    # ── raw technical histograms ─────────────────────────────────────────────
    focal_length_histogram = [v for v in ecol("focal_length_mm") if v is not None]
    aperture_histogram = [v for v in ecol("aperture_f") if v is not None]
    iso_histogram = [v for v in ecol("iso") if v is not None]

    # ── aesthetic by scene ───────────────────────────────────────────────────
    scene_scores: dict = defaultdict(list)
    for r in records:
        scene = r.get("scene", {}).get("scene_type")
        score = r.get("aesthetic_score")
        if scene and score is not None:
            scene_scores[scene].append(score)

    aesthetic_by_scene = {}
    for scene, scores in scene_scores.items():
        aesthetic_by_scene[scene] = {
            "avg_score": round(float(np.mean(scores)), 2),
            "std": round(float(np.std(scores)), 2),
            "count": len(scores),
        }

    # ── composition by scene ─────────────────────────────────────────────────
    comp_by_scene_data: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        scene = r.get("scene", {}).get("scene_type")
        comp = r.get("composition", {})
        if scene:
            for key in ("thirds_score", "symmetry_score", "negative_space"):
                v = comp.get(key)
                if v is not None:
                    comp_by_scene_data[scene][key].append(v)

    composition_by_scene = {}
    for scene, fields in comp_by_scene_data.items():
        composition_by_scene[scene] = {
            "avg_thirds": round(float(np.mean(fields["thirds_score"])), 3) if fields["thirds_score"] else None,
            "avg_symmetry": round(float(np.mean(fields["symmetry_score"])), 3) if fields["symmetry_score"] else None,
            "avg_negative_space": round(float(np.mean(fields["negative_space"])), 3) if fields["negative_space"] else None,
            "count": max(len(v) for v in fields.values()) if fields else 0,
        }

    # ── composition grid heatmap ─────────────────────────────────────────────
    grid_accumulator = np.zeros((3, 3))
    grid_count = 0
    for r in records:
        gw = r.get("composition", {}).get("grid_weights")
        if gw and len(gw) == 3 and len(gw[0]) == 3:
            grid_accumulator += np.array(gw)
            grid_count += 1
    if grid_count > 0:
        grid_heatmap = [[round(float(v), 4) for v in row] for row in (grid_accumulator / grid_count).tolist()]
    else:
        grid_heatmap = [[0.0] * 3 for _ in range(3)]

    # ── sharpness stats ──────────────────────────────────────────────────────
    sharpness_vals = [r.get("composition", {}).get("sharpness_score") for r in records]
    sharpness_by_scene: dict = defaultdict(list)
    for r in records:
        scene = r.get("scene", {}).get("scene_type")
        s = r.get("composition", {}).get("sharpness_score")
        if scene and s is not None:
            sharpness_by_scene[scene].append(s)

    sharpness_stats = {
        "avg": _mean(sharpness_vals),
        "std": _std(sharpness_vals),
        "by_scene": {
            scene: round(float(np.mean(vals)), 2)
            for scene, vals in sharpness_by_scene.items() if vals
        },
    }

    # ── exposure stats ───────────────────────────────────────────────────────
    highlights = [r.get("composition", {}).get("highlight_clipping") for r in records]
    shadows = [r.get("composition", {}).get("shadow_clipping") for r in records]
    biases = [r.get("composition", {}).get("exposure_bias") for r in records]
    highlight_threshold = 0.02  # >2% pixels blown = clipping
    shadow_threshold = 0.02

    clean_h = [v for v in highlights if v is not None]
    clean_s = [v for v in shadows if v is not None]
    clean_b = [v for v in biases if v is not None]

    exposure_stats = {
        "avg_bias": round(float(np.mean(clean_b)), 4) if clean_b else None,
        "avg_highlight_clipping": round(float(np.mean(clean_h)), 4) if clean_h else None,
        "avg_shadow_clipping": round(float(np.mean(clean_s)), 4) if clean_s else None,
        "highlight_clipped_pct": round(sum(1 for v in clean_h if v > highlight_threshold) / len(clean_h) * 100, 1) if clean_h else None,
        "shadow_clipped_pct": round(sum(1 for v in clean_s if v > shadow_threshold) / len(clean_s) * 100, 1) if clean_s else None,
    }

    # ── folder / trip breakdown ──────────────────────────────────────────────
    folder_data: dict = defaultdict(lambda: {"scores": [], "sats": [], "scenes": []})
    for r in records:
        path = r.get("path", "")
        parts = path.replace("\\", "/").split("/")
        folder = parts[-2] if len(parts) >= 2 else "root"
        score = r.get("aesthetic_score")
        sat = r.get("color", {}).get("avg_saturation")
        scene = r.get("scene", {}).get("scene_type")
        if score is not None:
            folder_data[folder]["scores"].append(score)
        if sat is not None:
            folder_data[folder]["sats"].append(sat)
        if scene:
            folder_data[folder]["scenes"].append(scene)

    folder_breakdown = {}
    for folder, data in sorted(folder_data.items(), key=lambda x: len(x[1]["scores"]), reverse=True):
        dominant_scene = Counter(data["scenes"]).most_common(1)[0][0] if data["scenes"] else "unknown"
        folder_breakdown[folder] = {
            "count": len(data["scores"]) + (len(folder_data[folder]["sats"]) - len(data["scores"])),
            "photo_count": sum(1 for r in records if (r.get("path", "").replace("\\", "/").split("/")[-2] if len(r.get("path", "").replace("\\", "/").split("/")) >= 2 else "root") == folder),
            "avg_aesthetic": round(float(np.mean(data["scores"])), 2) if data["scores"] else None,
            "dominant_scene": dominant_scene,
            "avg_saturation": round(float(np.mean(data["sats"])), 3) if data["sats"] else None,
        }

    # ── scene confidence ─────────────────────────────────────────────────────
    scene_conf_data: dict = defaultdict(list)
    for r in records:
        scene = r.get("scene", {}).get("scene_type")
        scores_dict = r.get("scene", {}).get("scene_scores", {})
        if scene and scene in scores_dict:
            scene_conf_data[scene].append(scores_dict[scene])

    scene_confidence = {
        scene: round(float(np.mean(vals)), 4)
        for scene, vals in scene_conf_data.items() if vals
    }

    # ── megapixel stats ──────────────────────────────────────────────────────
    mp_vals = [r.get("exif", {}).get("megapixels") for r in records]
    clean_mp = [v for v in mp_vals if v is not None]
    megapixel_stats = {
        "avg": round(float(np.mean(clean_mp)), 2) if clean_mp else None,
        "min": round(float(np.min(clean_mp)), 2) if clean_mp else None,
        "max": round(float(np.max(clean_mp)), 2) if clean_mp else None,
    }

    # ── color by scene ────────────────────────────────────────────────────────
    color_by_scene_data: dict = defaultdict(lambda: {"sats": [], "warm": [], "warmths": [], "brightness": [], "aesthetics": []})
    for r in records:
        scene = r.get("scene", {}).get("scene_type")
        col_d = r.get("color", {})
        if scene:
            sat = col_d.get("avg_saturation")
            wr = col_d.get("warm_ratio")
            wt = col_d.get("warmth")
            b = col_d.get("avg_brightness")
            a = r.get("aesthetic_score")
            if sat is not None:
                color_by_scene_data[scene]["sats"].append(sat)
            if wr is not None:
                color_by_scene_data[scene]["warm"].append(wr)
            if wt is not None:
                color_by_scene_data[scene]["warmths"].append(wt)
            if b is not None:
                color_by_scene_data[scene]["brightness"].append(b)
            if a is not None:
                color_by_scene_data[scene]["aesthetics"].append(a)

    color_by_scene = {}
    for scene, data in color_by_scene_data.items():
        dominant_warmth = Counter(data["warmths"]).most_common(1)[0][0] if data["warmths"] else "unknown"
        color_by_scene[scene] = {
            "avg_saturation": round(float(np.mean(data["sats"])), 3) if data["sats"] else None,
            "avg_brightness": round(float(np.mean(data["brightness"])), 3) if data["brightness"] else None,
            "avg_aesthetic": round(float(np.mean(data["aesthetics"])), 2) if data["aesthetics"] else None,
            "avg_warmth_ratio": round(float(np.mean(data["warm"])), 3) if data["warm"] else None,
            "dominant_warmth": dominant_warmth,
            "count": len(data["sats"]),
        }

    # ── color harmony distribution (from palette per photo) ──────────────────
    def _classify_harmony(palette):
        hues = []
        for p in (palette or [])[:4]:
            rgb = p.get("rgb", [])
            if len(rgb) == 3:
                h, s, _ = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
                if s > 0.15:
                    hues.append(h * 360)
        if len(hues) < 2:
            return "monochromatic"
        diffs = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                d = abs(hues[i] - hues[j])
                diffs.append(min(d, 360 - d))
        mx = max(diffs)
        if mx < 40:
            return "analogous"
        if 150 < mx < 215:
            return "complementary"
        if 100 < mx <= 150:
            return "triadic"
        return "mixed"

    harmony_counter: Counter = Counter()
    for r in records:
        harmony_counter[_classify_harmony(r.get("color", {}).get("palette", []))] += 1
    color_harmony_dist = dict(harmony_counter)

    # ── editing style patterns ────────────────────────────────────────────────
    cs = color_stats
    avg_sat = cs.get("avg_saturation") or 0
    avg_bri = cs.get("avg_brightness") or 0
    avg_con = cs.get("avg_contrast") or 0
    avg_wr = cs.get("avg_warmth_ratio") or 0
    avg_cr = cs.get("avg_cool_ratio") or 0
    sat_cv = editing_consistency.get("saturation_cv") or 0
    con_cv = editing_consistency.get("contrast_cv") or 0
    bri_cv = editing_consistency.get("brightness_cv") or 0

    def _conf(val, lo, hi):
        if val <= lo:
            return 0.0
        return round(min(1.0, (val - lo) / (hi - lo)), 2)

    raw_patterns = [
        ("warm_toned", "Warm Toned",
         _conf(avg_wr, 0.45, 0.70),
         f"avg warm-ratio {avg_wr:.2f} — dominant wavelengths skew orange/red/yellow. "
         "Common in golden-hour shooting, warm presets, or lifestyle editing."),
        ("cool_toned", "Cool & Desaturated",
         _conf(avg_cr, 0.38, 0.60),
         f"avg cool-ratio {avg_cr:.2f} — dominant wavelengths skew blue/cyan. "
         "Associated with moody, cinematic, or silver-tone editing styles."),
        ("vivid", "Vivid / Saturated",
         _conf(avg_sat, 0.35, 0.65),
         f"avg saturation {avg_sat:.3f} — colours are punchy and highly saturated. "
         "Typical of travel/landscape editing or vibrance-heavy presets."),
        ("muted", "Muted / Film-like",
         _conf(0.30 - avg_sat, 0.05, 0.25),
         f"avg saturation {avg_sat:.3f} — very low colour intensity, near-monochromatic palette. "
         "Common in film simulation presets (Kodak, Fuji) or faded/hazy aesthetics."),
        ("high_contrast", "High Contrast",
         _conf(avg_con, 0.38, 0.60),
         f"avg contrast {avg_con:.3f} — wide tonal separation between shadows and highlights. "
         "Adds drama and punch; common in black-and-white conversions and editorial work."),
        ("flat_matte", "Flat / Matte",
         _conf(0.22 - avg_con, 0.02, 0.18),
         f"avg contrast {avg_con:.3f} — compressed tonal range with lifted shadows and rolled highlights. "
         "Characteristic of the 'matte look' or log-profile editing without a strong tone curve."),
        ("high_key", "Bright / High Key",
         _conf(avg_bri, 0.58, 0.80),
         f"avg brightness {avg_bri:.3f} — photos lean bright and airy, often with open shadows. "
         "Prevalent in lifestyle, wedding, and portrait editing."),
        ("low_key", "Dark / Low Key",
         _conf(0.42 - avg_bri, 0.02, 0.32),
         f"avg brightness {avg_bri:.3f} — photos lean dark with heavy shadows. "
         "Found in moody portrait, fine-art, and night photography editing."),
        ("consistent", "Consistent Style",
         _conf(0.25 - max(sat_cv, con_cv, bri_cv), 0.0, 0.20),
         f"sat CV {sat_cv:.3f} · contrast CV {con_cv:.3f} · brightness CV {bri_cv:.3f} — "
         "very uniform look across the library, suggesting a defined preset or tight workflow."),
        ("varied", "Varied / Eclectic",
         _conf(max(sat_cv, con_cv, bri_cv), 0.25, 0.60),
         f"sat CV {sat_cv:.3f} · contrast CV {con_cv:.3f} · brightness CV {bri_cv:.3f} — "
         "high variation in colour and tone across photos. Could mean multiple shoots/projects "
         "with different styles, or no consistent preset applied."),
    ]

    editing_style_patterns = sorted(
        [{"key": k, "name": n, "confidence": c, "description": d}
         for k, n, c, d in raw_patterns if c >= 0.18],
        key=lambda x: x["confidence"], reverse=True,
    )

    # ── color grading stats ───────────────────────────────────────────────────
    # Tonal zone aggregates (new fields — may be None in old cache entries)
    shadow_pcts = [r.get("composition", {}).get("tonal_shadow_pct") for r in records]
    mid_pcts    = [r.get("composition", {}).get("tonal_mid_pct") for r in records]
    hi_pcts     = [r.get("composition", {}).get("tonal_highlight_pct") for r in records]
    clean_sp = [v for v in shadow_pcts if v is not None]
    clean_mp = [v for v in mid_pcts if v is not None]
    clean_hp = [v for v in hi_pcts if v is not None]

    avg_sp = float(np.mean(clean_sp)) if clean_sp else None
    avg_mp = float(np.mean(clean_mp)) if clean_mp else None
    avg_hp = float(np.mean(clean_hp)) if clean_hp else None

    if avg_bri >= 0.60:
        tonal_style = "high_key"
        tonal_desc = "Bright and airy — photos expose for the highlights, open shadows."
    elif avg_bri <= 0.38:
        tonal_style = "low_key"
        tonal_desc = "Dark and dramatic — heavy shadows dominate the frame."
    elif avg_con >= 0.38:
        tonal_style = "moody"
        tonal_desc = "Mid-tone brightness with elevated contrast — brooding, cinematic look."
    elif avg_con <= 0.22:
        tonal_style = "flat_matte"
        tonal_desc = "Compressed tonal range — lifted shadows, rolled highlights; the 'matte look'."
    else:
        tonal_style = "balanced"
        tonal_desc = "Balanced exposure and contrast — no dominant tonal treatment."

    if avg_wr > 0.52:
        color_temp = "warm"
        color_temp_desc = "Shifted toward orange/amber (~3000–5000K equivalent). Golden-hour or warm-preset tendency."
    elif avg_cr > 0.42:
        color_temp = "cool"
        color_temp_desc = "Shifted toward blue/cyan (~6500K+ equivalent). Shade, overcast, or cool-preset tendency."
    else:
        color_temp = "neutral"
        color_temp_desc = "No strong colour temperature bias (~5500K). Daylight-balanced or mixed conditions."

    dominant_harmony = harmony_counter.most_common(1)[0][0] if harmony_counter else "unknown"
    harmony_total = sum(harmony_counter.values()) or 1

    color_grading_stats = {
        "tonal_style": tonal_style,
        "tonal_description": tonal_desc,
        "color_temperature": color_temp,
        "color_temperature_description": color_temp_desc,
        "avg_tonal_shadow_pct": round(avg_sp, 3) if avg_sp is not None else None,
        "avg_tonal_mid_pct": round(avg_mp, 3) if avg_mp is not None else None,
        "avg_tonal_highlight_pct": round(avg_hp, 3) if avg_hp is not None else None,
        "color_harmony_dist": color_harmony_dist,
        "dominant_harmony": dominant_harmony,
        "harmony_pcts": {k: round(v / harmony_total * 100, 1) for k, v in harmony_counter.most_common()},
    }

    # ── composition patterns ──────────────────────────────────────────────────
    # Per-photo flags, then aggregate into prevalence
    comp_pattern_counts: Counter = Counter()
    n_comp = 0
    for r in records:
        c = r.get("composition", {})
        if not c:
            continue
        n_comp += 1
        ts = c.get("thirds_score")
        sy = c.get("symmetry_score")
        ns = c.get("negative_space")
        si = c.get("subject_isolation")
        fc = c.get("foreground_clutter")
        ll = c.get("leading_lines_score")
        if ts is not None and ts > 0.74:
            comp_pattern_counts["rule_of_thirds"] += 1
        if ts is not None and ts < 0.60:
            comp_pattern_counts["centered"] += 1
        if sy is not None and sy > 0.87:
            comp_pattern_counts["symmetric"] += 1
        if ns is not None and ns > 0.90:          # raised: top 36% are truly spacious/minimalist
            comp_pattern_counts["minimal"] += 1
        if ns is not None and ns < 0.18:
            comp_pattern_counts["frame_filling"] += 1
        if si is not None and si > 1.5:
            comp_pattern_counts["isolated_subject"] += 1
        if fc is not None and fc > 0.22:
            comp_pattern_counts["busy_foreground"] += 1
        # leading_lines excluded: diagonal-edge detection fires on ~88% of photos regardless
        # of intentional use, making it uninformative as a composition pattern.

    PATTERN_META = {
        "rule_of_thirds":   ("Rule of Thirds",        "Subject or key element placed near a thirds intersection — adds visual tension and directs the eye."),
        "symmetric":        ("Strong Symmetry",        "High left/right mirroring — architectural subjects, reflections, or deliberate geometric framing."),
        "minimal":          ("Negative Space",         "Unusually large areas of empty space — subject is isolated and given room to breathe in the frame."),
        "frame_filling":    ("Filling the Frame",      "Subject fills most of the frame with little empty space — maximises detail and impact."),
        "isolated_subject": ("Subject Isolation",      "Centre is significantly sharper than surroundings — subject pops cleanly from background (shallow DoF or high-contrast bg)."),
        "busy_foreground":  ("Foreground Interest",    "Complex edge density in the bottom zone — foreground elements add depth layering or texture."),
        "centered":         ("Centered Composition",   "Subject anchored at the frame centre — common in symmetry, direct portraits, and architecture."),
    }

    composition_patterns = []
    if n_comp > 0:
        for key, (name, desc) in PATTERN_META.items():
            cnt = comp_pattern_counts.get(key, 0)
            pct = round(cnt / n_comp * 100, 1)
            if pct >= 5:
                composition_patterns.append({
                    "key": key, "name": name, "description": desc,
                    "count": cnt, "pct": pct,
                })
        composition_patterns.sort(key=lambda x: x["pct"], reverse=True)

    # ── hue distribution ─────────────────────────────────────────────────────
    hue_buckets = [0] * 12
    for r in records:
        palette = r.get("color", {}).get("palette", [])
        if palette:
            rgb = palette[0].get("rgb", [])
            if len(rgb) == 3:
                h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
                if s > 0.15:
                    hue_buckets[int(h * 12) % 12] += 1

    hue_distribution = {
        "buckets": hue_buckets,
        "labels": ["Red", "Orange", "Yellow", "Yellow-Green", "Green", "Green-Cyan",
                   "Cyan", "Cyan-Blue", "Blue", "Blue-Violet", "Violet", "Pink-Red"],
    }

    # ── saturation histogram ──────────────────────────────────────────────────
    sat_buckets = [0] * 5
    for s in [v for v in sats if v is not None]:
        sat_buckets[min(4, int(s * 5))] += 1
    saturation_histogram = {
        "buckets": sat_buckets,
        "labels": ["0–0.2 (muted)", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0 (vivid)"],
    }

    # ── editing trends over time ──────────────────────────────────────────────
    trend_data: dict = defaultdict(lambda: {"aesthetics": [], "sats": [], "brightness": []})
    for r in records:
        ym = r.get("exif", {}).get("year_month")
        if not ym:
            continue
        score = r.get("aesthetic_score")
        sat = r.get("color", {}).get("avg_saturation")
        bri = r.get("color", {}).get("avg_brightness")
        if score is not None:
            trend_data[ym]["aesthetics"].append(score)
        if sat is not None:
            trend_data[ym]["sats"].append(sat)
        if bri is not None:
            trend_data[ym]["brightness"].append(bri)

    editing_trends = {}
    for ym in sorted(trend_data.keys()):
        d = trend_data[ym]
        editing_trends[ym] = {
            "count": max(len(d["aesthetics"]), len(d["sats"])),
            "avg_aesthetic": round(float(np.mean(d["aesthetics"])), 2) if d["aesthetics"] else None,
            "avg_saturation": round(float(np.mean(d["sats"])), 3) if d["sats"] else None,
            "avg_brightness": round(float(np.mean(d["brightness"])), 3) if d["brightness"] else None,
        }

    # ── depth stats ──────────────────────────────────────────────────────────
    depth_ranges = [r.get("depth", {}).get("depth_range") for r in records]
    depth_complexities = [r.get("depth", {}).get("depth_complexity") for r in records]
    depth_subject_scores = [r.get("depth", {}).get("subject_depth_score") for r in records]

    by_scene_depth: dict[str, list] = defaultdict(list)
    for r in records:
        scene = r.get("scene", {}).get("scene_type")
        dr = r.get("depth", {}).get("depth_range")
        if scene and dr is not None:
            by_scene_depth[scene].append(dr)

    depth_stats = {
        "avg_range": _mean(depth_ranges),
        "avg_complexity": _mean(depth_complexities),
        "avg_subject_depth_score": _mean(depth_subject_scores),
        "by_scene": {
            s: round(float(np.mean(v)), 4) for s, v in by_scene_depth.items() if v
        },
    }

    # ── BLIP VQA attribute distributions ─────────────────────────────────────
    VQA_KEYS = ["has_person", "setting", "time_of_day", "weather", "season"]
    vqa_counters: dict[str, Counter] = {k: Counter() for k in VQA_KEYS}
    vqa_total = 0
    for r in records:
        cap = r.get("caption", {})
        if not cap:
            continue
        vqa_total += 1
        for k in VQA_KEYS:
            val = cap.get(k)
            if val:
                vqa_counters[k][val] += 1

    def _vqa_dist(counter: Counter) -> list[dict]:
        total = sum(counter.values()) or 1
        return [
            {"label": label, "count": cnt, "pct": round(cnt / total * 100, 1)}
            for label, cnt in counter.most_common(6)
        ]

    visual_attributes = {
        k: _vqa_dist(vqa_counters[k]) for k in VQA_KEYS
    }
    visual_attributes["total_analyzed"] = vqa_total

    # ── Lightroom develop stats ───────────────────────────────────────────────
    DEVELOP_KEYS = [
        "Exposure", "Contrast", "Highlights", "Shadows", "Whites", "Blacks",
        "Clarity", "Vibrance", "Saturation", "Sharpness",
        "LuminanceSmoothing", "ColorNoiseReduction",
    ]
    develop_accum: dict[str, list] = {k: [] for k in DEVELOP_KEYS}
    develop_by_scene: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in DEVELOP_KEYS})
    lr_count = 0
    for r in records:
        dev = r.get("lightroom_develop", {})
        if not dev:
            continue
        lr_count += 1
        scene = r.get("scene", {}).get("scene_type")
        for k in DEVELOP_KEYS:
            v = dev.get(k)
            if v is not None:
                develop_accum[k].append(float(v))
                if scene:
                    develop_by_scene[scene][k].append(float(v))

    develop_stats = {
        "total_with_develop": lr_count,
        "avg": {k: _mean(v) for k, v in develop_accum.items() if v},
        "by_scene": {
            scene: {k: _mean(vals) for k, vals in sliders.items() if vals}
            for scene, sliders in develop_by_scene.items()
        },
    }

    # ── Signature edit — median slider values across the library ────────────
    _SIG_SLIDERS = [
        ("Exposure2012",  -5,   5,   "Exposure",   "tone"),
        ("Highlights2012",-100, 100, "Highlights", "tone"),
        ("Shadows2012",   -100, 100, "Shadows",    "tone"),
        ("Whites2012",    -100, 100, "Whites",     "tone"),
        ("Blacks2012",    -100, 100, "Blacks",     "tone"),
        ("Contrast2012",  -100, 100, "Contrast",   "tone"),
        ("Texture",       -100, 100, "Texture",    "presence"),
        ("Clarity2012",   -100, 100, "Clarity",    "presence"),
        ("Dehaze",        -100, 100, "Dehaze",     "presence"),
        ("Vibrance",      -100, 100, "Vibrance",   "color"),
        ("Saturation",    -100, 100, "Saturation", "color"),
    ]
    signature_edit: dict = {}
    for key, lo, hi, label, group in _SIG_SLIDERS:
        vals = [float(r["lightroom_develop"][key])
                for r in records
                if r.get("lightroom_develop", {}).get(key) is not None]
        if vals:
            med = float(np.median(vals))
            signature_edit[key] = {
                "label": label, "group": group,
                "median": round(med, 1),
                "std": round(float(np.std(vals)), 1),
                "min_range": lo, "max_range": hi,
                # normalised -1..+1 relative to the slider's full range
                "norm": round(med / hi if hi != 0 else 0, 3),
            }
    temp_vals = [float(r["lightroom_develop"]["Temperature"])
                 for r in records
                 if r.get("lightroom_develop", {}).get("Temperature") is not None]
    if temp_vals:
        med_temp = float(np.median(temp_vals))
        signature_edit["Temperature"] = {
            "label": "White Balance", "group": "color",
            "median": round(med_temp, 0),
            "std": round(float(np.std(temp_vals)), 0),
            "min_range": 2000, "max_range": 50000,
            "norm": round((med_temp - 5500) / 5500, 3),
        }

    # Derive plain-English narrative from signature values
    def _sig_narrative(sig: dict) -> str:
        parts = []
        hl = sig.get("Highlights2012", {}).get("median", 0)
        sh = sig.get("Shadows2012", {}).get("median", 0)
        if hl < -40 and sh > 30:
            parts.append(f"heavy tonal recovery — pulling highlights ({hl:+.0f}) while lifting shadows ({sh:+.0f}) for maximum dynamic range")
        elif hl < -20:
            parts.append(f"highlight recovery ({hl:+.0f})")
        elif sh > 20:
            parts.append(f"shadow lifting ({sh:+.0f})")
        vib = sig.get("Vibrance", {}).get("median", 0)
        sat = sig.get("Saturation", {}).get("median", 0)
        if vib > 20 and sat < 0:
            parts.append(f"rich selective colour — vibrance {vib:+.0f} over global saturation ({sat:+.0f}) keeps skin tones natural")
        elif vib > 15:
            parts.append(f"vivid colour (vibrance {vib:+.0f})")
        temp = sig.get("Temperature", {}).get("median", 5500)
        if temp > 6200:
            parts.append(f"warm white balance ({temp:.0f} K)")
        elif temp < 4800:
            parts.append(f"cool white balance ({temp:.0f} K)")
        exp = sig.get("Exposure2012", {}).get("median", 0)
        if abs(exp) > 0.4:
            parts.append(f"{'brightening' if exp > 0 else 'darkening'} exposure by {abs(exp):.1f} stop{'s' if abs(exp) >= 2 else ''}")
        if not parts:
            return ""
        return ". ".join(p.capitalize() for p in parts) + "."
    signature_edit["_narrative"] = _sig_narrative(signature_edit)

    # ── Monthly shooting distribution ────────────────────────────────────────
    monthly_counter: Counter = Counter()
    monthly_golden: Counter = Counter()
    for r in records:
        cap = r.get("lightroom_capture_date", "") or r.get("exif", {}).get("year_month", "")
        if cap and len(cap) >= 7:
            key = cap[:7]
            monthly_counter[key] += 1
            if _time_of_day(r) == "golden_hour":
                monthly_golden[key] += 1
    monthly_shooting = {
        m: {"total": monthly_counter[m], "golden_hour": monthly_golden.get(m, 0)}
        for m in sorted(monthly_counter)
    }

    # ── Lightroom ratings distribution ───────────────────────────────────────
    rating_counter: Counter = Counter()
    label_counter: Counter = Counter()
    keyword_counter: Counter = Counter()
    for r in records:
        rating = r.get("lightroom_rating")
        if rating is not None:
            rating_counter[int(rating)] += 1
        label = r.get("lightroom_label", "")
        if label:
            label_counter[label] += 1
        for kw in r.get("lightroom_keywords", []):
            if kw:
                keyword_counter[str(kw).lower()] += 1

    lightroom_stats = {
        "ratings": [{"stars": s, "count": rating_counter.get(s, 0)} for s in range(6)],
        "color_labels": [{"label": l, "count": c} for l, c in label_counter.most_common()],
        "top_keywords": [{"keyword": k, "count": c} for k, c in keyword_counter.most_common(30)],
        "total_with_lightroom": sum(1 for r in records if r.get("lightroom_id")),
    }

    # ── HSL channel fingerprint ───────────────────────────────────────────────
    HSL_CHANNELS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
    hsl_accum: dict[str, dict[str, list]] = {
        ch: {"hue": [], "sat": [], "lum": []} for ch in HSL_CHANNELS
    }
    for r in records:
        dev = r.get("lightroom_develop", {})
        if not dev:
            continue
        for ch in HSL_CHANNELS:
            h = dev.get(f"HueAdjustment{ch}")
            s = dev.get(f"SaturationAdjustment{ch}")
            l = dev.get(f"LuminanceAdjustment{ch}")
            if h is not None:
                hsl_accum[ch]["hue"].append(float(h))
            if s is not None:
                hsl_accum[ch]["sat"].append(float(s))
            if l is not None:
                hsl_accum[ch]["lum"].append(float(l))
    hsl_fingerprint = {
        ch: {
            "hue_adj": round(float(np.mean(d["hue"])), 2) if d["hue"] else 0.0,
            "sat_adj": round(float(np.mean(d["sat"])), 2) if d["sat"] else 0.0,
            "lum_adj": round(float(np.mean(d["lum"])), 2) if d["lum"] else 0.0,
        }
        for ch, d in hsl_accum.items()
    }

    # ── Editing intensity ─────────────────────────────────────────────────────
    _INTENSITY_DEFAULTS = {
        "Exposure2012": 0, "Contrast2012": 0, "Highlights2012": 0,
        "Shadows2012": 0, "Whites2012": 0, "Blacks2012": 0,
        "Texture": 0, "Clarity2012": 0, "Dehaze": 0,
        "Vibrance": 0, "Saturation": 0,
    }
    intensity_vals: list[float] = []
    intensity_by_scene: dict[str, list] = defaultdict(list)
    for r in records:
        dev = r.get("lightroom_develop", {})
        if not dev:
            continue
        score_i = sum(abs(float(dev[k]) - default) for k, default in _INTENSITY_DEFAULTS.items() if k in dev)
        intensity_vals.append(score_i)
        scene = r.get("scene", {}).get("scene_type")
        if scene:
            intensity_by_scene[scene].append(score_i)
    # 10-bucket histogram 0–500 (500 is a very heavily edited photo)
    intensity_hist = [0] * 10
    for v in intensity_vals:
        intensity_hist[min(9, int(v / 50))] += 1
    # aesthetic correlation: bucket intensity into deciles, avg aesthetic per decile
    intensity_aesthetic_corr: list[dict] = []
    if intensity_vals and a_scores:
        pairs = []
        for r in records:
            dev = r.get("lightroom_develop", {})
            if not dev:
                continue
            score_i = sum(abs(float(dev[k]) - default) for k, default in _INTENSITY_DEFAULTS.items() if k in dev)
            a = r.get("aesthetic_score")
            if a is not None:
                pairs.append((score_i, float(a)))
        if len(pairs) >= 2:
            int_arr = np.array([p[0] for p in pairs])
            aes_arr = np.array([p[1] for p in pairs])
            int_min, int_max = int_arr.min(), int_arr.max()
            if int_max > int_min:
                buckets_ia: dict[int, list] = defaultdict(list)
                for iv, av in zip(int_arr, aes_arr):
                    b = min(4, int((iv - int_min) / (int_max - int_min) * 5))
                    buckets_ia[b].append(av)
                for b in range(5):
                    vals_b = buckets_ia.get(b, [])
                    intensity_aesthetic_corr.append({
                        "bucket": b,
                        "label": ["Low", "Low-Mid", "Mid", "Mid-High", "High"][b],
                        "avg_aesthetic": round(float(np.mean(vals_b)), 2) if vals_b else None,
                        "count": len(vals_b),
                    })
    most_edited_scene = max(intensity_by_scene, key=lambda s: np.mean(intensity_by_scene[s]), default=None) if intensity_by_scene else None
    editing_intensity = {
        "avg": round(float(np.mean(intensity_vals)), 1) if intensity_vals else None,
        "std": round(float(np.std(intensity_vals)), 1) if intensity_vals else None,
        "histogram": intensity_hist,
        "by_scene": {s: round(float(np.mean(v)), 1) for s, v in intensity_by_scene.items() if v},
        "most_edited_scene": most_edited_scene,
        "aesthetic_by_intensity": intensity_aesthetic_corr,
    }

    # ── Pick / reject analysis ────────────────────────────────────────────────
    pick_buckets: dict[int, dict] = {1: {"aesthetics": [], "sharpness": [], "scenes": []},
                                      0: {"aesthetics": [], "sharpness": [], "scenes": []},
                                     -1: {"aesthetics": [], "sharpness": [], "scenes": []}}
    pick_by_scene: dict[str, dict[str, int]] = defaultdict(lambda: {"picks": 0, "total": 0})
    for r in records:
        pk = r.get("lightroom_pick")
        if pk is None:
            continue
        pk = int(pk)
        if pk not in pick_buckets:
            continue
        a = r.get("aesthetic_score")
        sh = r.get("composition", {}).get("sharpness_score")
        sc = r.get("scene", {}).get("scene_type")
        if a is not None:
            pick_buckets[pk]["aesthetics"].append(a)
        if sh is not None:
            pick_buckets[pk]["sharpness"].append(sh)
        if sc:
            pick_buckets[pk]["scenes"].append(sc)
            pick_by_scene[sc]["total"] += 1
            if pk == 1:
                pick_by_scene[sc]["picks"] += 1

    def _bucket_summary(data: dict) -> dict:
        aes = data["aesthetics"]
        sh = data["sharpness"]
        scenes = data["scenes"]
        return {
            "count": len(aes) or len(sh) or len(scenes),
            "avg_aesthetic": round(float(np.mean(aes)), 2) if aes else None,
            "avg_sharpness": round(float(np.mean(sh)), 2) if sh else None,
            "top_scene": Counter(scenes).most_common(1)[0][0] if scenes else None,
        }

    pick_stats = {
        "picks": _bucket_summary(pick_buckets[1]),
        "neutral": _bucket_summary(pick_buckets[0]),
        "rejects": _bucket_summary(pick_buckets[-1]),
        "pick_rate_by_scene": {
            sc: round(d["picks"] / d["total"] * 100, 1)
            for sc, d in pick_by_scene.items() if d["total"] > 0
        },
    }

    # ── Burst / near-duplicate detection (DBSCAN, cosine distance ≤ 0.05) ─────
    burst_groups: dict = {}
    storage_tiers: dict = {}
    dino_records = [(r["hash"], r["dinov2"], r.get("aesthetic_score") or 0,
                     r.get("scene", {}).get("scene_type", ""), r.get("path", ""))
                    for r in records if r.get("dinov2")]
    if len(dino_records) >= 2:
        hashes_d, vecs_d, scores_d, scenes_d, paths_d = zip(*dino_records)
        mat_d = np.array(vecs_d, dtype=np.float32)
        norms_d = np.linalg.norm(mat_d, axis=1, keepdims=True) + 1e-9
        mat_d = mat_d / norms_d
        sim_d = mat_d @ mat_d.T
        dist_d = np.clip(1.0 - sim_d, 0, 2).astype(np.float64)
        labels_d = DBSCAN(eps=0.05, min_samples=2, metric="precomputed").fit_predict(dist_d)

        cluster_ids = [l for l in set(labels_d) if l != -1]
        groups_d = []
        for cid in cluster_ids:
            idxs = [i for i, l in enumerate(labels_d) if l == cid]
            best = max(idxs, key=lambda i: scores_d[i])
            redundant = [paths_d[i] for i in idxs if i != best]
            groups_d.append({
                "size": len(idxs),
                "best_path": paths_d[best],
                "best_score": round(float(scores_d[best]), 2),
                "scene": scenes_d[best],
                "redundant_paths": redundant[:3],
            })
        groups_d.sort(key=lambda g: -g["size"])

        burst_groups = {
            "total_bursts": len(groups_d),
            "photos_in_bursts": sum(g["size"] for g in groups_d),
            "largest_burst": groups_d[0]["size"] if groups_d else 0,
            "groups": [{k: v for k, v in g.items() if k != "redundant_paths"} for g in groups_d[:10]],
        }

        total_w_emb = len(dino_records)
        redundant_count = sum(g["size"] - 1 for g in groups_d)
        storage_tiers = {
            "cluster_count": len(groups_d),
            "hero_count": len(groups_d),
            "redundant_count": redundant_count,
            "unclustered_count": int(np.sum(labels_d == -1)),
            "redundancy_pct": round(redundant_count / total_w_emb * 100, 1) if total_w_emb else 0.0,
            "tiers": groups_d[:10],
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
            captions = [r.get("caption", {}) for r in ev_records if r.get("caption")]
            if not captions:
                return ""
            # pick up to 3 most diverse by DINOv2, fallback to top aesthetic
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
            people = [c.get("people") for c in rep_caps if c.get("people")]
            tod = next((c.get("time_of_day") for c in rep_caps if c.get("time_of_day")), None)
            weather = next((c.get("weather") for c in rep_caps if c.get("weather")), None)
            setting_str = settings[0] if settings else "mixed"
            people_str = people[0] if people else ""
            parts = [f"A {setting_str} shoot"]
            if people_str:
                parts.append(people_str)
            if tod:
                parts.append(f"during {tod}")
            if weather and weather.lower() not in ("clear", "unknown"):
                parts.append(f"in {weather} conditions")
            return ". ".join(parts) + "."

        event_list = []
        for idx, group in enumerate(event_groups):
            ts_list = [t for t, _ in group]
            ev_records = [r for _, r in group]
            duration_mins = (ts_list[-1] - ts_list[0]).total_seconds() / 60
            scenes_ev = [r.get("scene", {}).get("scene_type") for r in ev_records if r.get("scene")]
            top_scene_ev = Counter(s for s in scenes_ev if s).most_common(1)
            hero_r = max(ev_records, key=lambda r: r.get("aesthetic_score") or 0)
            event_list.append({
                "event_id": idx + 1,
                "date": ts_list[0].strftime("%Y-%m-%d"),
                "photo_count": len(ev_records),
                "duration_mins": round(duration_mins, 1),
                "top_scene": top_scene_ev[0][0] if top_scene_ev else "",
                "hero_path": hero_r.get("path", ""),
                "hero_score": round(float(hero_r.get("aesthetic_score") or 0), 2),
                "narrative": _event_narrative(ev_records),
            })
        event_list.sort(key=lambda e: e["date"], reverse=True)
        events = {
            "total_events": len(event_list),
            "avg_photos_per_event": round(sum(e["photo_count"] for e in event_list) / len(event_list), 1) if event_list else 0,
            "largest_event": max((e["photo_count"] for e in event_list), default=0),
            "events": event_list,
        }

    # ── ELA stats ─────────────────────────────────────────────────────────────
    ela_records = [r.get("ela", {}) for r in records if r.get("ela")]
    ela_jpegs = [e for e in ela_records if e.get("ela_max_error") is not None]
    ela_stats = {
        "total_jpegs": len(ela_jpegs),
        "suspicious_count": sum(1 for e in ela_jpegs if e.get("ela_suspicious")),
        "suspicious_pct": round(sum(1 for e in ela_jpegs if e.get("ela_suspicious")) / len(ela_jpegs) * 100, 1) if ela_jpegs else 0.0,
        "avg_max_error": round(float(np.mean([e["ela_max_error"] for e in ela_jpegs])), 2) if ela_jpegs else None,
        "avg_mean_error": round(float(np.mean([e["ela_mean_error"] for e in ela_jpegs])), 4) if ela_jpegs else None,
    }

    # ── Keyword topic map (extended from lightroom_stats) ────────────────────
    kw_by_scene: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        sc = r.get("scene", {}).get("scene_type")
        for kw in r.get("lightroom_keywords", []):
            if kw and sc:
                kw_by_scene[sc][str(kw).lower()] += 1
    keyword_map = {
        "top_keywords": [{"word": k, "count": c} for k, c in keyword_counter.most_common(30)],
        "by_scene": {
            sc: [{"word": k, "count": c} for k, c in ctr.most_common(5)]
            for sc, ctr in kw_by_scene.items() if sum(ctr.values()) > 0
        },
    }

    # ── Saliency stats ────────────────────────────────────────────────────────
    sal_areas = [r.get("saliency", {}).get("subject_area_pct") for r in records]
    sal_off = [r.get("saliency", {}).get("subject_off_center") for r in records]
    sal_by_scene: dict[str, list] = defaultdict(list)
    placement_grid = [[0] * 3 for _ in range(3)]  # 3×3 zone counts
    for r in records:
        sal = r.get("saliency", {})
        if not sal or sal.get("subject_cx") is None:
            continue
        sc = r.get("scene", {}).get("scene_type")
        area = sal.get("subject_area_pct")
        if area is not None and sc:
            sal_by_scene[sc].append(area)
        cx = sal.get("subject_cx")
        cy = sal.get("subject_cy")
        if cx is not None and cy is not None:
            col_z = min(2, int(cx * 3))
            row_z = min(2, int(cy * 3))
            placement_grid[row_z][col_z] += 1
    # normalise placement grid
    total_placed = sum(sum(row) for row in placement_grid) or 1
    placement_grid_norm = [[round(v / total_placed, 4) for v in row] for row in placement_grid]
    saliency_stats = {
        "avg_subject_area": _mean([v for v in sal_areas if v is not None]),
        "avg_off_center": _mean([v for v in sal_off if v is not None]),
        "by_scene": {sc: round(float(np.mean(v)), 4) for sc, v in sal_by_scene.items() if v},
        "placement_distribution": placement_grid_norm,
        "total_analyzed": sum(1 for v in sal_areas if v is not None),
    }

    # ── Album / curation stats ────────────────────────────────────────────────
    album_records = [r for r in records if r.get("lightroom_album_names") is not None]
    in_album = [r for r in album_records if r.get("lightroom_album_names")]
    not_in_album = [r for r in album_records if not r.get("lightroom_album_names")]

    album_counter: dict[str, list] = defaultdict(list)
    for r in in_album:
        for name in r["lightroom_album_names"]:
            album_counter[name].append(r)

    album_breakdown = []
    for name, recs in sorted(album_counter.items(), key=lambda x: -len(x[1])):
        aes = [r.get("aesthetic_score") for r in recs if r.get("aesthetic_score") is not None]
        scenes = [r.get("scene", {}).get("scene_type") for r in recs if r.get("scene")]
        album_breakdown.append({
            "name": name,
            "count": len(recs),
            "avg_aesthetic": round(float(np.mean(aes)), 2) if aes else None,
            "top_scene": Counter(s for s in scenes if s).most_common(1)[0][0] if scenes else None,
        })

    scene_in_ctr: Counter = Counter()
    scene_total_ctr: Counter = Counter()
    for r in album_records:
        sc = r.get("scene", {}).get("scene_type")
        if sc:
            scene_total_ctr[sc] += 1
            if r.get("lightroom_album_names"):
                scene_in_ctr[sc] += 1
    scene_curation_rate = {
        sc: round(scene_in_ctr[sc] / scene_total_ctr[sc] * 100, 1)
        for sc in scene_total_ctr if scene_total_ctr[sc] > 0
    }

    in_aes = [r.get("aesthetic_score") for r in in_album if r.get("aesthetic_score") is not None]
    out_aes = [r.get("aesthetic_score") for r in not_in_album if r.get("aesthetic_score") is not None]

    album_stats = {
        "total_albums": len(album_counter),
        "total_in_albums": len(in_album),
        "total_trackable": len(album_records),
        "curation_rate": round(len(in_album) / len(album_records) * 100, 1) if album_records else 0.0,
        "avg_aesthetic_in": round(float(np.mean(in_aes)), 2) if in_aes else None,
        "avg_aesthetic_out": round(float(np.mean(out_aes)), 2) if out_aes else None,
        "albums": album_breakdown,
        "scene_curation_rate": scene_curation_rate,
    }

    return {
        "photo_count": len(records),
        "color_stats": color_stats,
        "composition_stats": composition_stats,
        "exif_stats": exif_stats,
        "aesthetic_stats": aesthetic_stats,
        "scene_stats": scene_stats,
        "clusters": clusters_out,
        "umap": umap_out,
        "editing_consistency": editing_consistency,
        # new
        "shooting_hours": shooting_hours,
        "focal_length_histogram": focal_length_histogram,
        "aperture_histogram": aperture_histogram,
        "iso_histogram": iso_histogram,
        "aesthetic_by_scene": aesthetic_by_scene,
        "composition_by_scene": composition_by_scene,
        "grid_heatmap": grid_heatmap,
        "sharpness_stats": sharpness_stats,
        "exposure_stats": exposure_stats,
        "folder_breakdown": folder_breakdown,
        "scene_confidence": scene_confidence,
        "megapixel_stats": megapixel_stats,
        "color_by_scene": color_by_scene,
        "hue_distribution": hue_distribution,
        "saturation_histogram": saturation_histogram,
        "editing_trends": editing_trends,
        "editing_style_patterns": editing_style_patterns,
        "color_grading_stats": color_grading_stats,
        "composition_patterns": composition_patterns,
        "depth_stats": depth_stats,
        "visual_attributes": visual_attributes,
        "develop_stats": develop_stats,
        "lightroom_stats": lightroom_stats,
        "signature_edit": signature_edit,
        "monthly_shooting": monthly_shooting,
        # tier-1 + saliency additions
        "hsl_fingerprint": hsl_fingerprint,
        "editing_intensity": editing_intensity,
        "pick_stats": pick_stats,
        "burst_groups": burst_groups,
        "keyword_map": keyword_map,
        "saliency_stats": saliency_stats,
        "storage_tiers": storage_tiers,
        "events": events,
        "ela_stats": ela_stats,
        "album_stats": album_stats,
    }

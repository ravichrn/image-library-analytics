import colorsys
from collections import Counter, defaultdict

import numpy as np

from extractors.scene import SCENE_LABELS

from ._helpers import _cv, _dist, _horizon_bucket, _mean, _scene_types_list, _score_bucket, _std, _time_of_day


def analyze(records: list[dict]) -> dict:
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
    def ccol(key):
        return [r.get("composition", {}).get(key) for r in records]

    def ecol(key):
        return [r.get("exif", {}).get(key) for r in records]

    thirds = ccol("thirds_score")
    syms = ccol("symmetry_score")
    neg_spaces = ccol("negative_space")
    horizons = ccol("horizon_position")
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
    exif_stats = {
        "time_of_day_distribution": _dist(
            [_time_of_day(r) for r in records],
            ["golden_hour", "midday", "morning_evening", "night"],
        ),
        "focal_category_distribution": _dist(ecol("focal_category"), ["wide", "normal", "telephoto"]),
        "dof_category_distribution": _dist(ecol("dof_category"), ["shallow", "mid", "deep"]),
        "light_category_distribution": _dist(ecol("light_category"), ["bright", "indoor", "low_light"]),
    }

    # ── aesthetic stats ──────────────────────────────────────────────────────
    a_scores = [r.get("aesthetic_score") for r in records if r.get("aesthetic_score") is not None]
    score_dist = Counter(_score_bucket(v) for v in a_scores)
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

    # ── scene stats (multi-label: one photo can belong to multiple scenes) ───
    # scene_distribution values = count of photos carrying that label (not normalised).
    # Percentages in the report use photo_count as denominator, so bars can sum >100%.
    _scene_label_counts: Counter = Counter()
    for r in records:
        for sc in _scene_types_list(r):
            _scene_label_counts[sc] += 1
    scene_dist = {sc: _scene_label_counts.get(sc, 0) for sc in SCENE_LABELS if _scene_label_counts.get(sc, 0) > 0}
    # dominant_scenes: all labels within 10% of the highest count, capped at 3.
    # Multi-label libraries can have genuinely tied or near-tied top scenes.
    if scene_dist:
        max_count = max(scene_dist.values())
        dominant_scenes = [sc for sc, cnt in sorted(scene_dist.items(), key=lambda x: -x[1]) if cnt >= max_count * 0.9][:3]
    else:
        dominant_scenes = []
    scene_stats = {"scene_distribution": scene_dist, "dominant_scenes": dominant_scenes}

    # ── editing consistency ──────────────────────────────────────────────────
    sat_cv = _cv(sats)
    con_cv = _cv(contrasts)
    bri_cv = _cv(brights)
    avg_cv = np.mean([v for v in [sat_cv, con_cv, bri_cv] if v is not None]) if any(v is not None for v in [sat_cv, con_cv, bri_cv]) else None

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

    # ── raw technical histograms ─────────────────────────────────────────────
    focal_length_histogram = [v for v in ecol("focal_length_mm") if v is not None]
    aperture_histogram = [v for v in ecol("aperture_f") if v is not None]
    iso_histogram = [v for v in ecol("iso") if v is not None]

    # ── aesthetic by scene (multi-label: photo contributes to all its scenes) ─
    scene_scores: dict = defaultdict(list)
    for r in records:
        score = r.get("aesthetic_score")
        if score is None:
            continue
        for scene in _scene_types_list(r):
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
        comp = r.get("composition", {})
        for scene in _scene_types_list(r):
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
        s = r.get("composition", {}).get("sharpness_score")
        if s is None:
            continue
        for scene in _scene_types_list(r):
            sharpness_by_scene[scene].append(s)

    sharpness_stats = {
        "avg": _mean(sharpness_vals),
        "std": _std(sharpness_vals),
        "by_scene": {scene: round(float(np.mean(vals)), 2) for scene, vals in sharpness_by_scene.items() if vals},
    }

    # ── exposure stats ───────────────────────────────────────────────────────
    highlights = [r.get("composition", {}).get("highlight_clipping") for r in records]
    shadows = [r.get("composition", {}).get("shadow_clipping") for r in records]
    biases = [r.get("composition", {}).get("exposure_bias") for r in records]
    highlight_threshold = 0.02
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
    folder_data: dict = defaultdict(lambda: {"scores": [], "sats": []})
    for r in records:
        path = r.get("path", "")
        parts = path.replace("\\", "/").split("/")
        folder = parts[-2] if len(parts) >= 2 else "root"
        score = r.get("aesthetic_score")
        sat = r.get("color", {}).get("avg_saturation")
        if score is not None:
            folder_data[folder]["scores"].append(score)
        if sat is not None:
            folder_data[folder]["sats"].append(sat)

    folder_breakdown = {}
    for folder, data in sorted(folder_data.items(), key=lambda x: len(x[1]["scores"]), reverse=True):
        folder_breakdown[folder] = {
            "count": len(data["scores"]) + (len(folder_data[folder]["sats"]) - len(data["scores"])),
            "photo_count": sum(
                1
                for r in records
                if (r.get("path", "").replace("\\", "/").split("/")[-2] if len(r.get("path", "").replace("\\", "/").split("/")) >= 2 else "root") == folder
            ),
            "avg_aesthetic": round(float(np.mean(data["scores"])), 2) if data["scores"] else None,
            "avg_saturation": round(float(np.mean(data["sats"])), 3) if data["sats"] else None,
        }

    # ── scene confidence ─────────────────────────────────────────────────────
    scene_conf_data: dict = defaultdict(list)
    for r in records:
        scores_dict = r.get("scene", {}).get("scene_scores", {})
        for scene in _scene_types_list(r):
            if scene in scores_dict:
                scene_conf_data[scene].append(scores_dict[scene])

    scene_confidence = {scene: round(float(np.mean(vals)), 4) for scene, vals in scene_conf_data.items() if vals}

    # ── megapixel stats ──────────────────────────────────────────────────────
    mp_vals = [r.get("exif", {}).get("megapixels") for r in records]
    clean_mp = [v for v in mp_vals if v is not None]
    megapixel_stats = {
        "avg": round(float(np.mean(clean_mp)), 2) if clean_mp else None,
        "min": round(float(np.min(clean_mp)), 2) if clean_mp else None,
        "max": round(float(np.max(clean_mp)), 2) if clean_mp else None,
    }

    # ── color by scene (multi-label) ──────────────────────────────────────────
    color_by_scene_data: dict = defaultdict(lambda: {"sats": [], "warm": [], "warmths": [], "brightness": [], "aesthetics": [], "contrasts": []})
    for r in records:
        col_d = r.get("color", {})
        a = r.get("aesthetic_score")
        for scene in _scene_types_list(r):
            for key2, bucket in [
                ("avg_saturation", "sats"),
                ("warm_ratio", "warm"),
                ("warmth", "warmths"),
                ("avg_brightness", "brightness"),
                ("contrast", "contrasts"),
            ]:
                v = col_d.get(key2)
                if v is not None:
                    color_by_scene_data[scene][bucket].append(v)
            if a is not None:
                color_by_scene_data[scene]["aesthetics"].append(a)

    color_by_scene = {}
    for scene, data in color_by_scene_data.items():
        dominant_warmth = Counter(data["warmths"]).most_common(1)[0][0] if data["warmths"] else "unknown"
        aes_s = aesthetic_by_scene.get(scene, {})
        color_by_scene[scene] = {
            "avg_saturation": round(float(np.mean(data["sats"])), 3) if data["sats"] else None,
            "avg_brightness": round(float(np.mean(data["brightness"])), 3) if data["brightness"] else None,
            "avg_contrast": round(float(np.mean(data["contrasts"])), 3) if data["contrasts"] else None,
            "avg_aesthetic": round(float(np.mean(data["aesthetics"])), 2) if data["aesthetics"] else None,
            "aesthetic_std": round(float(aes_s["std"]), 2) if aes_s.get("std") is not None else None,
            "avg_warmth_ratio": round(float(np.mean(data["warm"])), 3) if data["warm"] else None,
            "dominant_warmth": dominant_warmth,
            "count": len(data["sats"]),
        }

    # ── color harmony distribution ───────────────────────────────────────────
    def _classify_harmony_palette(palette) -> str:
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
        harmony_counter[_classify_harmony_palette(r.get("color", {}).get("palette", []))] += 1

    # ── tonal / color grading overview ───────────────────────────────────────
    avg_bri = color_stats.get("avg_brightness") or 0
    avg_con = color_stats.get("avg_contrast") or 0
    avg_wr = color_stats.get("avg_warmth_ratio") or 0
    avg_cr = color_stats.get("avg_cool_ratio") or 0

    shadow_pcts = [r.get("composition", {}).get("tonal_shadow_pct") for r in records]
    mid_pcts_tonal = [r.get("composition", {}).get("tonal_mid_pct") for r in records]
    hi_pcts = [r.get("composition", {}).get("tonal_highlight_pct") for r in records]
    avg_sp = float(np.mean([v for v in shadow_pcts if v is not None])) if any(v is not None for v in shadow_pcts) else None
    avg_mp2 = float(np.mean([v for v in mid_pcts_tonal if v is not None])) if any(v is not None for v in mid_pcts_tonal) else None
    avg_hp = float(np.mean([v for v in hi_pcts if v is not None])) if any(v is not None for v in hi_pcts) else None

    if avg_bri >= 0.60:
        tonal_style, tonal_desc = "high_key", "Bright and airy — photos expose for the highlights, open shadows."
    elif avg_bri <= 0.38:
        tonal_style, tonal_desc = "low_key", "Dark and dramatic — heavy shadows dominate the frame."
    elif avg_con >= 0.38:
        tonal_style, tonal_desc = "moody", "Mid-tone brightness with elevated contrast — brooding, cinematic look."
    elif avg_con <= 0.22:
        tonal_style, tonal_desc = "flat_matte", "Compressed tonal range — lifted shadows, rolled highlights; the 'matte look'."
    else:
        tonal_style, tonal_desc = "balanced", "Balanced exposure and contrast — no dominant tonal treatment."

    if avg_wr > 0.52:
        color_temp, color_temp_desc = "warm", "Shifted toward orange/amber (~3000–5000K equivalent). Golden-hour or warm-preset tendency."
    elif avg_cr > 0.42:
        color_temp, color_temp_desc = "cool", "Shifted toward blue/cyan (~6500K+ equivalent). Shade, overcast, or cool-preset tendency."
    else:
        color_temp, color_temp_desc = "neutral", "No strong colour temperature bias (~5500K). Daylight-balanced or mixed conditions."

    dominant_harmony = harmony_counter.most_common(1)[0][0] if harmony_counter else "unknown"
    harmony_total = sum(harmony_counter.values()) or 1

    color_harmony_dist = dict(harmony_counter)

    color_grading_stats = {
        "tonal_style": tonal_style,
        "tonal_description": tonal_desc,
        "color_temperature": color_temp,
        "color_temperature_description": color_temp_desc,
        "avg_tonal_shadow_pct": round(avg_sp, 3) if avg_sp is not None else None,
        "avg_tonal_mid_pct": round(avg_mp2, 3) if avg_mp2 is not None else None,
        "avg_tonal_highlight_pct": round(avg_hp, 3) if avg_hp is not None else None,
        "color_harmony_dist": color_harmony_dist,
        "dominant_harmony": dominant_harmony,
        "harmony_pcts": {k: round(v / harmony_total * 100, 1) for k, v in harmony_counter.most_common()},
        # split_toning populated by lightroom module when available
        "split_toning": None,
    }

    return {
        "color_stats": color_stats,
        "composition_stats": composition_stats,
        "exif_stats": exif_stats,
        "aesthetic_stats": aesthetic_stats,
        "scene_stats": scene_stats,
        "editing_consistency": editing_consistency,
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
        "color_grading_stats": color_grading_stats,
    }

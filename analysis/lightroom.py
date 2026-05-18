import colorsys
from collections import Counter, defaultdict

import numpy as np

from ._helpers import _INTENSITY_DEFAULTS, _TONING_KEYS, _hsl_to_hex, _mean


def analyze(records: list[dict]) -> dict:
    # ── Lightroom ratings / labels / keywords ────────────────────────────────
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
        "color_labels": [{"label": ll, "count": c} for ll, c in label_counter.most_common()],
        "top_keywords": [{"keyword": k, "count": c} for k, c in keyword_counter.most_common(30)],
        "total_with_lightroom": sum(1 for r in records if r.get("lightroom_id")),
    }

    # ── Develop stats ────────────────────────────────────────────────────────
    DEVELOP_KEYS = [
        "Exposure",
        "Contrast",
        "Highlights",
        "Shadows",
        "Whites",
        "Blacks",
        "Clarity",
        "Vibrance",
        "Saturation",
        "Sharpness",
        "LuminanceSmoothing",
        "ColorNoiseReduction",
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
        "by_scene": {scene: {k: _mean(vals) for k, vals in sliders.items() if vals} for scene, sliders in develop_by_scene.items()},
    }

    # ── Split toning / color grading from develop settings ──────────────────
    toning_accum: dict[str, list] = {k: [] for k in _TONING_KEYS}
    for r in records:
        dev = r.get("lightroom_develop", {})
        for k in _TONING_KEYS:
            v = dev.get(k)
            if v is not None:
                toning_accum[k].append(float(v))

    def _tavg_local(key: str) -> float | None:
        return _mean(toning_accum.get(key, []))

    def _hue_name_local(hue: float | None) -> str | None:
        if hue is None:
            return None
        h = hue % 360
        if h < 20 or h >= 340:
            return "red"
        if h < 45:
            return "orange"
        if h < 70:
            return "yellow"
        if h < 150:
            return "green"
        if h < 200:
            return "cyan/teal"
        if h < 260:
            return "blue"
        if h < 290:
            return "purple"
        return "magenta"

    _sh_sat = _tavg_local("SplitToningShadowSaturation") or 0
    _hl_sat = _tavg_local("SplitToningHighlightSaturation") or 0
    sh_name = _hue_name_local(_tavg_local("SplitToningShadowHue")) if _sh_sat > 3 else None
    hl_name = _hue_name_local(_tavg_local("SplitToningHighlightHue")) if _hl_sat > 3 else None
    if sh_name and hl_name:
        toning_style = f"{hl_name.capitalize()} highlights · {sh_name} shadows"
    elif sh_name:
        toning_style = f"{sh_name.capitalize()} shadows"
    elif hl_name:
        toning_style = f"{hl_name.capitalize()} highlights"
    else:
        toning_style = "Neutral (no toning)"

    split_toning = {
        "shadow_color": _hsl_to_hex(_tavg_local("SplitToningShadowHue"), _tavg_local("SplitToningShadowSaturation"), 0.22),
        "midtone_color": _hsl_to_hex(_tavg_local("ColorGradeMidtoneHue"), _tavg_local("ColorGradeMidtoneSat"), 0.50),
        "highlight_color": _hsl_to_hex(_tavg_local("SplitToningHighlightHue"), _tavg_local("SplitToningHighlightSaturation"), 0.78),
        "toning_style": toning_style,
        "avg_balance": round(_tavg_local("SplitToningBalance") or 0, 1),
        "shadow_lum_shift": round(_tavg_local("ColorGradeShadowLum") or 0, 1),
        "midtone_lum_shift": round(_tavg_local("ColorGradeMidtoneLum") or 0, 1),
        "highlight_lum_shift": round(_tavg_local("ColorGradeHighlightLum") or 0, 1),
    }

    # ── Signature edit ───────────────────────────────────────────────────────
    _SIG_SLIDERS_DEF = [
        ("Exposure2012", -5, 5, "Exposure", "tone"),
        ("Highlights2012", -100, 100, "Highlights", "tone"),
        ("Shadows2012", -100, 100, "Shadows", "tone"),
        ("Whites2012", -100, 100, "Whites", "tone"),
        ("Blacks2012", -100, 100, "Blacks", "tone"),
        ("Contrast2012", -100, 100, "Contrast", "tone"),
        ("Texture", -100, 100, "Texture", "presence"),
        ("Clarity2012", -100, 100, "Clarity", "presence"),
        ("Dehaze", -100, 100, "Dehaze", "presence"),
        ("Vibrance", -100, 100, "Vibrance", "color"),
        ("Saturation", -100, 100, "Saturation", "color"),
    ]
    signature_edit: dict = {}
    for key, lo, hi, label, group in _SIG_SLIDERS_DEF:
        vals = [float(r["lightroom_develop"][key]) for r in records if r.get("lightroom_develop", {}).get(key) is not None]
        if vals:
            med = float(np.median(vals))
            signature_edit[key] = {
                "label": label,
                "group": group,
                "median": round(med, 1),
                "std": round(float(np.std(vals)), 1),
                "min_range": lo,
                "max_range": hi,
                "norm": round(med / hi if hi != 0 else 0, 3),
            }
    temp_vals = [float(r["lightroom_develop"]["Temperature"]) for r in records if r.get("lightroom_develop", {}).get("Temperature") is not None]
    if temp_vals:
        med_temp = float(np.median(temp_vals))
        signature_edit["Temperature"] = {
            "label": "White Balance",
            "group": "color",
            "median": round(med_temp, 0),
            "std": round(float(np.std(temp_vals)), 0),
            "min_range": 2000,
            "max_range": 50000,
            "norm": round((med_temp - 5500) / 5500, 3),
        }

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

    # ── HSL channel fingerprint ───────────────────────────────────────────────
    HSL_CHANNELS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
    hsl_accum: dict[str, dict[str, list]] = {ch: {"hue": [], "sat": [], "lum": []} for ch in HSL_CHANNELS}
    for r in records:
        dev = r.get("lightroom_develop", {})
        if not dev:
            continue
        for ch in HSL_CHANNELS:
            h = dev.get(f"HueAdjustment{ch}")
            s = dev.get(f"SaturationAdjustment{ch}")
            lv = dev.get(f"LuminanceAdjustment{ch}")
            if h is not None:
                hsl_accum[ch]["hue"].append(float(h))
            if s is not None:
                hsl_accum[ch]["sat"].append(float(s))
            if lv is not None:
                hsl_accum[ch]["lum"].append(float(lv))
    hsl_fingerprint = {
        ch: {
            "hue_adj": round(float(np.mean(d["hue"])), 2) if d["hue"] else 0.0,
            "sat_adj": round(float(np.mean(d["sat"])), 2) if d["sat"] else 0.0,
            "lum_adj": round(float(np.mean(d["lum"])), 2) if d["lum"] else 0.0,
        }
        for ch, d in hsl_accum.items()
    }

    # ── Editing intensity ─────────────────────────────────────────────────────
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
    intensity_hist = [0] * 10
    for v in intensity_vals:
        intensity_hist[min(9, int(v / 50))] += 1

    # Aesthetic correlation by intensity decile
    a_scores_lr = [r.get("aesthetic_score") for r in records if r.get("aesthetic_score") is not None]
    intensity_aesthetic_corr: list[dict] = []
    if intensity_vals and a_scores_lr:
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
                for iv, av in zip(int_arr, aes_arr, strict=False):
                    b = min(4, int((iv - int_min) / (int_max - int_min) * 5))
                    buckets_ia[b].append(av)
                for b in range(5):
                    vals_b = buckets_ia.get(b, [])
                    intensity_aesthetic_corr.append(
                        {
                            "bucket": b,
                            "label": ["Low", "Low-Mid", "Mid", "Mid-High", "High"][b],
                            "avg_aesthetic": round(float(np.mean(vals_b)), 2) if vals_b else None,
                            "count": len(vals_b),
                        }
                    )

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
    pick_buckets: dict[int, dict] = {
        1: {"aesthetics": [], "sharpness": [], "scenes": []},
        0: {"aesthetics": [], "sharpness": [], "scenes": []},
        -1: {"aesthetics": [], "sharpness": [], "scenes": []},
    }
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

    def _bucket_summary_lr(data: dict) -> dict:
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
        "picks": _bucket_summary_lr(pick_buckets[1]),
        "neutral": _bucket_summary_lr(pick_buckets[0]),
        "rejects": _bucket_summary_lr(pick_buckets[-1]),
        "pick_rate_by_scene": {sc: round(d["picks"] / d["total"] * 100, 1) for sc, d in pick_by_scene.items() if d["total"] > 0},
    }

    # ── Keyword topic map ────────────────────────────────────────────────────
    kw_by_scene: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        sc = r.get("scene", {}).get("scene_type")
        for kw in r.get("lightroom_keywords", []):
            if kw and sc:
                kw_by_scene[sc][str(kw).lower()] += 1
    keyword_map = {
        "top_keywords": [{"word": k, "count": c} for k, c in keyword_counter.most_common(30)],
        "by_scene": {sc: [{"word": k, "count": c} for k, c in ctr.most_common(5)] for sc, ctr in kw_by_scene.items() if sum(ctr.values()) > 0},
    }

    # ── Album / curation stats ────────────────────────────────────────────────
    album_records = [r for r in records if r.get("lightroom_album_names") is not None]
    in_album = [r for r in album_records if r.get("lightroom_album_names")]
    not_in_album = [r for r in album_records if not r.get("lightroom_album_names")]

    album_counter_data: dict[str, list] = defaultdict(list)
    for r in in_album:
        for name in r["lightroom_album_names"]:
            album_counter_data[name].append(r)

    def _album_hsl_to_hex(hue_deg: float | None, sat_pct: float | None, lum: float = 0.5) -> str | None:
        if hue_deg is None or sat_pct is None or sat_pct < 2:
            return None
        r2, g2, b2 = colorsys.hls_to_rgb(hue_deg / 360, lum, sat_pct / 100)
        return f"#{int(r2 * 255):02x}{int(g2 * 255):02x}{int(b2 * 255):02x}"

    album_breakdown = []
    for name, recs in sorted(album_counter_data.items(), key=lambda x: -len(x[1])):
        aes = [r.get("aesthetic_score") for r in recs if r.get("aesthetic_score") is not None]
        sats_a = [r.get("color", {}).get("avg_saturation") for r in recs if (r.get("color") or {}).get("avg_saturation") is not None]
        brights_a = [r.get("color", {}).get("avg_brightness") for r in recs if (r.get("color") or {}).get("avg_brightness") is not None]
        warm_a = [r.get("color", {}).get("warm_ratio") for r in recs if (r.get("color") or {}).get("warm_ratio") is not None]
        merged_palette: dict[tuple, float] = {}
        for r in recs:
            for sw in (r.get("color") or {}).get("palette") or []:
                rgb = tuple(int(x) for x in sw["rgb"])
                weight = sw.get("weight", 0)
                best_key = None
                best_dist = 30
                for k in merged_palette:
                    d = (sum((a2 - b2) ** 2 for a2, b2 in zip(rgb, k, strict=False))) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_key = k
                if best_key:
                    merged_palette[best_key] += weight
                else:
                    merged_palette[rgb] = weight
        top_colors = ["#{:02x}{:02x}{:02x}".format(*k) for k in sorted(merged_palette, key=lambda k: -merged_palette[k])[:6]]

        def _tonal_vals(field, rec_list):
            return [float(r["lightroom_develop"][field]) for r in rec_list if r.get("lightroom_develop", {}).get(field) is not None]

        sh_hues_a = _tonal_vals("SplitToningShadowHue", recs)
        sh_sats_a = _tonal_vals("SplitToningShadowSaturation", recs)
        hl_hues_a = _tonal_vals("SplitToningHighlightHue", recs)
        hl_sats_a = _tonal_vals("SplitToningHighlightSaturation", recs)
        mt_hues_a = _tonal_vals("ColorGradeMidtoneHue", recs)
        mt_sats_a = _tonal_vals("ColorGradeMidtoneSat", recs)
        tonal_colors = {
            "shadow": _album_hsl_to_hex(_mean(sh_hues_a), _mean(sh_sats_a), 0.25),
            "midtone": _album_hsl_to_hex(_mean(mt_hues_a), _mean(mt_sats_a), 0.50),
            "highlight": _album_hsl_to_hex(_mean(hl_hues_a), _mean(hl_sats_a), 0.75),
        }
        intensities_a = [
            sum(abs(float(r["lightroom_develop"][k]) - default) for k, default in _INTENSITY_DEFAULTS.items() if k in r.get("lightroom_develop", {}))
            for r in recs
            if r.get("lightroom_develop")
        ]
        album_breakdown.append(
            {
                "name": name,
                "count": len(recs),
                "avg_aesthetic": round(float(np.mean(aes)), 2) if aes else None,
                "aesthetic_std": round(float(np.std(aes)), 2) if len(aes) > 1 else None,
                "avg_editing_intensity": round(float(np.mean(intensities_a)), 1) if intensities_a else None,
                "avg_saturation": round(float(np.mean(sats_a)), 3) if sats_a else None,
                "avg_brightness": round(float(np.mean(brights_a)), 3) if brights_a else None,
                "avg_warmth": round(float(np.mean(warm_a)), 2) if warm_a else None,
                "top_colors": top_colors,
                "tonal_colors": tonal_colors,
            }
        )

    scene_in_ctr: Counter = Counter()
    scene_total_ctr: Counter = Counter()
    for r in album_records:
        sc = r.get("scene", {}).get("scene_type")
        if sc:
            scene_total_ctr[sc] += 1
            if r.get("lightroom_album_names"):
                scene_in_ctr[sc] += 1
    scene_curation_rate = {sc: round(scene_in_ctr[sc] / scene_total_ctr[sc] * 100, 1) for sc in scene_total_ctr if scene_total_ctr[sc] > 0}

    in_aes = [r.get("aesthetic_score") for r in in_album if r.get("aesthetic_score") is not None]
    out_aes = [r.get("aesthetic_score") for r in not_in_album if r.get("aesthetic_score") is not None]

    album_stats = {
        "total_albums": len(album_counter_data),
        "total_in_albums": len(in_album),
        "total_trackable": len(album_records),
        "curation_rate": round(len(in_album) / len(album_records) * 100, 1) if album_records else 0.0,
        "avg_aesthetic_in": round(float(np.mean(in_aes)), 2) if in_aes else None,
        "avg_aesthetic_out": round(float(np.mean(out_aes)), 2) if out_aes else None,
        "albums": album_breakdown,
        "scene_curation_rate": scene_curation_rate,
    }

    return {
        "lightroom_stats": lightroom_stats,
        "develop_stats": develop_stats,
        "split_toning": split_toning,
        "signature_edit": signature_edit,
        "hsl_fingerprint": hsl_fingerprint,
        "editing_intensity": editing_intensity,
        "pick_stats": pick_stats,
        "keyword_map": keyword_map,
        "album_stats": album_stats,
    }

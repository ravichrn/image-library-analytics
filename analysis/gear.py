from collections import Counter, defaultdict

import numpy as np

from extractors.exif import camera_device_category

from ._helpers import _scene_types_list


def analyze(records: list[dict]) -> dict:
    # ── Device breakdown ─────────────────────────────────────────────────────
    device_data: dict[str, dict] = defaultdict(lambda: {"count": 0, "aesthetics": [], "iq_scores": []})
    top_lenses: Counter = Counter()
    for r in records:
        exif = r.get("exif", {})
        make = exif.get("camera_make")
        model = exif.get("camera_model")
        lens = exif.get("lens_model")
        cat = camera_device_category(make, model)
        device_data[cat]["count"] += 1
        a = r.get("aesthetic_score")
        iq = r.get("iq_score")
        if a is not None:
            device_data[cat]["aesthetics"].append(a)
        if iq is not None:
            device_data[cat]["iq_scores"].append(iq)
        if lens:
            top_lenses[lens] += 1

    device_breakdown = {
        cat: {
            "count": data["count"],
            "avg_aesthetic": round(float(np.mean(data["aesthetics"])), 2) if data["aesthetics"] else None,
            "avg_iq": round(float(np.mean(data["iq_scores"])), 3) if data["iq_scores"] else None,
        }
        for cat, data in device_data.items()
    }

    # ── Flash usage ───────────────────────────────────────────────────────────
    flash_records = [r for r in records if r.get("exif", {}).get("flash_fired") is not None]
    flash_fired = sum(1 for r in flash_records if r.get("exif", {}).get("flash_fired"))
    flash_by_scene: dict[str, dict] = defaultdict(lambda: {"fired": 0, "total": 0})
    for r in flash_records:
        for sc in _scene_types_list(r):
            flash_by_scene[sc]["total"] += 1
            if r.get("exif", {}).get("flash_fired"):
                flash_by_scene[sc]["fired"] += 1

    # ── Metering mode ─────────────────────────────────────────────────────────
    metering_vals = [r.get("exif", {}).get("metering_mode") for r in records if r.get("exif", {}).get("metering_mode")]
    metering_distribution = dict(Counter(metering_vals).most_common())

    # ── Shutter category distribution ─────────────────────────────────────────
    shutter_vals = [r.get("exif", {}).get("shutter_category") for r in records if r.get("exif", {}).get("shutter_category")]
    shutter_category_distribution = dict(Counter(shutter_vals))

    shutter_by_scene: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        sh = r.get("exif", {}).get("shutter_category")
        if sh:
            for sc in _scene_types_list(r):
                shutter_by_scene[sc][sh] += 1

    # ── GPS bounds & samples ──────────────────────────────────────────────────
    gps_records = [(r.get("exif", {}).get("gps_lat"), r.get("exif", {}).get("gps_lon"), r) for r in records if r.get("exif", {}).get("gps_lat") is not None]
    gps_data: dict = {}
    if gps_records:
        lats = [lat for lat, _, _ in gps_records]
        lons = [lon for _, lon, _ in gps_records]
        samples_full = [
            {
                "lat": lat,
                "lon": lon,
                "scenes": _scene_types_list(r),
                "aesthetic_score": r.get("aesthetic_score"),
            }
            for lat, lon, r in gps_records
        ]
        # cap at 500 to avoid bloating JSON
        import random as _random

        samples = _random.sample(samples_full, min(500, len(samples_full)))
        gps_data = {
            "count": len(gps_records),
            "pct": round(len(gps_records) / len(records) * 100, 1) if records else 0.0,
            "bounds": {"lat_min": min(lats), "lat_max": max(lats), "lon_min": min(lons), "lon_max": max(lons)},
            "samples": samples,
        }

    # ── Monthly technical trends ──────────────────────────────────────────────
    monthly_tech: dict[str, dict] = defaultdict(lambda: {"focal": [], "iso": [], "shutter": []})
    for r in records:
        ym = r.get("exif", {}).get("year_month")
        if not ym:
            continue
        fl = r.get("exif", {}).get("focal_length_mm")
        iso = r.get("exif", {}).get("iso")
        sh = r.get("exif", {}).get("shutter_speed")
        if fl is not None:
            monthly_tech[ym]["focal"].append(fl)
        if iso is not None:
            monthly_tech[ym]["iso"].append(iso)
        if sh is not None:
            monthly_tech[ym]["shutter"].append(sh)

    focal_iso_trend = {
        ym: {
            "median_focal": round(float(np.median(d["focal"])), 1) if d["focal"] else None,
            "median_iso": round(float(np.median(d["iso"])), 0) if d["iso"] else None,
            "median_shutter": round(float(np.median(d["shutter"])), 6) if d["shutter"] else None,
        }
        for ym, d in sorted(monthly_tech.items())
    }

    # ── Settings sweet spots per scene ───────────────────────────────────────
    scene_combos: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in records:
        exif = r.get("exif", {})
        fc = exif.get("focal_category")
        dof = exif.get("dof_category")
        a = r.get("aesthetic_score")
        if fc and dof and a is not None:
            for sc in _scene_types_list(r):
                scene_combos[sc][f"{fc}+{dof}"].append(a)

    sweet_spots: dict[str, list] = {}
    for sc, combos in scene_combos.items():
        ranked = sorted(
            [{"combo": combo, "avg_aesthetic": round(float(np.mean(scores)), 2), "count": len(scores)} for combo, scores in combos.items() if len(scores) >= 3],
            key=lambda x: -x["avg_aesthetic"],
        )
        if ranked:
            sweet_spots[sc] = ranked[:3]

    # ── Shooting profile by context (time_of_day × scene) ────────────────────
    ctx_data: dict[tuple, dict] = defaultdict(
        lambda: {"focal": [], "shutter": [], "aperture": [], "iso": [], "flash": [], "aesthetics": [], "combos": Counter()}
    )
    for r in records:
        exif = r.get("exif", {})
        tod = exif.get("time_of_day")
        if not tod:
            continue
        fl = exif.get("focal_length_mm")
        sh = exif.get("shutter_speed")
        ap = exif.get("aperture_f")
        iso = exif.get("iso")
        flash = exif.get("flash_fired")
        a = r.get("aesthetic_score")
        fc = exif.get("focal_category")
        dof = exif.get("dof_category")
        for sc in _scene_types_list(r):
            key = (tod, sc)
            if fl is not None:
                ctx_data[key]["focal"].append(fl)
            if sh is not None:
                ctx_data[key]["shutter"].append(sh)
            if ap is not None:
                ctx_data[key]["aperture"].append(ap)
            if iso is not None:
                ctx_data[key]["iso"].append(iso)
            if flash is not None:
                ctx_data[key]["flash"].append(int(flash))
            if a is not None:
                ctx_data[key]["aesthetics"].append(a)
            if fc and dof:
                ctx_data[key]["combos"][f"{fc}+{dof}"] += 1

    shooting_profile_by_context = []
    for (tod, sc), d in sorted(ctx_data.items()):
        best_combo = d["combos"].most_common(1)[0][0] if d["combos"] else None
        shooting_profile_by_context.append(
            {
                "time_of_day": tod,
                "scene": sc,
                "count": max(len(d["focal"]), len(d["aesthetics"]), 1),
                "median_focal": round(float(np.median(d["focal"])), 1) if d["focal"] else None,
                "median_shutter": round(float(np.median(d["shutter"])), 6) if d["shutter"] else None,
                "median_aperture": round(float(np.median(d["aperture"])), 1) if d["aperture"] else None,
                "median_iso": round(float(np.median(d["iso"])), 0) if d["iso"] else None,
                "flash_rate_pct": round(sum(d["flash"]) / len(d["flash"]) * 100, 1) if d["flash"] else 0.0,
                "avg_aesthetic": round(float(np.mean(d["aesthetics"])), 2) if d["aesthetics"] else None,
                "best_combo": best_combo,
            }
        )
    shooting_profile_by_context.sort(key=lambda x: -(x["count"]))

    return {
        "technical_settings_analysis": {
            "device_breakdown": device_breakdown,
            "top_lenses": dict(top_lenses.most_common(10)),
            "flash_usage_pct": round(flash_fired / len(flash_records) * 100, 1) if flash_records else 0.0,
            "flash_by_scene": {sc: round(d["fired"] / d["total"] * 100, 1) for sc, d in flash_by_scene.items() if d["total"] > 0},
            "metering_distribution": metering_distribution,
            "shutter_category_distribution": shutter_category_distribution,
            "shutter_by_scene": {sc: dict(ctr.most_common()) for sc, ctr in shutter_by_scene.items()},
            "gps": gps_data,
            "focal_iso_trend": focal_iso_trend,
            "sweet_spots_by_scene": sweet_spots,
        },
        "shooting_profile_by_context": shooting_profile_by_context,
    }

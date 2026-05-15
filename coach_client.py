"""
Local rule engine — flags compositional and technical issues per photo.
No API calls. Runs on all photos using already-cached metrics.
"""

from collections import Counter, defaultdict


def _comp(r, k):
    return r.get("composition", {}).get(k)


def _exif(r, k):
    return r.get("exif", {}).get(k)


def _scene(r):
    return r.get("scene", {}).get("scene_type", "unknown")


RULES = [
    (
        "subject_centered",
        "Subject too centered",
        lambda r: (1.0 - _comp(r, "thirds_score")) if _comp(r, "thirds_score") is not None and _comp(r, "thirds_score") < 0.72 else None,
        "The main subject falls near the center of the frame rather than on a rule-of-thirds intersection. "
        "Try placing your subject at one of the four power points (roughly 1/3 from any edge) to add visual tension and lead the eye.",
    ),
    (
        "horizon_tilted",
        "Horizon not level",
        lambda r: abs(_comp(r, "horizon_tilt_deg")) if _comp(r, "horizon_tilt_deg") is not None and abs(_comp(r, "horizon_tilt_deg")) > 1.5 else None,
        "A detected horizontal line (horizon, roofline, waterline) is tilted more than 1.5°. "
        "Even small tilts read as unsteady. Use your camera's electronic level, or straighten in post.",
    ),
    (
        "foreground_cluttered",
        "Busy / cluttered foreground",
        lambda r: _comp(r, "foreground_clutter") if _comp(r, "foreground_clutter") is not None and _comp(r, "foreground_clutter") > 0.22 else None,
        "Edge density in the bottom quarter of the frame is high, suggesting a cluttered or distracting foreground. "
        "Move closer, change angle, or use a wider aperture to blur foreground elements.",
    ),
    (
        "low_subject_isolation",
        "Subject not well isolated from background",
        lambda r: (1.0 - min(_comp(r, "subject_isolation"), 2.0) / 2.0)
        if _comp(r, "subject_isolation") is not None and _comp(r, "subject_isolation") < 0.85
        else None,
        "The center of the frame is no sharper than the surroundings — the subject blends into the background. "
        "Try a wider aperture (lower f-number) for shallower depth of field, or reframe to create tonal contrast between subject and background.",
    ),
    (
        "frame_too_busy",
        "Frame too busy — insufficient negative space",
        lambda r: (0.25 - _comp(r, "negative_space")) if _comp(r, "negative_space") is not None and _comp(r, "negative_space") < 0.25 else None,
        "Less than 25% of the frame is free of edge activity, leaving no visual breathing room. "
        "Simplify by stepping back, using a tighter crop, or choosing a cleaner background.",
    ),
    (
        "low_aesthetic",
        "Low overall aesthetic score",
        lambda r: (35.0 - r["aesthetic_score"]) / 35.0 if r.get("aesthetic_score") is not None and r["aesthetic_score"] < 35 else None,
        "The LAION aesthetic predictor (trained on human ratings) scored this photo below 35/100. "
        "Common causes: flat or cluttered composition, poor exposure, heavy noise, or an uninteresting subject. "
        "Typical edited photos score 40–60; professional work 60+.",
    ),
    (
        "deep_dof_portrait",
        "Portrait with deep depth-of-field",
        lambda r: 1.0 if _exif(r, "dof_category") == "deep" and _scene(r) == "people and portraits" else None,
        "EXIF data shows a small aperture (f/8+) on a portrait, meaning the background is in sharp focus and competes with the subject. "
        "For environmental portraits this can work, but for headshots/closeups, f/1.4–f/2.8 will separate the subject more cleanly.",
    ),
    (
        "low_light_quality",
        "Low-light shot with low aesthetic score",
        lambda r: 1.0 if _exif(r, "light_category") == "low_light" and r.get("aesthetic_score") is not None and r["aesthetic_score"] < 40 else None,
        "Shot at ISO > 1600 and the aesthetic score is below 40 — likely affected by noise, motion blur, or flat exposure. "
        "Consider a faster lens, stabilisation, or expose-to-the-right and denoise in post.",
    ),
    (
        "highlight_clipping",
        "Highlight clipping — overexposed areas",
        lambda r: _comp(r, "highlight_clipping") if _comp(r, "highlight_clipping") is not None and _comp(r, "highlight_clipping") > 0.03 else None,
        "More than 3% of pixels are blown to pure white (brightness > 250). "
        "Highlight clipping is irreversible — detail in those areas is lost. "
        "Dial in −1 to −2 stops of exposure compensation in bright scenes, or use spot metering on the highlights.",
    ),
    (
        "shadow_clipping",
        "Shadow clipping — crushed blacks",
        lambda r: _comp(r, "shadow_clipping") if _comp(r, "shadow_clipping") is not None and _comp(r, "shadow_clipping") > 0.05 else None,
        "More than 5% of pixels are at pure black (brightness < 5), losing shadow detail. "
        "This can be intentional for high-contrast moody shots, but if unintended, lift the shadows or use fill flash. "
        "Shooting RAW retains more recoverable shadow data.",
    ),
]


def flag_record(record: dict) -> dict:
    flags = {}
    for key, _label, check, _desc in RULES:
        sev = check(record)
        if sev is not None:
            flags[key] = round(float(sev), 4)
    return flags


def aggregate_flags(records: list[dict]) -> dict:
    n = len(records)
    flag_counts: Counter = Counter()
    flag_severity: dict = defaultdict(list)
    flag_by_scene: dict = defaultdict(Counter)
    scene_totals: Counter = Counter()

    for r in records:
        scene = r.get("scene", {}).get("scene_type", "unknown")
        scene_totals[scene] += 1
        for key, sev in flag_record(r).items():
            flag_counts[key] += 1
            flag_severity[key].append(sev)
            flag_by_scene[key][scene] += 1

    flags_out = {}
    for key, _label, _, _desc in RULES:
        count = flag_counts[key]
        if count == 0:
            continue
        sevs = flag_severity[key]
        top_scene = flag_by_scene[key].most_common(1)[0][0]
        flags_out[key] = {
            "label": _label,
            "description": _desc,
            "count": count,
            "pct": round(count / n * 100, 1),
            "avg_severity": round(sum(sevs) / len(sevs), 3),
            "top_scene": top_scene,
            "by_scene": dict(flag_by_scene[key]),
        }

    flags_out = dict(sorted(flags_out.items(), key=lambda x: x[1]["count"], reverse=True))
    return {"total_analyzed": n, "scene_totals": dict(scene_totals), "flags": flags_out}

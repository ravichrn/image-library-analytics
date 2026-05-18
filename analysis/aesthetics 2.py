import colorsys
from collections import Counter

from ._helpers import _cv, _mean


def analyze(records: list[dict]) -> dict:
    sats = [r.get("color", {}).get("avg_saturation") for r in records]
    brights = [r.get("color", {}).get("avg_brightness") for r in records]
    contrasts = [r.get("color", {}).get("contrast") for r in records]
    warm_ratios = [r.get("color", {}).get("warm_ratio") for r in records]
    cool_ratios = [r.get("color", {}).get("cool_ratio") for r in records]

    avg_sat = _mean(sats) or 0
    avg_bri = _mean(brights) or 0
    avg_con = _mean(contrasts) or 0
    avg_wr = _mean(warm_ratios) or 0
    avg_cr = _mean(cool_ratios) or 0
    sat_cv = _cv(sats) or 0
    con_cv = _cv(contrasts) or 0
    bri_cv = _cv(brights) or 0

    # ── editing style patterns ────────────────────────────────────────────────
    def _conf(val, lo, hi):
        if val <= lo:
            return 0.0
        return round(min(1.0, (val - lo) / (hi - lo)), 2)

    raw_patterns = [
        (
            "warm_toned",
            "Warm Toned",
            _conf(avg_wr, 0.45, 0.70),
            f"avg warm-ratio {avg_wr:.2f} — dominant wavelengths skew orange/red/yellow. Common in golden-hour shooting, warm presets, or lifestyle editing.",
        ),
        (
            "cool_toned",
            "Cool & Desaturated",
            _conf(avg_cr, 0.38, 0.60),
            f"avg cool-ratio {avg_cr:.2f} — dominant wavelengths skew blue/cyan. Associated with moody, cinematic, or silver-tone editing styles.",
        ),
        (
            "vivid",
            "Vivid / Saturated",
            _conf(avg_sat, 0.35, 0.65),
            f"avg saturation {avg_sat:.3f} — colours are punchy and highly saturated. Typical of travel/landscape editing or vibrance-heavy presets.",
        ),
        (
            "muted",
            "Muted / Film-like",
            _conf(0.30 - avg_sat, 0.05, 0.25),
            f"avg saturation {avg_sat:.3f} — very low colour intensity, near-monochromatic palette. "
            "Common in film simulation presets (Kodak, Fuji) or faded/hazy aesthetics.",
        ),
        (
            "high_contrast",
            "High Contrast",
            _conf(avg_con, 0.38, 0.60),
            f"avg contrast {avg_con:.3f} — wide tonal separation between shadows and highlights. "
            "Adds drama and punch; common in black-and-white conversions and editorial work.",
        ),
        (
            "flat_matte",
            "Flat / Matte",
            _conf(0.22 - avg_con, 0.02, 0.18),
            f"avg contrast {avg_con:.3f} — compressed tonal range with lifted shadows and rolled highlights. "
            "Characteristic of the 'matte look' or log-profile editing without a strong tone curve.",
        ),
        (
            "high_key",
            "Bright / High Key",
            _conf(avg_bri, 0.58, 0.80),
            f"avg brightness {avg_bri:.3f} — photos lean bright and airy, often with open shadows. Prevalent in lifestyle, wedding, and portrait editing.",
        ),
        (
            "low_key",
            "Dark / Low Key",
            _conf(0.42 - avg_bri, 0.02, 0.32),
            f"avg brightness {avg_bri:.3f} — photos lean dark with heavy shadows. Found in moody portrait, fine-art, and night photography editing.",
        ),
        (
            "consistent",
            "Consistent Style",
            _conf(0.25 - max(sat_cv, con_cv, bri_cv), 0.0, 0.20),
            f"sat CV {sat_cv:.3f} · contrast CV {con_cv:.3f} · brightness CV {bri_cv:.3f} — "
            "very uniform look across the library, suggesting a defined preset or tight workflow.",
        ),
        (
            "varied",
            "Varied / Eclectic",
            _conf(max(sat_cv, con_cv, bri_cv), 0.25, 0.60),
            "Your images don't follow a single visual recipe — saturation, contrast, and brightness vary widely. "
            "This usually means different shoots, different subjects, and multiple creative moods rather than one applied preset. "
            "It's a sign of range, not inconsistency.",
        ),
    ]

    editing_style_patterns = sorted(
        [{"key": k, "name": n, "confidence": c, "description": d} for k, n, c, d in raw_patterns if c >= 0.18],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # ── hue distribution ─────────────────────────────────────────────────────
    hue_buckets = [0] * 12
    for r in records:
        palette = r.get("color", {}).get("palette", [])
        if palette:
            rgb = palette[0].get("rgb", [])
            if len(rgb) == 3:
                h, s, _v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
                if s > 0.15:
                    hue_buckets[int(h * 12) % 12] += 1

    hue_distribution = {
        "buckets": hue_buckets,
        "labels": ["Red", "Orange", "Yellow", "Yellow-Green", "Green", "Green-Cyan", "Cyan", "Cyan-Blue", "Blue", "Blue-Violet", "Violet", "Pink-Red"],
    }

    # ── saturation histogram ──────────────────────────────────────────────────
    sat_buckets = [0] * 5
    for s in [v for v in sats if v is not None]:
        sat_buckets[min(4, int(s * 5))] += 1
    saturation_histogram = {
        "buckets": sat_buckets,
        "labels": ["0–0.2 (muted)", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0 (vivid)"],
    }

    # ── composition patterns ──────────────────────────────────────────────────
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
        if ts is not None and ts > 0.74:
            comp_pattern_counts["rule_of_thirds"] += 1
        if ts is not None and ts < 0.60:
            comp_pattern_counts["centered"] += 1
        if sy is not None and sy > 0.87:
            comp_pattern_counts["symmetric"] += 1
        if ns is not None and ns > 0.90:
            comp_pattern_counts["minimal"] += 1
        if ns is not None and ns < 0.18:
            comp_pattern_counts["frame_filling"] += 1
        if si is not None and si > 1.5:
            comp_pattern_counts["isolated_subject"] += 1
        if fc is not None and fc > 0.22:
            comp_pattern_counts["busy_foreground"] += 1

    PATTERN_META = {
        "rule_of_thirds": ("Rule of Thirds", "Subject or key element placed near a thirds intersection — adds visual tension and directs the eye."),
        "symmetric": ("Strong Symmetry", "High left/right mirroring — architectural subjects, reflections, or deliberate geometric framing."),
        "minimal": ("Negative Space", "Unusually large areas of empty space — subject is isolated and given room to breathe in the frame."),
        "frame_filling": ("Filling the Frame", "Subject fills most of the frame with little empty space — maximises detail and impact."),
        "isolated_subject": (
            "Subject Isolation",
            "Centre is significantly sharper than surroundings — subject pops cleanly from background (shallow DoF or high-contrast bg).",
        ),
        "busy_foreground": ("Foreground Interest", "Complex edge density in the bottom zone — foreground elements add depth layering or texture."),
        "centered": ("Centered Composition", "Subject anchored at the frame centre — common in symmetry, direct portraits, and architecture."),
    }

    composition_patterns = []
    if n_comp > 0:
        for key, (name, desc) in PATTERN_META.items():
            cnt = comp_pattern_counts.get(key, 0)
            pct = round(cnt / n_comp * 100, 1)
            if pct >= 5:
                composition_patterns.append({"key": key, "name": name, "description": desc, "count": cnt, "pct": pct})
        composition_patterns.sort(key=lambda x: x["pct"], reverse=True)

    return {
        "editing_style_patterns": editing_style_patterns,
        "hue_distribution": hue_distribution,
        "saturation_histogram": saturation_histogram,
        "composition_patterns": composition_patterns,
    }

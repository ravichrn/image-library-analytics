from collections import defaultdict
from datetime import datetime

import numpy as np

from ._helpers import _compute_edit_intensity


def analyze(records: list[dict]) -> dict:
    # ── Monthly editing parameter trends ─────────────────────────────────────
    CORE_SLIDERS = ["Exposure2012", "Clarity2012", "Saturation", "Vibrance", "Highlights2012", "Shadows2012", "Texture", "Dehaze"]
    CREATIVE_SLIDERS = ["GrainAmount", "PostCropVignetteAmount"]
    WORKFLOW_SLIDERS = ["Sharpness", "LuminanceSmoothing", "ColorNoiseReduction"]
    HSL_CHANNELS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]

    monthly: dict[str, dict] = defaultdict(
        lambda: {
            **{k: [] for k in CORE_SLIDERS + CREATIVE_SLIDERS + WORKFLOW_SLIDERS},
            "hsl_intensity": [],
            "edit_intensity": [],
            "toning_pct_n": 0,
            "toning_pct_d": 0,
            "crop_pct_n": 0,
            "crop_pct_d": 0,
            "custom_curve_pct_n": 0,
            "custom_curve_pct_d": 0,
            "lens_profile_pct_n": 0,
            "lens_profile_pct_d": 0,
            "auto_wb_pct_n": 0,
            "auto_wb_pct_d": 0,
            "auto_upright_n": 0,
            "auto_upright_d": 0,
            "auto_ca_n": 0,
            "auto_ca_d": 0,
        }
    )

    for r in records:
        dev = r.get("lightroom_develop", {})
        if not dev:
            continue
        cap = r.get("lightroom_capture_date", "") or r.get("exif", {}).get("year_month", "")
        if not cap or len(cap) < 7:
            continue
        ym = cap[:7]
        m = monthly[ym]

        for k in CORE_SLIDERS + CREATIVE_SLIDERS + WORKFLOW_SLIDERS:
            v = dev.get(k)
            if v is not None:
                m[k].append(float(v))

        # HSL intensity = sum of abs of all 24 HSL adjustments
        hsl_sum = sum(abs(float(dev.get(f"{adj}Adjustment{ch}", 0) or 0)) for ch in HSL_CHANNELS for adj in ("Hue", "Saturation", "Luminance"))
        m["hsl_intensity"].append(hsl_sum)

        m["edit_intensity"].append(_compute_edit_intensity(dev))

        # Adoption rates
        m["toning_pct_d"] += 1
        if float(dev.get("SplitToningShadowSaturation", 0) or 0) > 3 or float(dev.get("SplitToningHighlightSaturation", 0) or 0) > 3:
            m["toning_pct_n"] += 1
        m["crop_pct_d"] += 1
        if dev.get("HasCrop"):
            m["crop_pct_n"] += 1
        m["custom_curve_pct_d"] += 1
        if dev.get("ToneCurveName2012") == "Custom":
            m["custom_curve_pct_n"] += 1
        m["lens_profile_pct_d"] += 1
        if dev.get("LensProfileEnable") == 1 or dev.get("LensProfileEnable") == "1":
            m["lens_profile_pct_n"] += 1
        m["auto_wb_pct_d"] += 1
        if str(dev.get("WhiteBalance", "")).lower() == "auto":
            m["auto_wb_pct_n"] += 1
        m["auto_upright_d"] += 1
        if dev.get("PerspectiveUpright") and dev["PerspectiveUpright"] not in (0, "0", "Off", "off"):
            m["auto_upright_n"] += 1
        m["auto_ca_d"] += 1
        if dev.get("AutoLateralCA") == 1 or dev.get("AutoLateralCA") == "1":
            m["auto_ca_n"] += 1

    def _pct(n, d):
        return round(n / d * 100, 1) if d > 0 else 0.0

    editing_journey = {}
    for ym in sorted(monthly.keys()):
        m = monthly[ym]
        automation_score = np.mean(
            [
                _pct(m["auto_wb_pct_n"], m["auto_wb_pct_d"]) / 100,
                _pct(m["auto_upright_n"], m["auto_upright_d"]) / 100,
                _pct(m["lens_profile_pct_n"], m["lens_profile_pct_d"]) / 100,
                _pct(m["auto_ca_n"], m["auto_ca_d"]) / 100,
            ]
        )
        editing_journey[ym] = {
            "count": m["toning_pct_d"],
            **{k: round(float(np.mean(m[k])), 3) if m[k] else None for k in CORE_SLIDERS + CREATIVE_SLIDERS + WORKFLOW_SLIDERS},
            "hsl_intensity": round(float(np.mean(m["hsl_intensity"])), 1) if m["hsl_intensity"] else None,
            "avg_edit_intensity": round(float(np.mean(m["edit_intensity"])), 1) if m["edit_intensity"] else None,
            "toning_pct": _pct(m["toning_pct_n"], m["toning_pct_d"]),
            "crop_pct": _pct(m["crop_pct_n"], m["crop_pct_d"]),
            "custom_curve_pct": _pct(m["custom_curve_pct_n"], m["custom_curve_pct_d"]),
            "lens_profile_pct": _pct(m["lens_profile_pct_n"], m["lens_profile_pct_d"]),
            "auto_wb_pct": _pct(m["auto_wb_pct_n"], m["auto_wb_pct_d"]),
            "automation_score": round(float(automation_score), 3),
        }

    # ── Edit intensity ↔ aesthetic Pearson r ─────────────────────────────────
    pairs = []
    for r in records:
        dev = r.get("lightroom_develop", {})
        if not dev:
            continue
        a = r.get("aesthetic_score")
        if a is not None:
            pairs.append((_compute_edit_intensity(dev), float(a)))

    edit_intensity_aesthetic_r = None
    if len(pairs) >= 10:
        int_arr = np.array([p[0] for p in pairs])
        aes_arr = np.array([p[1] for p in pairs])
        if int_arr.std() > 0 and aes_arr.std() > 0:
            edit_intensity_aesthetic_r = round(float(np.corrcoef(int_arr, aes_arr)[0, 1]), 3)

    # ── Edit recency (capture → updated gap) ─────────────────────────────────
    recency_buckets = {"same_day": 0, "within_week": 0, "within_month": 0, "later": 0}
    for r in records:
        cap = r.get("lightroom_capture_date")
        upd = r.get("lightroom_updated")
        if not cap or not upd:
            continue
        try:
            dt_cap = datetime.fromisoformat(cap.replace("Z", "+00:00")).replace(tzinfo=None)
            dt_upd = datetime.fromisoformat(upd.replace("Z", "+00:00")).replace(tzinfo=None)
            days = (dt_upd - dt_cap).days
            if days <= 0:
                recency_buckets["same_day"] += 1
            elif days <= 7:
                recency_buckets["within_week"] += 1
            elif days <= 30:
                recency_buckets["within_month"] += 1
            else:
                recency_buckets["later"] += 1
        except Exception:
            pass

    # ── Period split: early vs recent half ───────────────────────────────────
    lr_records = [r for r in records if r.get("lightroom_develop")]
    period_stats: dict = {}
    if len(lr_records) >= 4:
        sorted_lr = sorted(lr_records, key=lambda r: r.get("lightroom_capture_date") or "")
        mid = len(sorted_lr) // 2
        halves = {"early": sorted_lr[:mid], "recent": sorted_lr[mid:]}
        for label, half in halves.items():
            intensities = [_compute_edit_intensity(r["lightroom_develop"]) for r in half]
            aesthetics = [r.get("aesthetic_score") for r in half if r.get("aesthetic_score") is not None]
            auto_scores = []
            for r in half:
                dev = r["lightroom_develop"]
                auto_scores.append(
                    np.mean(
                        [
                            1.0 if str(dev.get("WhiteBalance", "")).lower() == "auto" else 0.0,
                            1.0 if dev.get("PerspectiveUpright") and dev["PerspectiveUpright"] not in (0, "0", "Off", "off") else 0.0,
                            1.0 if dev.get("LensProfileEnable") in (1, "1") else 0.0,
                            1.0 if dev.get("AutoLateralCA") in (1, "1") else 0.0,
                        ]
                    )
                )
            period_stats[label] = {
                "count": len(half),
                "avg_edit_intensity": round(float(np.mean(intensities)), 1) if intensities else None,
                "avg_aesthetic": round(float(np.mean(aesthetics)), 2) if aesthetics else None,
                "automation_score": round(float(np.mean(auto_scores)), 3) if auto_scores else None,
            }

    # ── Camera profile distribution ───────────────────────────────────────────
    profile_counter: dict[str, int] = {}
    third_party_count = 0
    adobe_prefixes = ("adobe", "camera ", "embedded")
    for r in records:
        dev = r.get("lightroom_develop", {})
        if not dev:
            continue
        cp = dev.get("CameraProfile") or dev.get("CameraCalibrationProfile")
        if not cp:
            continue
        cp_str = str(cp).strip()
        profile_counter[cp_str] = profile_counter.get(cp_str, 0) + 1
        if not any(cp_str.lower().startswith(p) for p in adobe_prefixes):
            third_party_count += 1

    top_profiles = sorted(profile_counter.items(), key=lambda x: -x[1])[:5]
    total_with_profile = sum(profile_counter.values())
    uses_third_party_profile = round(third_party_count / total_with_profile * 100, 1) if total_with_profile else 0.0

    # ── Editing style signatures ──────────────────────────────────────────────
    sig_records = [r for r in records if r.get("lightroom_develop")]
    signatures: list[str] = []
    if sig_records:

        def _avg(key):
            vals = [float(r["lightroom_develop"].get(key, 0) or 0) for r in sig_records if r["lightroom_develop"].get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        if _avg("GrainAmount") > 20:
            signatures.append("film_grain")
        if _avg("Clarity2012") > 20:
            signatures.append("high_clarity")
        if _avg("Saturation") < -10:
            signatures.append("muted_palette")
        if _avg("Saturation") > 20 or _avg("Vibrance") > 30:
            signatures.append("vivid_palette")
        if _avg("PostCropVignetteAmount") < -20:
            signatures.append("heavy_vignette")
        custom_curve_pct = sum(1 for r in sig_records if r["lightroom_develop"].get("ToneCurveName2012") == "Custom") / len(sig_records) * 100
        if custom_curve_pct > 40:
            signatures.append("custom_curves")
        if uses_third_party_profile > 50:
            signatures.append("preset_heavy")
        all_auto_scores = []
        for r in sig_records:
            dev = r["lightroom_develop"]
            all_auto_scores.append(
                np.mean(
                    [
                        1.0 if str(dev.get("WhiteBalance", "")).lower() == "auto" else 0.0,
                        1.0 if dev.get("PerspectiveUpright") and dev["PerspectiveUpright"] not in (0, "0", "Off", "off") else 0.0,
                        1.0 if dev.get("LensProfileEnable") in (1, "1") else 0.0,
                        1.0 if dev.get("AutoLateralCA") in (1, "1") else 0.0,
                    ]
                )
            )
        if all_auto_scores and float(np.mean(all_auto_scores)) > 0.7:
            signatures.append("automation_reliant")
        _not_upright = (0, "0", "Off", "off")
        geo_pct = (
            sum(1 for r in sig_records if r["lightroom_develop"].get("PerspectiveUpright") and r["lightroom_develop"]["PerspectiveUpright"] not in _not_upright)
            / len(sig_records)
            * 100
        )
        if geo_pct > 20:
            signatures.append("geometry_corrector")

    return {
        "editing_journey": editing_journey,
        "edit_intensity_aesthetic_r": edit_intensity_aesthetic_r,
        "edit_recency": recency_buckets,
        "period_stats": period_stats,
        "camera_profile_distribution": [{"profile": p, "count": c} for p, c in top_profiles],
        "uses_third_party_profile_pct": uses_third_party_profile,
        "editing_style_signatures": signatures,
    }

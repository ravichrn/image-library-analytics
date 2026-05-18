import base64
import colorsys
import os
from collections import Counter
from io import BytesIO

import numpy as np
from PIL import Image as _PILImage

from extractors import hour_to_time_of_day


def _show_thumbnails() -> bool:
    return os.environ.get("SHOW_THUMBNAILS", "false").lower() == "true"


def _thumb_b64(rec: dict, size: int = 72) -> str | None:
    path = rec.get("path") or rec.get("rendition_path")
    if not path:
        return None
    try:
        with _PILImage.open(path) as img:
            img.thumbnail((size, size))
            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=60)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


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


def _horizon_bucket(v) -> str:
    if v is None:
        return "none"
    if v < 0.33:
        return "high"
    if v <= 0.66:
        return "mid"
    return "low"


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


def _score_bucket(v) -> str:
    if v < 33:
        return "low"
    if v <= 66:
        return "mid"
    return "high"


def _classify_harmony(hue_deg: float | None) -> str:
    if hue_deg is None:
        return "unknown"
    h = hue_deg % 360
    if h < 30 or h >= 330:
        return "warm"
    if 30 <= h < 90:
        return "yellow_green"
    if 90 <= h < 150:
        return "cool_green"
    if 150 <= h < 210:
        return "cyan"
    if 210 <= h < 270:
        return "cool_blue"
    return "purple_magenta"


def _hsl_to_hex(hue_deg: float | None, sat_pct: float | None, lum: float = 0.5) -> str | None:
    if hue_deg is None or sat_pct is None or sat_pct < 2:
        return None
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360, lum, sat_pct / 100)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _hue_name(hue_deg: float) -> str:
    h = hue_deg % 360
    names = [
        (15, "red"),
        (45, "orange"),
        (75, "yellow"),
        (105, "yellow_green"),
        (135, "green"),
        (165, "teal"),
        (195, "cyan"),
        (225, "sky_blue"),
        (255, "blue"),
        (285, "blue_purple"),
        (315, "purple"),
        (345, "magenta"),
    ]
    for threshold, name in names:
        if h < threshold:
            return name
    return "red"


def _conf(r: dict) -> float:
    return r.get("scene", {}).get("scene_confidence") or 0.0


def _tavg(records: list[dict], key: str) -> float | None:
    vals = [float(r.get("lightroom_develop", {}).get(key) or 0) for r in records if r.get("lightroom_develop", {}).get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _sig_narrative(sliders: dict, develop: dict) -> list[str]:
    parts = []
    exp = float(develop.get("Exposure2012") or 0)
    if abs(exp) > 0.3:
        parts.append(f"{'brightens' if exp > 0 else 'darkens'} exposure by {abs(exp):.1f} EV")
    sat = float(develop.get("Saturation") or 0)
    vib = float(develop.get("Vibrance") or 0)
    if sat > 15 or vib > 20:
        parts.append("boosts color richness")
    elif sat < -15:
        parts.append("desaturates toward B&W")
    clar = float(develop.get("Clarity2012") or 0)
    if clar > 20:
        parts.append("adds strong clarity/texture")
    elif clar < -20:
        parts.append("softens with negative clarity")
    vig = float(develop.get("PostCropVignetteAmount") or 0)
    if vig < -15:
        parts.append("applies a dark vignette")
    grain = float(develop.get("GrainAmount") or 0)
    if grain > 15:
        parts.append(f"adds film grain ({int(grain)})")
    return parts


def _vqa_dist(records: list[dict], key: str) -> dict:
    c: Counter = Counter()
    for r in records:
        v = (r.get("caption") or {}).get(key)
        if v:
            c[str(v).lower()] += 1
    return dict(c.most_common(10))


def _bucket_summary(vals: list, buckets: list[tuple[float, str]]) -> dict:
    result: dict = {}
    for _lo, label in buckets:
        result[label] = 0
    for v in vals:
        if v is None:
            continue
        for lo, label in reversed(buckets):
            if v >= lo:
                result[label] += 1
                break
    return result


def _compute_edit_intensity(dev: dict) -> float:
    keys = ["Exposure2012", "Highlights2012", "Shadows2012", "Clarity2012", "Saturation", "Vibrance"]
    return sum(abs(float(dev.get(k, 0) or 0)) for k in keys)


# Constants shared across lightroom / journey modules
_TONING_KEYS = [
    "SplitToningShadowHue",
    "SplitToningShadowSaturation",
    "SplitToningHighlightHue",
    "SplitToningHighlightSaturation",
    "SplitToningBalance",
    "ColorGradeShadowLum",
    "ColorGradeHighlightLum",
    "ColorGradeMidtoneLum",
    "ColorGradeShadowSat",
    "ColorGradeHighlightSat",
    "ColorGradeMidtoneSat",
    "ColorGradeShadowHue",
    "ColorGradeHighlightHue",
    "ColorGradeMidtoneHue",
    "ColorGradeBlending",
    "ColorGradeGlobalHue",
    "ColorGradeGlobalSat",
    "ColorGradeGlobalLum",
]

_DEVELOP_KEYS = [
    "Exposure2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    "Contrast2012",
    "Clarity2012",
    "Texture",
    "Dehaze",
    "Vibrance",
    "Saturation",
    "ParametricShadows",
    "ParametricDarks",
    "ParametricLights",
    "ParametricHighlights",
    "SharpenRadius",
    "SharpenDetail",
    "SharpenEdgeMasking",
    "Sharpness",
    "LuminanceSmoothing",
    "ColorNoiseReduction",
    "ColorNoiseReductionDetail",
    "GrainAmount",
    "GrainSize",
    "GrainFrequency",
    "PostCropVignetteAmount",
    "PostCropVignetteFeather",
    "PostCropVignetteMidpoint",
    "PostCropVignetteStyle",
    "LensProfileEnable",
    "AutoLateralCA",
    "VignetteAmount",
    "PerspectiveVertical",
    "PerspectiveHorizontal",
    "PerspectiveRotate",
    "PerspectiveScale",
    "PerspectiveAspect",
    "PerspectiveUpright",
]

_SIG_SLIDERS = [
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    "Clarity2012",
    "Texture",
    "Vibrance",
    "Saturation",
    "GrainAmount",
    "PostCropVignetteAmount",
    "Dehaze",
]

_INTENSITY_DEFAULTS: dict[str, float] = {
    "Exposure2012": 0,
    "Highlights2012": 0,
    "Shadows2012": 0,
    "Whites2012": 0,
    "Blacks2012": 0,
    "Contrast2012": 0,
    "Clarity2012": 0,
    "Texture": 0,
    "Vibrance": 0,
    "Saturation": 0,
}

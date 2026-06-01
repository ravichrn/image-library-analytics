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


def _scene_types_list(r: dict) -> list[str]:
    return (r.get("scene") or {}).get("scene_types") or []


def _hsl_to_hex(hue_deg: float | None, sat_pct: float | None, lum: float = 0.5) -> str | None:
    if hue_deg is None or sat_pct is None or sat_pct < 2:
        return None
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360, lum, sat_pct / 100)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _compute_edit_intensity(dev: dict) -> float:
    keys = [
        "Exposure2012",
        "Highlights2012",
        "Shadows2012",
        "Whites2012",
        "Blacks2012",
        "Contrast2012",
        "Clarity2012",
        "Texture",
        "Dehaze",
        "Saturation",
        "Vibrance",
    ]
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

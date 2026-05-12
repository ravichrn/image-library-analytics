from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


def _get_raw_exif(img: Image.Image) -> dict:
    try:
        raw = img._getexif()
        if not raw:
            return {}
        return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}


def _safe_rational(val) -> float | None:
    try:
        if hasattr(val, "numerator"):
            return val.numerator / val.denominator if val.denominator else None
        return float(val)
    except Exception:
        return None


def hour_to_time_of_day(h: int) -> str:
    if 5 <= h <= 7 or 17 <= h <= 19:
        return "golden_hour"
    if 10 <= h <= 15:
        return "midday"
    if h < 5 or h >= 20:
        return "night"
    return "morning_evening"


def extract_exif(img: Image.Image) -> dict:
    """Accept an already-open PIL image — caller opens the file once and reuses it."""
    result = {
        "focal_length_mm": None,
        "aperture_f": None,
        "iso": None,
        "time_of_day": None,
        "hour": None,
        "focal_category": None,
        "dof_category": None,
        "light_category": None,
        "megapixels": None,
    }
    try:
        w, h = img.size
        result["megapixels"] = round(w * h / 1_000_000, 2)
        exif = _get_raw_exif(img)

        fl = _safe_rational(exif.get("FocalLengthIn35mmFilm") or exif.get("FocalLength"))
        if fl:
            result["focal_length_mm"] = fl
            if fl < 35:
                result["focal_category"] = "wide"
            elif fl <= 70:
                result["focal_category"] = "normal"
            else:
                result["focal_category"] = "telephoto"

        apex = _safe_rational(exif.get("ApertureValue") or exif.get("FNumber"))
        if apex:
            result["aperture_f"] = apex
            result["dof_category"] = "shallow" if apex < 2.8 else ("mid" if apex < 8 else "deep")

        iso = exif.get("ISOSpeedRatings")
        if iso:
            result["iso"] = int(iso)
            result["light_category"] = "low_light" if int(iso) > 1600 else ("indoor" if int(iso) > 400 else "bright")

        dt_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if dt_str:
            try:
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                result["hour"] = dt.hour
                result["year_month"] = dt.strftime("%Y-%m")
                result["year"] = dt.year
                result["time_of_day"] = hour_to_time_of_day(dt.hour)
            except ValueError:
                pass
    except Exception:
        pass
    return result

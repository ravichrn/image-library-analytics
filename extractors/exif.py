from datetime import datetime

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


def camera_device_category(make: str | None, model: str | None) -> str:
    """Classify camera body into broad device category."""
    if not make and not model:
        return "unknown"
    combined = f"{make or ''} {model or ''}".lower()
    phone_brands = ("apple", "samsung", "google", "pixel", "iphone", "huawei", "xiaomi", "oneplus", "oppo", "vivo", "realme")
    mirrorless_keywords = (
        "ilce",
        "α",
        "a7",
        "a6",
        "zfc",
        "z50",
        "z6",
        "z7",
        "z8",
        "z9",
        "xt",
        "x-t",
        "x-s",
        "gfx",
        "om-1",
        "om-5",
        "e-m",
        "sl2",
        "sl3",
        "rp",
        "r5",
        "r6",
        "r7",
        "r8",
        "r10",
        "r50",
        "r100",
        "r3",
        "r1",
        "s5",
        "s9",
        "gh6",
        "gh7",
        "g9",
    )
    dslr_keywords = (
        "eos",
        "nikon d",
        "nikon z",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "d8",
        "d500",
        "d750",
        "d800",
        "d810",
        "d850",
        "7d",
        "5d",
        "6d",
        "rebel",
        "k-",
        "pentax",
    )
    if any(b in combined for b in phone_brands):
        return "phone"
    if any(k in combined for k in mirrorless_keywords):
        return "mirrorless"
    if any(k in combined for k in dslr_keywords):
        return "dslr"
    if make:
        return "other"
    return "unknown"


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
        # new fields
        "camera_make": None,
        "camera_model": None,
        "lens_model": None,
        "shutter_speed": None,
        "shutter_category": None,
        "flash_fired": None,
        "metering_mode": None,
        "gps_lat": None,
        "gps_lon": None,
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

        # Camera & lens make/model
        make = exif.get("Make")
        model = exif.get("Model")
        lens = exif.get("LensModel") or exif.get("LensInfo") or exif.get("Lens")
        result["camera_make"] = str(make).strip() if make else None
        result["camera_model"] = str(model).strip() if model else None
        result["lens_model"] = str(lens).strip() if lens else None

        # Shutter speed
        et = exif.get("ExposureTime")
        if et:
            et_f = _safe_rational(et)
            if et_f is not None:
                result["shutter_speed"] = round(et_f, 6)
                result["shutter_category"] = "freeze" if et_f <= 1 / 500 else "hand" if et_f <= 1 / 60 else "slow" if et_f <= 1 / 15 else "bulb"

        # Flash
        flash_val = exif.get("Flash")
        result["flash_fired"] = bool(flash_val & 0x1) if isinstance(flash_val, int) else None

        # Metering mode
        metering_map = {1: "average", 2: "center_weighted", 3: "spot", 5: "multi_spot", 6: "multi_segment"}
        m = exif.get("MeteringMode")
        result["metering_mode"] = metering_map.get(m, "other") if m is not None else None

        # GPS
        gps_info = exif.get("GPSInfo")
        if gps_info:

            def _dms(dms, ref):
                d, mn, s = [float(x) for x in dms]
                dd = d + mn / 60 + s / 3600
                return -dd if ref in ("S", "W") else dd

            try:
                result["gps_lat"] = round(_dms(gps_info.get(2, [0, 0, 0]), gps_info.get(1, "N")), 5)
                result["gps_lon"] = round(_dms(gps_info.get(4, [0, 0, 0]), gps_info.get(3, "E")), 5)
            except Exception:
                pass
    except Exception:
        pass
    return result

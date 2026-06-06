from pathlib import Path

import numpy as np
from PIL import Image

# JPEG Annex K standard luma quantization table (baseline for quality=50)
_ANNEX_K_LUMA = np.array(
    [
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ],
    dtype=np.float64,
)

_NULL = {"jpeg_quality_factor": None, "quant_table_nonstandard": None}


def _scale_to_quality(scale: float) -> int:
    """Convert IJG scale factor to quality (1-100)."""
    q = 5000 / scale if scale >= 100 else (200 - scale) / 2
    return max(1, min(100, round(q)))


def _estimate_quality_factor(luma_table: list[int]) -> int:
    """Reverse the IJG formula to estimate quality from the luma quant table."""
    t = np.array(luma_table, dtype=np.float64)
    # Per-entry scale: invert (std * scale + 50) / 100 → t
    scales = np.where(_ANNEX_K_LUMA > 0, (t * 100 - 50) / _ANNEX_K_LUMA, 100.0)
    return _scale_to_quality(float(np.median(scales)))


def _is_nonstandard(tables: dict, tolerance: float = 3.0) -> bool:
    """Return True if the luma table does not closely match any IJG-formula table.

    Camera-original JPEGs use proprietary quant tables that diverge from the IJG
    standard. Software re-exports (Lightroom, Photoshop, ImageMagick) typically use
    the IJG formula at their target quality, so they match closely (low residual).
    A high residual therefore suggests a camera-original or custom encoder.
    """
    if 0 not in tables:
        return True
    t = np.array(tables[0], dtype=np.float64)
    # Find the best-fit IJG quality and compute reconstruction residual
    scales = np.where(_ANNEX_K_LUMA > 0, (t * 100 - 50) / _ANNEX_K_LUMA, 100.0)
    scale = float(np.median(scales))
    expected = np.clip((_ANNEX_K_LUMA * scale + 50) / 100, 1, 255).round()
    return float(np.abs(t - expected).mean()) > tolerance


def extract_jpeg_quality(path_or_img) -> dict:
    """Accept either a file path or an already-open PIL Image (reads header only, no pixel decode)."""
    try:
        if isinstance(path_or_img, Image.Image):
            img = path_or_img
            if img.format != "JPEG":
                return dict(_NULL)
        else:
            if path_or_img is None:
                return dict(_NULL)
            p = Path(path_or_img)
            if p.suffix.lower() not in (".jpg", ".jpeg"):
                return dict(_NULL)
            img = Image.open(p)
        tables = getattr(img, "quantization", None)
        if not tables or 0 not in tables:
            return dict(_NULL)
        quality = _estimate_quality_factor(tables[0])
        nonstandard = _is_nonstandard(tables)
        return {
            "jpeg_quality_factor": quality,
            "quant_table_nonstandard": nonstandard,
        }
    except Exception:
        return dict(_NULL)

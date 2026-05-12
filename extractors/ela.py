from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

_NULL = {"ela_max_error": None, "ela_mean_error": None, "ela_suspicious": False}


def extract_ela(path) -> dict:
    if path is None:
        return dict(_NULL)
    path = Path(path)
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return dict(_NULL)
    try:
        img = Image.open(path).convert("RGB")
        buf = BytesIO()
        img.save(buf, "JPEG", quality=75)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")
        diff = np.abs(np.array(img, dtype=np.float32) - np.array(recomp, dtype=np.float32))
        max_err = float(diff.max())
        mean_err = float(diff.mean())
        return {
            "ela_max_error": round(max_err, 2),
            "ela_mean_error": round(mean_err, 4),
            "ela_suspicious": max_err > 35.0,
        }
    except Exception:
        return dict(_NULL)

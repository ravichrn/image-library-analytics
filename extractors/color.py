import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def extract_color(img: Image.Image) -> dict:
    """Accept an already-open PIL image — caller opens the file once and reuses it."""
    try:
        img = img.convert("RGB").resize((200, 200))
    except Exception:
        return {}

    pixels = np.array(img).reshape(-1, 3).astype(float)
    sample = pixels[np.random.choice(len(pixels), min(2000, len(pixels)), replace=False)]

    km = KMeans(n_clusters=5, n_init=3, random_state=42)
    km.fit(sample)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(km.labels_, minlength=5)
    weights = (counts / counts.sum()).tolist()

    palette = [{"rgb": c.tolist(), "weight": round(w, 3)} for c, w in zip(centers, weights)]

    hsv = np.array(img.convert("HSV")) / 255.0
    avg_saturation = float(hsv[:, :, 1].mean())
    avg_brightness = float(hsv[:, :, 2].mean())

    hue = hsv[:, :, 0]
    warm_mask = (hue <= 0.17) | (hue >= 0.9)
    cool_mask = (hue >= 0.5) & (hue <= 0.75)
    warm_ratio = float(warm_mask.mean())
    cool_ratio = float(cool_mask.mean())
    warmth = "warm" if warm_ratio > cool_ratio else ("cool" if cool_ratio > warm_ratio else "neutral")
    contrast = float(hsv[:, :, 2].std())

    return {
        "palette": palette,
        "avg_saturation": round(avg_saturation, 3),
        "avg_brightness": round(avg_brightness, 3),
        "contrast": round(contrast, 3),
        "warmth": warmth,
        "warm_ratio": round(warm_ratio, 3),
        "cool_ratio": round(cool_ratio, 3),
    }

import os

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms

# 512×512 for the segmentation model; palette sampling at 128×128 is fast and sufficient
_INPUT_SIZE = 512
_PALETTE_SIZE = 128

_transform = transforms.Compose(
    [
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

_null = {
    "subject_area_pct": None,
    "subject_cx": None,
    "subject_cy": None,
    "subject_off_center": None,
    "fg_palette": None,
    "bg_palette": None,
}


def _palette_from_pixels(pixels: np.ndarray, n: int = 5) -> list[dict]:
    """KMeans palette from an (N, 3) float pixel array. Returns [] if too few pixels."""
    if len(pixels) < n:
        return []
    rng = np.random.default_rng(42)
    sample = pixels[rng.choice(len(pixels), min(2000, len(pixels)), replace=False)]
    n_actual = min(n, len(np.unique(sample.astype(np.uint8), axis=0)))
    if n_actual < 1:
        return []
    km = KMeans(n_clusters=n_actual, n_init=1, random_state=42)
    km.fit(sample)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(km.labels_, minlength=n_actual)
    weights = counts / counts.sum()
    return [{"rgb": c.tolist(), "weight": round(float(w), 3)} for c, w in zip(centers, weights, strict=False)]


def load_saliency_model(device: str):
    from transformers import AutoModelForImageSegmentation

    token = os.environ.get("HF_TOKEN") or None
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True, token=token)
    model.eval()
    if device in ("mps", "cuda"):
        model = model.half()
    model.to(device)
    return model


def _parse_pred(output) -> torch.Tensor:
    pred = output
    if isinstance(pred, list | tuple):
        pred = pred[-1]
    if isinstance(pred, list | tuple):
        pred = pred[-1]
    return pred


def extract_saliency_batch(paths: list, model, batch_size: int = 4) -> list[dict]:
    results: list[dict] = [dict(_null)] * len(paths)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    valid_idx: list[int] = []
    imgs: list[Image.Image] = []

    for i, path in enumerate(paths):
        if path is None:
            continue
        try:
            img = Image.open(path).convert("RGB")
            imgs.append(img)
            valid_idx.append(i)
        except Exception:
            pass

    for batch_start in range(0, len(imgs), batch_size):
        batch_imgs = imgs[batch_start : batch_start + batch_size]
        batch_idx = valid_idx[batch_start : batch_start + batch_size]

        tensors = torch.stack([_transform(img) for img in batch_imgs]).to(device=device, dtype=dtype)
        with torch.no_grad():
            pred = _parse_pred(model(tensors))  # (B, 1, 512, 512)

        # Work at 512×512 — no upsampling to original resolution needed.
        # Area %, centroid, and off-center are scale-invariant so values are identical.
        masks = pred.float().sigmoid().squeeze(1).cpu().numpy()  # (B, 512, 512)

        for j, mask512 in enumerate(masks):
            area_pct = float(mask512.mean())
            ys, xs = np.where(mask512 > 0.5)
            cx = float(xs.mean() / _INPUT_SIZE) if len(xs) else 0.5
            cy = float(ys.mean() / _INPUT_SIZE) if len(ys) else 0.5

            # Palette extraction at _PALETTE_SIZE × _PALETTE_SIZE
            img_small = np.array(batch_imgs[j].convert("RGB").resize((_PALETTE_SIZE, _PALETTE_SIZE))).reshape(-1, 3).astype(float)
            mask_small = np.array(Image.fromarray((mask512 * 255).astype(np.uint8)).resize((_PALETTE_SIZE, _PALETTE_SIZE), Image.NEAREST)).reshape(-1) / 255.0

            fg_palette = _palette_from_pixels(img_small[mask_small > 0.5])
            bg_palette = _palette_from_pixels(img_small[mask_small <= 0.5])

            results[batch_idx[j]] = {
                "subject_area_pct": round(area_pct, 4),
                "subject_cx": round(cx, 4),
                "subject_cy": round(cy, 4),
                "subject_off_center": round(float(np.hypot(cx - 0.5, cy - 0.5)), 4),
                "fg_palette": fg_palette,
                "bg_palette": bg_palette,
            }

    return results

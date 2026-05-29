import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms

# 512×512 for the segmentation model; palette sampling at 128×128 is fast and sufficient
_INPUT_SIZE = 384
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
    import torch
    from transformers import AutoModelForImageSegmentation

    token = os.environ.get("HF_TOKEN") or None
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True, token=token)
    model.eval()
    if device in ("mps", "cuda"):
        model = model.half()
    model.to(device)
    if device == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return model


def _parse_pred(output) -> torch.Tensor:
    pred = output
    if isinstance(pred, list | tuple):
        pred = pred[-1]
    if isinstance(pred, list | tuple):
        pred = pred[-1]
    return pred


def _compute_palette(img: Image.Image, mask512: np.ndarray) -> tuple[list, list]:
    """CPU-only palette extraction — runs in a background thread while GPU processes next batch."""
    img_small = np.array(img.convert("RGB").resize((_PALETTE_SIZE, _PALETTE_SIZE))).reshape(-1, 3).astype(float)
    mask_small = np.array(Image.fromarray((mask512 * 255).astype(np.uint8)).resize((_PALETTE_SIZE, _PALETTE_SIZE), Image.NEAREST)).reshape(-1) / 255.0
    return _palette_from_pixels(img_small[mask_small > 0.5]), _palette_from_pixels(img_small[mask_small <= 0.5])


def extract_saliency_batch(paths: list, model, batch_size: int = 4) -> list[dict]:
    from .prefetch import iter_prefetched

    results: list[dict] = [dict(_null)] * len(paths)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Each entry: (future, result_index, partial_result_without_palettes)
    pending: list[tuple[Future, int, dict]] = []

    with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as pool:
        for _batch_paths, batch_imgs, batch_idx in iter_prefetched(paths, batch_size):
            if not batch_imgs:
                continue

            # Collect previous batch's palette futures — they ran while GPU processed this batch
            for fut, idx, base in pending:
                fg_palette, bg_palette = fut.result()
                results[idx] = {**base, "fg_palette": fg_palette, "bg_palette": bg_palette}
            pending = []

            tensors = torch.stack([_transform(img) for img in batch_imgs]).to(device=device, dtype=dtype)
            with torch.no_grad():
                pred = _parse_pred(model(tensors))  # (B, 1, 512, 512)

            masks = pred.float().sigmoid().squeeze(1).cpu().numpy()  # (B, 512, 512)

            for j, mask512 in enumerate(masks):
                area_pct = float(mask512.mean())
                ys, xs = np.where(mask512 > 0.5)
                cx = float(xs.mean() / _INPUT_SIZE) if len(xs) else 0.5
                cy = float(ys.mean() / _INPUT_SIZE) if len(ys) else 0.5
                base = {
                    "subject_area_pct": round(area_pct, 4),
                    "subject_cx": round(cx, 4),
                    "subject_cy": round(cy, 4),
                    "subject_off_center": round(float(np.hypot(cx - 0.5, cy - 0.5)), 4),
                }
                # Submit palette work — runs while GPU processes the next batch
                pending.append((pool.submit(_compute_palette, batch_imgs[j], mask512), batch_idx[j], base))

        # Collect the final batch's palette futures
        for fut, idx, base in pending:
            fg_palette, bg_palette = fut.result()
            results[idx] = {**base, "fg_palette": fg_palette, "bg_palette": bg_palette}

    return results

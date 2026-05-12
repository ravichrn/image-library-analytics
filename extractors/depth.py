import numpy as np
from PIL import Image


def load_depth_model(device: str):
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )


def _depth_stats(depth_img) -> dict:
    depth = np.array(depth_img, dtype=np.float32)
    max_val = depth.max()
    if max_val < 1e-6:
        return {}
    norm = depth / max_val
    depth_range = round(float(np.percentile(norm, 95) - np.percentile(norm, 5)), 4)
    depth_complexity = round(float(norm.std()), 4)
    h, w = norm.shape
    cx, cy = w // 2, h // 2
    cr = max(1, min(cx, cy) // 3)
    center_depth = float(norm[cy - cr:cy + cr, cx - cr:cx + cr].mean())
    edges = np.concatenate(
        [norm[:10, :].ravel(), norm[-10:, :].ravel(),
         norm[:, :10].ravel(), norm[:, -10:].ravel()]
    )
    subject_depth_score = round(abs(center_depth - float(edges.mean())), 4)
    return {
        "depth_range": depth_range,
        "depth_complexity": depth_complexity,
        "subject_depth_score": subject_depth_score,
    }


def extract_depth_batch(paths: list, pipe, batch_size: int = 8) -> list[dict]:
    """Process images in batches through the depth pipeline."""
    imgs = []
    valid_idx = []
    for i, path in enumerate(paths):
        try:
            imgs.append(Image.open(str(path)).convert("RGB"))
            valid_idx.append(i)
        except Exception:
            pass

    results = [{}] * len(paths)
    for start in range(0, len(imgs), batch_size):
        batch_imgs = imgs[start:start + batch_size]
        batch_idx = valid_idx[start:start + batch_size]
        try:
            outputs = pipe(batch_imgs)
            for out_i, output in zip(batch_idx, outputs):
                results[out_i] = _depth_stats(output["depth"])
        except Exception:
            pass

    return results

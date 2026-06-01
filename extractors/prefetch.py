"""Async image prefetcher — loads the next batch from disk while GPU runs the current one."""

import os
from concurrent.futures import Future, ThreadPoolExecutor

from PIL import Image

_DECODE_WORKERS = min(4, os.cpu_count() or 4)
_TARGET_SIZE = (640, 640)


def _load_one(path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.draft("RGB", _TARGET_SIZE)
        return img.convert("RGB")
    except Exception:
        return None


def _load_pil_batch(paths: list) -> tuple[list, list[int]]:
    imgs, valid_idx = [], []
    with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as pool:
        results = list(pool.map(_load_one, (p for p in paths if p)))
    result_iter = iter(results)
    for i, path in enumerate(paths):
        if not path:
            continue
        img = next(result_iter)
        if img is not None:
            imgs.append(img)
            valid_idx.append(i)
    return imgs, valid_idx


def iter_prefetched(all_paths: list, batch_size: int):
    """Yield (batch_paths, imgs, valid_idx) with the next batch loading in background.

    GPU inference on batch N overlaps with disk IO + PIL decode for batch N+1.
    Within each batch, up to _DECODE_WORKERS threads decode in parallel — safe
    because libjpeg-turbo releases the GIL during JPEG decode.
    """
    batches = [all_paths[i : i + batch_size] for i in range(0, len(all_paths), batch_size)]
    if not batches:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut: Future = pool.submit(_load_pil_batch, batches[0])
        for i, batch_paths in enumerate(batches):
            imgs, valid_idx = fut.result()
            if i + 1 < len(batches):
                fut = pool.submit(_load_pil_batch, batches[i + 1])
            yield batch_paths, imgs, valid_idx

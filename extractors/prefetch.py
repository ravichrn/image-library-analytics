"""Async image prefetcher — loads the next batch from disk while GPU runs the current one."""

from concurrent.futures import Future, ThreadPoolExecutor

from PIL import Image


def _load_pil_batch(paths: list) -> tuple[list, list[int]]:
    imgs, valid_idx = [], []
    for i, path in enumerate(paths):
        if not path:
            continue
        try:
            imgs.append(Image.open(path).convert("RGB"))
            valid_idx.append(i)
        except Exception:
            pass
    return imgs, valid_idx


def iter_prefetched(all_paths: list, batch_size: int):
    """Yield (batch_paths, imgs, valid_idx) with the next batch loading in background.

    GPU inference on batch N overlaps with disk IO + PIL decode for batch N+1.
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

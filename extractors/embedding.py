import os

import torch
from transformers import AutoImageProcessor, AutoModel

from .device import empty_cache, get_device

_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def load_dino_model() -> tuple:
    device = get_device()
    token = os.environ.get("HF_TOKEN") or None
    processor = AutoImageProcessor.from_pretrained(_MODEL_ID, token=token)
    model = AutoModel.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        token=token,
    )
    model.eval()
    model.to(device)
    if device == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return model, processor, device


def unload_model(model) -> None:
    import gc

    del model
    gc.collect()
    empty_cache()


def extract_embedding_batch(
    paths: list,
    dino_model,
    dino_processor,
    device: str,
    batch_size: int = 16,
    on_batch=None,
) -> list[list[float] | None]:
    """Extract DINOv3-B CLS embeddings for all paths.

    Uses a prefetch thread to overlap JPEG decode with GPU inference — the next
    batch loads from disk while the current batch runs on MPS/CUDA.

    Args:
        paths:      All image paths to embed (may include None for missing files).
        batch_size: GPU forward-pass batch size (scheduler-driven from caller).
        on_batch:   Optional callback(n_images) called after each batch completes
                    — used by main.py to advance the progress bar.
    """
    from .prefetch import iter_prefetched

    results: list[list[float] | None] = [None] * len(paths)
    if not paths:
        return results

    dtype = next(dino_model.parameters()).dtype
    batch_start = 0

    for batch_paths, imgs, valid_local_idx in iter_prefetched(paths, batch_size):
        if imgs:
            inputs = dino_processor(images=imgs, return_tensors="pt").to(device)
            inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = dino_model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().float().tolist()
            for local_i, emb in zip(valid_local_idx, cls_tokens, strict=False):
                results[batch_start + local_i] = emb

        if on_batch is not None:
            on_batch(len(batch_paths))
        batch_start += len(batch_paths)

    return results

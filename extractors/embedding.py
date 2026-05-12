import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_dino_model() -> tuple:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained(
        "facebook/dinov2-small",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.eval()
    model.to(device)
    return model, processor, device


def unload_model(model) -> None:
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def extract_embedding_batch(
    paths: list, dino_model, dino_processor, device: str
) -> list[list[float] | None]:
    """Process a batch of images in one forward pass. Returns one embedding per path."""
    imgs = []
    valid_idx = []
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as img:
                imgs.append(img.convert("RGB"))
            valid_idx.append(i)
        except Exception:
            pass

    results: list[list[float] | None] = [None] * len(paths)
    if not imgs:
        return results

    inputs = dino_processor(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().float().tolist()

    for out_i, emb in zip(valid_idx, cls_tokens):
        results[out_i] = emb

    return results

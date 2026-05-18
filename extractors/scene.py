import torch
import torchvision.transforms as _TV
from PIL import Image

# CLIP-IQA+ expects 512×512 RGB tensors in [0, 1]
_IQ_TRANSFORM = _TV.Compose([_TV.Resize((512, 512)), _TV.ToTensor()])

SCENE_LABELS = [
    "nature",
    "architecture",
    "urban street",
    "people and portraits",
    "abstract",
    "travel and landmarks",
    "food",
    "interior",
    "animals",
    "night scene",
]


def load_siglip_model(device: str) -> tuple:
    from transformers import AutoModel, AutoProcessor

    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=dtype).eval().to(device)
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    return model, processor


def encode_scene_labels_siglip(model, processor, device: str) -> torch.Tensor:
    inputs = processor(text=SCENE_LABELS, return_tensors="pt", padding="max_length").to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        text_features = out.pooler_output if hasattr(out, "pooler_output") else out
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def classify_scene_batch(
    paths: list,
    model,
    processor,
    device: str,
    text_features: torch.Tensor,
) -> tuple[list[dict], torch.Tensor | None]:
    """
    SigLIP 2 sigmoid scoring.
    Returns (scene_results, image_features) — image_features reused for VQA.
    scene_results: one {"scene": {...}} dict per path.
    image_features: (N_valid, D) normalised tensor, or None if no valid images.
    """
    imgs: list = []
    valid_idx: list[int] = []
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as img:
                imgs.append(img.convert("RGB"))
            valid_idx.append(i)
        except Exception:
            pass

    results = [{"scene": {"scene_type": "unknown", "scene_scores": {}}} for _ in paths]
    if not imgs:
        return results, None

    inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probs = torch.sigmoid(image_features @ text_features.T).cpu().tolist()

    for out_i, prob_row in zip(valid_idx, probs, strict=False):
        scores = {label: round(p, 4) for label, p in zip(SCENE_LABELS, prob_row, strict=False)}
        results[out_i] = {"scene": {"scene_type": max(scores, key=scores.__getitem__), "scene_scores": scores}}

    return results, image_features


_VQA_LABELS: dict[str, list[str]] = {
    "has_person": ["a photo with a person", "a photo without a person"],
    "setting": ["an indoor photo", "an outdoor photo"],
    "time_of_day": ["morning light", "afternoon daylight", "evening golden hour", "night or low light"],
    "weather": ["clear sunny weather", "cloudy overcast weather", "rainy or wet weather", "foggy or misty weather", "snowy weather"],
    "season": [
        "spring with green growth or blossoms",
        "summer with bright sun and full foliage",
        "autumn with golden or red leaves",
        "winter with bare trees or snow",
    ],
}

_VQA_SHORT: dict[str, list[str]] = {
    "has_person": ["yes", "no"],
    "setting": ["indoors", "outdoors"],
    "time_of_day": ["morning", "afternoon", "evening", "night"],
    "weather": ["clear", "cloudy", "rainy", "foggy", "snowy"],
    "season": ["spring", "summer", "autumn", "winter"],
}


def encode_vqa_labels_siglip(model, processor, device: str) -> dict[str, torch.Tensor]:
    """Pre-encode all VQA label sets. Call once after loading SigLIP."""
    encoded: dict[str, torch.Tensor] = {}
    for key, labels in _VQA_LABELS.items():
        inputs = processor(text=labels, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
        encoded[key] = feats
    return encoded


def extract_vqa_batch(
    image_features: torch.Tensor,
    vqa_feats: dict[str, torch.Tensor],
) -> list[dict]:
    """
    Zero-shot SigLIP VQA using pre-computed image features.
    image_features: (N, D) normalised tensor from classify_scene_batch.
    Returns one dict of VQA answers per image.
    """
    results: list[dict] = [{} for _ in range(image_features.shape[0])]
    for key, text_feats in vqa_feats.items():
        probs = torch.sigmoid(image_features @ text_feats.T).cpu()
        short_labels = _VQA_SHORT[key]
        for i in range(image_features.shape[0]):
            best = int(probs[i].argmax())
            results[i][key] = short_labels[best]
    return results


def load_aesthetic_model(device: str):
    """Load aesthetic-predictor-v2-5 (SigLIP-based MLP, 1-10 scale → normalised to 0-100)."""
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(dtype).to(device).eval()
    return model, preprocessor


def extract_aesthetic_batch(paths: list, aesthetic, batch_size: int = 4) -> list[float | None]:
    """Run aesthetic-predictor-v2-5 on a batch. Returns 0-100 score per path.

    Scores are on a 1-10 scale internally; normalised to 0-100 for consistency.
    batch_size capped at 4 — model is SigLIP SO400M-based and memory-heavy.
    """
    results: list[float | None] = [None] * len(paths)
    if aesthetic is None:
        return results

    model, preprocessor = aesthetic
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    valid = [(i, p) for i, p in enumerate(paths) if p is not None]

    for start in range(0, len(valid), batch_size):
        chunk = valid[start : start + batch_size]
        imgs: list = []
        indices: list[int] = []
        for i, path in chunk:
            try:
                imgs.append(Image.open(path).convert("RGB"))
                indices.append(i)
            except Exception:
                pass
        if not imgs:
            continue
        try:
            pixel_values = preprocessor(images=imgs, return_tensors="pt").pixel_values.to(dtype).to(device)
            with torch.no_grad():
                logits = model(pixel_values).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            for idx, raw in zip(indices, logits, strict=False):
                # raw is on 1-10 scale; normalise to 0-100
                results[idx] = round(max(0.0, min(100.0, (raw - 1) / 9 * 100)), 2)
        except Exception:
            pass

    return results


def load_clipiqa_metric(device: str):
    """Load CLIP-IQA+ on MPS/CUDA/CPU for technical IQ scoring."""
    try:
        import pyiqa

        return pyiqa.create_metric("clipiqa+", device=device)
    except Exception:
        return None


def extract_iq_batch(paths: list, iq_metric, batch_size: int = 16) -> list[float | None]:
    """Run CLIP-IQA+ in batches. Returns a 0-100 IQ score per path."""
    results: list[float | None] = [None] * len(paths)
    valid: list[tuple[int, object]] = [(i, p) for i, p in enumerate(paths) if p is not None]

    for start in range(0, len(valid), batch_size):
        chunk = valid[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        indices: list[int] = []
        device = next(iter(iq_metric.parameters()), torch.tensor(0)).device
        for i, path in chunk:
            try:
                img = Image.open(path).convert("RGB")
                t = _IQ_TRANSFORM(img).to(device)
                tensors.append(t)
                indices.append(i)
            except Exception:
                pass
        if not tensors:
            continue
        try:
            batch_tensor = torch.stack(tensors)
            with torch.no_grad():
                scores = iq_metric(batch_tensor)
            if isinstance(scores, torch.Tensor):
                scores = scores.flatten().tolist()
            for idx, score in zip(indices, scores, strict=False):
                results[idx] = round(float(score) * 100, 2)
        except Exception:
            for idx, t in zip(indices, tensors, strict=False):
                try:
                    with torch.no_grad():
                        s = iq_metric(t.unsqueeze(0))
                    results[idx] = round(float(s.item()) * 100, 2)
                except Exception:
                    pass

    return results

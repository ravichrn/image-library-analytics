import hashlib
import json as _json
from pathlib import Path

import torch
import torchvision.transforms as _TV
from PIL import Image

_TEXT_FEATS_CACHE = Path("artifacts/siglip_text_feats.pt")
_SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"

# ARNIQA is trained on 224×224 crops; resize before batching to keep memory bounded
_IQ_INPUT_SIZE = 224
_IQ_TRANSFORM = _TV.Compose([_TV.Resize((_IQ_INPUT_SIZE, _IQ_INPUT_SIZE), antialias=True), _TV.ToTensor()])

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
    model = AutoModel.from_pretrained(_SIGLIP_MODEL_ID, torch_dtype=dtype).eval().to(device)
    processor = AutoProcessor.from_pretrained(_SIGLIP_MODEL_ID)
    if device == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass
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
    _preloaded: tuple[list, list[int]] | None = None,
) -> tuple[list[dict], torch.Tensor | None, None]:
    """
    SigLIP2-base cosine scoring.
    Returns (scene_results, image_features, None) — image_features reused for VQA.
    scene_results: one {"scene": {...}} dict per path.
    image_features: (N_valid, D) normalised tensor, or None if no valid images.
    _preloaded: optional (imgs, valid_idx) from prefetcher to skip disk IO.
    """
    if _preloaded is not None:
        imgs, valid_idx = _preloaded
    else:
        imgs, valid_idx = [], []
        for i, path in enumerate(paths):
            try:
                with Image.open(path) as img:
                    imgs.append(img.convert("RGB"))
                valid_idx.append(i)
            except Exception:
                pass

    results = [{"scene": {"scene_types": [], "scene_scores": {}}} for _ in paths]
    if not imgs:
        return results, None, None

    inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Raw cosine similarities — no scale, no bias.
        # SigLIP2 logit_scale pushes all scores to sigmoid ≈ 1.0 for short generic
        # labels, making absolute thresholds useless. Cosine similarity gives
        # meaningful relative spread: top label ~0.5–0.7, unrelated labels ~0.2–0.3.
        cosines = (image_features @ text_features.T).cpu().tolist()

    for out_i, cos_row in zip(valid_idx, cosines, strict=False):
        scores = {label: round(c, 4) for label, c in zip(SCENE_LABELS, cos_row, strict=False)}
        max_cos = max(scores.values()) if scores else 0.0
        # Multi-label: include all labels within 85% of the top cosine similarity.
        # This catches near-ties (portrait in nature) while excluding weak matches.
        scene_types = sorted(
            [label for label, s in scores.items() if max_cos > 0 and s >= max_cos * 0.85],
            key=scores.__getitem__,
            reverse=True,
        ) or [max(scores, key=scores.__getitem__)]
        results[out_i] = {"scene": {"scene_types": scene_types, "scene_scores": scores}}

    return results, image_features, None


# time_of_day and season are excluded — both are derived from EXIF metadata
# (timestamp → hour → time bucket; month → season) and never need visual inference.
_VQA_LABELS: dict[str, list[str]] = {
    "has_person": ["a photo with a person", "a photo without a person"],
    "setting": ["an indoor photo", "an outdoor photo"],
    "weather": ["clear sunny weather", "cloudy overcast weather", "rainy or wet weather", "foggy or misty weather", "snowy weather"],
}

_VQA_SHORT: dict[str, list[str]] = {
    "has_person": ["yes", "no"],
    "setting": ["indoors", "outdoors"],
    "weather": ["clear", "cloudy", "rainy", "foggy", "snowy"],
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


def _text_label_hash() -> str:
    """SHA-256 (first 16 hex chars) of model ID + all label strings.
    Used to invalidate the text-features cache when labels or model change."""
    payload = _json.dumps(
        {"model": _SIGLIP_MODEL_ID, "scene": SCENE_LABELS, "vqa": _VQA_LABELS},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode_text_features_siglip(model, processor, device: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Encode scene + VQA text labels, with a disk cache.

    The cache (artifacts/siglip_text_feats.pt) is reused across runs.
    It is invalidated automatically when any label string or the model ID changes.
    Returns (scene_feats, vqa_feats) on `device`.
    """
    expected_hash = _text_label_hash()
    if _TEXT_FEATS_CACHE.exists():
        try:
            data = torch.load(_TEXT_FEATS_CACHE, map_location="cpu", weights_only=True)
            if data.get("label_hash") == expected_hash:
                scene_feats = data["scene"].to(device)
                vqa_feats = {k: v.to(device) for k, v in data["vqa"].items()}
                return scene_feats, vqa_feats
        except Exception:
            pass

    scene_feats = encode_scene_labels_siglip(model, processor, device)
    vqa_feats = encode_vqa_labels_siglip(model, processor, device)

    try:
        _TEXT_FEATS_CACHE.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "label_hash": expected_hash,
                "scene": scene_feats.cpu(),
                "vqa": {k: v.cpu() for k, v in vqa_feats.items()},
            },
            _TEXT_FEATS_CACHE,
        )
    except Exception:
        pass

    return scene_feats, vqa_feats


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
        probs = (image_features @ text_feats.T).cpu()
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
    if device == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return model, preprocessor


def extract_aesthetic_batch(paths: list, aesthetic, batch_size: int = 4) -> list[float | None]:
    """Run aesthetic-predictor-v2-5 on a batch. Returns 0-100 score per path.

    Scores are on a 1-10 scale internally; normalised to 0-100 for consistency.
    batch_size capped at 4 — model is SigLIP SO400M-based and memory-heavy.
    """
    from .prefetch import iter_prefetched

    results: list[float | None] = [None] * len(paths)
    if aesthetic is None:
        return results

    model, preprocessor = aesthetic
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    valid_paths = [p if p is not None else None for p in paths]

    for _batch_paths, imgs, indices in iter_prefetched(valid_paths, batch_size):
        if not imgs:
            continue
        try:
            pixel_values = preprocessor(images=imgs, return_tensors="pt").pixel_values.to(dtype).to(device)
            with torch.no_grad():
                logits = model(pixel_values).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            for idx, raw in zip(indices, logits, strict=False):
                results[idx] = round(max(0.0, min(100.0, (raw - 1) / 9 * 100)), 2)
        except Exception:
            pass

    return results


def load_iq_metric(device: str):
    """Load ARNIQA on MPS/CUDA/CPU for technical IQ scoring."""
    try:
        import pyiqa

        return pyiqa.create_metric("arniqa", device=device)
    except Exception as e:
        return e  # caller logs the real error


def extract_iq_batch(
    paths: list,
    iq_metric,
    batch_size: int = 16,
    on_batch=None,
) -> list[float | None]:
    """Run ARNIQA over all paths, overlapping JPEG decode with GPU inference.

    Args:
        paths:      All image paths (may include None for missing files).
        batch_size: GPU forward-pass batch size.
        on_batch:   Optional callback(n_images) for progress tracking.
    """
    from .prefetch import iter_prefetched

    results: list[float | None] = [None] * len(paths)
    device = next(iter(iq_metric.parameters()), torch.tensor(0)).device
    batch_start = 0

    for batch_paths, imgs, valid_local_idx in iter_prefetched(paths, batch_size):
        if imgs:
            tensors: list[torch.Tensor] = []
            indices: list[int] = []
            for local_i, img in zip(valid_local_idx, imgs, strict=False):
                try:
                    tensors.append(_IQ_TRANSFORM(img).to(device))
                    indices.append(batch_start + local_i)
                except Exception:
                    pass
            if tensors:
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

        if on_batch is not None:
            on_batch(len(batch_paths))
        batch_start += len(batch_paths)

    return results

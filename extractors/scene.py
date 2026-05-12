from pathlib import Path

import torch
import torch.nn as nn
import open_clip
from PIL import Image

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

_AESTHETIC_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor"
    "/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
_AESTHETIC_CACHE = Path.home() / ".cache" / "aesthetic_predictor" / "mlp_weights.pth"


class _AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024), nn.ReLU(),
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_clip_models() -> tuple:
    model, preprocess, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, preprocess, tokenizer


def load_aesthetic_predictor(device: str) -> nn.Module:
    _AESTHETIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _AESTHETIC_CACHE.exists():
        import requests
        r = requests.get(_AESTHETIC_WEIGHTS_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(_AESTHETIC_CACHE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    mlp = _AestheticMLP()
    state = torch.load(_AESTHETIC_CACHE, map_location="cpu", weights_only=True)
    mlp.load_state_dict(state)
    mlp.eval()
    mlp.to(device)
    return mlp


def encode_scene_labels(clip_model, clip_tokenizer, device: str) -> torch.Tensor:
    tokens = clip_tokenizer(SCENE_LABELS).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def classify_scene_and_aesthetic_batch(
    paths: list,
    clip_model,
    clip_preprocess,
    device: str,
    scene_text_features: torch.Tensor,
    aesthetic_mlp: nn.Module,
) -> list[dict]:
    """Process a batch of images in one forward pass. Returns one result dict per path."""
    imgs = []
    valid_idx = []
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as img:
                imgs.append(clip_preprocess(img.convert("RGB")))
            valid_idx.append(i)
        except Exception:
            pass

    results = [{"scene": {"scene_type": "unknown", "scene_scores": {}}, "aesthetic_score": None}
               for _ in paths]

    if not imgs:
        return results

    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scene_sims = image_features @ scene_text_features.T
        scene_probs = scene_sims.softmax(dim=-1).cpu().tolist()
        raw_scores = aesthetic_mlp(image_features.float()).squeeze(-1).cpu().tolist()

    for out_i, (probs, raw) in zip(valid_idx, zip(scene_probs, raw_scores)):
        scores = {label: round(p, 4) for label, p in zip(SCENE_LABELS, probs)}
        results[out_i] = {
            "scene": {"scene_type": max(scores, key=scores.__getitem__), "scene_scores": scores},
            "aesthetic_score": round(max(0.0, min(100.0, raw * 10)), 2),
        }

    return results

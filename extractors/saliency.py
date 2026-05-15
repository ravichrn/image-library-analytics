import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_briarmbg(repo_dir: str):
    """Load BriaRMBG from snapshot dir, handling relative imports."""
    repo_path = Path(repo_dir)
    pkg = "briaai_rmbg"

    # Create package namespace so relative imports resolve
    pkg_mod = types.ModuleType(pkg)
    pkg_mod.__path__ = [str(repo_path)]
    pkg_mod.__package__ = pkg
    sys.modules[pkg] = pkg_mod

    # Load each py file as a submodule
    for fname in ["MyConfig", "utilities", "briarmbg"]:
        fpath = repo_path / f"{fname}.py"
        if not fpath.exists():
            continue
        spec = importlib.util.spec_from_file_location(f"{pkg}.{fname}", fpath)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[f"{pkg}.{fname}"] = mod
        spec.loader.exec_module(mod)

    return sys.modules[f"{pkg}.briarmbg"]


def load_saliency_model(device: str):
    repo_dir = snapshot_download("briaai/RMBG-1.4")
    briarmbg = _load_briarmbg(repo_dir)
    model = briarmbg.BriaRMBG()
    weights = torch.load(f"{repo_dir}/model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def extract_saliency_batch(paths: list, model, batch_size: int = 8) -> list[dict]:
    null_result = {"subject_area_pct": None, "subject_cx": None, "subject_cy": None, "subject_off_center": None}
    device = next(model.parameters()).device
    results = []
    for path in paths:
        if path is None:
            results.append(dict(null_result))
            continue
        try:
            img = Image.open(path).convert("RGB")
            tensor = _TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor)[0][0]
            mask = torch.sigmoid(pred).squeeze().cpu().numpy()
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
            area_pct = float(mask.mean())
            ys, xs = np.where(mask > 0.5)
            if len(xs):
                cx = float(xs.mean() / mask.shape[1])
                cy = float(ys.mean() / mask.shape[0])
            else:
                cx, cy = 0.5, 0.5
            off_center = float(np.hypot(cx - 0.5, cy - 0.5))
            results.append(
                {
                    "subject_area_pct": round(area_pct, 4),
                    "subject_cx": round(cx, 4),
                    "subject_cy": round(cy, 4),
                    "subject_off_center": round(off_center, 4),
                }
            )
        except Exception:
            results.append(dict(null_result))
    return results

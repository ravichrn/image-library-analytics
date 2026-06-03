from pathlib import Path

import numpy as np

from .device import empty_cache

_MODEL_NAME = "yolo26n-pose.pt"
_CACHE_PATH = Path("artifacts") / _MODEL_NAME

# COCO keypoint indices used for pose classification
_KP_L_SHOULDER, _KP_R_SHOULDER = 5, 6
_KP_L_HIP, _KP_R_HIP = 11, 12
_KP_L_ANKLE, _KP_R_ANKLE = 15, 16


def load_pose_model(device: str):
    import shutil

    from ultralytics import YOLO

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _CACHE_PATH.exists():
        # Download via ultralytics default mechanism, then move to artifacts/
        model = YOLO(_MODEL_NAME)  # downloads to ~/.cache/ultralytics/
        ul_default = Path.home() / ".cache" / "ultralytics" / _MODEL_NAME
        if ul_default.exists():
            shutil.copy(ul_default, _CACHE_PATH)
        return model
    return YOLO(str(_CACHE_PATH))


def unload_pose_model(model) -> None:
    import gc

    del model
    gc.collect()
    empty_cache()


def _classify_pose(kpts_xy: np.ndarray) -> str | None:
    """Derive standing/sitting/crouching/lying from shoulder→hip→ankle ratios."""
    if kpts_xy is None or len(kpts_xy) < 17:
        return None
    shoulder_y = np.mean(kpts_xy[[_KP_L_SHOULDER, _KP_R_SHOULDER], 1])
    hip_y = np.mean(kpts_xy[[_KP_L_HIP, _KP_R_HIP], 1])
    ankle_y = np.mean(kpts_xy[[_KP_L_ANKLE, _KP_R_ANKLE], 1])
    if shoulder_y == 0 or hip_y == 0:
        return None
    torso = abs(hip_y - shoulder_y)
    legs = abs(ankle_y - hip_y)
    ratio = legs / (torso + 1e-6)
    if ratio > 1.2:
        return "standing"
    if ratio > 0.5:
        return "sitting"
    if ratio > 0.1:
        return "crouching"
    return "lying"


def _parse_pred(pred, model) -> dict:
    detected_objects: list[dict] = []
    if pred.boxes is not None:
        for box in pred.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= 0.3:
                detected_objects.append({"label": model.names[cls_id], "confidence": round(conf, 3)})

    person_count = 0
    pose_type = None
    body_coverage = None

    if pred.keypoints is not None:
        person_count = len(pred.keypoints)
        if person_count > 0:
            kpts_xy = pred.keypoints.xy[0].cpu().numpy()
            pose_type = _classify_pose(kpts_xy)

            person_boxes = [b for b in (pred.boxes or []) if model.names[int(b.cls[0])] == "person"]
            if person_boxes:
                img_h, img_w = pred.orig_shape
                best = max(person_boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = best.xyxy[0].tolist()
                body_coverage = round((x2 - x1) * (y2 - y1) / (img_h * img_w), 3)

    return {
        "detected_objects": detected_objects,
        "pose": {"pose_type": pose_type, "person_count": person_count, "body_coverage": body_coverage},
    }


def extract_pose_batch(
    paths: list,
    model,
    device: str,
    batch_size: int = 16,
    on_batch=None,
) -> list[dict]:
    """Run YOLO26n-pose over all paths, passing pre-decoded PIL images to skip
    YOLO's internal C++ decode and overlapping decode with GPU inference.

    Measured 1.5x throughput gain vs passing file paths (42 -> 64 img/s on real files).

    Args:
        paths:      All image paths (may include None for missing files).
        batch_size: YOLO forward-pass batch size.
        on_batch:   Optional callback(n_images) for progress tracking.
    """
    from .prefetch import iter_prefetched

    results: list[dict] = [{}] * len(paths)
    batch_start = 0

    for batch_paths, imgs, valid_local_idx in iter_prefetched(paths, batch_size):
        if imgs:
            global_indices = [batch_start + li for li in valid_local_idx]
            try:
                # Pass PIL images directly — YOLO skips its internal OpenCV decode
                batch_preds = model(imgs, verbose=False, device=device)
                for gi, pred in zip(global_indices, batch_preds, strict=False):
                    try:
                        results[gi] = _parse_pred(pred, model)
                    except Exception:
                        results[gi] = {}
            except Exception:
                for gi, img in zip(global_indices, imgs, strict=False):
                    try:
                        pred = model([img], verbose=False, device=device)[0]
                        results[gi] = _parse_pred(pred, model)
                    except Exception:
                        results[gi] = {}

        if on_batch is not None:
            on_batch(len(batch_paths))
        batch_start += len(batch_paths)

    return results

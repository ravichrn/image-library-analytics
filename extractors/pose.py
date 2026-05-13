import numpy as np
from pathlib import Path

_MODEL_NAME = "yolov8n-pose.pt"
_CACHE_PATH = Path.home() / ".cache" / "ultralytics" / _MODEL_NAME

# COCO keypoint indices used for pose classification
_KP_L_SHOULDER, _KP_R_SHOULDER = 5, 6
_KP_L_HIP, _KP_R_HIP = 11, 12
_KP_L_ANKLE, _KP_R_ANKLE = 15, 16


def load_pose_model(device: str):
    from ultralytics import YOLO
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(_CACHE_PATH))


def unload_pose_model(model) -> None:
    import gc
    del model
    gc.collect()


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


def extract_pose_batch(paths: list, model, device: str) -> list[dict]:
    """Run YOLOv8n-pose on each path. Returns object detections + pose per photo."""
    results: list[dict] = [{}] * len(paths)

    for i, path in enumerate(paths):
        if path is None:
            continue
        try:
            preds = model(str(path), verbose=False, device=device)[0]

            detected_objects: list[dict] = []
            if preds.boxes is not None:
                for box in preds.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= 0.3:
                        detected_objects.append({
                            "label": model.names[cls_id],
                            "confidence": round(conf, 3),
                        })

            person_count = 0
            pose_type = None
            body_coverage = None

            if preds.keypoints is not None:
                person_count = len(preds.keypoints)
                if person_count > 0:
                    kpts_xy = preds.keypoints.xy[0].cpu().numpy()
                    pose_type = _classify_pose(kpts_xy)

                    person_boxes = [
                        b for b in (preds.boxes or [])
                        if model.names[int(b.cls[0])] == "person"
                    ]
                    if person_boxes:
                        img_h, img_w = preds.orig_shape
                        best = max(person_boxes, key=lambda b: float(b.conf[0]))
                        x1, y1, x2, y2 = best.xyxy[0].tolist()
                        body_coverage = round((x2 - x1) * (y2 - y1) / (img_h * img_w), 3)

            results[i] = {
                "detected_objects": detected_objects,
                "pose": {
                    "pose_type": pose_type,
                    "person_count": person_count,
                    "body_coverage": body_coverage,
                },
            }
        except Exception:
            results[i] = {}

    return results

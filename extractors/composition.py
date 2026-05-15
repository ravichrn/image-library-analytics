import cv2
import numpy as np
from PIL import Image


def extract_composition(img: Image.Image) -> dict:
    """Accept an already-open PIL image — caller opens the file once and reuses it."""
    try:
        arr = np.array(img.convert("L").resize((300, 300)), dtype=np.uint8)
    except Exception:
        return {}

    edges = cv2.Canny(arr, 50, 150)

    h, w = edges.shape
    cell_h, cell_w = h // 3, w // 3
    grid = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            cell = edges[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            grid[r, c] = cell.mean()
    total = grid.sum() or 1
    grid_weights = (grid / total).tolist()

    center_weight = grid[1, 1] / total
    thirds_score = round(1.0 - float(center_weight), 3)

    left = arr[:, : w // 2]
    right = np.fliplr(arr[:, w // 2 :])
    min_w = min(left.shape[1], right.shape[1])
    diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
    symmetry_score = round(1.0 - diff.mean() / 255.0, 3)

    low_var_mask = edges < 10
    negative_space = round(float(low_var_mask.mean()), 3)

    horizon_position = None
    horizon_tilt_deg = None
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is not None:
        horizontal = [l[0] for l in lines if abs(l[0][1] - np.pi / 2) < 0.2]
        if horizontal:
            rhos = [abs(l[0]) for l in horizontal]
            avg_rho = np.mean(rhos)
            horizon_position = round(float(avg_rho) / h, 3)
            thetas = [l[0][1] for l in lines if abs(l[0][1] - np.pi / 2) < 0.2]
            avg_theta = np.mean(thetas)
            tilt_rad = avg_theta - np.pi / 2
            horizon_tilt_deg = round(float(np.degrees(tilt_rad)), 2)

    bottom_quarter = edges[int(h * 0.75) :, :]
    foreground_clutter = round(float(bottom_quarter.mean()) / 255.0, 3)

    cx, cy = w // 2, h // 2
    cr = min(cx, cy) // 2
    center_crop = arr[cy - cr : cy + cr, cx - cr : cx + cr]
    center_lap = float(cv2.Laplacian(center_crop, cv2.CV_64F).var())
    full_lap = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    subject_isolation = round(center_lap / (full_lap + 1e-6), 3)
    sharpness_score = round(full_lap, 2)

    mean_brightness = float(arr.mean())
    highlight_clipping = round(float((arr > 250).mean()), 4)
    shadow_clipping = round(float((arr < 5).mean()), 4)
    exposure_bias = round((mean_brightness - 128.0) / 128.0, 4)

    tonal_shadow_pct = round(float((arr < 85).mean()), 4)
    tonal_mid_pct = round(float(((arr >= 85) & (arr < 170)).mean()), 4)
    tonal_highlight_pct = round(float((arr >= 170).mean()), 4)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=40, maxLineGap=12)
    leading_lines_score = 0.0
    if lines is not None:
        count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            if 15 < angle < 75:
                count += 1
        leading_lines_score = round(min(1.0, count / 15.0), 3)

    return {
        "thirds_score": thirds_score,
        "symmetry_score": symmetry_score,
        "negative_space": negative_space,
        "horizon_position": horizon_position,
        "horizon_tilt_deg": horizon_tilt_deg,
        "foreground_clutter": foreground_clutter,
        "subject_isolation": subject_isolation,
        "sharpness_score": sharpness_score,
        "highlight_clipping": highlight_clipping,
        "shadow_clipping": shadow_clipping,
        "exposure_bias": exposure_bias,
        "grid_weights": grid_weights,
        "tonal_shadow_pct": tonal_shadow_pct,
        "tonal_mid_pct": tonal_mid_pct,
        "tonal_highlight_pct": tonal_highlight_pct,
        "leading_lines_score": leading_lines_score,
    }

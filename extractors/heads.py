from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS_DIR = Path("artifacts")
_METADATA_PATH = MODELS_DIR / "heads_metadata.json"

# ── EXIF-derived time_of_day and season ──────────────────────────────────────


def _hour_to_caption_tod(h: int) -> str:
    if 5 <= h <= 9:
        return "morning"
    if 10 <= h <= 15:
        return "afternoon"
    if 16 <= h <= 19:
        return "evening"
    return "night"


def _month_to_season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def _exif_time_of_day(r: dict) -> str | None:
    """Derive time_of_day from EXIF hour or Lightroom capture date."""
    hour = (r.get("exif") or {}).get("hour")
    if hour is not None:
        return _hour_to_caption_tod(int(hour))
    cap = r.get("lightroom_capture_date") or ""
    if cap and "T" in cap:
        try:
            return _hour_to_caption_tod(int(cap.split("T")[1].split(":")[0]))
        except (IndexError, ValueError):
            pass
    return None


def _exif_season(r: dict) -> str | None:
    """Derive season from EXIF year_month or Lightroom capture date."""
    ym = (r.get("exif") or {}).get("year_month")
    if not ym:
        cap = r.get("lightroom_capture_date") or ""
        if len(cap) >= 7:
            ym = cap[:7]
    if ym and len(ym) >= 7:
        try:
            return _month_to_season(int(ym[5:7]))
        except ValueError:
            pass
    return None


# ── Aesthetic regressor (Ridge on DINOv3-B embeddings) ───────────────────────

_AESTHETIC_REG_PATH = MODELS_DIR / "aesthetic_regressor.joblib"

# Learning-curve plateau: marginal MAE gain/100ph drops below 0.02 at K=1500
# (measured empirically). At 2.5 img/s this equals ~10 min of SigLIP — the
# validated accuracy/time sweet spot. Beyond 1500, gains are negligible.
_MAX_SEED = 1500


def _seed_count(n_photos: int) -> int:
    """K clusters to run through SigLIP. Formula: 5*sqrt(N), capped at _MAX_SEED.
    Self-scales with library size — single source of truth, never duplicated."""
    import math

    return min(max(2, round(5 * math.sqrt(n_photos))), _MAX_SEED)


def aesthetic_regressor_available() -> bool:
    """True when the trained Ridge aesthetic regressor exists on disk."""
    return _AESTHETIC_REG_PATH.exists()


def load_aesthetic_regressor() -> dict:
    """Load the Ridge regressor artifact. Returns dict with at minimum
    {"pipe", "r2", "mae", "alpha"}. New-format artifacts also contain
    {"X_seed", "y_seed", "coverage_threshold"} used for OOD detection."""
    import joblib

    return joblib.load(_AESTHETIC_REG_PATH)


def predict_aesthetic_scores(records: list[dict], reg: dict) -> list[float]:
    """Predict aesthetic_score (0-100) for records with a dinov3 embedding."""
    X = np.array([r["dinov3"] for r in records], dtype=np.float32)
    raw = reg["pipe"].predict(X)
    return [round(float(max(0.0, min(100.0, s))), 2) for s in raw]


def select_aesthetic_seed(records: list[dict]) -> tuple[list[dict], np.ndarray]:
    """k-means on dinov3 embeddings → 1 centroid-nearest photo per cluster.

    K = min(_seed_count(N), N//2) so SigLIP never runs on >50% of the library.
    Returns (seed_records, X_seed) where X_seed is shape (K, 768).
    Caller must ensure every record has a "dinov3" key.
    """
    from sklearn.cluster import MiniBatchKMeans

    n = len(records)
    K = min(_seed_count(n), n // 2)
    X = np.array([r["dinov3"] for r in records], dtype=np.float32)

    km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42)
    km.fit(X)

    seed_idx: list[int] = []
    for c in range(K):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1)
        seed_idx.append(int(members[dists.argmin()]))

    seed_records = [records[i] for i in seed_idx]
    X_seed = X[seed_idx]
    return seed_records, X_seed


def compute_coverage_threshold(X_all: np.ndarray, X_seed: np.ndarray) -> float:
    """p95 of min-distances from every photo in X_all to its nearest seed.
    Defines the OOD boundary: photos beyond this on future runs get new seeds."""
    batch = 512
    min_dists: list[float] = []
    for i in range(0, len(X_all), batch):
        chunk = X_all[i : i + batch]
        d = np.linalg.norm(chunk[:, None, :] - X_seed[None, :, :], axis=2).min(axis=1)
        min_dists.extend(d.tolist())
    return float(np.percentile(min_dists, 95))


def check_ood(X_query: np.ndarray, X_seed: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask: True = photo's nearest-seed distance exceeds threshold."""
    batch = 512
    min_dists: list[float] = []
    for i in range(0, len(X_query), batch):
        chunk = X_query[i : i + batch]
        d = np.linalg.norm(chunk[:, None, :] - X_seed[None, :, :], axis=2).min(axis=1)
        min_dists.extend(d.tolist())
    return np.array(min_dists) > threshold


def train_and_save_aesthetic_regressor(
    X_seed: np.ndarray,
    y_seed: np.ndarray,
    coverage_threshold: float,
) -> None:
    """Fit Ridge on (X_seed, y_seed) with CV alpha selection, save full artifact.

    Artifact keys: pipe, X_seed, y_seed, coverage_threshold, r2, mae, alpha.
    X_seed and y_seed are stored so subsequent runs can retrain on old+new seeds
    without re-running SigLIP on the original seed photos.
    """
    import json as _json
    from datetime import datetime

    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(5, len(y_seed))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_r2, best_alpha = -np.inf, 100.0
    for alpha in [1.0, 10.0, 100.0, 500.0, 1000.0]:
        pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=alpha))])
        y_pred = cross_val_predict(pipe, X_seed, y_seed, cv=kf)
        r2 = r2_score(y_seed, y_pred)
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=best_alpha))])
    y_pred_cv = cross_val_predict(pipe, X_seed, y_seed, cv=kf)
    r2 = r2_score(y_seed, y_pred_cv)
    mae = mean_absolute_error(y_seed, y_pred_cv)

    pipe.fit(X_seed, y_seed)

    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "pipe": pipe,
        "X_seed": X_seed,
        "y_seed": y_seed,
        "coverage_threshold": coverage_threshold,
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 3),
        "alpha": best_alpha,
    }
    joblib.dump(artifact, _AESTHETIC_REG_PATH)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "teacher": "aesthetic-predictor-v2-5 (SigLIP SO400M)",
        "n_seed": int(len(y_seed)),
        "feature_dim": int(X_seed.shape[1]),
        "alpha": best_alpha,
        "coverage_threshold": round(coverage_threshold, 4),
        "cv_r2": round(float(r2), 4),
        "cv_mae": round(float(mae), 3),
    }
    (MODELS_DIR / "aesthetic_regressor_meta.json").write_text(_json.dumps(meta, indent=2))


# ── Auto-training helper (called from main.py after aesthetic pass) ───────────


def auto_train_aesthetic_regressor(all_cached_records: list[dict]) -> bool:
    """Retrain Ridge on all SigLIP-labelled records accumulated so far.

    Filters to records with aesthetic_score_source=="siglip" (or no source
    field, treated as siglip for backward compat) to keep pseudo-labels from
    regressor predictions out of the training set.

    Returns True if saved, False if fewer than 2 labelled records found.
    """
    import json as _json
    from datetime import datetime

    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    labelled = [
        r for r in all_cached_records if r.get("dinov3") and r.get("aesthetic_score") is not None and r.get("aesthetic_score_source", "siglip") == "siglip"
    ]
    n = len(labelled)
    if n < 2:
        return False

    X = np.array([r["dinov3"] for r in labelled], dtype=np.float32)
    y = np.array([r["aesthetic_score"] for r in labelled], dtype=np.float32)

    n_splits = min(5, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_r2, best_alpha = -np.inf, 100.0
    for alpha in [1.0, 10.0, 100.0, 500.0, 1000.0]:
        pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=alpha))])
        y_pred = cross_val_predict(pipe, X, y, cv=kf)
        r2 = r2_score(y, y_pred)
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=best_alpha))])
    y_pred_cv = cross_val_predict(pipe, X, y, cv=kf)
    r2 = r2_score(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)

    pipe.fit(X, y)

    # Preserve X_seed / y_seed / coverage_threshold from existing artifact
    existing: dict = {}
    if _AESTHETIC_REG_PATH.exists():
        try:
            existing = joblib.load(_AESTHETIC_REG_PATH)
        except Exception:
            pass

    MODELS_DIR.mkdir(exist_ok=True)
    artifact: dict = {
        "pipe": pipe,
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 3),
        "alpha": best_alpha,
    }
    for key in ("X_seed", "y_seed", "coverage_threshold"):
        if key in existing:
            artifact[key] = existing[key]
    joblib.dump(artifact, _AESTHETIC_REG_PATH)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "teacher": "aesthetic-predictor-v2-5 (SigLIP SO400M)",
        "n_samples": n,
        "feature_dim": int(X.shape[1]),
        "alpha": best_alpha,
        "cv_r2": round(float(r2), 4),
        "cv_mae": round(float(mae), 3),
    }
    (MODELS_DIR / "aesthetic_regressor_meta.json").write_text(_json.dumps(meta, indent=2))
    return True

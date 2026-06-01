"""
Train a Ridge regression head that predicts aesthetic_score (0-100) from
cached DINOv3-B embeddings, using the SigLIP aesthetic-predictor-v2-5 scores
already in the cache as ground-truth labels.

After training, main.py uses this model in Pass 4a instead of loading the
full SigLIP aesthetic model, cutting ~22 min off a full pipeline run.

Output:
  models/aesthetic_regressor.joblib   — {"pipe": Pipeline, "r2": float, "mae": float}
  models/aesthetic_regressor_meta.json

Usage:
  uv run python scripts/train_aesthetic_regressor.py
  uv run python scripts/train_aesthetic_regressor.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path("artifacts")
OUT_PATH = MODELS_DIR / "aesthetic_regressor.joblib"
META_PATH = MODELS_DIR / "aesthetic_regressor_meta.json"


def load_data() -> tuple[np.ndarray, np.ndarray]:
    from cache import load_all_cached

    records = [r for r in load_all_cached() if r.get("dinov3") and r.get("aesthetic_score") is not None]
    if not records:
        raise SystemExit("No records with both DINOv3-B embeddings and aesthetic_score in cache. Run the full pipeline first.")

    print(f"Training set: {len(records)} photos")
    X = np.array([r["dinov3"] for r in records], dtype=np.float32)
    y = np.array([r["aesthetic_score"] for r in records], dtype=np.float32)
    print(f"  X: {X.shape}  y: [{y.min():.1f}, {y.max():.1f}]  mean={y.mean():.1f}  std={y.std():.1f}")
    return X, y


def sweep_alpha(X: np.ndarray, y: np.ndarray) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("\nAlpha sweep (5-fold CV):")
    best_r2, best_alpha = -np.inf, 10.0
    for alpha in [0.1, 1.0, 10.0, 100.0, 500.0]:
        pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=alpha))])
        y_pred = cross_val_predict(pipe, X, y, cv=kf)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        marker = " ◀" if r2 > best_r2 else ""
        print(f"  alpha={alpha:>6}  R²={r2:.4f}  MAE={mae:.3f}{marker}")
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha
    return best_alpha


def evaluate(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=alpha))])
    y_pred = cross_val_predict(pipe, X, y, cv=kf)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    residuals = y - y_pred
    print(f"\n5-fold CV  R²={r2:.4f}  MAE={mae:.3f}")
    print(f"  Within ±2  pts: {(np.abs(residuals) <= 2).mean() * 100:.1f}%")
    print(f"  Within ±5  pts: {(np.abs(residuals) <= 5).mean() * 100:.1f}%")
    print(f"  Within ±10 pts: {(np.abs(residuals) <= 10).mean() * 100:.1f}%")
    print(f"  >±10 pts (large error): {(np.abs(residuals) > 10).mean() * 100:.1f}%")

    # Inference speed
    pipe.fit(X, y)
    t0 = time.perf_counter()
    _ = pipe.predict(X)
    infer_ms = (time.perf_counter() - t0) * 1000
    print(f"  Inference: {infer_ms:.1f}ms for {len(X)} photos ({len(X) / (infer_ms / 1000):.0f} img/s)")
    print(f"  vs SigLIP: ~1320s for {len(X)} photos → {len(X) / 1320:.1f} img/s")
    return r2, mae


def train_and_save(X: np.ndarray, y: np.ndarray, alpha: float, r2: float, mae: float) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=alpha))])
    pipe.fit(X, y)

    artifact = {"pipe": pipe, "r2": round(float(r2), 4), "mae": round(float(mae), 3), "alpha": alpha}
    joblib.dump(artifact, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")

    meta = {
        "trained_at": datetime.now().isoformat(),
        "backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "teacher": "aesthetic-predictor-v2-5 (SigLIP SO400M)",
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "alpha": alpha,
        "cv_r2": round(float(r2), 4),
        "cv_mae": round(float(mae), 3),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Metadata → {META_PATH}")
    print("\nDone. main.py will use this model in Pass 4a automatically.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="CV metrics only, no save")
    args = parser.parse_args()

    X, y = load_data()
    best_alpha = sweep_alpha(X, y)
    r2, mae = evaluate(X, y, best_alpha)

    if not args.eval_only:
        train_and_save(X, y, best_alpha, r2, mae)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate KMeans cluster coherence on DINOv3-B embeddings.

Uses SigLIP scene_type as proxy labels — no manual annotation needed.

Metrics:
  NMI (Normalized Mutual Information) — how well KMeans clusters align with
    scene_type labels. 0 = random, 1 = perfect alignment. A score > 0.3 on
    10 scene classes with a real library means the embeddings carry meaningful
    scene structure.
  Within-cluster cosine similarity — mean similarity of all pairs inside each
    cluster. High = visually coherent clusters.
  Between-cluster cosine similarity — mean similarity between cluster centroids.
    Low = well-separated clusters.
  Separation ratio (within/between) — the key combined metric. > 1.5 means
    intra-cluster similarity substantially exceeds inter-cluster similarity.

Run from project root:
    uv run python scripts/eval_clusters.py
    uv run python scripts/eval_clusters.py --n-clusters 8
    uv run python scripts/eval_clusters.py --output docs/cluster_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))

from cache import load_all_cached


def main() -> dict:
    parser = argparse.ArgumentParser(description="Evaluate KMeans cluster coherence on DINOv3-B embeddings")
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters (default: auto, same as pipeline)")
    parser.add_argument("--output", type=Path, default=Path("docs/cluster_eval.json"), help="Write JSON results here (default: docs/cluster_eval.json)")
    args = parser.parse_args()

    print("Loading embeddings from cache...")
    records = load_all_cached()
    valid = [
        (r["hash"], r["dinov3"], r.get("scene", {}).get("scene_type", "unknown")) for r in records if r.get("dinov3") and r.get("scene", {}).get("scene_type")
    ]
    print(f"  {len(valid)} records with DINOv3-B + scene_type")

    if len(valid) < 30:
        print("ERROR: fewer than 30 records have both DINOv3-B embeddings and scene_type", file=sys.stderr)
        sys.exit(1)

    _hashes, vecs_raw, scene_types = zip(*valid, strict=False)
    scene_types = list(scene_types)
    matrix = np.array(vecs_raw, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix_norm = matrix / norms

    n_clusters = args.n_clusters or max(2, min(10, len(valid) // 30))
    print(f"  Fitting KMeans (k={n_clusters}, n_init=5)...")
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    cluster_labels = km.fit_predict(matrix_norm)

    # NMI vs scene_type
    le = LabelEncoder()
    scene_encoded = le.fit_transform(scene_types)
    nmi = float(normalized_mutual_info_score(scene_encoded, cluster_labels))

    # Within-cluster cosine similarity and per-cluster purity
    within_sims: list[float] = []
    per_cluster: dict[str, dict] = {}
    for cid in range(n_clusters):
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) < 2:
            continue
        vecs_c = matrix_norm[idx]
        sim_mat = vecs_c @ vecs_c.T
        n = len(idx)
        mean_sim = float(sim_mat[np.triu_indices(n, k=1)].mean())
        within_sims.append(mean_sim)
        scene_counts = Counter(scene_types[i] for i in idx)
        top_scene, top_count = scene_counts.most_common(1)[0]
        per_cluster[str(cid)] = {
            "size": int(n),
            "top_scene": top_scene,
            "purity": round(top_count / n, 3),
            "within_cosine_sim": round(mean_sim, 4),
        }

    mean_within = float(np.mean(within_sims)) if within_sims else 0.0

    # Between-cluster centroid similarity
    centroids_norm = km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-9)
    centroid_sims = centroids_norm @ centroids_norm.T
    between_vals = centroid_sims[np.triu_indices(n_clusters, k=1)]
    mean_between = float(between_vals.mean()) if len(between_vals) else 0.0
    separation = mean_within / (mean_between + 1e-9)

    print(f"\nCluster coherence  (k={n_clusters}, n={len(valid)} photos):")
    print(f"  NMI vs scene_type        : {nmi:.4f}  [0=random, 1=perfect]")
    print(f"  Within-cluster cosine sim: {mean_within:.4f}")
    print(f"  Between-cluster cosine sim: {mean_between:.4f}")
    print(f"  Separation ratio (W/B)   : {separation:.2f}x  [>1.5 = meaningful structure]")
    print()
    print(f"  {'Cluster':>8}  {'Size':>6}  {'Top scene':>24}  {'Purity':>8}  {'Within sim':>10}")
    print("  " + "-" * 64)
    for cid, info in sorted(per_cluster.items(), key=lambda x: -x[1]["size"]):
        print(f"  {cid:>8}  {info['size']:>6}  {info['top_scene']:>24}  {info['purity']:>8.3f}  {info['within_cosine_sim']:>10.4f}")

    results = {
        "n_clusters": n_clusters,
        "n_photos": len(valid),
        "nmi_vs_scene_type": round(nmi, 4),
        "mean_within_cluster_cosine_sim": round(mean_within, 4),
        "mean_between_cluster_cosine_sim": round(mean_between, 4),
        "separation_ratio": round(separation, 2),
        "clusters": per_cluster,
    }

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written → {args.output}")

    return results


if __name__ == "__main__":
    main()

"""Quantitative validation for two heuristics whose thresholds were previously
unjustified magic numbers:

  - near-duplicate clustering (DBSCAN eps=0.05 on cosine distance)
  - pose classification (shoulder->hip->ankle vertical ratios)

These tests build labeled inputs with known ground truth and report precision /
recall, turning "it detects near-duplicates" into a measured number.

Synthetic fixture uses 768-d vectors (matching real DINOv3-B output) with three
difficulty tiers:
  - Near-duplicates: same burst frame, tiny perturbation (cosine dist < 0.001)
  - Hard negatives:  related scene, same base + 0.6x noise (cosine dist ~0.15) -
                     similar enough to be visually related but outside eps=0.05
  - True singletons: independent random vectors (cosine dist >> 0.05)
"""

import itertools

import numpy as np

from analysis.embeddings import NEAR_DUP_EPS, near_duplicate_labels
from extractors.pose import _classify_pose

DIM = 768  # matches DINOv3-B embedding dimension


def _make_embeddings(seed: int = 0):
    """Build 768-d embeddings with three tiers:
      - 5 near-duplicate groups (2-3 frames each, cosine dist < 0.001)
      - 5 hard negatives (related to group bases, cosine dist ~0.15)
      - 10 true singletons (independent random vectors)
    Returns (vectors, ground_truth_group_id_per_vector).
    Hard negatives and singletons get group id >= 100 (no partner → not duplicates).
    """
    rng = np.random.default_rng(seed)
    vectors: list[np.ndarray] = []
    groups: list[int] = []
    bases: list[np.ndarray] = []

    # Near-duplicate groups: a base vector + tiny perturbation per frame.
    # At DIM=768, noise scale 0.01 gives cosine dist ~ 0.0001 — well inside eps=0.05.
    group_sizes = [3, 2, 2, 3, 2]
    for gid, size in enumerate(group_sizes):
        base = rng.standard_normal(DIM)
        bases.append(base)
        for _ in range(size):
            vectors.append(base + 0.01 * rng.standard_normal(DIM))
            groups.append(gid)

    # Hard negatives: correlated with a group base but cosine dist ~0.15 (outside eps).
    # v = base + 0.6*noise → after normalization cosine_sim ≈ 1/sqrt(1+0.36) ≈ 0.857
    for i, base in enumerate(bases):
        vectors.append(base + 0.6 * rng.standard_normal(DIM))
        groups.append(200 + i)  # singleton group id — no partner

    # True singletons: independent vectors, far from everything.
    for s in range(10):
        vectors.append(rng.standard_normal(DIM))
        groups.append(100 + s)

    return np.array(vectors, dtype=np.float32), groups


def _pairs_from_labels(labels) -> set:
    """All unordered index pairs sharing a non-(-1) label = predicted duplicates."""
    pairs = set()
    by_label: dict = {}
    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue
        by_label.setdefault(lbl, []).append(i)
    for members in by_label.values():
        pairs.update(itertools.combinations(sorted(members), 2))
    return pairs


def _pairs_from_groups(groups) -> set:
    pairs = set()
    by_group: dict = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    for members in by_group.values():
        if len(members) >= 2:
            pairs.update(itertools.combinations(sorted(members), 2))
    return pairs


def test_near_duplicate_precision_recall():
    """Synthetic 768-d fixture with near-duplicates, hard negatives, and singletons."""
    vectors, groups = _make_embeddings()
    labels = near_duplicate_labels(vectors, eps=NEAR_DUP_EPS)

    predicted = _pairs_from_labels(labels)
    truth = _pairs_from_groups(groups)

    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(truth) if truth else 1.0

    # eps=0.05 must perfectly separate near-identical frames (dist < 0.001) from
    # hard negatives (dist ~0.15) and singletons — no false merges, no missed duplicates.
    assert precision == 1.0, f"near-dup precision={precision:.3f} (false merges present)"
    assert recall == 1.0, f"near-dup recall={recall:.3f} (missed duplicates)"


def _keypoints(shoulder_y: float, hip_y: float, ankle_y: float) -> np.ndarray:
    """17x2 COCO keypoints with only the shoulder(5,6)/hip(11,12)/ankle(15,16) rows set."""
    kp = np.ones((17, 2), dtype=np.float32)  # nonzero default so guards don't trip
    for idx in (5, 6):
        kp[idx, 1] = shoulder_y
    for idx in (11, 12):
        kp[idx, 1] = hip_y
    for idx in (15, 16):
        kp[idx, 1] = ankle_y
    return kp


def test_pose_classification_accuracy():
    # (shoulder_y, hip_y, ankle_y) -> expected label. Image y grows downward.
    cases = [
        # standing: legs/torso > 1.2
        ((100, 200, 420), "standing"),
        ((50, 180, 400), "standing"),
        ((80, 160, 360), "standing"),
        # sitting: 0.5 < ratio <= 1.2
        ((100, 200, 280), "sitting"),
        ((100, 200, 300), "sitting"),
        ((120, 220, 320), "sitting"),
        # crouching: 0.1 < ratio <= 0.5
        ((100, 200, 230), "crouching"),
        ((100, 200, 240), "crouching"),
        # lying: ratio <= 0.1
        ((100, 200, 205), "lying"),
        ((100, 200, 208), "lying"),
    ]

    labels = sorted({c[1] for c in cases})
    tp = dict.fromkeys(labels, 0)
    fp = dict.fromkeys(labels, 0)
    fn = dict.fromkeys(labels, 0)
    correct = 0

    for (sh, hip, ank), expected in cases:
        pred = _classify_pose(_keypoints(sh, hip, ank))
        if pred == expected:
            correct += 1
            tp[expected] += 1
        else:
            fn[expected] += 1
            if pred in fp:
                fp[pred] += 1

    accuracy = correct / len(cases)
    # Per-class precision/recall (macro) — documents that the ratio thresholds
    # separate the four poses on canonical geometry.
    macro_recall = np.mean([tp[c] / (tp[c] + fn[c]) for c in labels if (tp[c] + fn[c])])

    assert accuracy == 1.0, f"pose accuracy={accuracy:.3f} on canonical cases"
    assert macro_recall == 1.0, f"pose macro recall={macro_recall:.3f}"

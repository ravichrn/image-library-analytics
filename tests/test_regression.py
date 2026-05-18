"""Regression tests for specific past bugs."""

import pytest

# ── IQ histogram bucketing ────────────────────────────────────────────────────


def _iq_distribution(iq_scores: list[float]) -> list[int]:
    """Replicate the bucketing logic from analysis/advanced.py."""
    buckets = [0] * 10
    for s in iq_scores:
        buckets[min(9, int(s / 10))] += 1
    return buckets


def test_iq_histogram_buckets_correct():
    # Score 100 must land in bucket 9, not overflow.
    dist = _iq_distribution([0.0, 10.0, 50.0, 99.9, 100.0])
    assert dist[0] == 1  # 0.0  → bucket 0
    assert dist[1] == 1  # 10.0 → bucket 1
    assert dist[5] == 1  # 50.0 → bucket 5
    assert dist[9] == 2  # 99.9 and 100.0 both clamped to bucket 9
    assert sum(dist) == 5


def test_iq_histogram_buckets_never_overflow():
    # Old bug: int(s * 10) for s=100 → index 1000, IndexError.
    try:
        dist = _iq_distribution([100.0])
    except IndexError:
        pytest.fail("Score 100.0 caused IndexError — bucketing formula is wrong")
    assert dist[9] == 1


# ── UMAP points include hash ──────────────────────────────────────────────────


def test_umap_points_include_hash():
    """analysis.embeddings.analyze must embed 'hash' in every UMAP point."""
    import numpy as np

    from analysis.embeddings import analyze

    rng = np.random.default_rng(0)
    records = [
        {
            "hash": f"hash{i:03d}",
            "path": f"/fake/img{i}.jpg",
            "dinov2": rng.random(768).tolist(),
            "aesthetic_score": 50.0 + i,
            "scene": {"scene_type": "nature"},
        }
        for i in range(20)
    ]
    result = analyze(records)
    points = result["umap"]["points"]
    assert points, "UMAP produced no points"
    for pt in points:
        assert "hash" in pt, f"UMAP point missing 'hash': {pt}"
        assert pt["hash"].startswith("hash"), f"hash looks wrong: {pt['hash']}"


# ── Strict dedup skips stem merge ─────────────────────────────────────────────


def test_strict_dedup_skips_stem_merge(monkeypatch):
    """With STRICT_DEDUP=true, same-stem records from local and Lightroom must stay separate."""
    local_record = {"hash": "aaabbb", "path": "/local/DSC0001.jpg", "source": "local"}
    lr_record = {"hash": "cccddd", "lightroom_filename_stem": "DSC0001", "source": "lightroom"}

    monkeypatch.setenv("STRICT_DEDUP", "true")
    monkeypatch.setenv("SOURCES", "local,lightroom")

    # Patch both source loaders to return our two fake records.
    import sources
    import sources.lightroom as _lr_mod
    import sources.local as _local_mod

    monkeypatch.setattr(_local_mod, "load_local", lambda **_kw: [local_record])
    monkeypatch.setattr(_lr_mod, "load_lightroom", lambda **_kw: [lr_record])

    result = sources.load_sources()
    hashes = {r["hash"] for r in result}
    assert len(result) == 2, f"Expected 2 records, got {len(result)}: {hashes}"
    assert "aaabbb" in hashes
    assert "cccddd" in hashes


def test_stem_merge_happens_without_strict_dedup(monkeypatch):
    """Without STRICT_DEDUP, same-stem local+Lightroom records should merge into one."""
    local_record = {"hash": "aaabbb", "path": "/local/DSC0001.jpg", "source": "local"}
    lr_record = {"hash": "cccddd", "lightroom_filename_stem": "DSC0001", "source": "lightroom"}

    monkeypatch.setenv("STRICT_DEDUP", "false")
    monkeypatch.setenv("SOURCES", "local,lightroom")

    import sources
    import sources.lightroom as _lr_mod
    import sources.local as _local_mod

    monkeypatch.setattr(_local_mod, "load_local", lambda **_kw: [local_record])
    monkeypatch.setattr(_lr_mod, "load_lightroom", lambda **_kw: [lr_record])

    result = sources.load_sources()
    assert len(result) == 1, f"Expected stem merge to produce 1 record, got {len(result)}"
    assert result[0]["source"] == "both"

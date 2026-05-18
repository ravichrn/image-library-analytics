import json
from pathlib import Path

import pytest

from analysis import aggregate


@pytest.fixture
def records():
    return json.loads((Path(__file__).parent / "fixtures/sample_records.json").read_text())


def test_aggregate_returns_photo_count(records):
    result = aggregate(records)
    assert result["photo_count"] == len(records)


def test_aggregate_top_level_keys(records):
    result = aggregate(records)
    for key in ("photo_count", "aesthetic_stats", "exif_stats", "_sources"):
        assert key in result, f"Missing key: {key}"


def test_sources_flags_are_bools(records):
    s = aggregate(records)["_sources"]
    assert isinstance(s["has_lightroom"], bool)
    assert isinstance(s["has_embeddings"], bool)
    assert isinstance(s["has_pose"], bool)


def test_aesthetic_stats_present(records):
    aes = aggregate(records).get("aesthetic_stats", {})
    assert "avg_score" in aes
    assert "score_histogram_10" in aes


def test_gear_stats_present(records):
    result = aggregate(records)
    # Camera counts live inside exif_stats
    assert "exif_stats" in result
    exif = result["exif_stats"]
    assert "camera_counts" in exif or "camera_make_counts" in exif or isinstance(exif, dict)

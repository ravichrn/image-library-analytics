import json
from pathlib import Path

import pytest

from coach_client import aggregate_flags, flag_record


@pytest.fixture
def records():
    return json.loads((Path(__file__).parent / "fixtures/sample_records.json").read_text())


def test_flag_record_returns_dict(records):
    flags = flag_record(records[0])
    assert isinstance(flags, dict)


def test_flag_record_values_are_floats(records):
    flags = flag_record(records[0])
    for key, val in flags.items():
        assert isinstance(val, float), f"Flag {key} value is not float: {val!r}"


def test_aggregate_flags_structure(records):
    result = aggregate_flags(records)
    assert "flags" in result
    assert "total_analyzed" in result
    assert result["total_analyzed"] == len(records)


def test_aggregate_flags_pct_sane(records):
    result = aggregate_flags(records)
    for key, info in result["flags"].items():
        assert 0 <= info["pct"] <= 100, f"Flag {key} pct out of range: {info['pct']}"


def test_iq_low_aesthetic_high_triggered(records):
    # records[4]: iq_score=30 (<40), aesthetic_score=70 (>65)
    flags = flag_record(records[4])
    assert "iq_low_aesthetic_high" in flags, "Expected iq_low_aesthetic_high flag on record 4"


def test_low_aesthetic_triggered(records):
    # records[3]: aesthetic_score=28 (<35)
    flags = flag_record(records[3])
    assert "low_aesthetic" in flags

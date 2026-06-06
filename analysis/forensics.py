import numpy as np


def analyze(records: list[dict]) -> dict:
    jpeg_records = [r.get("jpeg_quality", {}) for r in records if r.get("jpeg_quality")]
    jpeg_valid = [j for j in jpeg_records if j.get("jpeg_quality_factor") is not None]
    if jpeg_valid:
        _qf_vals = np.array([j["jpeg_quality_factor"] for j in jpeg_valid])
        _q1, _q3 = float(np.percentile(_qf_vals, 25)), float(np.percentile(_qf_vals, 75))
        _qf_fence = _q1 - 3 * (_q3 - _q1)  # low-quality outliers (lower is worse)
        _qf_extreme_count = int((_qf_vals < _qf_fence).sum())
        _qf_fence_rounded = round(_qf_fence, 1)
    else:
        _qf_extreme_count = 0
        _qf_fence_rounded = None

    # JPEG quality + IQ disagreement: heavily re-compressed JPEG with poor technical quality
    # Signals a re-exported or heavily edited JPEG that also has poor technical quality
    _jpeg_iq_conflict = sum(
        1
        for r in records
        if r.get("jpeg_quality", {}).get("jpeg_quality_factor") is not None
        and r.get("iq_score") is not None
        and r["jpeg_quality"]["jpeg_quality_factor"] < 75
        and r["iq_score"] < 40
    )

    jpeg_stats = {
        "total_jpegs": len(jpeg_valid),
        "reexported_count": sum(1 for j in jpeg_valid if not j.get("quant_table_nonstandard")),
        "reexported_pct": round(sum(1 for j in jpeg_valid if not j.get("quant_table_nonstandard")) / len(jpeg_valid) * 100, 1) if jpeg_valid else 0.0,
        "avg_quality_factor": round(float(np.mean([j["jpeg_quality_factor"] for j in jpeg_valid])), 1) if jpeg_valid else None,
        "low_quality_outlier_count": _qf_extreme_count,
        "low_quality_outlier_threshold": _qf_fence_rounded,
        "jpeg_iq_conflict_count": _jpeg_iq_conflict,
    }

    return {"jpeg_stats": jpeg_stats}

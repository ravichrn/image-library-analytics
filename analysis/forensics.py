import numpy as np


def analyze(records: list[dict]) -> dict:
    ela_records = [r.get("ela", {}) for r in records if r.get("ela")]
    ela_jpegs = [e for e in ela_records if e.get("ela_max_error") is not None]
    if ela_jpegs:
        _ela_vals = np.array([e["ela_max_error"] for e in ela_jpegs])
        _q1, _q3 = float(np.percentile(_ela_vals, 25)), float(np.percentile(_ela_vals, 75))
        _ela_fence = _q3 + 3 * (_q3 - _q1)
        _ela_extreme_count = int((_ela_vals > _ela_fence).sum())
        _ela_fence_rounded = round(_ela_fence, 1)
    else:
        _ela_extreme_count = 0
        _ela_fence_rounded = None

    # ELA/IQ disagreement: high compression artifact error + low IQ score
    # Signals a re-exported or heavily edited JPEG that also has poor technical quality
    _ela_iq_conflict = sum(
        1
        for r in records
        if r.get("ela", {}).get("ela_max_error") is not None and r.get("iq_score") is not None and r["ela"]["ela_max_error"] > 18 and r["iq_score"] < 40
    )

    ela_stats = {
        "total_jpegs": len(ela_jpegs),
        "suspicious_count": sum(1 for e in ela_jpegs if e.get("ela_suspicious")),
        "suspicious_pct": round(sum(1 for e in ela_jpegs if e.get("ela_suspicious")) / len(ela_jpegs) * 100, 1) if ela_jpegs else 0.0,
        "avg_max_error": round(float(np.mean([e["ela_max_error"] for e in ela_jpegs])), 2) if ela_jpegs else None,
        "avg_mean_error": round(float(np.mean([e["ela_mean_error"] for e in ela_jpegs])), 4) if ela_jpegs else None,
        "extreme_outlier_count": _ela_extreme_count,
        "extreme_outlier_threshold": _ela_fence_rounded,
        "ela_iq_conflict_count": _ela_iq_conflict,
    }

    return {"ela_stats": ela_stats}

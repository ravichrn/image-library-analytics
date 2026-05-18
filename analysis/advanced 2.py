from collections import Counter, defaultdict

import numpy as np

from coach_client import THRESHOLDS

from ._helpers import _mean


def analyze(records: list[dict]) -> dict:
    # ── VQA attribute distributions ──────────────────────────────────────────
    VQA_KEYS = ["has_person", "setting", "time_of_day", "weather", "season"]
    vqa_counters: dict[str, Counter] = {k: Counter() for k in VQA_KEYS}
    vqa_total = 0
    for r in records:
        cap = r.get("caption", {})
        if not cap:
            continue
        vqa_total += 1
        for k in VQA_KEYS:
            val = cap.get(k)
            if val:
                vqa_counters[k][val] += 1

    def _vqa_dist(counter: Counter) -> list[dict]:
        total = sum(counter.values()) or 1
        return [{"label": label, "count": cnt, "pct": round(cnt / total * 100, 1)} for label, cnt in counter.most_common(6)]

    visual_attributes = {k: _vqa_dist(vqa_counters[k]) for k in VQA_KEYS}
    visual_attributes["total_analyzed"] = vqa_total

    # ── Aesthetic score by VQA condition ────────────────────────────────────
    aesthetic_by_vqa: dict[str, dict[str, float]] = {}
    for field in VQA_KEYS:
        buckets: dict[str, list] = {}
        for r in records:
            val = (r.get("caption") or {}).get(field, "")
            score = r.get("aesthetic_score")
            if val and score:
                buckets.setdefault(val, []).append(score)
        aesthetic_by_vqa[field] = {k: round(sum(v) / len(v), 1) for k, v in sorted(buckets.items(), key=lambda x: -(sum(x[1]) / len(x[1]))) if len(v) >= 3}

    # ── Saliency stats ────────────────────────────────────────────────────────
    sal_areas = [r.get("saliency", {}).get("subject_area_pct") for r in records]
    sal_off = [r.get("saliency", {}).get("subject_off_center") for r in records]
    sal_by_scene: dict[str, list] = defaultdict(list)
    placement_grid = [[0] * 3 for _ in range(3)]
    for r in records:
        sal = r.get("saliency", {})
        if not sal or sal.get("subject_cx") is None:
            continue
        sc = r.get("scene", {}).get("scene_type")
        area = sal.get("subject_area_pct")
        if area is not None and sc:
            sal_by_scene[sc].append(area)
        cx = sal.get("subject_cx")
        cy = sal.get("subject_cy")
        if cx is not None and cy is not None:
            col_z = min(2, int(cx * 3))
            row_z = min(2, int(cy * 3))
            placement_grid[row_z][col_z] += 1
    total_placed = sum(sum(row) for row in placement_grid) or 1
    placement_grid_norm = [[round(v / total_placed, 4) for v in row] for row in placement_grid]
    saliency_stats = {
        "avg_subject_area": _mean([v for v in sal_areas if v is not None]),
        "avg_off_center": _mean([v for v in sal_off if v is not None]),
        "by_scene": {sc: round(float(np.mean(v)), 4) for sc, v in sal_by_scene.items() if v},
        "placement_distribution": placement_grid_norm,
        "total_analyzed": sum(1 for v in sal_areas if v is not None),
    }

    # ── MUSIQ IQ score stats ─────────────────────────────────────────────────
    iq_scores = [r.get("iq_score") for r in records if r.get("iq_score") is not None]
    if iq_scores:
        buckets_iq = [0] * 10
        for s in iq_scores:
            buckets_iq[min(int(s * 10), 9)] += 1
        iq_stats = {
            "avg": _mean(iq_scores),
            "std": float(np.std(iq_scores)),
            "distribution": buckets_iq,
            "high_aesthetic_low_iq_count": sum(
                1
                for r in records
                if (r.get("aesthetic_score") or 0) > THRESHOLDS["aesthetic_high"] and r.get("iq_score") is not None and r["iq_score"] < THRESHOLDS["iq_floor"]
            ),
        }
    else:
        iq_stats = {"avg": None, "std": None, "distribution": [], "high_aesthetic_low_iq_count": 0, "unavailable": True}

    # ── Object frequency (from YOLOv8-Pose) ─────────────────────────────────
    obj_counter: Counter = Counter()
    for r in records:
        for obj in (r.get("pose_data") or {}).get("detected_objects", []):
            obj_counter[obj["label"]] += 1
    object_frequency = dict(obj_counter.most_common(20))

    # ── Pose stats (portrait photos only) ────────────────────────────────────
    portrait_records = [r for r in records if r.get("pose_data")]
    pose_types = [r.get("pose_data", {}).get("pose", {}).get("pose_type") for r in portrait_records if r.get("pose_data", {}).get("pose", {}).get("pose_type")]
    framing_tiers: dict[str, int] = {"tight": 0, "medium": 0, "wide": 0}
    framing_aesthetic: dict[str, list] = {"tight": [], "medium": [], "wide": []}
    pose_aesthetic: dict[str, list] = {}
    person_count_dist: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4+": 0}
    solo_vs_group: dict[str, int] = {"solo": 0, "group": 0}
    for r in portrait_records:
        pose = r.get("pose_data", {}).get("pose", {})
        coverage = pose.get("body_coverage")
        person_count = pose.get("person_count", 1) or 1
        pose_type = pose.get("pose_type")
        aesthetic = r.get("aesthetic_score")
        if coverage is not None:
            tier = "tight" if coverage < 0.25 else ("wide" if coverage > 0.55 else "medium")
            framing_tiers[tier] += 1
            if aesthetic:
                framing_aesthetic[tier].append(aesthetic)
        if pose_type and aesthetic:
            pose_aesthetic.setdefault(pose_type, []).append(aesthetic)
        key = str(min(person_count, 3)) if person_count <= 3 else "4+"
        person_count_dist[key] += 1
        solo_vs_group["solo" if person_count == 1 else "group"] += 1
    coverages = [
        r.get("pose_data", {}).get("pose", {}).get("body_coverage")
        for r in portrait_records
        if r.get("pose_data", {}).get("pose", {}).get("body_coverage") is not None
    ]
    pose_stats = {
        "pose_type_distribution": dict(Counter(pose_types)),
        "avg_body_coverage": _mean(coverages),
        "portrait_count": len(portrait_records),
        "framing_tiers": framing_tiers,
        "framing_aesthetic": {k: round(sum(v) / len(v), 1) if v else None for k, v in framing_aesthetic.items()},
        "pose_type_aesthetic": {k: round(sum(v) / len(v), 1) for k, v in sorted(pose_aesthetic.items(), key=lambda x: -(sum(x[1]) / len(x[1]))) if v},
        "person_count_dist": person_count_dist,
        "solo_vs_group": solo_vs_group,
    }

    return {
        "visual_attributes": visual_attributes,
        "aesthetic_by_vqa": aesthetic_by_vqa,
        "saliency_stats": saliency_stats,
        "iq_stats": iq_stats,
        "object_frequency": object_frequency,
        "pose_stats": pose_stats,
    }

from collections import Counter, defaultdict
from datetime import datetime as _dt

import numpy as np

from ._helpers import _time_of_day


def analyze(records: list[dict]) -> dict:
    # ── shooting hours ───────────────────────────────────────────────────────
    shooting_hours = {str(h): 0 for h in range(24)}
    for r in records:
        hour = r.get("exif", {}).get("hour")
        if hour is None:
            cap = r.get("lightroom_capture_date", "")
            if cap and "T" in cap:
                try:
                    hour = int(cap.split("T")[1].split(":")[0])
                except Exception:
                    pass
        if hour is not None:
            shooting_hours[str(hour)] = shooting_hours.get(str(hour), 0) + 1

    # ── monthly shooting distribution ────────────────────────────────────────
    monthly_counter: Counter = Counter()
    monthly_golden: Counter = Counter()
    for r in records:
        cap = r.get("lightroom_capture_date", "") or r.get("exif", {}).get("year_month", "")
        if cap and len(cap) >= 7:
            key = cap[:7]
            monthly_counter[key] += 1
            if _time_of_day(r) == "golden_hour":
                monthly_golden[key] += 1
    monthly_shooting = {m: {"total": monthly_counter[m], "golden_hour": monthly_golden.get(m, 0)} for m in sorted(monthly_counter)}

    # ── editing trends over time ──────────────────────────────────────────────
    trend_data: dict = defaultdict(lambda: {"aesthetics": [], "sats": [], "brightness": []})
    for r in records:
        ym = r.get("exif", {}).get("year_month")
        if not ym:
            continue
        score = r.get("aesthetic_score")
        sat = r.get("color", {}).get("avg_saturation")
        bri = r.get("color", {}).get("avg_brightness")
        if score is not None:
            trend_data[ym]["aesthetics"].append(score)
        if sat is not None:
            trend_data[ym]["sats"].append(sat)
        if bri is not None:
            trend_data[ym]["brightness"].append(bri)

    editing_trends = {}
    for ym in sorted(trend_data.keys()):
        d = trend_data[ym]
        editing_trends[ym] = {
            "count": max(len(d["aesthetics"]), len(d["sats"])),
            "avg_aesthetic": round(float(np.mean(d["aesthetics"])), 2) if d["aesthetics"] else None,
            "avg_saturation": round(float(np.mean(d["sats"])), 3) if d["sats"] else None,
            "avg_brightness": round(float(np.mean(d["brightness"])), 3) if d["brightness"] else None,
        }

    # ── day-of-week distribution (requires full date — Lightroom source) ───────
    dow_counts = {str(i): 0 for i in range(7)}  # 0=Monday … 6=Sunday
    for r in records:
        dt_str = r.get("lightroom_capture_date", "")
        if dt_str and len(dt_str) >= 10:
            try:
                dow_counts[str(_dt.fromisoformat(dt_str[:10]).weekday())] += 1
            except Exception:
                pass

    return {
        "shooting_hours": shooting_hours,
        "monthly_shooting": monthly_shooting,
        "editing_trends": editing_trends,
        "dow_distribution": dow_counts,
    }

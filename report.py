import json
from collections import Counter
from pathlib import Path

import jinja2
from markupsafe import Markup

OUTPUT_DIR = Path("docs")


def _best_conditions(photos: list) -> list[dict]:
    """Rank shooting conditions by avg aesthetic score (min 10 samples each)."""
    buckets: dict[str, list] = {}
    for p in photos:
        score = p.get("aesthetic_score")
        if score is None:
            continue
        exif = p.get("exif") or {}
        for key in ("time_of_day", "focal_category", "dof_category", "light_category", "shutter_category"):
            val = exif.get(key)
            if val:
                buckets.setdefault(f"{key}:{val}", []).append(score)
        scene = (p.get("scene") or {}).get("scene_type")
        if scene:
            buckets.setdefault(f"scene:{scene}", []).append(score)
    return sorted(
        [{"condition": k, "avg_score": round(sum(v) / len(v), 1), "count": len(v)} for k, v in buckets.items() if len(v) >= 10],
        key=lambda x: -x["avg_score"],
    )[:10]


def _compute_key_insights(agg: dict, photos: list, prime_pct: int | None) -> list[dict]:
    """Generate up to 10 actionable suggestions and quality issues."""
    insights: list[dict] = []

    scene_stats = agg.get("scene_stats", {})
    aesthetic_stats = agg.get("aesthetic_stats", {})
    iq_stats = agg.get("iq_stats", {})
    shooting_hours = agg.get("shooting_hours", {})
    color_stats = agg.get("color_stats", {})
    aesthetic_by_scene = agg.get("aesthetic_by_scene", {})
    visual_attributes = agg.get("visual_attributes", {})

    dom_scene = scene_stats.get("dominant_scene")

    # Best-scoring scene ≠ dominant scene → actionable redirect
    if aesthetic_by_scene:

        def _scene_score(item):
            val = item[1]
            if isinstance(val, dict):
                return val.get("avg_score") or val.get("avg_aesthetic") or 0
            return val or 0

        best_scene_name, best_val = max(aesthetic_by_scene.items(), key=_scene_score)
        best_score = _scene_score((best_scene_name, best_val))
        if best_score and best_scene_name != dom_scene:
            insights.append(
                {
                    "type": "suggestion",
                    "text": (
                        f"Your {best_scene_name} shots score highest aesthetically"
                        f" ({best_score:.1f} avg) but aren't your most-shot subject — consider shooting more of it."
                    ),
                }
            )

    # Low golden-hour shooting
    if shooting_hours:
        golden_count = sum(shooting_hours.get(str(h), 0) for h in [5, 6, 7, 17, 18, 19])
        total_count = sum(shooting_hours.values())
        if total_count > 0:
            golden_pct = round(golden_count / total_count * 100)
            if golden_pct < 8:
                insights.append(
                    {
                        "type": "suggestion",
                        "text": (f"Only {golden_pct}% of shots during golden hour — early morning or late afternoon light could lift scores meaningfully."),
                    }
                )

    # High score variance → cull more aggressively
    score_std = aesthetic_stats.get("score_std")
    if score_std and score_std > 10:
        insights.append(
            {
                "type": "issue",
                "text": (f"High aesthetic score variance (σ={score_std:.1f}) — consider culling more aggressively or standardising your edit style."),
            }
        )

    # Low avg IQ → technical quality concern
    if iq_stats and not iq_stats.get("unavailable"):
        avg_iq = iq_stats.get("avg") or iq_stats.get("avg_score")
        if avg_iq is not None and avg_iq < 50:
            insights.append(
                {
                    "type": "issue",
                    "text": (f"Average technical IQ score is {avg_iq:.0f}/100 — check for focus, motion blur, or exposure issues during culling."),
                }
            )

        # High IQ–aesthetic mismatch (intentional creative choices)
        low_iq_high_aes = iq_stats.get("high_aesthetic_low_iq_count", 0)
        if low_iq_high_aes > 0:
            insights.append(
                {
                    "type": "suggestion",
                    "text": (
                        f"{low_iq_high_aes} images score high aesthetically but low technically"
                        " — likely intentional (grain, motion blur, shallow DoF). Worth reviewing."
                    ),
                }
            )

    # Overcast / diffuse light heavy library
    if visual_attributes:
        for item in visual_attributes.get("weather", []):
            if item.get("label") in ("cloudy", "rainy", "foggy") and item.get("pct", 0) >= 35:
                insights.append(
                    {
                        "type": "suggestion",
                        "text": (
                            f"{item['pct']}% of shots in overcast/cloudy conditions"
                            " — diffuse light is forgiving; try lifting shadows in post to restore contrast."
                        ),
                    }
                )
                break

    # Cool-dominant palette
    warmth = color_stats.get("warmth_distribution", {})
    wt = sum(warmth.values()) if warmth else 0
    if wt > 0:
        cool_pct = round(warmth.get("cool", 0) / wt * 100)
        if cool_pct >= 55:
            insights.append(
                {
                    "type": "suggestion",
                    "text": (f"{cool_pct}% cool-toned images — strong blue/teal palette. If unintentional, try warming white balance slightly in your preset."),
                }
            )

    # Zoom-heavy kit
    if prime_pct is not None and prime_pct < 30:
        insights.append(
            {
                "type": "suggestion",
                "text": (f"Zoom-heavy kit ({100 - prime_pct}% zoom lenses) — primes may improve sharpness and force more deliberate composition."),
            }
        )

    # Centred compositions dominate → weak thirds usage
    comp_stats = agg.get("composition_stats", {})
    thirds = comp_stats.get("thirds_score")
    if thirds is not None and thirds < 0.65:
        insights.append(
            {
                "type": "suggestion",
                "text": (
                    f"Thirds score {thirds:.2f} — most shots are centred (1.0 = fully off-centre). Try placing subjects near rule-of-thirds intersections."
                ),
            }
        )

    # Best focal category ≠ most-used focal category
    focal_dist = agg.get("exif_stats", {}).get("focal_category_distribution", {})
    aesthetic_by_scene = agg.get("aesthetic_by_scene", {})
    if focal_dist:
        dom_focal = max(focal_dist, key=focal_dist.get)
        # look for focal hint in best_conditions via aesthetic_by_vqa — skip if not enough data
        # simple version: if wide is dominant but telephoto has higher avg aes (from shooting_profile)
        sp = agg.get("shooting_profile_by_context", [])
        focal_aes: dict[str, list] = {}
        for row in sp:
            fc = row.get("focal_category")
            aes = row.get("avg_aesthetic")
            if fc and aes:
                focal_aes.setdefault(fc, []).append(aes)
        if focal_aes and dom_focal in focal_aes:
            best_focal = max(focal_aes, key=lambda k: sum(focal_aes[k]) / len(focal_aes[k]))
            if best_focal != dom_focal:
                insights.append(
                    {
                        "type": "suggestion",
                        "text": (
                            f"Your {best_focal} shots score highest but you shoot mostly {dom_focal} — try experimenting more with {best_focal} focal lengths."
                        ),
                    }
                )

    # Small subjects on average
    saliency_stats = agg.get("saliency_stats", {})
    avg_subject_area = saliency_stats.get("avg_subject_area")
    if avg_subject_area is not None and avg_subject_area < 0.06:
        insights.append(
            {
                "type": "suggestion",
                "text": (
                    f"Average subject fills only {avg_subject_area * 100:.1f}% of the frame. Getting closer or using longer focal lengths can add impact."
                ),
            }
        )

    return insights[:10]


SCHEMA_VERSION = "1"


def generate_json(data: dict, path: Path = OUTPUT_DIR / "results.json") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = {"schema_version": SCHEMA_VERSION, **data}
    path.write_text(json.dumps(output, indent=2))


def generate_html(data: dict, path: Path = OUTPUT_DIR / "report.html") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates"), autoescape=True)
    env.filters["tojson"] = lambda v: Markup(json.dumps(v))
    template = env.get_template("report.html.j2")

    agg = data.get("aggregated", {})
    photos = data.get("photos", [])

    sources = agg.get("_sources", {})
    # ── Aggregate fg/bg palettes from saliency data ───────────────────────
    _fg_palette_colors: list = []
    _bg_palette_colors: list = []
    for p in photos:
        sal = p.get("saliency") or {}
        for sw in sal.get("fg_palette") or []:
            if len(_fg_palette_colors) < 40:
                _fg_palette_colors.append(sw["rgb"])
        for sw in sal.get("bg_palette") or []:
            if len(_bg_palette_colors) < 40:
                _bg_palette_colors.append(sw["rgb"])

    # ── Enrich UMAP points with EXIF for tooltip ──────────────────────────
    _hash_exif: dict = {}
    for p in photos:
        h = p.get("hash")
        ex = p.get("exif") or {}
        if h and (ex.get("focal_length_mm") or ex.get("aperture_f") or ex.get("iso")):
            _hash_exif[h] = {
                "focal_length_mm": ex.get("focal_length_mm"),
                "aperture_f": ex.get("aperture_f"),
                "iso": ex.get("iso"),
            }
    umap_points = []
    for pt in agg.get("umap", {}).get("points", []):
        h = pt.get("hash", "")
        umap_points.append({**pt, "exif": _hash_exif.get(h, {})})

    # ── Derived stats for hero tiles ──────────────────────────────────────
    fl_raw = agg.get("focal_length_histogram", [])
    fl_counter = Counter(round(f) for f in fl_raw if f is not None)
    focal_length_exact = fl_counter.most_common(20)

    iso_raw = agg.get("iso_histogram", [])
    iso_median = sorted(iso_raw)[len(iso_raw) // 2] if iso_raw else None
    iso_label = None
    if iso_median is not None:
        if iso_median <= 100:
            iso_label = "bright-light shooter"
        elif iso_median <= 400:
            iso_label = "mixed-light shooter"
        elif iso_median <= 1600:
            iso_label = "indoor-light shooter"
        else:
            iso_label = "low-light shooter"

    tsa = agg.get("technical_settings_analysis", {})
    top_lenses = tsa.get("top_lenses", {}) or {}
    prime_count = zoom_count = 0
    for lens, cnt in top_lenses.items():
        lens_str = str(lens)
        # zoom lenses typically have a dash/range in their name (e.g. "24-70mm")
        if ("-" in lens_str or "–" in lens_str) and any(c.isdigit() for c in lens_str):
            zoom_count += cnt
        else:
            prime_count += cnt
    total_lens = prime_count + zoom_count
    prime_pct = round(prime_count / total_lens * 100) if total_lens else None

    shutter_dist = tsa.get("shutter_category_distribution", {}) or {}
    shutter_label_map = {"freeze": "Fast Shutter", "hand": "Handheld", "slow": "Slow Shutter", "bulb": "Bulb"}
    dominant_shutter = None
    dominant_shutter_pct = None
    if shutter_dist:
        dom_cat = max(shutter_dist, key=shutter_dist.get)
        dom_cnt = shutter_dist[dom_cat]
        total_sh = sum(shutter_dist.values())
        dominant_shutter = shutter_label_map.get(dom_cat, dom_cat)
        dominant_shutter_pct = round(dom_cnt / total_sh * 100) if total_sh else None

    # ── IQ score per scene ────────────────────────────────────────────────
    _iq_by_scene: dict = {}
    for p in photos:
        scene = (p.get("scene") or {}).get("scene_type")
        iq = p.get("iq_score")
        if scene and iq is not None:
            _iq_by_scene.setdefault(scene, []).append(iq)
    iq_by_scene = {s: round(sum(v) / len(v), 1) for s, v in _iq_by_scene.items()}

    # ── IQ score per focal category ───────────────────────────────────────
    _iq_by_focal: dict = {}
    for p in photos:
        fc = (p.get("exif") or {}).get("focal_category")
        iq = p.get("iq_score")
        if fc and iq is not None:
            _iq_by_focal.setdefault(fc, []).append(iq)
    iq_by_focal = {fc: round(sum(v) / len(v), 1) for fc, v in _iq_by_focal.items() if len(v) >= 5}

    # ── Enrich editing_trends with avg IQ per month ───────────────────────
    _iq_by_month: dict = {}
    for p in photos:
        ym = (p.get("exif") or {}).get("year_month")
        iq = p.get("iq_score")
        if ym and iq is not None:
            _iq_by_month.setdefault(ym, []).append(iq)
    editing_trends_enriched: dict = {}
    for month, v in agg.get("editing_trends", {}).items():
        entry = dict(v)
        if month in _iq_by_month:
            entry["avg_iq"] = round(sum(_iq_by_month[month]) / len(_iq_by_month[month]), 2)
        editing_trends_enriched[month] = entry

    # ── Enrich editing_trends with avg edit intensity per month ───────────
    _ej = agg.get("editing_journey", {})
    for _month, _entry in editing_trends_enriched.items():
        _ej_month = _ej.get(_month, {})
        if _ej_month.get("avg_edit_intensity") is not None:
            _entry["avg_edit_intensity"] = round(_ej_month["avg_edit_intensity"], 2)

    # ── Editing style patterns for subsets (top 10%, curated albums) ──────
    from analysis.aesthetics import analyze as _analyze_aesthetics

    _scored = [p for p in photos if p.get("aesthetic_score") is not None]
    if len(_scored) >= 10:
        _p90_idx = int(len(_scored) * 0.9)
        _p90_val = sorted(_scored, key=lambda p: p["aesthetic_score"])[_p90_idx]["aesthetic_score"]
        editing_style_patterns_top10 = _analyze_aesthetics([p for p in _scored if p["aesthetic_score"] >= _p90_val]).get("editing_style_patterns", [])
    else:
        editing_style_patterns_top10 = []

    _album_photos = [p for p in photos if p.get("lightroom_album_names")]
    editing_style_patterns_albums = _analyze_aesthetics(_album_photos).get("editing_style_patterns", []) if len(_album_photos) >= 5 else []

    # ── Sort shooting profile by time of day ──────────────────────────────
    _time_order = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
    shooting_profile_sorted = sorted(
        agg.get("shooting_profile_by_context", []),
        key=lambda r: (_time_order.get((r.get("time_of_day") or "").lower(), 99), r.get("scene", "")),
    )

    # ── Best shooting conditions ──────────────────────────────────────────
    best_conditions = _best_conditions(photos)

    # ── Key insights ──────────────────────────────────────────────────────
    key_insights = _compute_key_insights(agg, photos, prime_pct)

    html = template.render(
        photos=photos,
        agg=agg,
        photo_count=len(photos),
        umap_points=umap_points,
        umap_total=agg.get("umap", {}).get("total", 0),
        scene_dist=agg.get("scene_stats", {}).get("scene_distribution", {}),
        color_stats=agg.get("color_stats", {}),
        composition_stats=agg.get("composition_stats", {}),
        aesthetic_stats=agg.get("aesthetic_stats", {}),
        clusters=agg.get("clusters", {}),
        shooting_hours=agg.get("shooting_hours", {}),
        focal_length_histogram=agg.get("focal_length_histogram", []),
        aperture_histogram=agg.get("aperture_histogram", []),
        iso_histogram=agg.get("iso_histogram", []),
        aesthetic_by_scene=agg.get("aesthetic_by_scene", {}),
        grid_heatmap=agg.get("grid_heatmap", []),
        sharpness_stats=agg.get("sharpness_stats", {}),
        exposure_stats=agg.get("exposure_stats", {}),
        megapixel_stats=agg.get("megapixel_stats", {}),
        color_by_scene=agg.get("color_by_scene", {}),
        hue_distribution=agg.get("hue_distribution", {}),
        saturation_histogram=agg.get("saturation_histogram", {}),
        editing_trends=editing_trends_enriched,
        editing_style_patterns=agg.get("editing_style_patterns", []),
        editing_style_patterns_top10=editing_style_patterns_top10,
        editing_style_patterns_albums=editing_style_patterns_albums,
        color_grading_stats=agg.get("color_grading_stats", {}),
        composition_patterns=agg.get("composition_patterns", []),
        visual_attributes=agg.get("visual_attributes", {}),
        develop_stats=agg.get("develop_stats", {}),
        lightroom_stats=agg.get("lightroom_stats", {}),
        signature_edit=agg.get("signature_edit", {}),
        monthly_shooting=agg.get("monthly_shooting", {}),
        hsl_fingerprint=agg.get("hsl_fingerprint", {}),
        editing_intensity=agg.get("editing_intensity", {}),
        pick_stats=agg.get("pick_stats", {}),
        keyword_map=agg.get("keyword_map", {}),
        saliency_stats=agg.get("saliency_stats", {}),
        fg_palette_colors=_fg_palette_colors,
        bg_palette_colors=_bg_palette_colors,
        storage_tiers=agg.get("storage_tiers", {}),
        events=agg.get("events", {}),
        ela_stats=agg.get("ela_stats", {}),
        album_stats=agg.get("album_stats", {}),
        coach=data.get("coach"),
        iq_stats=agg.get("iq_stats", {}),
        pose_stats=agg.get("pose_stats", {}),
        _sources=sources,
        technical_settings_analysis=tsa,
        shooting_profile_by_context=shooting_profile_sorted,
        editing_journey=agg.get("editing_journey", {}),
        edit_intensity_aesthetic_r=agg.get("edit_intensity_aesthetic_r"),
        period_stats=agg.get("period_stats", {}),
        camera_profile_distribution=agg.get("camera_profile_distribution", []),
        # derived vars
        focal_length_exact=focal_length_exact,
        iso_median=iso_median,
        iso_label=iso_label,
        prime_pct=prime_pct,
        dominant_shutter=dominant_shutter,
        dominant_shutter_pct=dominant_shutter_pct,
        key_insights=key_insights,
        iq_by_scene=iq_by_scene,
        # newly surfaced computed data
        aesthetic_by_vqa=agg.get("aesthetic_by_vqa", {}),
        edit_recency=agg.get("edit_recency", {}),
        editing_consistency=agg.get("editing_consistency", {}),
        uses_third_party_profile_pct=agg.get("uses_third_party_profile_pct"),
        dow_distribution=agg.get("dow_distribution", {}),
        best_conditions=best_conditions,
        split_toning=agg.get("split_toning", {}),
        burst_groups=agg.get("burst_groups", {}),
        iq_by_focal=iq_by_focal,
        score_histogram_10=agg.get("aesthetic_stats", {}).get("score_histogram_10", []),
        high_aes_low_iq=agg.get("iq_stats", {}).get("high_aesthetic_low_iq_count", 0),
    )
    path.write_text(html)

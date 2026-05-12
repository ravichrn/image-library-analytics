import json
from pathlib import Path

import jinja2
from markupsafe import Markup

OUTPUT_DIR = Path("output")


def generate_json(data: dict, path: Path = OUTPUT_DIR / "results.json") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def generate_html(data: dict, path: Path = OUTPUT_DIR / "report.html") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates"), autoescape=True)
    env.filters["tojson"] = lambda v: Markup(json.dumps(v))
    template = env.get_template("report.html.j2")

    agg = data.get("aggregated", {})
    html = template.render(
        photos=data.get("photos", []),
        agg=agg,
        photo_count=len(data.get("photos", [])),
        # existing
        umap_points=agg.get("umap", {}).get("points", []),
        umap_total=agg.get("umap", {}).get("total", 0),
        scene_dist=agg.get("scene_stats", {}).get("scene_distribution", {}),
        color_stats=agg.get("color_stats", {}),
        composition_stats=agg.get("composition_stats", {}),
        exif_stats=agg.get("exif_stats", {}),
        aesthetic_stats=agg.get("aesthetic_stats", {}),
        editing_consistency=agg.get("editing_consistency", {}),
        clusters=agg.get("clusters", {}),
        # new
        shooting_hours=agg.get("shooting_hours", {}),
        focal_length_histogram=agg.get("focal_length_histogram", []),
        aperture_histogram=agg.get("aperture_histogram", []),
        iso_histogram=agg.get("iso_histogram", []),
        aesthetic_by_scene=agg.get("aesthetic_by_scene", {}),
        composition_by_scene=agg.get("composition_by_scene", {}),
        grid_heatmap=agg.get("grid_heatmap", []),
        sharpness_stats=agg.get("sharpness_stats", {}),
        exposure_stats=agg.get("exposure_stats", {}),
        folder_breakdown=agg.get("folder_breakdown", {}),
        scene_confidence=agg.get("scene_confidence", {}),
        megapixel_stats=agg.get("megapixel_stats", {}),
        color_by_scene=agg.get("color_by_scene", {}),
        hue_distribution=agg.get("hue_distribution", {}),
        saturation_histogram=agg.get("saturation_histogram", {}),
        editing_trends=agg.get("editing_trends", {}),
        editing_style_patterns=agg.get("editing_style_patterns", []),
        color_grading_stats=agg.get("color_grading_stats", {}),
        composition_patterns=agg.get("composition_patterns", []),
        depth_stats=agg.get("depth_stats", {}),
        visual_attributes=agg.get("visual_attributes", {}),
        develop_stats=agg.get("develop_stats", {}),
        lightroom_stats=agg.get("lightroom_stats", {}),
        signature_edit=agg.get("signature_edit", {}),
        monthly_shooting=agg.get("monthly_shooting", {}),
        # tier-1 + saliency additions
        hsl_fingerprint=agg.get("hsl_fingerprint", {}),
        editing_intensity=agg.get("editing_intensity", {}),
        pick_stats=agg.get("pick_stats", {}),
        burst_groups=agg.get("burst_groups", {}),
        keyword_map=agg.get("keyword_map", {}),
        saliency_stats=agg.get("saliency_stats", {}),
        storage_tiers=agg.get("storage_tiers", {}),
        events=agg.get("events", {}),
        ela_stats=agg.get("ela_stats", {}),
        album_stats=agg.get("album_stats", {}),
        coach=data.get("coach"),
    )
    path.write_text(html)

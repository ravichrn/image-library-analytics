from analysis import advanced, aesthetics, core, forensics, gear, temporal


def aggregate(records: list[dict]) -> dict:
    result: dict = {}
    result["photo_count"] = len(records)

    result.update(core.analyze(records))
    result.update(temporal.analyze(records))
    result.update(aesthetics.analyze(records))
    result.update(advanced.analyze(records))
    result.update(forensics.analyze(records))
    result.update(gear.analyze(records))

    has_embeddings = any(r.get("dinov2") for r in records)
    if has_embeddings:
        from analysis import embeddings

        result.update(embeddings.analyze(records))
    else:
        result.update(
            {
                "clusters": {"n_clusters": 0, "labels": [], "centers": []},
                "umap": {"points": [], "total": 0, "sampled": 0},
                "burst_groups": {},
                "storage_tiers": {},
                "events": {},
            }
        )

    has_lightroom = any(r.get("lightroom_develop") for r in records)
    if has_lightroom:
        from analysis import journey, lightroom

        lr_result = lightroom.analyze(records)
        result.update(lr_result)
        result.update(journey.analyze(records))
        # Merge split_toning into color_grading_stats
        if "split_toning" in lr_result and "color_grading_stats" in result:
            result["color_grading_stats"]["split_toning"] = lr_result["split_toning"]
        # Merge editing_intensity into color_by_scene
        if "editing_intensity" in lr_result and "color_by_scene" in result:
            for scene, val in lr_result["editing_intensity"].get("by_scene", {}).items():
                if scene in result["color_by_scene"]:
                    result["color_by_scene"][scene]["editing_intensity"] = val
    else:
        result.update(
            {
                "lightroom_stats": None,
                "develop_stats": None,
                "signature_edit": None,
                "hsl_fingerprint": None,
                "editing_intensity": None,
                "pick_stats": None,
                "keyword_map": None,
                "album_stats": None,
                "editing_journey": None,
                "edit_intensity_aesthetic_r": None,
                "edit_recency": None,
                "period_stats": None,
                "camera_profile_distribution": None,
                "editing_style_signatures": [],
            }
        )

    result["_sources"] = {
        "has_lightroom": has_lightroom,
        "has_gps": any(r.get("exif", {}).get("gps_lat") for r in records),
        "has_embeddings": has_embeddings,
        "has_pose": any(r.get("pose_data") for r in records),
        "has_lightroom_ratings": any(r.get("lightroom_rating") is not None for r in records),
    }

    return result

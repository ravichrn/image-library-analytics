# Image Library Analytics

Generates an interactive HTML dashboard with deep metrics across composition, color, aesthetics, editing style, and shooting patterns. All ML inference runs on-device. Supports local folders and [Adobe Lightroom](https://lightroom.adobe.com) cloud.

**[Live report →](https://ravichrn.github.io/image-library-analytics/report.html)**

---

## Models

~8GB total. MPS (Apple Silicon) or CUDA used automatically; falls back to CPU. Batch sizes are tuned automatically based on available device memory at startup.

| Model | Purpose |
|---|---|
| SigLIP 2 SO400M | Scene classification + zero-shot VQA (people, setting, time, weather, season) — single pass, features reused |
| aesthetic-predictor-v2-5 | Aesthetic score 0–100 (SigLIP-based MLP, photography-tuned) |
| CLIP-IQA+ | Technical IQ score — blur, noise, exposure — batched tensor inference |
| DINOv2-base | 768-dim visual embeddings → UMAP, clustering, duplicate detection |
| RMBG-2.0 | Subject mask → area, centroid, placement |
| YOLO11n-pose | Object detection + pose estimation (portrait photos only) |
| ELA | Error Level Analysis — JPEG compression artifact detection |

---

## Pipeline

6 passes, each cached by SHA-256 — repeat runs only process new or changed photos. Each model is unloaded and GPU memory flushed before the next pass.

1. **EXIF · Color · Composition · ELA** — parallel CPU (6 threads)
2. **SigLIP 2** — scene classification + zero-shot VQA (image features computed once, reused for both)
3a. **aesthetic-predictor-v2-5** — aesthetic score
3b. **CLIP-IQA+** — technical IQ score
4. **DINOv2-base** — embeddings → UMAP + clustering + near-duplicate detection
5. **RMBG-2.0** — subject saliency
6. **YOLO11n-pose** — object detection + pose (portraits only)

Lightroom metadata is fetched via delta sync — only changed assets are pulled. Unchanged libraries make no API calls. Camera EXIF (make, model, lens, ISO, shutter, aperture, focal length, flash, metering) is read directly from the Lightroom API payload — not from the rendition JPEG, which Lightroom strips.

---

## Analysis

The `analysis/` package is split into focused modules and runs conditionally based on available data:

| Module | What it computes | Requires |
|---|---|---|
| `core` | Color, composition, EXIF, aesthetic, scene, folder, exposure, sharpness, megapixels | all sources |
| `temporal` | Shooting hours, monthly distribution, editing trends over time | all sources |
| `aesthetics` | Editing style patterns, hue/sat histograms, composition patterns | all sources |
| `advanced` | VQA attributes, saliency, pose, object frequency, CLIP-IQA+ | all sources |
| `forensics` | ELA suspicious-image stats, ELA/IQ conflict count | all sources |
| `embeddings` | UMAP scatter, KMeans clusters, burst/duplicate detection, event grouping | DINOv2 embeddings |
| `gear` | Device breakdown, top lenses, shutter/flash/metering mix, GPS, shooting profile by context | all sources |
| `lightroom` | Develop stats, signature edit, HSL fingerprint, editing intensity, pick/reject, albums | Lightroom source |
| `journey` | Monthly editing parameter trends, automation score, edit recency, style signatures | Lightroom source |

Lightroom-specific sections in the report are hidden automatically when the source is a local folder.

---

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

**HuggingFace auth:** RMBG-2.0 is a gated model. Accept the license at [huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), then add your token (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) to `.env`:

```ini
HF_TOKEN=hf_your_token_here
```

**Sources** — set in `.env`:

```ini
SOURCES=local            # default
SOURCES=lightroom
SOURCES=lightroom,local  # merged and deduplicated by SHA-256
```

**Adobe Lightroom (one-time auth):**

1. Create a project at [developer.adobe.com/console](https://developer.adobe.com/console)
2. Add the Lightroom API with an OAuth Web App credential
3. Set redirect URI to `https://localhost:8765/callback`
4. Copy Client ID and Secret into `.env`
5. Run `uv run python auth/lightroom.py` — after login, press **Cmd+L Cmd+C** to copy the redirect URL; the script saves the token automatically

---

## Usage

```bash
uv run python main.py                          # full library
uv run python main.py --sample 50             # quick test on 50 random photos
uv run python main.py --report-only           # regenerate report from cache, no ML

# Selective passes — skip or isolate specific ML steps
uv run python main.py --skip pose             # skip YOLO pose pass
uv run python main.py --skip iq,dino         # skip CLIP-IQA+ and DINOv2
uv run python main.py --only scene            # run only SigLIP scene pass

# Cache management
uv run python main.py --clear-key saliency    # wipe saliency from all cached records, then exit
uv run python main.py --prune                 # remove entries for photos no longer in library
uv run python main.py --prune --dry-run       # preview what --prune would delete

# Lightroom scoped runs
uv run python main.py --lightroom-album "Best of 2024"
uv run python main.py --lightroom-since 2024-06-01

open docs/report.html
```

**Pass names for `--skip`/`--only`:** `exif`, `scene`, `aesthetic`, `iq`, `dino`, `saliency`, `pose`

**Skip IQ via env var** (useful for CI or large libraries):
```ini
# .env
SKIP_IQ=true
```

---

## Coaching

The local rule engine (`coach_client.py`) flags compositional and technical issues without any API calls. Rules fire on cached metrics only.

**Configurable thresholds** — copy `coach_rules.yaml` into the project root and edit:

```yaml
# coach_rules.yaml
aesthetic_floor: 35       # low_aesthetic fires below this
iq_floor: 40              # iq_low_aesthetic_high fires below this
horizon_tilt_deg: 1.5     # horizon_tilted fires above this
# ... see coach_rules.yaml for all thresholds
```

Changes take effect on the next run — no code edits needed.

**Active rules:**

| Rule | Signals used |
|---|---|
| Subject too centered | `thirds_score` |
| Horizon not level | `horizon_tilt_deg` |
| Busy / cluttered foreground | `foreground_clutter` |
| Subject not well isolated | `subject_isolation` |
| Frame too busy | `negative_space` |
| Low overall aesthetic | `aesthetic_score` |
| Portrait with deep DOF | `dof_category` + `scene_type` |
| Low-light + low aesthetic | `light_category` + `aesthetic_score` |
| Highlight clipping | `highlight_clipping` |
| Shadow clipping | `shadow_clipping` |
| Strong feel, weak technical quality | `iq_score` + `aesthetic_score` |
| Subject off-center but not on thirds | `subject_off_center` + `thirds_score` |

---

## Output

| File | |
|---|---|
| `docs/report.html` | Interactive dashboard (mobile-responsive, URL deep links) |
| `docs/results.json` | Raw metrics + aggregations (`schema_version: "1"`) |
| `cache/cache.db` | SQLite cache keyed by SHA-256 |
| `cache/renditions/` | Lightroom renditions (2048px) |
| `cache/.sync_state.json` | Last Lightroom sync timestamp — auto-managed |

**Adobe Lightroom renditions** — controlled via `.env`:

```ini
LIGHTROOM_DOWNLOAD_RENDITIONS=false  # download 2048px renditions for ML
LIGHTROOM_KEEP_RENDITIONS=true       # keep on disk for faster re-runs
```

---

## Dev

```bash
uv sync --group dev         # install dev deps (includes ruff, pre-commit, pytest)
uv run pre-commit install   # wire hooks into .git
uv run pytest tests/ -v     # run fixture tests (no ML required)
```

Ruff (lint + format) and file hygiene checks run automatically on `git commit` and auto-fix what they can.

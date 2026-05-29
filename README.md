# On-Device Image Analytics

On-device multi-model image analytics pipeline. Runs 7 models (SigLIP 2, DINOv2, RMBG-2.0, YOLO-pose, aesthetic predictor, CLIP-IQA+, ELA) on Apple Silicon — no cloud, no API calls for inference. Models are sequentially loaded, run, and fully unloaded with device cache flushes between passes. Outputs an interactive analytics dashboard covering composition, color, aesthetics, editing style, shooting patterns, and near-duplicate detection.

Supports local folders and [Adobe Lightroom](https://lightroom.adobe.com) cloud with delta sync. **[Live report](https://ravichrn.github.io/image-library-analytics/report.html)**

---

## Models

| Model | Purpose |
|---|---|
| SigLIP 2 SO400M | Scene classification + zero-shot VQA (people, setting, time, weather, season) — image features computed once, reused for both |
| aesthetic-predictor-v2-5 | Aesthetic score 0–100 (SigLIP-based MLP, photography-tuned) |
| CLIP-IQA+ | Technical IQ score — blur, noise, exposure — batched tensor inference |
| DINOv2-base | 768-dim visual embeddings → UMAP, clustering, near-duplicate detection |
| RMBG-2.0 | Subject mask → area, centroid, placement |
| YOLO11n-pose | Object detection + pose estimation (portrait photos only) |
| ELA | Error Level Analysis — JPEG compression artifact detection |

---

## Pipeline

Sequential loading — each pass loads its model, runs inference, then explicitly unloads weights and flushes the device cache before the next pass begins. Batch sizes are tuned automatically at startup based on available memory.

Pipeline profile (M3 Pro 18 GB, 5,107-photo library):

| Pass | Model | Photos | Time | img/s | Load | Peak mem |
|---|---|---:|---:|---:|---:|---:|
| 1 | EXIF · Color · Composition · ELA (CPU, 6 threads) | 5,107 | — | — | — | — |
| 2 | SigLIP 2 SO400M | 5,107 | 1342 s | 3.8 | 8.8 s | 2,273 MB |
| 3a | aesthetic-predictor-v2-5 | 5,107 | 1346 s | 3.8 | 2.8 s | 3,132 MB |
| 3b | CLIP-IQA+ | 5,107 | 231 s | 22.1 | 1.6 s | 2,273 MB |
| 4 | DINOv2-base | 5,107 | 154 s | 33.2 | 0.9 s | 2,446 MB |
| 5 | RMBG-2.0 | 5,107 | 892 s | 5.7 | 2.2 s | 444 MB |
| 6 | YOLO11n-pose (portraits only) | 2,385 | 65 s | 36.3 | 0.1 s | 14 MB |

Total wall-clock time: ~30 min. Numbers written to `docs/pipeline_profile.json` after each run.

Sequential loading keeps peak at 3.1 GB; all 7 models loaded simultaneously would require 10.3 GB (3.4×). Lightroom delta sync fetches only changed assets.

---

## Analysis

The `analysis/` package is split into focused modules and runs conditionally based on available data:

| Module | What it computes |
|---|---|
| `core` | Color, composition, EXIF, aesthetic, scene, folder, exposure, sharpness, megapixels |
| `temporal` | Shooting hours, monthly distribution, editing trends over time |
| `aesthetics` | Editing style patterns, hue/sat histograms, composition patterns |
| `advanced` | VQA attributes, saliency, pose, object frequency, CLIP-IQA+ |
| `forensics` | ELA suspicious-image stats, ELA/IQ conflict count |
| `embeddings` | UMAP scatter, KMeans clusters, burst/duplicate detection, event grouping |
| `gear` | Device breakdown, top lenses, shutter/flash/metering mix, GPS, shooting profile by context |
| `lightroom` | Develop stats, signature edit, HSL fingerprint, editing intensity, pick/reject, albums |
| `journey` | Monthly editing parameter trends, automation score, edit recency, style signatures |

`lightroom` and `journey` require Lightroom source. `embeddings` requires DINOv2 embeddings. Lightroom-specific sections in the report are hidden automatically when the source is a local folder.

---

## Evaluation

### Near-duplicate detection

Evaluated against 101 hand-labeled pairs from the library (67 clear burst frames with cosine sim ≥ 0.97; 34 hard cases at sim 0.95–0.97 near the eps=0.05 threshold):

| eps | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| 0.03 | 1.000 | 0.663 | 0.798 | zero false merges; misses sim 0.95–0.97 |
| **0.05** | **0.878** | **1.000** | **0.935** | configured default |
| 0.08 | 0.795 | 1.000 | 0.886 | |

2 of the 14 FPs are unlabeled A↔C pairs within GT bursts (transitive closure); precision after closure = 0.896. The remaining 12 are visually similar shots at the detection boundary not captured in the ground truth.

Results written to `docs/near_dup_eval.json`.

### Cluster coherence

KMeans (k=10) on L2-normalised DINOv2 embeddings, evaluated using SigLIP `scene_type` as proxy labels — no manual annotation needed:

| Metric | Value |
|---|---|
| NMI vs scene_type | 0.28 |
| Within-cluster cosine sim | 0.346 |

Three clusters achieve 85–93% scene purity (architecture 92.8%, portraits 91.4%, portraits 85.0%). Mixed clusters reflect genuine visual overlap between portrait and nature content in DINOv2 space.

```bash
uv run python scripts/eval_clusters.py
```

Results written to `docs/cluster_eval.json`.

---

## Apple Silicon backend benchmark

Benchmarks PyTorch-MPS across DINOv2, SigLIP2, and YOLO11n-pose. MLX and CoreML were evaluated but are currently unavailable: CoreML conversion fails on torch 2.11.0 + coremltools 9.0 (integer op incompatibility in both DINOv2 and YOLO); no mlx-community DINOv2 encoder model exists (`mlx_lm.convert` targets language models only).

**Metrics per batch size:**
- **Latency** — warmup 10 iters, measure 100; p50 and p95 per-batch and per-image (model load time reported separately)
- **Throughput** — imgs/s at batch sizes 1, 4, 8, 16
- **Peak unified memory** — `torch.mps.current_allocated_memory()` delta around batch
- **Perf-per-watt** — images/joule via `powermetrics` over a 30s sustained loop (`--energy`, requires `sudo`)

```bash
uv run python scripts/benchmark_backends.py \
  --model dinov2 siglip yolo \
  --batch-sizes 1 4 8 16

# With energy measurement (requires sudo):
sudo uv run python scripts/benchmark_backends.py \
  --model dinov2 siglip yolo \
  --batch-sizes 1 4 8 16 --energy
```

Results written to `docs/backend_comparison.json`.

**Notable findings:**
- DINOv2 scales well with batch size (50→64 img/s, bs=1→16) — compute-bound on MPS
- SigLIP2 is flat at ~4 img/s across all batch sizes — memory-bound; 400M params saturate bandwidth before compute
- YOLO requires explicit `device='mps'` on each predict call — omitting it causes silent CPU fallback at all batch sizes (3.7–19× throughput loss confirmed via `torch.mps.current_allocated_memory()` showing 0 MB GPU allocation). With the fix: 101→154 img/s (bs=1→16), clean diminishing-returns curve, no collapse.

---

## Coaching

The local rule engine (`coach_client.py`) flags compositional and technical issues without any API calls. Thresholds are auto-calibrated from the library's own distribution on each run (e.g. `aesthetic_floor` = p15 of your library's aesthetic scores) so the coach flags the weakest slice of *your* library rather than applying global defaults. Fixed technical thresholds (`horizon_tilt_deg`, `highlight_clipping`, `shadow_clipping`) are not auto-tuned — they encode photographic standards. Manual overrides in `coach_rules.yaml` always win over calibrated values.

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

**Token expiry:** Adobe refresh tokens expire after ~14 days of inactivity. If you see `RuntimeError: Refresh token invalid`, re-run `uv run python auth/lightroom.py` — the expired token is cleared from `.env` automatically, and the new one is saved after login.

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
uv run python main.py --clear-key ml                  # clear all ML keys (scene, embeddings, scores, pose)
uv run python main.py --clear-key saliency            # clear a single key
uv run python main.py --clear-key saliency,iq_score  # clear multiple keys
uv run python main.py --prune                         # remove entries for photos no longer in library
uv run python main.py --prune --dry-run               # preview what --prune would delete

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

## Output

| File | |
|---|---|
| `docs/report.html` | Interactive dashboard (mobile-responsive, URL deep links) |
| `docs/results.json` | Raw metrics + aggregations (`schema_version: "1"`) |
| `cache/cache.db` | SQLite cache keyed by SHA-256 |
| `cache/renditions/` | Lightroom renditions (2048px) |

---

## Dev

```bash
uv sync --group dev         # install dev deps (includes ruff, pre-commit, pytest)
uv run pre-commit install   # wire hooks into .git
uv run pytest tests/ -v     # run fixture tests (no ML required)
```

Ruff (lint + format) and file hygiene checks run automatically on `git commit` and auto-fix what they can.

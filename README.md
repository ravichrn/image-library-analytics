# On-Device Image Analytics

An on-device inference pipeline that runs seven vision models within a fixed memory budget on Apple Silicon through sequential model loading and unloading. Performs scene classification, aesthetic and quality scoring, pose detection, background removal, and near-duplicate clustering. Built around a profiling-driven optimization approach that identified image decode rather than model compute as the primary bottleneck, with a benchmark-driven batch scheduler, content-addressed caching for incremental runs, and a lightweight regressor that reduces the cost of the most expensive model.

Supports local folders and [Adobe Lightroom](https://lightroom.adobe.com) cloud with per-asset delta sync.

**[Analytics report](https://ravichrn.github.io/on-device-image-analytics/analytics_report.html)** · **[Performance report](https://ravichrn.github.io/on-device-image-analytics/performance_report.html)**

---

## Key Results

| What | Number | Context |
|---|---|---|
| Sequential peak memory | **1.74 GB** | vs 9.04 GB naive (all models at once + SO400M for all photos) |
| RMBG cascade gate | **87% filtered** | 659 / 5,107 photos actually run through RMBG-2.0 |
| Aesthetic regressor coverage | **~93%** | of photos scored at <1% of SigLIP compute |
| Decode bottleneck (before fix) | **35 img/s** | image decode dominated GPU idle time |
| Decode throughput (after fix) | **247 img/s** | PIL draft mode + background prefetch + 4 decode threads |
| YOLO pre-decode speedup | **42 → 64 img/s** | bypassing YOLO's internal OpenCV path |
| Incremental run | **~30 s** | vs ~8.5 min full run (5,107-photo library, M3 Pro) |
| Cache hit rate (warm) | **100%** | zero model loads on unchanged photos |

---

## Features/Optimisations

**GPU (MPS)**
- Sequential model loading: each pass loads, runs, and fully unloads with a device cache flush — peak memory 1.74 GB vs 9.04 GB naive
- `torch.float16` for all MPS forward passes

**Batch sizing — benchmark-driven scheduler**
- Microbenchmark finds the smallest batch reaching ≥99% peak throughput per model; UCB bandit adapts per-machine at runtime
- Downgrades batch size only on an observed latency spike (p95 > 2× profiled AND throughput < 90% profiled)
- Memory guardrail at launch: free device memory ≥12 GB → 2× default batch; <6 GB → half

**I/O — CPU → GPU pipeline**
- 1 background thread prefetches batch N+1 while GPU runs batch N; 4 decode threads within each batch — GPU idle time eliminated
- PIL draft mode on every JPEG: libjpeg-turbo downscales before full decode — 35 → 247 img/s
- YOLO receives pre-decoded PIL images, bypassing its internal OpenCV path: 42 → 64 img/s end-to-end
- Pass 1 (EXIF · color · composition · JPEG quality) runs `min(CPU, 6)` threads across photos
- Lightroom renditions: 16-worker parallel download completes before any ML pass begins

**Aesthetic scoring — warm-start**
- k-means selects K=`5×√N` diverse seeds (≤1,500); only seeds run through the aesthetic predictor; a Ridge regressor predicts the remaining N−K. Incremental runs: OOD check per new photo — zero SigLIP calls when in-distribution.

**Scene / VQA**
- Multi-label: `scene_types` includes all labels within 85% of the top cosine score — relative thresholding avoids the absolute sigmoid collapse at high logit_scale
- `time_of_day` and `season` derived from EXIF timestamp only — no visual inference

**RMBG-2.0 — subject filtering**
- Saliency only runs on photos with an isolatable subject: `has_person = yes` from VQA, or `{people and portraits, animals, food, interior}` scoring ≥ 0.35 cosine similarity — cuts candidates from 5,107 → ~660 on this library (~75 s vs ~495 s without filtering; count is library-dependent)

**Cache**
- SHA-256 content-addressed SQLite cache; passes with 100% hit rate are skipped without loading any model
- `aesthetic_score_source` tags each score as `"siglip"` or `"regressor"` — prevents pseudo-labels from polluting retraining

---

## Models

| Model | Purpose | Pipeline role |
|---|---|---|
| DINOv3-B | 768-dim visual embeddings — backbone for all downstream tasks | Pass 2: all new photos; cached embeddings reused in all subsequent passes |
| Aesthetic Ridge regressor | Aesthetic score 0–100 from DINOv3-B features; trained on k-means seed labels | Pass 4a: seeds → SigLIP, remainder → regressor; incremental runs use OOD check |
| aesthetic-predictor-v2-5 | SigLIP-based MLP — ground-truth labels for the regressor | Pass 4a: seed photos only; `--teacher` forces full-library scoring for regressor retraining |
| ARNIQA | Technical IQ score — blur, noise, exposure; self-supervised ResNet-50, fewer MACs than CLIP-based models | Pass 4b: bs=16 |
| RMBG-2.0 | Subject mask → area, centroid, fg/bg palette | Pass 5: 256 px input; filtered to photos with subject signal (library-dependent) |
| YOLO26n-pose | Object detection + pose estimation | Pass 6: portraits only; +7.2% mAP50-95 vs YOLO11n; PIL input |
| JPEG quant tables | Quality factor (0–100) from quantization tables; `quant_table_nonstandard` flag distinguishes camera-original from software re-exports | Pass 1: CPU, alongside EXIF/color/composition |
| SigLIP2-base (224 px) | Multi-label scene classification + VQA | Pass 3: all photos; 87.7 img/s at bs=16 on MPS |

---

## Pipeline

**Measured timings** (M3 Pro 18 GB, 5,107-photo library, last full run):

| Pass | Model | Photos | Time | img/s | Peak mem |
|---|---|---:|---:|---:|---:|
| 1 | EXIF · Color · Composition · JPEG Quality | 5,107 | cached | — | — |
| 2 | DINOv3-B | 5,107 | 91 s | 56.1 | 171 MB |
| 3 | SigLIP2-base | 5,107 | 77 s | 66.2 | 922 MB |
| 4a | aesthetic-predictor on K=357 seeds | 5,107 | 99 s | 51.7 | 1,781 MB |
| 4a | Ridge regressor on remainder | ~4,750 | < 1 s | — | — |
| 4a | OOD check + regressor *(incremental)* | new photos | < 1 s + seeds | — | — |
| 4b | ARNIQA | 5,107 | — | — | — |
| 5 | RMBG-2.0 @ 256 px | library-dep. ¹ | 76 s ¹ | 8.6 | 1,366 MB |
| 6 | YOLO26n-pose | library-dep. ¹ | 9 s ¹ | 75.5 | 922 MB |

¹ Passes 5–6 photo counts depend on library content (portrait/animal/food/interior ratio). Measured on this library: ~660 photos each.

**Full run:** ~8.5 min · **Incremental (new/changed photos only):** ~30 s

---

## Analysis

| Module | What it computes |
|---|---|
| `core` | Color, composition, EXIF, aesthetic, scene, folder breakdown, exposure, sharpness, megapixels |
| `temporal` | Shooting hours, monthly distribution, editing trends (month-over-month deltas) |
| `aesthetics` | Editing style patterns, hue/sat histograms, composition patterns |
| `advanced` | VQA attributes, saliency, pose, object frequency, IQ scores |
| `forensics` | JPEG quality factor stats, re-export rate, JPEG/IQ conflict count |
| `embeddings` | UMAP scatter, KMeans clusters, burst/duplicate detection, event grouping |
| `gear` | Device breakdown, top lenses, shutter/flash/metering mix, GPS, shooting profile |
| `lightroom` | Develop stats, signature edit, HSL fingerprint, editing intensity, pick/reject, albums |
| `journey` | Monthly editing parameter trends (12 sliders), automation score, edit recency, style signatures |

`lightroom` and `journey` require Lightroom source. `embeddings` requires DINOv3-B embeddings. Lightroom sections are hidden automatically for local-folder runs.

**Editing Journey chart:** σ-normalized by default; raw values toggle available.

---

## Evaluation

### Near-duplicate detection

101 hand-labeled pairs (67 clear burst frames cosine sim ≥ 0.97; 34 hard cases at 0.95–0.97):

| eps | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| 0.03 | 1.000 | 0.663 | 0.798 | zero false merges; misses sim 0.95–0.97 |
| **0.05** | **0.878** | **1.000** | **0.935** | configured default |
| 0.08 | 0.795 | 1.000 | 0.886 | |

2 of 14 FPs are unlabeled A↔C pairs within GT bursts; precision after transitive closure = 0.896. Results → `docs/near_dup_eval.json`.

### Cluster coherence

KMeans (k=10) on L2-normalised DINOv3-B embeddings. UMAP colored by cluster index — scene labels not used.

```bash
uv run python scripts/eval_clusters.py   # results → docs/cluster_eval.json
```

### Aesthetic regressor

| Setup | R² | MAE |
|---|---:|---:|
| Warm-start (K seeds) | 0.42–0.49 | 4.2–4.4 |
| Full library | **0.49** | **3.86** |

```bash
uv run python scripts/train_aesthetic_regressor.py --eval-only  # metrics only
uv run python scripts/train_aesthetic_regressor.py              # train + save
```

---

## Coaching

Local rule engine (`coach_client.py`) — no API calls. Thresholds auto-calibrated from your library distribution (e.g. `aesthetic_floor` = p15); fixed photographic standards (`horizon_tilt_deg`, clipping) are not auto-tuned. `coach_rules.yaml` overrides always win.

| Rule | Signals |
|---|---|
| Subject too centered | `thirds_score` |
| Horizon not level | `horizon_tilt_deg` |
| Busy foreground | `foreground_clutter` |
| Subject not isolated | `subject_isolation` |
| Frame too busy | `negative_space` |
| Low aesthetic | `aesthetic_score` |
| Portrait with deep DOF | `dof_category` + `scene_types` |
| Low-light + low aesthetic | `light_category` + `aesthetic_score` |
| Highlight clipping | `highlight_clipping` |
| Shadow clipping | `shadow_clipping` |
| Strong feel, weak IQ | `iq_score` + `aesthetic_score` |
| Off-center but not on thirds | `subject_off_center` + `thirds_score` |

---

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

**HuggingFace auth:** RMBG-2.0 and DINOv3-B are gated. Accept licenses at [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) and [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), then set `HF_TOKEN=hf_...` in `.env`.

**Sources** (`.env`):

```ini
SOURCES=local            # default
SOURCES=lightroom
SOURCES=lightroom,local  # merged and deduplicated by SHA-256
```

**Adobe Lightroom (one-time auth):**

1. Create a project at [developer.adobe.com/console](https://developer.adobe.com/console) with a Lightroom API OAuth Web App credential
2. Set redirect URI to `https://localhost:8765/callback`, copy Client ID + Secret into `.env`
3. Run `uv run python auth/lightroom.py` — after login, press **Cmd+L Cmd+C** to copy the redirect URL

**Token expiry:** Re-run `auth/lightroom.py` if you see `RuntimeError: Refresh token invalid` (~14 days inactivity).

**Lightroom sync:** Scans all asset metadata each run (~11 API calls for 5k photos); re-processes only assets whose `updated` timestamp changed.

**Develop settings:** Two mutually exclusive formats — new API (crsVersion ≥ 16.4, 2024+) fetches XMP via `/xmp/develop` with 12-worker pre-fetch; old API inlines `xmpCameraRaw` in the asset payload. Both XMP attribute and element styles are parsed. `lightroom_develop = {}` means confirmed empty; absent key means never fetched.

---

## Usage

```bash
uv run python main.py                  # full library
uv run python main.py --sample 50     # quick test on 50 random photos
uv run python main.py --report-only   # regenerate reports from latest run

# Selective passes (names: exif, dino, scene, aesthetic, iq, saliency, pose)
uv run python main.py --skip iq,saliency
uv run python main.py --only dino

# Cache management
uv run python main.py --clear-key ml      # clear all ML keys
uv run python main.py --prune             # remove stale entries

# Lightroom scoped runs
uv run python main.py --lightroom-album "Best of 2024"
uv run python main.py --lightroom-since 2024-06-01
```

Reports land in `docs/` — open `analytics_report.html` and `performance_report.html`. All generated files and model weights go in `artifacts/`; delete it to fully reset.

---

## Dev

```bash
uv sync --group dev
uv run pre-commit install
uv run pytest tests/ -v
```

Ruff (lint + format) and file hygiene checks run on `git commit`.

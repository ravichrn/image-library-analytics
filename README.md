# On-Device Image Analytics

On-device multi-model image analytics pipeline for Apple Silicon. Runs 7 models locally — no cloud, no API calls for inference. Outputs an interactive dashboard covering composition, color, aesthetics, editing style, shooting patterns, near-duplicate detection, and curation scoring. Supports local folders and [Adobe Lightroom](https://lightroom.adobe.com) cloud with per-asset delta sync.

**[Live report](https://ravichrn.github.io/image-library-analytics/report.html)**

---

## Features/Optimisations

**GPU (MPS)**
- Sequential model loading: each pass loads, runs, fully unloads with a device cache flush — peak memory ~3.1 GB vs ~10 GB all-at-once
- `torch.float16` for all MPS forward passes; `torch.compile` applied best-effort on CUDA
- Explicit `del model`, `gc.collect()`, and `empty_cache()` after every pass — frees Python references and device allocations before the next model loads

**Batch sizing — benchmark-driven scheduler**
- Microbenchmark finds the smallest batch reaching ≥99% peak throughput per model; UCB bandit adapts per-machine at runtime
- Benchmark config (model IDs, input sizes) imported from extractors — pipeline changes flow through automatically
- Runtime only downgrades batch size on an observed latency spike (p95 > 2× profiled AND throughput < 90% profiled)
- Memory guardrail at launch: free device memory ≥12 GB → 2× default batch; <6 GB → half

**I/O — CPU → GPU pipeline**
- Prefetch pipeline: 1 background thread submits the next batch load while GPU runs the current one; within each batch, 4 threads decode images in parallel — GPU idle time between batches eliminated
- PIL draft mode on every JPEG: libjpeg-turbo decodes to nearest power-of-2 downscale before full decode; cuts per-image decode cost 2.4×. GIL is released during JPEG decode, making 4-thread parallelism safe: 35 → 247 img/s measured
- YOLO receives pre-decoded PIL images, bypassing its internal C++ OpenCV path: 42 → 64 img/s end-to-end
- Pass 1 (EXIF · color · composition · ELA) runs `min(CPU, 6)` threads across photos — one image open per photo feeds all 4 cheap extractors
- Lightroom renditions: 16-worker parallel download completes before any ML pass begins

**Aesthetic scoring — warm-start**
- k-means selects K=`5×√N` diverse seeds (≤1,500); only seeds go to the aesthetic predictor; a Ridge regressor trained on seed labels predicts the remaining N−K photos in the same run. Incremental runs: OOD check per new photo — zero SigLIP calls when in-distribution.
- After each SigLIP aesthetic pass, the regressor retrains on all accumulated SigLIP-labelled records (not just this run's seeds) — quality improves monotonically as the library grows

**Scene / VQA**
- SigLIP2-base computes raw cosine similarities between image and all 10 label embeddings. `scene_types` includes all labels within 85% of the top cosine score (catches near-ties like portrait in nature). Absolute sigmoid thresholds are not used: SigLIP2-base logit_scale pushes all scores to ≈1.0 with short generic labels, so thresholding is relative.
- `time_of_day` and `season` are derived from EXIF timestamp only — no visual inference needed
- Scene + VQA text embeddings cached to `artifacts/siglip_text_feats.pt` with a SHA-256 label hash — text encoding skipped on warm runs

**RMBG-2.0 — subject filtering**
- Saliency only runs on photos with an isolatable subject: `has_person = yes` from VQA (catches portraits regardless of scene label), or any of `{people and portraits, animals, food, interior}` scoring ≥ 0.35 raw cosine similarity in `scene_scores`.
- Cuts RMBG candidates from 5,107 → ~2,000–2,500 photos (~240 s vs ~490 s). Photos outside the filter get null saliency; report and coaching handle this gracefully.

**Cache**
- SHA-256 content-addressed SQLite cache; passes with 100% hit rate are skipped without loading any model
- `aesthetic_score_source` tags each score as `"siglip"` or `"regressor"` — prevents pseudo-labels from polluting future retraining

---

## Models

| Model | Purpose | Pipeline role |
|---|---|---|
| DINOv3-B | 768-dim visual embeddings — backbone for all downstream tasks | Pass 2: all new photos; cached embeddings reused in all subsequent passes |
| Aesthetic Ridge regressor | Aesthetic score 0–100 from DINOv3-B features; trained on k-means seed labels | Pass 4a: seeds → SigLIP, remainder → regressor; incremental runs use OOD check |
| aesthetic-predictor-v2-5 | SigLIP-based MLP — ground-truth labels for the regressor | Pass 4a: seed photos only; `--teacher` forces full-library scoring for regressor retraining |
| CLIP-IQA+ | Technical IQ score — blur, noise, exposure | Pass 4b: bs=16 |
| RMBG-2.0 | Subject mask → area, centroid, fg/bg palette | Pass 5: 256 px input; filtered to photos with subject signal (~2,000–2,500 of 5,107) |
| YOLO26n-pose | Object detection + pose estimation | Pass 6: portraits only; +7.2% mAP50-95 vs YOLO11n; PIL input |
| ELA | JPEG compression artifact detection | Pass 1: CPU, alongside EXIF/color/composition |
| SigLIP2-base (224 px) | Multi-label scene classification + VQA | Pass 3: all photos; 87.7 img/s at bs=16 on MPS |

---

## Pipeline

**Estimated timings** (M3 Pro 18 GB, 5,107 photos, benchmark-derived):

| Pass | Model | Photos | Est. time | img/s | Peak mem |
|---|---|---:|---:|---:|---:|
| 1 | EXIF · Color · Composition · ELA | 5,107 | — | — | — |
| 2 | DINOv3-B | 5,107 | ~74 s | 68.6 | — |
| 3 | Scene/VQA heads | 5,107 | < 1 s | — | — |
| 3 | SigLIP2-base *(cold, no heads)* | 5,107 | ~58 s | 87.7 | — |
| 4a | aesthetic-predictor on K seeds | ~357 | ~87 s | 4.1 | 3,132 MB |
| 4a | Ridge regressor on remainder | ~4,750 | < 1 s | — | — |
| 4a | OOD check + regressor *(incremental)* | new photos | < 1 s + seeds | — | — |
| 4b | CLIP-IQA+ | 5,107 | ~135 s | 37.8 | 2,273 MB |
| 5 | RMBG-2.0 @ 256 px | ~2,000–2,500 | ~200–240 s | 10.3 | 444 MB |
| 6 | YOLO26n-pose | ~2,385 | ~16 s | 146.3 | — |

**First/Cold run (no heads + regressor training):** ~10 min · **Incremental (300 new photos):** ~30 s

---

## Trained heads

**Cross-validated accuracy (5-fold, 5,107 photos):**

| Head | Accuracy | Classes |
|---|---:|---|
| scene (multi-label) | **90.8%** | 10 categories — each photo can carry multiple labels |
| has_person | **88.3%** | yes / no |
| setting | **94.5%** | indoors / outdoors |
| weather | **84.0%** | clear / cloudy / rainy / foggy / snowy |

`time_of_day` and `season` are EXIF-derived (hour → morning/afternoon/evening/night; month → season) — not predicted visually. `has_person` uses threshold 0.25 (recall-heavy) — a false negative silently skips the YOLO pose pass.

```bash
# Refresh aesthetic regressor ground-truth labels
uv run python main.py --teacher --skip scene,iq,saliency,pose,dino
```

---

## Analysis

| Module | What it computes |
|---|---|
| `core` | Color, composition, EXIF, aesthetic, scene, folder breakdown, exposure, sharpness, megapixels |
| `temporal` | Shooting hours, monthly distribution, editing trends (month-over-month deltas) |
| `aesthetics` | Editing style patterns, hue/sat histograms, composition patterns |
| `advanced` | VQA attributes, saliency, pose, object frequency, CLIP-IQA+ |
| `forensics` | ELA suspicious-image stats, ELA/IQ conflict count |
| `embeddings` | UMAP scatter, KMeans clusters, burst/duplicate detection, event grouping |
| `gear` | Device breakdown, top lenses, shutter/flash/metering mix, GPS, shooting profile |
| `lightroom` | Develop stats, signature edit, HSL fingerprint, editing intensity, pick/reject, albums |
| `journey` | Monthly editing parameter trends (12 sliders), automation score, edit recency, style signatures |

`lightroom` and `journey` require Lightroom source. `embeddings` and `curation` require DINOv3-B embeddings. Lightroom-specific sections in the report are hidden automatically when the source is a local folder.

**Editing Journey chart:** σ-normalized (Z-score) by default — each slider centred on its library mean, scaled by std dev (0σ = your average, ±1σ = one std dev). Raw values toggle available.

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

## Apple Silicon backend benchmark

MLX and CoreML evaluated but unavailable: CoreML conversion fails on torch 2.11.0 + coremltools 9.0; no mlx-community DINOv3 encoder exists.

```bash
uv run python scripts/benchmark_backends.py                # all models, bs=1 4 8 16, 3 runs
uv run python scripts/benchmark_backends.py --model dinov3      # single model
uv run python scripts/benchmark_backends.py --full         # micro + e2e + scheduler profile regeneration
uv run python scripts/generate_scheduler_profile.py        # recompute scheduler profile from existing data
uv run python scripts/generate_edge_ai_report.py           # compile performance summary from all data files
```

Results → `docs/backend_comparison.json` · Scheduler profile → `docs/scheduler_profile.json` · Edge-AI report → `docs/edge_ai_report.md`.

`generate_edge_ai_report.py` reads the existing JSON files (no model inference) and produces a single Markdown document covering: memory budget per model, microbench vs e2e vs pipeline throughput side-by-side, batch-scaling curves with flat/scaling annotation, scheduler profile and bandit state, backend compatibility matrix (MPS / CUDA / CoreML / MLX), and on-device architecture notes.

**Microbench throughput (img/s, forward pass only on synthetic inputs):**

| Model | bs=1 | bs=4 | bs=8 | bs=16 | bs=32 | Scheduler picks |
|---|---:|---:|---:|---:|---:|:---:|
| DINOv3-B | 57.1 | 64.2 | 67.2 | **68.6** | 64.4 | bs=16 |
| SigLIP2-base | 60.1 | 79.6 | 84.5 | **87.7** | 88.8 | bs=16 |
| aesthetic-predictor | 3.7 | 4.0 | **4.1** | 4.1 | — | bs=8 |
| RMBG-2.0 @ 256 px | 9.0 | **10.3** | 10.3 | 10.4 | — | bs=4 (flat curve — transformer compute bound) |
| BiRefNet-lite @ 256 px | 19.9 | 38.5 | **40.5** | 39.7 | — | evaluated; rejected — mask quality diverges significantly from RMBG-2.0 for non-product scenes |
| CLIP-IQA+ | 27.1 | 34.6 | 36.8 | **37.8** | — | bs=16 |
| YOLO26n-pose | 85.3 | 129.2 | 140.0 | **146.3** | — | bs=16 |

**End-to-end (real files, includes decode + prefetch + preprocessing):**

| Model | E2E before optimisations | E2E after |
|---|---:|---:|
| YOLO26n-pose | 42.6 img/s | **63.9 img/s** (+50%) |
| DINOv3-B | ~35 img/s | ~247 img/s decode throughput |

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

**Lightroom sync:** Every run scans all asset metadata (~11 API calls for 5k photos) and re-processes assets whose `updated` timestamp changed (catches develop edits, rating/pick changes, keyword updates, album moves).

**Develop settings fetch:** Two mutually exclusive formats per photo — new API (crsVersion ≥ 16.4, 2024+) serves XMP via `/xmp/develop` link; old API (pre-2024) inlines `xmpCameraRaw` string in the asset payload. Both attribute-style (`crs:KEY="val"`) and element-style (`<crs:KEY>val</crs:KEY>`) XMP are parsed. Parallel pre-fetch (12 workers) for new-format photos; serial fetch fallback if pre-fetch returns empty. XMP fetch errors are logged with URL and exception. `lightroom_develop = {}` means fetched and confirmed empty (not re-fetched); absent key means never fetched (re-fetched on next run). Delta sync re-processes only photos whose `updated` timestamp changed.

---

## Usage

```bash
uv run python main.py                          # full library
uv run python main.py --sample 50             # quick test on 50 random photos
uv run python main.py --report-only           # regenerate report from cache, no ML

# Selective passes
uv run python main.py --skip pose
uv run python main.py --skip iq,saliency
uv run python main.py --only dino

# Refresh aesthetic regressor labels (teacher = full SigLIP on all photos)
uv run python main.py --teacher --skip scene,iq,saliency,pose,dino

# Cache management
uv run python main.py --clear-key ml                  # clear all ML keys
uv run python main.py --clear-key dinov3              # clear embeddings
uv run python main.py --clear-key saliency,iq_score
uv run python main.py --clear-key lightroom_develop   # force XMP re-fetch on next sync
uv run python main.py --prune                         # remove stale entries
uv run python main.py --prune --dry-run

# Lightroom scoped runs
uv run python main.py --lightroom-album "Best of 2024"
uv run python main.py --lightroom-since 2024-06-01

open docs/report.html
```

**Pass names for `--skip`/`--only`:** `exif`, `dino`, `scene`, `aesthetic`, `iq`, `saliency`, `pose`

```ini
SKIP_IQ=true  # .env — skip IQ for CI or large libraries
```

---

## Output

| File | |
|---|---|
| `docs/report.html` | Interactive dashboard (mobile-responsive, URL deep links) |
| `docs/results.json` | Raw metrics + aggregations |
| `docs/embeddings_cache.json` | UMAP + near-duplicate results; auto-invalidates on embedding change |
| `artifacts/` | Single folder for all generated and downloaded files — delete to fully reset |
| `artifacts/*.joblib` | Trained LogReg heads + Ridge aesthetic regressor (with seed embeddings + coverage threshold) |
| `artifacts/*.json` | Head metadata, regressor metadata |
| `artifacts/yolo26n-pose.pt` | YOLO model weights (downloaded on first run) |
| `artifacts/cache/cache.db` | SQLite cache keyed by SHA-256 |
| `artifacts/cache/renditions/` | Lightroom renditions (2048px) |

---

## Dev

```bash
uv sync --group dev
uv run pre-commit install
uv run pytest tests/ -v
```

Ruff (lint + format) and file hygiene checks run on `git commit`.

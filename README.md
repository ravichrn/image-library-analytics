# Image Library Analytics

An image library analysis tool that generates an interactive HTML dashboard with deep metrics across composition, color, aesthetics, editing style, and shooting patterns. All ML inference runs on-device. Supports local folders and Adobe Lightroom cloud.

**[Live report →](https://ravichrn.github.io/image-library-analytics/report.html)**

## Models (all local, ~6.5GB total)

| Model | Purpose |
|---|---|
| CLIP ViT-L/14 | Zero-shot scene classification (10 labels) + aesthetic score |
| LAION aesthetic predictor MLP | Aesthetic score 0–100 (768→1024→128→64→16→1) |
| MUSIQ (pyiqa) | Technical IQ score — noise, blur, exposure quality; runs on CPU |
| DINOv2-base | 768-dim visual embeddings for UMAP + clustering + burst detection |
| Depth Anything v2 Small | Monocular depth estimation — depth range, complexity, subject separation |
| Florence-2-base | VQA (people, setting, time, weather, season) + detailed caption per photo |
| RMBG-1.4 (briaai/RMBG-1.4) | Salient object detection — subject area, centroid, off-center placement |
| YOLOv8n-pose | Object detection (80 COCO classes) + human pose estimation for portrait photos |
| ELA (built-in) | Error Level Analysis — JPEG re-compression at quality 75, detects inconsistent compression artifacts |

MPS (Apple Silicon) or CUDA used automatically if available; falls back to CPU. Requires ~8GB RAM.

## Report sections

| Category | Sections |
|---|---|
| **Library overview** | Summary stats, scene preferences, visual attributes (Florence-2 VQA), visual similarity map (UMAP) |
| **Composition & depth** | Composition style patterns, 3×3 composition grid (edge activity, subject placement, zone interpretation), depth range & complexity by scene |
| **Technical** | Focal length / aperture / ISO histograms, sharpness & exposure, shooting hours, monthly shooting |
| **Aesthetics & color** | Aesthetic score histogram + by-scene dot-plot, color grading, color & mood palette, color profile by scene |
| **Editing style** | Editing style patterns, signature edit (median sliders), Lightroom develop heatmap, HSL DNA, trends over time |
| **Curation & culling** | Quality issues, picks vs. rejects, portfolio albums, photo events with auto-narrative (BLIP captions), similar image groups (DINOv2 + DBSCAN), ratings & keywords |
| **Forensics** | ELA JPEG compression analysis |

## Pipeline

8 passes — each caches results by SHA-256 so repeat runs only process new or changed photos. All ML passes use batched inference; Pass 1 runs in parallel across CPU cores.

1. **EXIF · Color · Composition · ELA** — parallel CPU pass; extracts focal length, aperture, ISO, shooting time, palette, saturation, brightness, contrast, sharpness, tonal zones, rule of thirds, negative space, horizon, subject isolation; ELA re-compresses each JPEG at quality 75 and flags compression inconsistencies
2. **CLIP (ViT-L/14) + LAION MLP** — batched; scene classification (10 labels) + aesthetic score 0–100
3. **MUSIQ (pyiqa)** — CPU pass; technical IQ score per photo — blur, noise, compression, exposure quality; flags photos with high aesthetic but low technical quality
4. **DINOv2-base** — batched; 768-dim visual embeddings → UMAP 2D + KMeans clustering + DBSCAN burst/near-duplicate detection + event narrative diversity selection
5. **Depth Anything v2 (Small)** — batched depth estimation → depth range, complexity, subject-background separation
6. **Florence-2-base** — batched; detailed caption + 5 VQA questions per photo (people, setting, time of day, weather, season); captions used for event narratives
7. **RMBG-1.4 saliency (briaai/RMBG-1.4)** — subject mask per photo → area %, centroid, off-center distance, 3×3 placement distribution
8. **YOLOv8n-pose** — portrait photos only (where Florence-2 detected a person); 80-class object detection + human pose keypoints → pose type, body coverage

Lightroom metadata (develop settings, ratings, keywords) is fetched via delta sync — only new or updated assets are pulled from the API on each run. If nothing has changed, all data is served from local SQLite cache with no API calls.

## Sources

Configure which sources to load in `.env`:

```ini
SOURCES=local          # local folder only (default)
SOURCES=lightroom      # Lightroom cloud only
SOURCES=lightroom,local  # both, deduplicated by SHA-256
```

Photos present in both sources are merged into a single record — local file is used for ML passes, Lightroom metadata enriches it with develop settings and ratings.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

### Lightroom setup (one-time)

1. Create a project at [developer.adobe.com/console](https://developer.adobe.com/console)
2. Add the **Lightroom API**, choose **OAuth Web App** credential
3. Set redirect URI to `https://localhost:8765/callback`
4. Copy `Client ID` and `Client Secret` into `.env`
5. Run the auth flow:

```bash
uv run python auth/lightroom.py
```

After Adobe login, your browser redirects to a failing localhost page — this is expected. Adobe requires the redirect URI to use `https://`, since there is no local SSL certificate, the auth code present in the URL is needed.

Press **Cmd+L** then **Cmd+C** in the browser — the script detects the URL from clipboard automatically and saves the refresh token to `.env`.

## Usage

```bash
uv run python main.py                    # full library
uv run python main.py --sample 50        # quick test on 50 random photos
uv run python main.py --batch-size 32    # larger batches if you have more VRAM
uv run python main.py --prune            # remove cache entries for deleted photos
open docs/report.html
```

### Cache pruning

Over time the cache accumulates entries for photos you've deleted from your library. Run `--prune` periodically to keep the report in sync:

- **Local source** — compares the cache against a fresh filesystem scan; any file no longer on disk is removed.
- **Lightroom source** — performs a full API fetch (no delta filter) to get the true current asset list, then removes anything not returned. This is the only reliable way to detect Lightroom deletions since the delta sync only surfaces new and updated assets, not deleted ones.
- **Both sources** — combines the above.

Orphaned rendition files in `cache/renditions/` are also deleted for any pruned hash.

## Rendition download behaviour (Lightroom)

Controlled by two `.env` flags:

| Flag | Default | Effect |
|---|---|---|
| `LIGHTROOM_DOWNLOAD_RENDITIONS` | `false` | Download 2048px renditions for ML analysis (~400KB each) |
| `LIGHTROOM_KEEP_RENDITIONS` | `true` | Keep renditions on disk after processing (faster re-runs) |

When renditions are enabled, all missing files are downloaded in parallel (16 workers) before ML passes begin. On re-runs, already-downloaded renditions are reused instantly.

## Output

| File | Description |
|---|---|
| `docs/report.html` | Interactive dashboard — open in any browser |
| `docs/results.json` | Raw per-photo metrics + all aggregations |
| `cache/cache.db` | SQLite database — all per-photo results keyed by SHA-256 |
| `cache/renditions/<hash>.jpg` | Downloaded Lightroom renditions (2048px, edits applied) |
| `cache/.sync_state.json` | Last Lightroom sync timestamp — auto-managed |

## Partial cache refresh

Clear a single analysis key from the SQLite cache (preserves all other ML results):

```bash
python -c "
import sqlite3, json
db = sqlite3.connect('cache/cache.db')
for h, data in db.execute('SELECT hash, data FROM photos').fetchall():
    d = json.loads(data)
    d.pop('saliency', None)   # replace with whichever key to clear
    db.execute('UPDATE photos SET data=? WHERE hash=?', (json.dumps(d), h))
db.commit()
"
uv run python main.py
```

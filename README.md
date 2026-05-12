# Image Library Analytics

Analyzes your photo library locally and creates an interactive HTML dashboard with deep metrics across composition, color, aesthetics, editing style, and shooting patterns. Supports local folders and Adobe Lightroom cloud.

## Models (all local, ~5GB total)

| Model | Purpose |
|---|---|
| CLIP ViT-L/14 | Zero-shot scene classification (10 labels) + aesthetic score |
| LAION aesthetic predictor MLP | Aesthetic score 0–100 (768→1024→128→64→16→1) |
| DINOv2-small | 768-dim visual embeddings for UMAP + clustering + burst detection |
| Depth Anything v2 Small | Monocular depth estimation — depth range, complexity, subject separation |
| BLIP VQA base | Visual question answering — people, setting, time of day, weather, season per photo |
| RMBG-1.4 (briaai/RMBG-1.4) | Salient object detection — subject area, centroid, off-center placement |
| ELA (built-in) | Error Level Analysis — JPEG re-compression at quality 75, detects inconsistent compression artifacts |

MPS (Apple Silicon) or CUDA used automatically if available; falls back to CPU. Requires ~6GB RAM.

## Report sections

| Section | What it shows |
|---|---|
| Library Overview | Photo count, dominant scene, avg aesthetic, resolution, sharpness, exposure bias |
| Scene Preferences | Doughnut chart with % breakdown of scene types |
| Visual Attributes | BLIP VQA distributions — people, indoor/outdoor, time of day, weather, season |
| Visual Similarity Map | UMAP scatter colored by scene type — shows which scenes cluster visually |
| Focal Length / Aperture / ISO | Histograms of technical shooting choices |
| Compositional Style | Named pattern detection: Rule of Thirds, Symmetry, Negative Space, Frame-filling, Subject Isolation, Foreground Interest, Centered |
| Composition Grid | 3×3 avg edge density + RMBG-1.4 subject placement heatmap (where subjects actually land) |
| Depth & Spatial Composition | Depth Anything v2 — depth range, complexity, subject-background separation by scene |
| Sharpness & Exposure | Per-scene sharpness bars, highlight/shadow clipping %, exposure bias |
| Editing Style Analysis | Multi-pattern detection (warm-toned, high contrast, muted, high-key, etc.) with confidence scores |
| Aesthetic Quality | 10-bucket score histogram + avg/median/std |
| Aesthetic Score by Scene | Dot-plot with ±1σ consistency bands — score position + volume per scene |
| Color Grading | Tonal style, color temperature, tonal zone split (shadows/mids/highlights), color harmony distribution |
| Color & Mood | Palette swatches, saturation/brightness stats, hue distribution, warm/cool split |
| Signature Edit | Median Lightroom slider values across your library — the recipe behind your look |
| Lightroom Develop Settings | Per-scene avg slider heatmap (scenes × sliders) — from Lightroom cloud metadata |
| Editing DNA | Per-channel HSL fingerprint (8 colour channels × Hue/Sat/Lum) + editing intensity by scene |
| Color Profile by Scene | Per-scene: saturation, brightness, warmth ratio, dominant tone |
| Editing Trends Over Time | Monthly avg aesthetic score, saturation, and brightness |
| Shooting Hours | 24-hour bar chart of when photos were taken |
| Monthly Shooting | Monthly photo count stacked by golden hour vs. other times |
| Quality Issues | Local rule engine: 10 compositional/technical checks with fix suggestions |
| What You Keep vs. Discard | Picks vs. rejects comparison — avg aesthetic, sharpness, top scene per bucket; pick rate by scene |
| Portfolio Albums | Lightroom album curation analysis — curation rate, AI aesthetic agreement, scene curation rates, per-album breakdown |
| Burst & Near-Duplicates | DBSCAN on DINOv2 cosine distance ≤ 0.05 — groups of near-identical photos for culling |
| Ratings, Labels & Keywords | Star rating distribution, color labels, top 20 keywords + per-scene keyword breakdown |
| ELA Forensics | JPEG compression inconsistency analysis — suspicious photo count, avg max/mean error |
| Smart Culling | Hero vs. redundant breakdown per DBSCAN cluster — redundancy %, top clusters with hero score |
| Photo Events | Shooting sessions grouped by 4-hour time gaps — photo count, duration, top scene, auto-narrative from BLIP captions |

## Pipeline

6 passes — each caches results by SHA-256 so repeat runs only process new or changed photos. All ML passes use batched inference; Pass 1 runs in parallel across CPU cores.

1. **EXIF · Color · Composition · ELA** — parallel CPU pass; extracts focal length, aperture, ISO, shooting time, palette, saturation, brightness, contrast, sharpness, tonal zones, rule of thirds, negative space, horizon, subject isolation; ELA re-compresses each JPEG at quality 75 and flags compression inconsistencies
2. **CLIP (ViT-L/14) + LAION MLP** — batched; scene classification (10 labels) + aesthetic score 0–100
3. **DINOv2-small** — batched; 768-dim visual embeddings → UMAP 2D + KMeans clustering + DBSCAN burst/near-duplicate detection + event narrative diversity selection
4. **Depth Anything v2 (Small)** — batched depth estimation → depth range, complexity, subject-background separation
5. **BLIP VQA base** — 5 questions per photo in a single `generate()` call; batched across photos
6. **RMBG-1.4 saliency (briaai/RMBG-1.4)** — subject mask per photo → area %, centroid, off-center distance, 3×3 placement distribution

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
open output/report.html
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
| `output/report.html` | Interactive dashboard — open in any browser |
| `output/results.json` | Raw per-photo metrics + all aggregations |
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

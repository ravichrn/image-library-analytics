# Image Library Analytics

Generates an interactive HTML dashboard with deep metrics across composition, color, aesthetics, editing style, and shooting patterns. All ML inference runs on-device. Supports local folders and [Adobe Lightroom](https://lightroom.adobe.com) cloud.

**[Live report →](https://ravichrn.github.io/image-library-analytics/report.html)**

---

## Models

~6.5GB total. MPS (Apple Silicon) or CUDA used automatically; falls back to CPU.

| Model | Purpose |
|---|---|
| CLIP ViT-L/14 | Scene classification + aesthetic score |
| LAION aesthetic MLP | Aesthetic score 0–100 |
| MUSIQ | Technical IQ score — blur, noise, exposure |
| DINOv2-base | 768-dim Visual embeddings → UMAP, clustering, duplicate detection |
| Depth Anything v2 | Depth range, complexity, subject separation |
| Florence-2-base | Captions + VQA (people, setting, time, weather, season) |
| RMBG-1.4 | Subject mask → area, centroid, placement |
| YOLOv8n-pose | Object detection + pose estimation (portrait photos only) |
| ELA | Error Level Analysis — JPEG compression artifact detection |

---

## Pipeline

8 passes, each cached by SHA-256 — repeat runs only process new or changed photos.

1. **EXIF · Color · Composition · ELA** — parallel CPU
2. **CLIP + LAION MLP** — scene classification + aesthetic score
3. **MUSIQ** — technical IQ score
4. **DINOv2** — embeddings → UMAP + clustering + near-duplicate detection
5. **Depth Anything v2** — depth estimation
6. **Florence-2** — captions + VQA
7. **RMBG-1.4** — subject saliency
8. **YOLOv8n-pose** — object detection + pose (portraits only)

Lightroom metadata is fetched via delta sync — only changed assets are pulled. Unchanged libraries make no API calls.

---

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
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
uv run python main.py                 # full library
uv run python main.py --sample 50    # quick test
uv run python main.py --batch-size 32
uv run python main.py --prune        # remove cache entries for deleted photos
open docs/report.html
```

`--prune` does a full asset fetch (bypassing delta sync) to detect deletions, then removes stale cache entries and orphaned renditions.

---

## Output

| File | |
|---|---|
| `docs/report.html` | Interactive dashboard |
| `docs/results.json` | Raw metrics + aggregations |
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
uv sync --group dev       # install dev deps
uv run pre-commit install # wire hooks into .git
```

Ruff (lint + format) and file hygiene checks run automatically on `git commit` and auto-fix what they can.

---

**Partial cache refresh** — clear one analysis key without re-running everything:

```bash
python -c "
import sqlite3, json
db = sqlite3.connect('cache/cache.db')
for h, data in db.execute('SELECT hash, data FROM photos').fetchall():
    d = json.loads(data)
    d.pop('saliency', None)
    db.execute('UPDATE photos SET data=? WHERE hash=?', (json.dumps(d), h))
db.commit()
"
uv run python main.py
```

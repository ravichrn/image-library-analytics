import hashlib
import os
import random
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}  # .heic requires pillow-heif


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_local(sample: int | None = None) -> list[dict]:
    photo_dir = os.path.expanduser(os.environ.get("PHOTO_DIR", ""))
    if not photo_dir:
        return []
    root = Path(photo_dir)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if sample and sample < len(paths):
        paths = random.sample(paths, sample)
    records = []
    for path in sorted(paths):
        records.append(
            {
                "path": str(path),
                "hash": _file_hash(path),
                "source": "local",
            }
        )
    return records

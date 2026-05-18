import json
import sqlite3
from pathlib import Path

CACHE_DIR = Path("cache")
DB_PATH = CACHE_DIR / "cache.db"

_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        CACHE_DIR.mkdir(exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("CREATE TABLE IF NOT EXISTS photos (hash TEXT PRIMARY KEY, data TEXT NOT NULL)")
        _conn.commit()
    return _conn


def load_cache(file_hash: str) -> dict | None:
    row = _db().execute("SELECT data FROM photos WHERE hash=?", (file_hash,)).fetchone()
    return json.loads(row[0]) if row else None


def save_cache(file_hash: str, data: dict) -> None:
    db = _db()
    db.execute(
        "INSERT INTO photos (hash, data) VALUES (?, ?) ON CONFLICT(hash) DO UPDATE SET data=excluded.data",
        (file_hash, json.dumps(data)),
    )
    db.commit()


def load_all_cached() -> list[dict]:
    rows = _db().execute("SELECT data FROM photos").fetchall()
    return [json.loads(r[0]) for r in rows]


def prune_cache(keep: set[str]) -> int:
    """Delete cache entries whose hash is not in `keep`. Returns count removed."""
    db = _db()
    stale = [row[0] for row in db.execute("SELECT hash FROM photos").fetchall() if row[0] not in keep]
    if stale:
        db.executemany("DELETE FROM photos WHERE hash=?", [(h,) for h in stale])
        db.commit()
    return len(stale)

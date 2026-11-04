"""Convert ADT4 image stores that hold sub-keyframes.

Older ADT4 stores (the C++ store writer, the first indexer, ``import-legacy``) have
frames of kind 1 (sub-keyframes): keyframe images between agent keyframes, anchored to a
nearby agent node with their relative pose. hydra_image_store has a single keyframe kind
(0, every keyframe archive) and observation frames (2), and rejects kind 1, so a
sub-keyframe simply becomes a keyframe; its anchor is kept.
"""

from __future__ import annotations

import os
import sqlite3

from hydra_python.image_store.store import DB_NAME, _resolve_db, schema_sql

#: the frame kind older ADT4 stores used for sub-keyframes
SUBKEYFRAME_KIND = 1


def _db_path(path: str | os.PathLike):
    db = _resolve_db(path)
    if not db.exists() and (db.parent / "image_store" / DB_NAME).exists():
        db = db.parent / "image_store" / DB_NAME  # a run directory
    if not db.exists():
        raise FileNotFoundError(f"no image store at {db}")
    return db


def convert_store(path: str | os.PathLike, *, dry_run: bool = False) -> dict:
    """Turn the sub-keyframes (kind 1) of an ADT4 store into keyframes (kind 0), in
    place. Idempotent. Close every writer of the store first (Hydra, a tailer, the
    console)."""
    db = _db_path(path)
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        before = dict(conn.execute("SELECT kind, COUNT(*) FROM frames GROUP BY kind"))
        stats = {
            "store": str(db.parent),
            "subkeyframes": before.get(SUBKEYFRAME_KIND, 0),
            "kinds_before": before,
        }
        if dry_run:
            return stats
        conn.execute("UPDATE frames SET kind = 0 WHERE kind = ?", (SUBKEYFRAME_KIND,))
        conn.executescript(schema_sql())  # tables added since (frame_descriptors)
        stats["kinds_after"] = dict(
            conn.execute("SELECT kind, COUNT(*) FROM frames GROUP BY kind")
        )
        return stats
    finally:
        conn.close()


def current_kind(kind: int) -> int:
    """The current frame kind of a frame row of an older ADT4 store."""
    return 0 if int(kind) == SUBKEYFRAME_KIND else int(kind)


__all__ = ["SUBKEYFRAME_KIND", "convert_store", "current_kind"]

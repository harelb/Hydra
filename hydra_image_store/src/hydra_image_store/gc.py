"""Garbage collection of image files that no kept observation needs.

Do not run gc on a store Hydra is still writing: the C++ writer keeps image paths of
frames and observations it has not committed yet, and decides what to keep from its own
in-memory state, so gc can delete files that a later commit references (and ``--orphans``
treats every file of an uncommitted row as unreferenced). :func:`gc_store` refuses when
another process has the store's sqlite files open (Linux ``/proc``; a read-only
viewer counts too) unless ``force=True`` (CLI ``--force``).
"""

from __future__ import annotations

import os
import pathlib
from collections import Counter

from .store import FrameKind, ImageStoreWriter


class StoreInUseError(RuntimeError):
    """Another process has the store open (see :func:`processes_using_store`)."""


def processes_using_store(db_path: str | os.PathLike) -> list[tuple[int, str]]:
    """``(pid, command)`` of other processes holding ``store.sqlite`` (or its ``-wal`` /
    ``-shm``) open. Reads ``/proc/<pid>/fd``, so it is Linux-only and sees only
    processes this user may inspect; elsewhere it returns ``[]``."""
    db = os.path.realpath(db_path)
    targets = {db, db + "-wal", db + "-shm"}
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        return []
    me = os.getpid()
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            fds = list((entry / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target in targets:
                try:
                    comm = (entry / "comm").read_text().strip()
                except OSError:
                    comm = "?"
                found.append((int(entry.name), comm))
                break
    return sorted(found)


def _collect(root: pathlib.Path, rel: str | None, stats: Counter, doomed: list) -> None:
    if rel is None or rel in stats.setdefault("_seen", set()):
        return
    stats["_seen"].add(rel)
    path = root / rel
    if os.path.lexists(path):
        stats["bytes_freed"] += path.lstat().st_size
        stats["files_removed"] += 1
        doomed.append(path)


def gc_store(
    writer: ImageStoreWriter,
    max_images_per_object: int | None = None,
    *,
    orphans: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Drop pixels nobody keeps.

    1. With ``max_images_per_object``: per merged node set (canonical node + aliases),
       only the best N observations that still have pixels stay ``kept``.
    2. Masks of unkept observations are deleted; rgb/depth of observation frames
       (kind 2) that no kept observation references are deleted. Keyframe and
       sub-keyframe images are never touched. Rows are kept (paths set to NULL).
    3. ``orphans``: files under ``frames/`` and ``masks/`` no row references are deleted.

    Row updates are committed before any file is unlinked, so a failure part way leaves
    files without rows (which ``orphans`` removes later), never rows without files.
    Raises :class:`StoreInUseError` when another process has the store open (Hydra must
    not be writing it), unless ``force``.
    """
    if writer._batch_depth:
        # the rows must be committed before files are unlinked
        raise RuntimeError("gc_store must not run inside writer.batch()")
    if not force:
        users = processes_using_store(writer._db.db_path)
        if users:
            raise StoreInUseError(
                f"{writer._db.db_path} is open in "
                + ", ".join(f"pid {pid} ({comm})" for pid, comm in users)
                + "; gc must not run while Hydra writes the store. Stop it first, or "
                "pass force=True / --force if only readers have it open."
            )
    doomed: list[pathlib.Path] = []
    stats: Counter = Counter()
    stats["_seen"] = set()
    root = writer.root
    reader = writer.reader()
    conn = writer.conn
    with writer.batch():
        conn.execute(
            "SAVEPOINT gc"
        )  # dry runs apply the row updates, then roll them back
        if max_images_per_object is not None:
            n = max(0, int(max_images_per_object))
            for node in reader.object_nodes():
                obs = [
                    o
                    for o in reader.observations(node, order="score")
                    if o.kept or o.mask_path is not None
                ]
                for o in obs[n:]:
                    if o.kept:
                        stats["observations_unkept"] += 1
                    conn.execute(
                        "UPDATE observations SET kept = 0 WHERE obs_id = ?", (o.obs_id,)
                    )
        rows = conn.execute(
            "SELECT obs_id, mask_path FROM observations WHERE mask_path IS NOT NULL AND kept = 0"
        ).fetchall()
        for oid, rel in rows:
            _collect(root, rel, stats, doomed)
            stats["masks_dropped"] += 1
            conn.execute(
                "UPDATE observations SET mask_path = NULL WHERE obs_id = ?", (oid,)
            )
        rows = conn.execute(
            "SELECT f.frame_id, f.rgb_path, f.depth_path FROM frames f "
            "WHERE f.kind = ? AND (f.rgb_path IS NOT NULL OR f.depth_path IS NOT NULL) "
            "AND NOT EXISTS (SELECT 1 FROM observations o WHERE o.frame_id = f.frame_id AND o.kept != 0)",
            (int(FrameKind.OBSERVATION),),
        ).fetchall()
        for fid, rgb, depth in rows:
            _collect(root, rgb, stats, doomed)
            _collect(root, depth, stats, doomed)
            stats["frames_dropped"] += 1
            conn.execute(
                "UPDATE frames SET rgb_path = NULL, depth_path = NULL WHERE frame_id = ?",
                (fid,),
            )
        if orphans:
            referenced = set()
            for rgb, depth in conn.execute("SELECT rgb_path, depth_path FROM frames"):
                referenced.update(p for p in (rgb, depth) if p)
            referenced.update(
                r[0]
                for r in conn.execute(
                    "SELECT mask_path FROM observations WHERE mask_path IS NOT NULL"
                )
            )
            for sub in ("frames", "masks"):
                base = root / sub
                if not base.exists():
                    continue
                for dirpath, _, files in os.walk(base):
                    for name in files:
                        rel = os.path.relpath(os.path.join(dirpath, name), root)
                        if rel not in referenced and rel not in stats["_seen"]:
                            _collect(root, rel, stats, doomed)
                            stats["orphans"] += 1
        if dry_run:
            conn.execute("ROLLBACK TO gc")
        conn.execute("RELEASE gc")
    # committed: only now delete the files the rows no longer reference
    if not dry_run:
        for path in doomed:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if not dry_run and max_images_per_object is not None:
        writer.set_meta("max_images_per_object", int(max_images_per_object))
    stats.pop("_seen")
    return dict(stats)

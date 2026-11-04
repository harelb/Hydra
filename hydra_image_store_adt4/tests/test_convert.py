import sqlite3

import pytest
from conftest import write_image
from hydra_image_store_adt4.cli import main
from hydra_image_store_adt4.convert import convert_store
from hydra_python.image_store import FrameKind, ImageStore, make_node_id

A0 = make_node_id("a", 0)


def old_store(root):
    """An older ADT4 store: keyframe (0), sub-keyframe (1), observation frame (2)."""
    w = ImageStore.open_writer(root, robot_id=0)
    sid = w.add_sensor("cam", 8, 6, 1, 1, 1, 1)
    img = write_image(root.parent / "src.jpg")
    w.add_frame_ref(sid, 100, kind=0, rgb=img, anchor_node_id=A0, link="copy")
    w.add_frame_ref(sid, 150, kind=1, anchor_node_id=A0)
    fid = w.add_frame_ref(sid, 160, kind=2)
    w.add_observation(7, fid, kept=True)
    w.close()
    return root


def kinds(root):
    with sqlite3.connect(root / "store.sqlite") as conn:
        return dict(conn.execute("SELECT timestamp_ns, kind FROM frames"))


def test_convert_old_store(tmp_path):
    root = old_store(tmp_path / "st")
    with ImageStore.open(root) as store, pytest.raises(ValueError, match="kind 1"):
        store.frames()  # rejected until converted
    stats = convert_store(root)
    assert stats["subkeyframes"] == 1 and stats["kinds_before"] == {0: 1, 1: 1, 2: 1}
    assert kinds(root) == {100: 0, 150: 0, 160: 2}
    with ImageStore.open(root) as store:
        assert [f.timestamp_ns for f in store.frames_for_agent(A0)] == [100, 150]
        assert [f.kind for f in store.frames()] == [
            FrameKind.KEYFRAME,
            FrameKind.KEYFRAME,
            FrameKind.OBSERVATION,
        ]
        assert [o.track_id for o in store.track_observations(0, 7)] == [7]
    again = convert_store(root)  # idempotent
    assert again["subkeyframes"] == 0 and kinds(root) == {100: 0, 150: 0, 160: 2}


def test_convert_dry_run_and_run_dir(tmp_path):
    old_store(tmp_path / "run/image_store")
    stats = convert_store(tmp_path / "run", dry_run=True)
    assert stats["subkeyframes"] == 1 and kinds(tmp_path / "run/image_store")[150] == 1
    assert main(["convert", str(tmp_path / "run")]) == 0
    assert kinds(tmp_path / "run/image_store")[150] == 0
    assert main(["convert", str(tmp_path / "nope")]) == 1

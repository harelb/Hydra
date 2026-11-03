import numpy as np
from hydra_image_store_adt4.sam3 import (
    SAM3_TRACK_BASE,
    add_sam3_observation,
    is_sam3_track,
    sam3_track_id,
)

from hydra_image_store import EXTERNAL_TRACK_BASE, ImageStore, index, make_node_id


def test_sam3_track_ids():
    node = make_node_id("O", 12)
    assert SAM3_TRACK_BASE == EXTERNAL_TRACK_BASE
    assert sam3_track_id(node) == EXTERNAL_TRACK_BASE | 12
    assert is_sam3_track(sam3_track_id(node)) and not is_sam3_track(12)


def test_sam3_writeback_survives_reindex(tmp_path):
    run = tmp_path / "run"
    (run / "tracks").mkdir(parents=True)  # a run the indexer knows
    node = make_node_id("O", 40)
    m = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], bool)
    with ImageStore.open_writer(run, robot_id=0) as w:
        sid = w.add_sensor("cam", 4, 2, 1, 1, 1, 1)
        fid = w.add_frame_ref(sid, 1)
        oid = add_sam3_observation(w, node, fid, mask=m, bbox2d=(1, 0, 3, 1), kept=True)
    index(run)
    with ImageStore.open(run, index=False) as store:
        (o,) = store.observations(node)
        assert o.obs_id == oid and o.track_id == sam3_track_id(node)
        assert o.mask_rle == "[1,2,3,2]"
        np.testing.assert_array_equal(store.mask(o), m)

"""SAM3 object discovery write-back on the generic image store writer.

SAM3-discovered objects get one track per scene graph node in the store's external
track-id range (``EXTERNAL_TRACK_BASE`` up), which the indexer never rebinds or deletes.
"""

from __future__ import annotations

from hydra_python.image_store import EXTERNAL_TRACK_BASE, ImageStoreWriter

#: SAM3 tracks: ``SAM3_TRACK_BASE | (node_id & SAM3_NODE_MASK)``, one per node
SAM3_TRACK_BASE = EXTERNAL_TRACK_BASE
SAM3_NODE_MASK = (1 << 40) - 1


def sam3_track_id(node_id: int) -> int:
    """Track id of the SAM3 observations bound to ``node_id`` (one track per node)."""
    return SAM3_TRACK_BASE | (int(node_id) & SAM3_NODE_MASK)


def is_sam3_track(track_id: int) -> bool:
    return SAM3_TRACK_BASE <= int(track_id) <= SAM3_TRACK_BASE | SAM3_NODE_MASK


def add_sam3_observation(
    writer: ImageStoreWriter, node_id: int, frame_id: int, **kwargs
) -> int:
    """Observation of a SAM3-discovered object in ``frame_id``; binds the node's SAM3
    track to ``node_id``. ``kwargs`` are those of ``ImageStoreWriter.add_observation``
    (``mask``, ``mask_rle``, ``mask_path``, ``bbox2d``, ``score``, ``kept``, ...)."""
    track = sam3_track_id(node_id)
    with writer.batch():
        writer.bind_track(track, node_id)
        return writer.add_observation(track, frame_id, **kwargs)


__all__ = [
    "SAM3_NODE_MASK",
    "SAM3_TRACK_BASE",
    "add_sam3_observation",
    "is_sam3_track",
    "sam3_track_id",
]

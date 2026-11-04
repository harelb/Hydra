"""ADT4 tools on top of hydra_python.image_store (not part of upstream Hydra).

- :mod:`.sam3`: SAM3 object discovery write-back (external track ids)
- :mod:`.legacy`: import pre-store ADT4 run directories
- :mod:`.fuse`: fuse per-robot stores into one keyed by a Hydra-Multi fused graph
- :mod:`.convert`: turn the sub-keyframes of older ADT4 stores into keyframes
"""

import hydra_image_store  # noqa: F401 - ADT4 shim: puts hydra_python on sys.path

from .convert import convert_store
from .sam3 import SAM3_TRACK_BASE, add_sam3_observation, is_sam3_track, sam3_track_id

__all__ = [
    "SAM3_TRACK_BASE",
    "add_sam3_observation",
    "convert_store",
    "is_sam3_track",
    "sam3_track_id",
]

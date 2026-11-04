# hydra_image_store_adt4

ADT4's own tools on top of `hydra_python.image_store` (kept out of the upstream package; the
`hydra_image_store` shim next to this directory keeps the old import name for ADT4 code):

| module | what |
| --- | --- |
| `sam3` | SAM3 object discovery write-back: `add_sam3_observation(writer, node_id, frame_id, **kw)`, `sam3_track_id(node)` = `EXTERNAL_TRACK_BASE \| (node & (2**40-1))`, one track per node |
| `legacy` | `import_legacy(run_dir, out, dsg=None)`: pre-store run directories (`agents/`, `subkeyframes/`, `images/O_*`, `sam3_objects/O*`); agent keyframes and sub-keyframes become keyframes; Khronos folders get `LEGACY_TRACK_BASE \| index` (`1 << 41`); `strip_legacy_dsg` drops SubKeyframe nodes and `image_folder` |
| `fuse` | `fuse_stores(out, sources, node_map, merges)`: per-robot stores -> one store keyed by a Hydra-Multi fused graph (`backend/fusion_node_map.json`) |
| `convert` | `convert_store(path)`: older ADT4 stores have sub-keyframes (`frames.kind` 1), which `hydra_python.image_store` rejects; they become keyframes (kind 0), anchors kept; idempotent |

```
hydra-image-store-adt4 convert STORE_OR_RUN... [--dry-run]
hydra-image-store-adt4 import-legacy --run-dir RUN --out STORE [--dsg dsg.json] [--robot-id 0]
                                     [--sam3-dir sam3_objects ...] [--copy] [--write-dsg OUT]
hydra-image-store-adt4 fuse FUSED_DIR --source ROBOT=STORE[:SESSION] ... --out DIR
                            [--node-map FILE] [--overwrite] [--copy | --symlink]
```

Convert every existing ADT4 store (stop Hydra, tailers and the console first) before the
current `hydra_python.image_store` reads it: it raises on frames of kind 1. `fuse` converts
older sources on the fly.

Install with `pip install hydra/ -e hydra_image_store -e hydra_image_store_adt4` (or put
`hydra_image_store/src` and `hydra_image_store_adt4/src` on `PYTHONPATH`; the shim finds
`hydra/python/src`); tests: `cd hydra_image_store_adt4 && python -m pytest`.

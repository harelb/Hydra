"""``hydra-image-store-adt4``: ADT4 image store tools (not part of upstream Hydra)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _cmd_convert(args) -> int:
    from .convert import convert_store

    rc = 0
    for store in args.store:
        try:
            stats = convert_store(store, dry_run=args.dry_run)
        except FileNotFoundError as exc:
            print(f"{store}: {exc}", file=sys.stderr)
            rc = 1
            continue
        print(json.dumps(stats, default=str))
    return rc


def _cmd_import_legacy(args) -> int:
    from .legacy import find_dsg, import_legacy, strip_legacy_dsg

    stats = import_legacy(
        args.run_dir,
        args.out,
        args.dsg,
        robot_id=args.robot_id,
        link="copy" if args.copy else "hardlink",
        sam3_dirs=tuple(args.sam3_dir) if args.sam3_dir else ("sam3_objects",),
    )
    if args.write_dsg:
        src = args.dsg or find_dsg(pathlib.Path(args.run_dir))
        if src is None:
            raise SystemExit(
                "--write-dsg needs --dsg (no scene graph found in the run dir)"
            )
        stats["stripped_dsg"] = strip_legacy_dsg(src, args.write_dsg)
    print(json.dumps(stats, indent=1))
    return 0


def _cmd_fuse(args) -> int:
    from .fuse import FuseSource, fuse_stores, load_node_map, load_world_T_robot

    fused_dir = pathlib.Path(args.fused)
    node_map, merges = load_node_map(
        args.node_map or fused_dir / "backend" / "fusion_node_map.json"
    )
    sources = []
    for spec in args.source:
        robot, _, rest = spec.partition("=")
        store, _, session = rest.partition(":")
        if not store:
            print(
                f"bad --source {spec!r}; expected ROBOT_ID=STORE[:SESSION]",
                file=sys.stderr,
            )
            return 1
        world_T_robot = None
        pose_path = fused_dir / "interfaces" / f"robot_{robot}" / "world_T_robot.txt"
        if pose_path.exists():
            world_T_robot = load_world_T_robot(pose_path)
        sources.append(
            FuseSource(pathlib.Path(store), int(robot), session or None, world_T_robot)
        )

    link = "copy" if args.copy else ("symlink" if args.symlink else "hardlink")
    try:
        stats = fuse_stores(
            args.out, sources, node_map, merges, link=link, overwrite=args.overwrite
        )
    except (FileExistsError, ValueError, RuntimeError) as exc:
        print(f"fuse failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=1, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hydra-image-store-adt4", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "convert",
        help="turn the sub-keyframes (kind 1) of older ADT4 stores into keyframes; "
        "idempotent",
    )
    s.add_argument("store", nargs="+", help="store directory, store.sqlite or run dir")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=_cmd_convert)

    s = sub.add_parser("import-legacy", help="convert a pre-store run directory")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--dsg", help="default: <run-dir>/hydra/backend/dsg.json if present")
    s.add_argument("--out", required=True, help="new store directory")
    s.add_argument("--robot-id", type=int, default=0)
    s.add_argument(
        "--sam3-dir", action="append", help="SAM3 object dirs (default: sam3_objects)"
    )
    s.add_argument(
        "--write-dsg",
        help="also write a copy of the scene graph without pre-store SubKeyframe "
        "nodes and image_folder attributes (loadable by upstream spark_dsg)",
    )
    s.add_argument(
        "--copy", action="store_true", help="copy images (default: hardlink)"
    )
    s.set_defaults(func=_cmd_import_legacy)

    s = sub.add_parser(
        "fuse",
        help="merge per-robot / per-session stores into one store keyed "
        "by a Hydra-Multi fused scene graph",
    )
    s.add_argument(
        "fused",
        help="Hydra-Multi output directory (backend/fusion_node_map.json, "
        "interfaces/robot_<id>/world_T_robot.txt)",
    )
    s.add_argument(
        "--source",
        action="append",
        required=True,
        help="ROBOT_ID=STORE[:SESSION]: a source store and its robot id in the fused graph",
    )
    s.add_argument("--out", required=True, help="fused store directory")
    s.add_argument("--node-map", help="default: <fused>/backend/fusion_node_map.json")
    s.add_argument("--overwrite", action="store_true")
    g = s.add_mutually_exclusive_group()
    g.add_argument(
        "--copy", action="store_true", help="copy images (default: hardlink)"
    )
    g.add_argument(
        "--symlink", action="store_true", help="symlink images (default: hardlink)"
    )
    s.set_defaults(func=_cmd_fuse)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

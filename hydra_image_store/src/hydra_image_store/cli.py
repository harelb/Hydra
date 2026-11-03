"""``hydra-image-store`` command line."""

from __future__ import annotations

import argparse
import json
import sys

from .store import FrameKindError, ImageStore, SchemaVersionError


def _cmd_info(args) -> int:
    with ImageStore.open(*args.store) as store:
        info = store.info()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    for s in info:
        print(f"{s['path']}")
        print(
            f"  schema_version {s['meta'].get('schema_version')}  robots {s['robots']}  sensors {s['sensors']}"
        )
        frames = ", ".join(f"{k} {v}" for k, v in s["frames"].items()) or "none"
        print(
            f"  frames: {frames}  (with rgb {s['frames_with_rgb']}, unanchored {s['frames_unanchored']})"
        )
        print(
            f"  observations {s['observations']} (kept {s['observations_kept']}) over {s['tracks']} tracks"
        )
        print(
            f"  track bindings {s['track_bindings']} -> {s['bound_nodes']} nodes, node aliases {s['node_aliases']}"
        )
        extra = {k: v for k, v in s["meta"].items() if k != "schema_version"}
        if extra:
            print(f"  meta {extra}")
    return 0


def _cmd_export(args) -> int:
    from .export import export_store
    from .poses import load_dsg

    pose_source = load_dsg(args.dsg) if args.dsg else None
    with ImageStore.open(*args.store) as store:
        stats = export_store(
            store, args.out, pose_source, copy=args.copy, kept_only=args.kept_only
        )
    print(json.dumps(stats, indent=1))
    return 0


def _cmd_check(args) -> int:
    from .check import check_store

    report = check_store(args.store, dsg=args.dsg)
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        for name, ok in report["checks"].items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print("ALL PASS" if report["ok"] else "FAILURES")
    return 0 if report["ok"] else 1


def _cmd_index(args) -> int:
    from .indexer import index

    stats = index(
        args.run,
        keyframes=args.keyframes,
        tracks=args.tracks,
        hydra=args.hydra,
        store=args.store,
        dsg=args.dsg,
        events=args.events,
        robot_id=args.robot_id,
        agent_prefix=args.agent_prefix,
        max_anchor_dist_m=args.max_anchor_dist,
        link="copy" if args.copy else "hardlink",
        final=not args.partial,
    )
    print(json.dumps(stats, indent=1))
    return 0


def _cmd_tail(args) -> int:
    import logging
    import pathlib
    import signal
    import time

    from .tail import Tailer

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    tailer = Tailer(
        args.run,
        keyframes=args.keyframes,
        tracks=args.tracks,
        hydra=args.hydra,
        events=args.events,
        store=args.store,
        dsg=args.dsg,
        robot_id=args.robot_id,
        agent_prefix=args.agent_prefix,
        max_anchor_dist_m=args.max_anchor_dist,
        link="copy" if args.copy else "hardlink",
        settle_s=args.settle,
        finalize=not args.no_finalize,
    )

    def stop(*_):
        tailer._stop.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def on_poll(stats):
        if args.verbose or stats.get("final"):
            print(json.dumps({"time": round(time.time(), 3), **stats}), flush=True)

    if args.once:
        on_poll(tailer.poll())
    else:
        tailer.run(args.interval, timeout_s=args.timeout, on_poll=on_poll)
    summary = {**tailer.summary(), "final_stats": tailer.final_stats}
    text = json.dumps(summary, indent=1)
    if args.summary:
        pathlib.Path(args.summary).write_text(text)
    print(text)
    return 0


def _add_index_args(s) -> None:
    s.add_argument("--keyframes", help="KeyframeSelector.log_path (<ts>.zip archives)")
    s.add_argument(
        "--tracks", help="Khronos track logger log_path (<track_id>/track.json)"
    )
    s.add_argument("--hydra", help="Hydra log dir holding backend/ and frontend/")
    s.add_argument(
        "--dsg", help="scene graph for anchoring (default: <hydra>/backend/dsg.json)"
    )
    s.add_argument("--store", help="store directory (default: <run>/image_store)")
    s.add_argument(
        "--robot-id",
        type=int,
        default=None,
        help="default: the robot_id of the track logs, else the one the store was "
        "indexed with, else of the event logs, else inferred from the scene graph's "
        "agent prefix, else 0 (tail waits instead of guessing 0)",
    )
    s.add_argument(
        "--agent-prefix", help="agent NodeSymbol prefix (default: 'a' + robot id)"
    )
    s.add_argument("--max-anchor-dist", type=float, default=2.0, help="[m]")
    s.add_argument(
        "--copy", action="store_true", help="copy track images (default: hardlink)"
    )
    s.add_argument("--events", help="Hydra event log directory (default: <run>/events)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hydra-image-store", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="summarize one or more stores")
    s.add_argument("store", nargs="+", help="store directory or store.sqlite")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_info)

    s = sub.add_parser(
        "export", help="write one folder of images and metadata per node"
    )
    s.add_argument("store", nargs="+")
    s.add_argument(
        "--dsg",
        help="scene graph used to correct poses (json, or anything spark_dsg loads)",
    )
    s.add_argument("--out", required=True)
    s.add_argument(
        "--copy", action="store_true", help="copy files instead of hard/symlinking"
    )
    s.add_argument("--kept-only", action="store_true", help="only kept observations")
    s.set_defaults(func=_cmd_export)

    s = sub.add_parser(
        "index",
        help="build/update <RUN>/image_store from Hydra/Khronos image logs "
        "(keyframe archives, track logs, track_bindings.json, merges.json)",
    )
    s.add_argument(
        "run",
        help="run directory (inputs default to its keyframes/, "
        "tracks/ and hydra/ subdirectories)",
    )
    _add_index_args(s)
    s.add_argument(
        "--partial",
        action="store_true",
        help="Hydra is still running: leave frames newer than the newest "
        "agent node pending",
    )
    s.set_defaults(func=_cmd_index)

    s = sub.add_parser(
        "tail",
        help="index <RUN> live while Hydra writes it (keyframe archives, track logs, "
        "event logs); reconcile with the save-time files and exit when Hydra saved",
    )
    s.add_argument("run", help="run directory (same defaults as index)")
    _add_index_args(s)
    s.add_argument("--interval", type=float, default=1.0, help="[s] between passes")
    s.add_argument(
        "--settle",
        type=float,
        default=2.0,
        help="[s] the save-time files must stay unchanged before reconciling",
    )
    s.add_argument("--timeout", type=float, default=None, help="[s] stop after this")
    s.add_argument("--once", action="store_true", help="one pass, then exit")
    s.add_argument(
        "--no-finalize",
        action="store_true",
        help="never reconcile with the save-time files (live passes only)",
    )
    s.add_argument("--summary", help="also write the lag/pass summary (json) here")
    s.add_argument("-v", "--verbose", action="store_true", help="print every pass")
    s.set_defaults(func=_cmd_tail)

    s = sub.add_parser("check", help="structural consistency checks of a store")
    s.add_argument("store", help="run directory or store directory")
    s.add_argument(
        "--dsg",
        help="also check keyframe anchors / bound nodes against this scene graph",
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_check)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FrameKindError, SchemaVersionError) as err:
        print(f"hydra-image-store: error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

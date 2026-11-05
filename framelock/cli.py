"""Command-line interface: run on footage, render the demo, or launch the web UI."""

from __future__ import annotations

import argparse
import os
import sys

from framelock import overlay
from framelock.control import FramingConfig, FramingController
from framelock.lock import LockManager
from framelock.pipeline import Pipeline
from framelock.tracker import Tracker


def _build_detector(args):
    if args.detector == "motion":
        from framelock.detectors.motion import MotionDetector

        return MotionDetector(min_area=args.min_area)
    if args.detector == "cloud":
        from framelock.detectors.cloud import CloudConfig, CloudDetector

        if not args.cloud_endpoint:
            raise SystemExit("--cloud-endpoint is required for the cloud detector")
        key = args.cloud_api_key or os.environ.get("FRAMELOCK_CLOUD_KEY")
        return CloudDetector(
            CloudConfig(
                endpoint=args.cloud_endpoint, api_key=key, prompt=args.prompt,
                model=args.model, timeout=args.cloud_timeout,
            )
        )
    if args.detector == "onnx":
        from framelock.detectors.onnx import OnnxYoloDetector

        if not args.model:
            raise SystemExit("--model <path.onnx> is required for the onnx detector")
        allow = tuple(args.labels.split(",")) if args.labels else ("person",)
        return OnnxYoloDetector(args.model, label_allow=allow)
    raise SystemExit(f"unknown detector {args.detector!r}")


def _build_gimbal(args):
    if args.gimbal == "sim":
        from framelock.gimbal import SimulatedGimbal

        return SimulatedGimbal()
    if args.gimbal == "udp":
        from framelock.gimbal import UdpJsonGimbal

        return UdpJsonGimbal(args.udp_host, args.udp_port)
    if args.gimbal == "mavlink":
        from pymavlink import mavutil

        from framelock.gimbal import MavlinkGimbal

        if not args.mavlink_conn:
            raise SystemExit("--mavlink-conn is required for the mavlink gimbal")
        conn = mavutil.mavlink_connection(args.mavlink_conn)
        return MavlinkGimbal(conn)
    raise SystemExit(f"unknown gimbal {args.gimbal!r}")


def _cmd_run(args) -> int:
    import cv2

    from framelock.sources import VideoSource

    source = VideoSource(int(args.source) if args.source.isdigit() else args.source)
    width, height = source.size
    fps = source.fps
    detector = _build_detector(args)
    gimbal = _build_gimbal(args)
    pipe = Pipeline(
        detector, gimbal, width, height,
        tracker=Tracker(min_hits=args.min_hits, max_misses=args.max_misses),
        lock=LockManager(max_coast_seconds=args.max_coast),
        controller=FramingController(FramingConfig(max_rate=args.max_rate)),
        detect_every=args.detect_every,
        latency_comp=args.latency_comp,
    )

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    dt = 1.0 / fps
    locked = False
    count = 0
    try:
        for frame in source.frames():
            result = pipe.process(frame, dt)
            if args.auto_lock and not locked and count >= args.lock_after:
                # Explicit operator opt-in: lock the subject nearest frame center.
                res = pipe.designate(width / 2, height / 2)
                locked = res.track_id is not None
            annotated = overlay.draw(
                frame, result, 0.0, 0.0, fps, detector.name
            )
            if writer:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("FrameLock", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    pipe.release()
                    locked = False
            count += 1
            if args.max_frames and count >= args.max_frames:
                break
    finally:
        source.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        close = getattr(detector, "close", None)
        if close:
            close()
    print(f"processed {count} frames; final lock state: {pipe.lock.state.value}")
    return 0


def _cmd_render(args) -> int:
    from framelock.render import render_sim_video

    metrics = render_sim_video(
        args.output, frames=args.frames, fps=args.fps,
        auto_lock=not args.no_auto_lock, seed=args.seed, snapshot=args.snapshot,
    )
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return 0


def _cmd_demo(args) -> int:
    from framelock.web import run_demo

    run_demo(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="framelock", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run on a video file, webcam, or RTSP stream")
    r.add_argument("--source", default="0", help="video path, RTSP URL, or webcam index")
    r.add_argument("--detector", choices=["motion", "cloud", "onnx"], default="motion")
    r.add_argument("--gimbal", choices=["sim", "udp", "mavlink"], default="sim")
    r.add_argument("--output", help="write annotated MP4 here")
    r.add_argument("--no-display", action="store_true", help="headless (no GUI window)")
    r.add_argument("--auto-lock", action="store_true",
                   help="operator opt-in: lock the central subject after warmup")
    r.add_argument("--lock-after", type=int, default=15)
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--min-area", type=float, default=400.0)
    r.add_argument("--min-hits", type=int, default=3)
    r.add_argument("--max-misses", type=int, default=30)
    r.add_argument("--max-coast", type=float, default=1.5)
    r.add_argument("--max-rate", type=float, default=60.0)
    r.add_argument("--detect-every", type=int, default=1)
    r.add_argument("--latency-comp", type=float, default=0.0)
    r.add_argument("--cloud-endpoint")
    r.add_argument("--cloud-api-key")
    r.add_argument("--cloud-timeout", type=float, default=2.0)
    r.add_argument("--model", help="ONNX model path or cloud model name")
    r.add_argument("--prompt", help="open-vocabulary description for the cloud detector")
    r.add_argument("--labels", help="comma-separated allowed labels (onnx)")
    r.add_argument("--udp-host", default="127.0.0.1")
    r.add_argument("--udp-port", type=int, default=14550)
    r.add_argument("--mavlink-conn", help="pymavlink connection string, e.g. udpin:0.0.0.0:14550")
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser("render", help="render the closed-loop sim demo to an MP4")
    d.add_argument("--output", default="demo.mp4")
    d.add_argument("--snapshot", help="also write the final frame as an image")
    d.add_argument("--frames", type=int, default=360)
    d.add_argument("--fps", type=float, default=30.0)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--no-auto-lock", action="store_true")
    d.set_defaults(func=_cmd_render)

    w = sub.add_parser("demo", help="launch the interactive web demo")
    w.add_argument("--host", default="0.0.0.0")
    w.add_argument("--port", type=int, default=8000)
    w.set_defaults(func=_cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Headless renderer: produce an annotated proof video of the closed-loop demo."""

from __future__ import annotations

import cv2
import numpy as np

from framelock import overlay
from framelock.sim import SimRunner
from framelock.types import LockState


def render_sim_video(
    output: str,
    frames: int = 360,
    fps: float = 30.0,
    auto_lock: bool = True,
    lock_after: int = 30,
    seed: int = 7,
    snapshot: str | None = None,
) -> dict:
    runner = SimRunner(seed=seed)
    world = runner.world
    dt = 1.0 / fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (world.view_w, world.view_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    locked = False
    errors: list[float] = []
    last_annotated = None
    final_state = LockState.IDLE
    try:
        for i in range(frames):
            frame, result = runner.step(dt)
            if auto_lock and not locked and i >= lock_after:
                people = [t for t in result.tracks if t.label == "person"]
                if people:
                    t = people[0]
                    runner.designate(t.bbox.cx, t.bbox.cy)
                    locked = True
            if result.lock.state == LockState.LOCKED and result.control_bbox is not None:
                errors.append(
                    float(np.hypot(
                        result.control_bbox.cx - world.view_w / 2,
                        result.control_bbox.cy - world.view_h / 2,
                    ))
                )
            final_state = result.lock.state
            last_annotated = overlay.draw(
                frame, result, world.camera.pan, world.camera.tilt, fps, "sim"
            )
            writer.write(last_annotated)
    finally:
        writer.release()

    if snapshot and last_annotated is not None:
        cv2.imwrite(snapshot, last_annotated)

    settled = errors[len(errors) // 2:] if errors else []
    return {
        "output": output,
        "frames": frames,
        "locked": locked,
        "final_state": final_state.value,
        "mean_error_px": round(float(np.mean(errors)), 2) if errors else None,
        "settled_mean_error_px": round(float(np.mean(settled)), 2) if settled else None,
        "max_error_px": round(float(np.max(errors)), 2) if errors else None,
    }

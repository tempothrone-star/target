"""Heads-up display drawing for the framing pipeline (pure OpenCV, no I/O)."""

from __future__ import annotations

import cv2
import numpy as np

from framelock.pipeline import PipelineResult
from framelock.types import LockState, TrackState

_COLORS = {
    LockState.IDLE: (160, 160, 160),
    LockState.LOCKED: (0, 220, 0),
    LockState.COASTING: (0, 200, 255),
    LockState.LOST: (0, 0, 255),
}


def _reticle(img, box, color, label):
    x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
    c = max(8, min(w, h) // 4)
    for (px, py, dx, dy) in (
        (x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)
    ):
        cv2.line(img, (px, py), (px + dx * c, py), color, 2)
        cv2.line(img, (px, py), (px, py + dy * c), color, 2)
    cv2.putText(img, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw(
    frame: np.ndarray,
    result: PipelineResult,
    pan: float,
    tilt: float,
    fps: float,
    detector_name: str,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # Faint center crosshair (the framing goal).
    cv2.drawMarker(img, (w // 2, h // 2), (90, 90, 90), cv2.MARKER_CROSS, 18, 1)

    for tr in result.tracks:
        if tr.state == TrackState.COASTING:
            color = (0, 160, 200)
        else:
            color = (200, 200, 200)
        b = tr.bbox
        cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x2), int(b.y2)), color, 1)
        cv2.putText(
            img, f"#{tr.track_id} {tr.label}", (int(b.x), int(b.y2) + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

    lock = result.lock
    color = _COLORS.get(lock.state, (255, 255, 255))
    if lock.bbox is not None:
        _reticle(img, lock.bbox, color, f"LOCK #{lock.track_id}")
    if result.control_bbox is not None and lock.state == LockState.LOCKED:
        cb = result.control_bbox
        cv2.drawMarker(
            img, (int(cb.cx), int(cb.cy)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 1
        )

    cmd = result.command
    lines = [
        f"STATE: {lock.state.value.upper()}",
        f"target: {lock.track_id if lock.track_id is not None else '-'}",
        f"pan/tilt: {pan:+.1f} / {tilt:+.1f} deg",
        f"rate: {cmd.pan_rate:+.1f} / {cmd.tilt_rate:+.1f} deg/s",
        f"detector: {detector_name}   fps: {fps:.0f}",
    ]
    panel_h = 16 + len(lines) * 20
    overlay_panel = img.copy()
    cv2.rectangle(overlay_panel, (6, 6), (300, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_panel, 0.55, img, 0.45, 0, img)
    for i, text in enumerate(lines):
        y = 24 + i * 20
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if lock.state == LockState.LOST:
        cv2.putText(
            img, "TARGET LOST - click a subject to re-lock", (w // 2 - 200, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
    elif lock.state == LockState.IDLE:
        cv2.putText(
            img, "click a subject to lock", (w // 2 - 130, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2,
        )
    return img

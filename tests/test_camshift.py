"""Behavior tests for the CamShift click-to-lock tracker (synthetic color blob)."""

import cv2
import numpy as np

from framelock.detectors.camshift import CamShiftLock
from framelock.types import BBox


def _frame_with_blob(cx, cy, size=40, color=(40, 60, 220)):
    f = np.full((480, 640, 3), (30, 30, 30), np.uint8)
    cv2.rectangle(f, (cx - size, cy - size), (cx + size, cy + size), color, -1)
    return f


def test_locks_and_follows_moving_blob():
    lock = CamShiftLock()
    f0 = _frame_with_blob(200, 240)
    assert lock.init(f0, BBox.from_center(200, 240, 80, 80))
    assert lock.active
    cx = 200
    for _ in range(30):
        cx += 6
        box = lock.update(_frame_with_blob(cx, 240))
        assert box is not None
    assert abs(box.cx - cx) < 25  # followed the blob to the right


def test_reports_loss_when_subject_disappears():
    lock = CamShiftLock()
    assert lock.init(_frame_with_blob(200, 240), BBox.from_center(200, 240, 80, 80))
    blank = np.full((480, 640, 3), (30, 30, 30), np.uint8)
    results = [lock.update(blank) for _ in range(10)]
    assert any(r is None for r in results)


def test_init_rejects_tiny_roi():
    lock = CamShiftLock()
    assert not lock.init(_frame_with_blob(200, 240), BBox(200, 240, 2, 2))


def test_reset_deactivates():
    lock = CamShiftLock()
    lock.init(_frame_with_blob(200, 240), BBox.from_center(200, 240, 80, 80))
    lock.reset()
    assert not lock.active
    assert lock.update(_frame_with_blob(200, 240)) is None

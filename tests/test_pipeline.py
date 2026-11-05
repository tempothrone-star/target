"""End-to-end pipeline tests with a scripted detector and simulated gimbal."""

import numpy as np

from framelock.control import FramingConfig, FramingController
from framelock.gimbal import SimulatedGimbal
from framelock.lock import LockManager
from framelock.pipeline import Pipeline
from framelock.tracker import Tracker
from framelock.types import BBox, Detection, LockState


class ScriptedDetector:
    """Returns one box whose center follows a supplied path of (cx, cy)."""

    name = "scripted"

    def __init__(self, path):
        self.path = path
        self.i = 0

    def detect(self, frame):
        cx, cy = self.path[min(self.i, len(self.path) - 1)]
        self.i += 1
        return [Detection(BBox.from_center(cx, cy, 40, 40), 0.99, "person")]


def _frame():
    return np.zeros((480, 640, 3), np.uint8)


def _pipeline(path, **kw):
    det = ScriptedDetector(path)
    gimbal = SimulatedGimbal()
    pipe = Pipeline(
        det, gimbal, 640, 480,
        tracker=Tracker(min_hits=1),
        lock=LockManager(max_coast_seconds=1.0),
        controller=FramingController(FramingConfig(kp=40, kd=4, smoothing=0.5, max_rate=300)),
        **kw,
    )
    return pipe, gimbal


def test_designate_then_lock_persists():
    path = [(320, 240)] * 5
    pipe, _ = _pipeline(path)
    for _ in range(3):
        pipe.process(_frame(), 1 / 30)
    res = pipe.designate(320, 240)
    assert res.state == LockState.LOCKED
    out = pipe.process(_frame(), 1 / 30)
    assert out.lock.state == LockState.LOCKED


def test_closed_loop_recenters_world_target():
    """A world-fixed target off-center must be driven to image center by the loop."""
    gimbal = SimulatedGimbal(px_per_deg=5.0, fixed_dt=1 / 30)
    world_x = 520.0  # fixed in world; image center is 320

    class WorldDetector:
        name = "world"

        def detect(self, frame):
            img_cx = world_x - gimbal.pan * gimbal.px_per_deg
            return [Detection(BBox.from_center(img_cx, 240, 40, 40), 0.99, "person")]

    pipe = Pipeline(
        WorldDetector(), gimbal, 640, 480,
        tracker=Tracker(min_hits=1),
        lock=LockManager(max_coast_seconds=1.0),
        controller=FramingController(),
    )
    last_pan = 0.0
    for _ in range(3):
        pipe.process(_frame(), 1 / 30)
    pipe.designate(world_x, 240)
    for _ in range(300):
        ego = ((gimbal.pan - last_pan) * gimbal.px_per_deg, 0.0)
        last_pan = gimbal.pan
        out = pipe.process(_frame(), 1 / 30, ego_shift=ego)
    assert gimbal.pan > 0
    assert abs(out.control_bbox.cx - 320) < 12.0
    assert out.lock.state == LockState.LOCKED


def test_loss_stops_gimbal_and_does_not_reacquire():
    path = [(320, 240)] * 5
    pipe, gimbal = _pipeline(path)
    for _ in range(3):
        pipe.process(_frame(), 1 / 30)
    pipe.designate(320, 240)
    # Feed empty detections directly to force loss.
    for _ in range(60):
        out = pipe.process(_frame(), 1 / 30, detections=[])
    assert out.lock.state == LockState.LOST
    assert out.command.valid is False
    assert out.command.pan_rate == 0.0 and out.command.tilt_rate == 0.0


def test_detect_every_skips_detector_but_keeps_prediction():
    path = [(100 + i * 5, 240) for i in range(40)]
    pipe, _ = _pipeline(path, detect_every=3)
    ran = []
    for _ in range(9):
        out = pipe.process(_frame(), 1 / 30)
        ran.append(out.extra["ran_detector"])
    assert ran == [True, False, False, True, False, False, True, False, False]


def test_latency_compensation_leads_the_target():
    path = [(100 + i * 10, 240) for i in range(60)]
    pipe, _ = _pipeline(path, latency_comp=0.2)
    for _ in range(30):
        pipe.process(_frame(), 1 / 30)
    pipe.designate(pipe._tracks[0].bbox.cx, 240)
    out = pipe.process(_frame(), 1 / 30)
    # The control box should lead the measured box in the direction of motion.
    assert out.control_bbox.cx > out.lock.bbox.cx


def test_invalid_detect_every_rejected():
    import pytest

    with pytest.raises(ValueError):
        Pipeline(ScriptedDetector([(0, 0)]), SimulatedGimbal(), 640, 480, detect_every=0)

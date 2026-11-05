"""A synthetic moving world + virtual pan/tilt camera for the live demo.

This closes the loop without hardware: subjects move in a world, the camera
viewport pans/tilts under gimbal control, and a synthetic detector reports what
is visible (with noise and optional dropout) so the real tracker/lock/controller
are exercised end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from framelock.gimbal import SimulatedGimbal
from framelock.types import BBox, Detection


@dataclass
class SimObject:
    oid: int
    label: str
    x: float
    y: float
    vx: float
    vy: float
    w: float
    h: float
    color: tuple[int, int, int]


class SimWorld:
    def __init__(
        self,
        world_w: int = 2400,
        world_h: int = 1400,
        view_w: int = 960,
        view_h: int = 540,
        noise_px: float = 2.0,
        dropout_prob: float = 0.0,
        seed: int = 7,
    ) -> None:
        self.world_w = world_w
        self.world_h = world_h
        self.view_w = view_w
        self.view_h = view_h
        self.noise_px = noise_px
        self.dropout_prob = dropout_prob
        self.rng = np.random.default_rng(seed)
        self.camera = SimulatedGimbal(
            pan_limit=180.0, tilt_limit=90.0, px_per_deg=6.0, fixed_dt=1 / 30
        )
        self.base_cx = world_w / 2
        self.base_cy = world_h / 2
        # Subjects bounce inside a region the camera can always center, so the
        # demo shows true framing rather than hitting world/gimbal limits.
        self.min_x = self.view_w / 2 + 60
        self.max_x = self.world_w - self.view_w / 2 - 60
        self.min_y = self.view_h / 2 + 60
        self.max_y = self.world_h - self.view_h / 2 - 60
        self.objects = [
            SimObject(1, "person", 900, 600, 60, 22, 46, 96, (80, 200, 255)),
            SimObject(2, "person", 1450, 760, -50, 34, 46, 96, (120, 120, 255)),
            SimObject(3, "car", 1150, 880, 80, -18, 120, 60, (255, 180, 80)),
            SimObject(4, "ball", 1700, 520, -70, 55, 40, 40, (180, 255, 120)),
        ]

    def step(self, dt: float) -> None:
        for o in self.objects:
            o.x += o.vx * dt
            o.y += o.vy * dt
            if o.x < self.min_x or o.x > self.max_x:
                o.vx *= -1
                o.x = float(np.clip(o.x, self.min_x, self.max_x))
            if o.y < self.min_y or o.y > self.max_y:
                o.vy *= -1
                o.y = float(np.clip(o.y, self.min_y, self.max_y))

    def origin(self) -> tuple[float, float]:
        cam_cx = self.base_cx + self.camera.pan * self.camera.px_per_deg
        cam_cy = self.base_cy + self.camera.tilt * self.camera.px_per_deg
        ox = float(np.clip(cam_cx - self.view_w / 2, 0, self.world_w - self.view_w))
        oy = float(np.clip(cam_cy - self.view_h / 2, 0, self.world_h - self.view_h))
        return ox, oy

    def render(self) -> np.ndarray:
        ox, oy = self.origin()
        img = np.full((self.view_h, self.view_w, 3), (38, 32, 28), dtype=np.uint8)
        for gx in range(0, self.view_w, 60):
            cv2.line(img, (gx, 0), (gx, self.view_h), (52, 46, 42), 1)
        for gy in range(0, self.view_h, 60):
            cv2.line(img, (0, gy), (self.view_w, gy), (52, 46, 42), 1)
        for o in self.objects:
            ix, iy = o.x - ox, o.y - oy
            x, y = int(ix - o.w / 2), int(iy - o.h / 2)
            if x + o.w < 0 or x > self.view_w or y + o.h < 0 or y > self.view_h:
                continue
            if o.label == "person":
                cv2.rectangle(img, (x, y + int(o.h * 0.25)), (x + int(o.w), y + int(o.h)),
                              o.color, -1)
                cv2.circle(img, (int(ix), y + int(o.h * 0.18)), int(o.w * 0.32), o.color, -1)
            elif o.label == "ball":
                cv2.circle(img, (int(ix), int(iy)), int(o.w / 2), o.color, -1)
            else:
                cv2.rectangle(img, (x, y), (x + int(o.w), y + int(o.h)), o.color, -1)
            cv2.putText(img, o.label, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        return img

    def detections(self) -> list[Detection]:
        ox, oy = self.origin()
        out: list[Detection] = []
        for o in self.objects:
            if self.dropout_prob > 0 and self.rng.random() < self.dropout_prob:
                continue
            ix = o.x - ox + float(self.rng.normal(0, self.noise_px))
            iy = o.y - oy + float(self.rng.normal(0, self.noise_px))
            box = BBox.from_center(ix, iy, o.w, o.h)
            if box.x2 < 0 or box.x > self.view_w or box.y2 < 0 or box.y > self.view_h:
                continue
            out.append(Detection(box.clipped(self.view_w, self.view_h), 0.97, o.label))
        return out


class SimDetector:
    """Detector facade over the synthetic world (ignores the rendered pixels)."""

    name = "sim"

    def __init__(self, world: SimWorld) -> None:
        self.world = world

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.world.detections()


class SimRunner:
    """World + pipeline with camera ego-motion compensation wired in."""

    def __init__(self, seed: int = 7, controller=None) -> None:
        from framelock.control import FramingConfig, FramingController
        from framelock.lock import LockManager
        from framelock.pipeline import Pipeline
        from framelock.tracker import Tracker

        self.world = SimWorld(seed=seed)
        self.pipe = Pipeline(
            SimDetector(self.world), self.world.camera, self.world.view_w, self.world.view_h,
            tracker=Tracker(min_hits=2, max_misses=45),
            lock=LockManager(max_coast_seconds=1.5),
            controller=controller or FramingController(
                FramingConfig(kp=150.0, ki=40.0, kd=10.0, smoothing=0.6, max_rate=70.0,
                              deadband=0.015)
            ),
        )
        self._last_pan = 0.0
        self._last_tilt = 0.0

    def step(self, dt: float):
        self.world.step(dt)
        px = self.world.camera.px_per_deg
        cam = self.world.camera
        ego = ((cam.pan - self._last_pan) * px, (cam.tilt - self._last_tilt) * px)
        self._last_pan, self._last_tilt = cam.pan, cam.tilt
        frame = self.world.render()
        result = self.pipe.process(frame, dt, ego_shift=ego)
        return frame, result

    def designate(self, x: float, y: float):
        return self.pipe.designate(x, y)

    def release(self):
        return self.pipe.release()

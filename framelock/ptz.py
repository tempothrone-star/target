"""Digital pan/tilt/zoom viewport for fixed cameras (e.g. a laptop webcam).

A laptop camera cannot physically move, so framing is done by steering a crop
window inside the full frame. The same bounded angular-rate command that drives
a drone gimbal drives this crop, keeping the locked subject centered.
"""

from __future__ import annotations

import numpy as np

from framelock.types import BBox, GimbalCommand


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class DigitalPTZ:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        viewport_scale: float = 0.62,
        px_per_deg: float = 4.0,
    ) -> None:
        if not 0 < viewport_scale < 1:
            raise ValueError("viewport_scale must be in (0, 1)")
        self.fw = frame_w
        self.fh = frame_h
        self.vw = max(2, int(frame_w * viewport_scale))
        self.vh = max(2, int(frame_h * viewport_scale))
        self.px_per_deg = px_per_deg
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0

    def step(self, command: GimbalCommand, dt: float) -> None:
        if not command.valid:
            return
        self.cx = _clamp(
            self.cx + command.pan_rate * self.px_per_deg * dt,
            self.vw / 2, self.fw - self.vw / 2,
        )
        self.cy = _clamp(
            self.cy + command.tilt_rate * self.px_per_deg * dt,
            self.vh / 2, self.fh - self.vh / 2,
        )

    def origin(self) -> tuple[float, float]:
        return self.cx - self.vw / 2.0, self.cy - self.vh / 2.0

    def viewport_bbox(self) -> BBox:
        x, y = self.origin()
        return BBox(x, y, self.vw, self.vh)

    def to_viewport(self, bbox: BBox) -> BBox:
        x, y = self.origin()
        return BBox(bbox.x - x, bbox.y - y, bbox.w, bbox.h)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x, y = self.origin()
        x0, y0 = int(round(x)), int(round(y))
        return frame[y0 : y0 + self.vh, x0 : x0 + self.vw].copy()

"""Bounded PID framing controller: target-box error -> gimbal angular rates.

Command convention (degrees/second):
- pan_rate  > 0 moves the camera toward the image right.
- tilt_rate > 0 moves the camera toward the image bottom.

Both signs therefore drive the camera toward the subject. Rates are deadbanded,
clamped, and low-pass smoothed so the gimbal never jerks or exceeds safe limits.
When there is no valid lock the controller emits a zero-rate stop.
"""

from __future__ import annotations

from dataclasses import dataclass

from framelock.types import BBox, GimbalCommand


@dataclass
class FramingConfig:
    kp: float = 120.0
    ki: float = 30.0
    kd: float = 8.0
    deadband: float = 0.02
    max_rate: float = 90.0
    smoothing: float = 0.6  # 0..1, higher = snappier
    integral_limit: float = 1.5  # anti-windup clamp on the integral accumulator


class FramingController:
    def __init__(self, config: FramingConfig | None = None) -> None:
        self.config = config or FramingConfig()
        self._prev_err = (0.0, 0.0)
        self._prev_out = (0.0, 0.0)
        self._integral = (0.0, 0.0)
        self._has_prev = False

    def reset(self) -> None:
        self._prev_err = (0.0, 0.0)
        self._prev_out = (0.0, 0.0)
        self._integral = (0.0, 0.0)
        self._has_prev = False

    def _axis(
        self, err: float, prev_err: float, prev_out: float, integral: float, dt: float
    ) -> tuple[float, float]:
        cfg = self.config
        if abs(err) < cfg.deadband:
            err = 0.0
        deriv = (err - prev_err) / dt if dt > 0 else 0.0
        integral = max(-cfg.integral_limit, min(cfg.integral_limit, integral + err * dt))
        rate = cfg.kp * err + cfg.ki * integral + cfg.kd * deriv
        rate = max(-cfg.max_rate, min(cfg.max_rate, rate))
        out = cfg.smoothing * rate + (1.0 - cfg.smoothing) * prev_out
        return out, integral

    def compute(
        self, lock_bbox: BBox | None, frame_w: float, frame_h: float, dt: float
    ) -> GimbalCommand:
        if lock_bbox is None or frame_w <= 0 or frame_h <= 0:
            self.reset()
            return GimbalCommand(0.0, 0.0, valid=False)

        ex = (lock_bbox.cx - frame_w / 2.0) / (frame_w / 2.0)
        ey = (lock_bbox.cy - frame_h / 2.0) / (frame_h / 2.0)

        prev_ex, prev_ey = self._prev_err if self._has_prev else (ex, ey)
        prev_pan, prev_tilt = self._prev_out if self._has_prev else (0.0, 0.0)
        int_x, int_y = self._integral

        pan, int_x = self._axis(ex, prev_ex, prev_pan, int_x, dt)
        tilt, int_y = self._axis(ey, prev_ey, prev_tilt, int_y, dt)

        self._prev_err = (ex, ey)
        self._prev_out = (pan, tilt)
        self._integral = (int_x, int_y)
        self._has_prev = True

        return GimbalCommand(
            pan_rate=pan,
            tilt_rate=tilt,
            valid=True,
            meta={"error_x": ex, "error_y": ey},
        )

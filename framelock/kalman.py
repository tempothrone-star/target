"""Constant-velocity Kalman filter for bounding-box center and size.

State vector: [cx, cy, w, h, vx, vy, vw, vh].
Measurement:  [cx, cy, w, h].

This runs locally at frame rate so the control loop stays smooth even when the
(cloud) detector is slow or briefly unavailable.
"""

from __future__ import annotations

import numpy as np

from framelock.types import BBox

_DIM = 8
_MEAS = 4


class BoxKalmanFilter:
    """Predict/update a single box with a constant-velocity motion model."""

    def __init__(
        self,
        bbox: BBox,
        process_noise: float = 1.0,
        measurement_noise: float = 4.0,
    ) -> None:
        self.x = np.zeros(_DIM, dtype=float)
        self.x[0] = bbox.cx
        self.x[1] = bbox.cy
        self.x[2] = max(bbox.w, 1.0)
        self.x[3] = max(bbox.h, 1.0)

        self.P = np.eye(_DIM, dtype=float) * 10.0
        # Velocity is initially unknown -> large covariance.
        self.P[4:, 4:] *= 100.0

        self.H = np.zeros((_MEAS, _DIM), dtype=float)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self._q = float(process_noise)
        self._r = float(measurement_noise)
        self.R = np.eye(_MEAS, dtype=float) * self._r

    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(_DIM, dtype=float)
        for i in range(_MEAS):
            F[i, i + _MEAS] = dt
        return F

    def _process_covariance(self, dt: float) -> np.ndarray:
        # Piecewise white-noise jerk model per axis pair (pos, vel).
        Q = np.zeros((_DIM, _DIM), dtype=float)
        dt2 = dt * dt
        dt3 = dt2 * dt
        for i in range(_MEAS):
            p, v = i, i + _MEAS
            Q[p, p] = dt3 / 3.0
            Q[p, v] = Q[v, p] = dt2 / 2.0
            Q[v, v] = dt
        return Q * self._q

    def predict(self, dt: float) -> BBox:
        if dt <= 0:
            return self.box()
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_covariance(dt)
        return self.box()

    def update(self, bbox: BBox) -> BBox:
        z = np.array([bbox.cx, bbox.cy, max(bbox.w, 1.0), max(bbox.h, 1.0)], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(_DIM) - K @ self.H
        # Joseph form for numerical stability.
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.box()

    def box(self) -> BBox:
        w = max(float(self.x[2]), 1.0)
        h = max(float(self.x[3]), 1.0)
        return BBox.from_center(float(self.x[0]), float(self.x[1]), w, h)

    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])

    def shift(self, dx: float, dy: float) -> None:
        """Translate the position state (camera ego-motion compensation)."""
        self.x[0] += dx
        self.x[1] += dy

    def predict_ahead(self, seconds: float) -> BBox:
        """Non-mutating lookahead used to compensate detector latency."""
        cx = self.x[0] + self.x[4] * seconds
        cy = self.x[1] + self.x[5] * seconds
        w = max(self.x[2] + self.x[6] * seconds, 1.0)
        h = max(self.x[3] + self.x[7] * seconds, 1.0)
        return BBox.from_center(float(cx), float(cy), float(w), float(h))

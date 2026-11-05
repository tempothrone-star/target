"""Behavior tests for the box Kalman filter."""

import numpy as np

from framelock.kalman import BoxKalmanFilter
from framelock.types import BBox


def _run_constant_velocity(speed_px_s=120.0, frames=60, fps=30.0):
    kf = BoxKalmanFilter(BBox(100, 100, 40, 40))
    dt = 1.0 / fps
    truth_cx = 120.0
    for _ in range(frames):
        truth_cx += speed_px_s * dt
        kf.predict(dt)
        kf.update(BBox.from_center(truth_cx, 120.0, 40, 40))
    return kf


def test_converges_to_constant_velocity():
    kf = _run_constant_velocity()
    vx, vy = kf.velocity()
    assert abs(vx - 120.0) < 10.0
    assert abs(vy) < 5.0


def test_prediction_reduces_error_vs_last_measurement():
    kf = _run_constant_velocity()
    last = kf.box()
    ahead = kf.predict_ahead(0.2)
    truth_future_cx = last.cx + 120.0 * 0.2
    assert abs(ahead.cx - truth_future_cx) < abs(last.cx - truth_future_cx)


def test_predict_ahead_does_not_mutate_state():
    kf = _run_constant_velocity()
    before = kf.x.copy()
    kf.predict_ahead(1.0)
    assert np.array_equal(before, kf.x)


def test_update_pulls_toward_measurement():
    kf = BoxKalmanFilter(BBox(0, 0, 20, 20))
    kf.update(BBox.from_center(200, 200, 20, 20))
    assert kf.box().cx > 10.0  # moved toward the measurement
    assert kf.box().cx < 200.0  # but not all the way in one step


def test_zero_dt_predict_is_noop():
    kf = BoxKalmanFilter(BBox(5, 5, 10, 10))
    before = kf.x.copy()
    kf.predict(0.0)
    assert np.array_equal(before, kf.x)


def test_covariance_stays_symmetric_psd():
    kf = _run_constant_velocity()
    assert np.allclose(kf.P, kf.P.T)
    assert np.all(np.linalg.eigvalsh(kf.P) > -1e-6)


def test_size_never_collapses():
    kf = BoxKalmanFilter(BBox(0, 0, 10, 10))
    for _ in range(50):
        kf.predict(1 / 30)
        kf.update(BBox.from_center(0, 0, 0.0, 0.0))
    box = kf.box()
    assert box.w >= 1.0 and box.h >= 1.0

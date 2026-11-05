"""Behavior tests for the bounded framing controller."""

from framelock.control import FramingConfig, FramingController
from framelock.types import BBox


def test_no_lock_emits_stop():
    ctl = FramingController()
    cmd = ctl.compute(None, 640, 480, dt=1 / 30)
    assert not cmd.valid
    assert cmd.pan_rate == 0.0 and cmd.tilt_rate == 0.0


def test_target_right_commands_positive_pan():
    ctl = FramingController()
    cmd = ctl.compute(BBox.from_center(480, 240, 40, 40), 640, 480, dt=1 / 30)
    assert cmd.valid and cmd.pan_rate > 0
    assert abs(cmd.tilt_rate) < 1e-6


def test_target_below_commands_positive_tilt():
    ctl = FramingController()
    cmd = ctl.compute(BBox.from_center(320, 360, 40, 40), 640, 480, dt=1 / 30)
    assert cmd.tilt_rate > 0
    assert abs(cmd.pan_rate) < 1e-6


def test_centered_target_within_deadband_is_zero():
    ctl = FramingController()
    cmd = ctl.compute(BBox.from_center(320, 240, 40, 40), 640, 480, dt=1 / 30)
    assert cmd.pan_rate == 0.0 and cmd.tilt_rate == 0.0


def test_rate_is_clamped_to_max():
    ctl = FramingController(FramingConfig(max_rate=10.0, smoothing=1.0))
    cmd = ctl.compute(BBox.from_center(639, 240, 2, 2), 640, 480, dt=1 / 30)
    assert cmd.pan_rate <= 10.0


def test_pid_tracks_constant_velocity_with_no_steady_state_lag():
    ctl = FramingController(
        FramingConfig(kp=50.0, ki=20.0, kd=5.0, smoothing=0.6, max_rate=300.0)
    )
    w, h = 640, 480
    px_per_deg, dt = 5.0, 1 / 30
    cam_x, target_x, velocity = 0.0, 200.0, 80.0
    for _ in range(600):
        target_x += velocity * dt
        img_cx = target_x - cam_x
        cmd = ctl.compute(BBox.from_center(img_cx, 240, 40, 40), w, h, dt)
        cam_x += cmd.pan_rate * px_per_deg * dt
    assert abs((target_x - cam_x) - w / 2) < 5.0


def test_integral_anti_windup_bounds_output():
    ctl = FramingController(FramingConfig(ki=100.0, max_rate=10.0, smoothing=1.0))
    for _ in range(1000):
        cmd = ctl.compute(BBox.from_center(639, 240, 2, 2), 640, 480, 1 / 30)
    assert abs(cmd.pan_rate) <= 10.0


def test_closed_loop_converges_to_center():
    """Simulate a first-order camera and verify the subject ends up centered."""
    ctl = FramingController()
    w, h = 640, 480
    cam_x, cam_y = 0.0, 0.0  # camera center offset in pixels
    target_x, target_y = 500.0, 400.0  # fixed world target
    px_per_deg = 5.0
    dt = 1 / 30
    for _ in range(300):
        img_cx = target_x - cam_x
        img_cy = target_y - cam_y
        cmd = ctl.compute(BBox.from_center(img_cx, img_cy, 40, 40), w, h, dt)
        cam_x += cmd.pan_rate * px_per_deg * dt
        cam_y += cmd.tilt_rate * px_per_deg * dt
    assert abs((target_x - cam_x) - w / 2) < 12.0
    assert abs((target_y - cam_y) - h / 2) < 12.0

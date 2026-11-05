"""Behavior tests for the laptop-webcam framing engine (synthetic frames)."""

import base64

import cv2
import numpy as np
import pytest

from framelock.webcam import WebcamEngine


def _blob(cx, cy, size=40, color=(40, 60, 220)):
    f = np.full((480, 640, 3), (30, 30, 30), np.uint8)
    cv2.rectangle(f, (cx - size, cy - size), (cx + size, cy + size), color, -1)
    return f


def _jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_designate_before_frame_is_idle():
    eng = WebcamEngine()
    assert eng.designate(0.5, 0.5)["lock_state"] == "idle"


def test_process_returns_two_valid_views_and_state():
    eng = WebcamEngine()
    out = eng.process(_jpeg(_blob(320, 240)))
    assert out["state"]["mode"] == "webcam"
    for key in ("full", "gimbal"):
        raw = base64.b64decode(out[key])
        assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None


def test_lock_then_digital_ptz_centers_static_subject():
    eng = WebcamEngine()
    eng.process(_jpeg(_blob(420, 240)))
    res = eng.designate(420 / 640, 240 / 480)
    assert res["lock_state"] == "locked"
    for _ in range(90):
        out = eng.process(_jpeg(_blob(420, 240)))
    assert out["state"]["lock_state"] == "locked"
    # The viewport center should have steered onto the subject (x=420).
    assert abs(eng.ptz.cx - 420) < 20


def test_loss_coasts_then_stops_and_resets():
    eng = WebcamEngine()
    eng.process(_jpeg(_blob(320, 240)))
    eng.designate(0.5, 0.5)
    blank = np.full((480, 640, 3), (30, 30, 30), np.uint8)
    states = [eng.process(_jpeg(blank))["state"]["lock_state"] for _ in range(150)]
    assert "coasting" in states
    assert states[-1] == "lost"
    assert not eng.camshift.active


def test_release_clears_lock():
    eng = WebcamEngine()
    eng.process(_jpeg(_blob(320, 240)))
    eng.designate(0.5, 0.5)
    assert eng.release()["lock_state"] == "idle"


def test_invalid_bytes_raise():
    eng = WebcamEngine()
    with pytest.raises(ValueError):
        eng.process(b"not a jpeg")

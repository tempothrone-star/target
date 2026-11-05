"""Smoke tests for the live-demo web app (no browser, no hardware)."""

import time

from fastapi.testclient import TestClient

from framelock.web import create_app


def test_endpoints_and_click_to_lock():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert "FrameLock" in client.get("/").text

        # Let the background loop produce a few frames so tracks exist.
        time.sleep(0.4)

        state = client.get("/state").json()
        assert "lock_state" in state
        assert state["fps"] > 0

        snap = client.get("/snapshot.jpg")
        assert snap.status_code == 200
        assert snap.headers["content-type"] == "image/jpeg"
        assert len(snap.content) > 0

        # Click the center: a track should be designated (nearest subject).
        res = client.post("/lock", json={"x": 0.5, "y": 0.5}).json()
        assert res["lock_state"] == "locked"
        assert res["track_id"] is not None

        rel = client.post("/release", json={}).json()
        assert rel["lock_state"] == "idle"

        drop = client.post("/dropout", json={}).json()
        assert drop["dropout"] > 0


def test_webcam_frame_lock_and_release():
    import cv2
    import numpy as np

    frame = np.full((480, 640, 3), (30, 30, 30), np.uint8)
    cv2.rectangle(frame, (280, 200), (360, 280), (40, 60, 220), -1)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    jpeg = buf.tobytes()

    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/api/webcam/frame", content=jpeg, headers={"Content-Type": "image/jpeg"}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["state"]["mode"] == "webcam"
        assert data["full"] and data["gimbal"]

        lock = client.post("/api/webcam/lock", json={"x": 0.5, "y": 0.5}).json()
        assert lock["lock_state"] == "locked"

        rel = client.post("/api/webcam/release", json={}).json()
        assert rel["lock_state"] == "idle"


def test_webcam_invalid_frame_returns_400():
    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/api/webcam/frame", content=b"garbage", headers={"Content-Type": "image/jpeg"}
        )
        assert r.status_code == 400


def test_lock_coordinates_are_clamped():
    app = create_app()
    with TestClient(app) as client:
        time.sleep(0.3)
        # Out-of-range clicks must not crash and must clamp into the frame.
        res = client.post("/lock", json={"x": 5.0, "y": -3.0})
        assert res.status_code == 200

"""Laptop-webcam framing engine.

The browser captures webcam frames and posts them here. The operator clicks a
subject; CamShift locks it, a Kalman filter smooths/predicts it, the lock manager
enforces coast-then-stop on loss, and a digital PTZ viewport steers to keep it
centered. Two views are returned: the full scene (clickable, with the gimbal-aim
rectangle) and the stabilized gimbal output (the crop a real gimbal would frame).
"""

from __future__ import annotations

import base64
import threading
import time

import cv2
import numpy as np

from framelock import overlay
from framelock.control import FramingConfig, FramingController
from framelock.detectors.camshift import CamShiftLock
from framelock.detectors.motion import MotionDetector
from framelock.kalman import BoxKalmanFilter
from framelock.lock import LockManager, LockResult
from framelock.pipeline import PipelineResult
from framelock.ptz import DigitalPTZ
from framelock.tracker import Tracker
from framelock.types import BBox, GimbalCommand, LockState, Track, TrackState

_LOCK_ID = 1


def _b64_jpeg(img: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("failed to encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class WebcamEngine:
    def __init__(self, viewport_scale: float = 0.62, max_rate: float = 90.0) -> None:
        self.viewport_scale = viewport_scale
        self.camshift = CamShiftLock()
        self.lock = LockManager(max_coast_seconds=1.5)
        self.controller = FramingController(FramingConfig(max_rate=max_rate, deadband=0.015))
        self.motion = MotionDetector(min_area=500.0)
        self.awareness = Tracker(min_hits=2, max_misses=15)
        self.kf: BoxKalmanFilter | None = None
        self.ptz: DigitalPTZ | None = None
        self._mutex = threading.Lock()
        self._last_t: float | None = None
        self._last_frame: np.ndarray | None = None
        self._size: tuple[int, int] | None = None
        self._fps = 0.0

    def _ensure_ptz(self, w: int, h: int) -> None:
        if self._size != (w, h):
            self.ptz = DigitalPTZ(w, h, viewport_scale=self.viewport_scale)
            self._size = (w, h)
            self.camshift.reset()
            self.kf = None
            self.lock.release()

    def process(self, jpeg: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode frame")
        h, w = frame.shape[:2]
        with self._mutex:
            self._last_frame = frame
            self._ensure_ptz(w, h)
            now = time.monotonic()
            dt = 1 / 30 if self._last_t is None else min(max(now - self._last_t, 1 / 60), 0.25)
            if self._last_t is not None and dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_t = now

            self.awareness.update(self.motion.detect(frame), dt)

            tracks: list[Track] = []
            if self.camshift.active:
                box = self.camshift.update(frame)
                if box is not None:
                    if self.kf is None:
                        self.kf = BoxKalmanFilter(box)
                    else:
                        self.kf.predict(dt)
                        self.kf.update(box)
                    tracks = [Track(
                        track_id=_LOCK_ID, bbox=self.kf.box(), label="locked", score=1.0,
                        state=TrackState.CONFIRMED, misses=0, velocity=self.kf.velocity(),
                    )]
            lock_res = self.lock.update(tracks, dt)
            if lock_res.state == LockState.LOST:
                self.camshift.reset()
                self.kf = None

            cmd = GimbalCommand(0.0, 0.0, valid=False)
            if lock_res.bbox is not None:
                vp_box = self.ptz.to_viewport(lock_res.bbox)
                cmd = self.controller.compute(vp_box, self.ptz.vw, self.ptz.vh, dt)
            else:
                self.controller.reset()
            self.ptz.step(cmd, dt)

            full = self._draw_full(frame, lock_res)
            gimbal = self._draw_gimbal(frame, lock_res, cmd, w, h)
            state = self._state(lock_res, cmd, w, h)
            return {"full": _b64_jpeg(full), "gimbal": _b64_jpeg(gimbal), "state": state}

    def designate(self, nx: float, ny: float) -> dict:
        with self._mutex:
            if self._last_frame is None:
                return {"lock_state": LockState.IDLE.value, "track_id": None}
            h, w = self._last_frame.shape[:2]
            x, y = nx * w, ny * h
            containing = [t for t in self.awareness.tracks if t.bbox.contains(x, y)]
            if containing:
                roi = min(containing, key=lambda t: t.bbox.area).bbox
            else:
                roi = BBox.from_center(x, y, max(40, w * 0.12), max(40, h * 0.18))
            if not self.camshift.init(self._last_frame, roi):
                return {"lock_state": LockState.IDLE.value, "track_id": None}
            self.kf = BoxKalmanFilter(roi)
            self.controller.reset()
            res = self.lock.designate(
                [Track(_LOCK_ID, roi, "locked", 1.0, TrackState.CONFIRMED)], x, y
            )
            return {"lock_state": res.state.value, "track_id": res.track_id}

    def release(self) -> dict:
        with self._mutex:
            self.camshift.reset()
            self.kf = None
            self.controller.reset()
            res = self.lock.release()
            return {"lock_state": res.state.value}

    def _draw_full(self, frame: np.ndarray, lock_res: LockResult) -> np.ndarray:
        img = frame.copy()
        for t in self.awareness.tracks:
            b = t.bbox
            cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x2), int(b.y2)), (150, 150, 150), 1)
        vp = self.ptz.viewport_bbox()
        cv2.rectangle(img, (int(vp.x), int(vp.y)), (int(vp.x2), int(vp.y2)), (255, 220, 0), 2)
        cv2.putText(img, "gimbal aim", (int(vp.x) + 4, int(vp.y) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1, cv2.LINE_AA)
        if lock_res.bbox is not None:
            color = overlay._COLORS.get(lock_res.state, (255, 255, 255))
            overlay._reticle(img, lock_res.bbox, color, "LOCK")
        cv2.putText(img, "click a subject to lock", (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def _draw_gimbal(
        self, frame: np.ndarray, lock_res: LockResult, cmd: GimbalCommand, w: int, h: int
    ) -> np.ndarray:
        crop = self.ptz.crop(frame)
        vp_lock = LockResult(
            lock_res.state, lock_res.track_id,
            self.ptz.to_viewport(lock_res.bbox) if lock_res.bbox else None,
            lock_res.coast_elapsed,
        )
        pan = (self.ptz.cx - w / 2) / self.ptz.px_per_deg
        tilt = (self.ptz.cy - h / 2) / self.ptz.px_per_deg
        res = PipelineResult(
            tracks=[], detections=[], lock=vp_lock, command=cmd,
            control_bbox=self.ptz.to_viewport(lock_res.bbox) if lock_res.bbox else None,
            frame_index=0,
        )
        return overlay.draw(crop, res, pan, tilt, self._fps, "camshift")

    def _state(self, lock_res: LockResult, cmd: GimbalCommand, w: int, h: int) -> dict:
        return {
            "lock_state": lock_res.state.value,
            "track_id": lock_res.track_id,
            "pan": round((self.ptz.cx - w / 2) / self.ptz.px_per_deg, 2),
            "tilt": round((self.ptz.cy - h / 2) / self.ptz.px_per_deg, 2),
            "pan_rate": round(cmd.pan_rate, 2),
            "tilt_rate": round(cmd.tilt_rate, 2),
            "fps": round(self._fps, 1),
            "mode": "webcam",
        }

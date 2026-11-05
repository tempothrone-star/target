"""End-to-end framing pipeline: detect -> track -> lock -> control -> gimbal.

The detector may run every frame (local) or every Nth frame (slow cloud). Between
detections the Kalman tracker predicts forward, and an optional latency lead
compensates for cloud round-trip delay so the camera aims where the subject is
*now*, not where it was when the frame was captured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from framelock.control import FramingController
from framelock.detectors.base import Detector
from framelock.gimbal import Gimbal
from framelock.lock import LockManager, LockResult
from framelock.tracker import Tracker
from framelock.types import BBox, Detection, GimbalCommand, LockState, Track


@dataclass
class PipelineResult:
    tracks: list[Track]
    detections: list[Detection]
    lock: LockResult
    command: GimbalCommand
    control_bbox: BBox | None
    frame_index: int
    extra: dict = field(default_factory=dict)


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        gimbal: Gimbal,
        frame_w: float,
        frame_h: float,
        tracker: Tracker | None = None,
        lock: LockManager | None = None,
        controller: FramingController | None = None,
        detect_every: int = 1,
        latency_comp: float = 0.0,
    ) -> None:
        if detect_every < 1:
            raise ValueError("detect_every must be >= 1")
        self.detector = detector
        self.gimbal = gimbal
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.tracker = tracker or Tracker()
        self.lock = lock or LockManager()
        self.controller = controller or FramingController()
        self.detect_every = detect_every
        self.latency_comp = latency_comp
        self._frame_index = 0
        self._tracks: list[Track] = []

    def designate(self, x: float, y: float) -> LockResult:
        result = self.lock.designate(self._tracks, x, y)
        if result.state != LockState.LOCKED:
            self.controller.reset()
        return result

    def release(self) -> LockResult:
        self.controller.reset()
        return self.lock.release()

    def process(
        self,
        frame: np.ndarray,
        dt: float,
        detections: list[Detection] | None = None,
        ego_shift: tuple[float, float] = (0.0, 0.0),
    ) -> PipelineResult:
        """Process one frame.

        ego_shift is the camera motion in pixels since the previous frame
        (pan_px, tilt_px); it is compensated so tracking survives camera moves.
        """
        self._frame_index += 1
        ran_detector = False
        if detections is None:
            if (self._frame_index - 1) % self.detect_every == 0:
                detections = self.detector.detect(frame)
                ran_detector = True
            else:
                detections = []

        self._tracks = self.tracker.update(detections, dt, ego_shift)
        lock_result = self.lock.update(self._tracks, dt)

        control_bbox = lock_result.bbox
        if (
            lock_result.state == LockState.LOCKED
            and self.latency_comp > 0
            and lock_result.track_id is not None
        ):
            kf = self.tracker.filter_for(lock_result.track_id)
            if kf is not None:
                control_bbox = kf.predict_ahead(self.latency_comp)

        command = self.controller.compute(control_bbox, self.frame_w, self.frame_h, dt)
        self.gimbal.send(command)

        return PipelineResult(
            tracks=self._tracks,
            detections=detections,
            lock=lock_result,
            command=command,
            control_bbox=control_bbox,
            frame_index=self._frame_index,
            extra={"ran_detector": ran_detector},
        )

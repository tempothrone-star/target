"""Frame sources for real footage: video files, webcams, and RTSP streams."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class VideoSource:
    def __init__(self, source: str | int) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source: {source!r}")

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    @property
    def size(self) -> tuple[int, int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield frame

    def release(self) -> None:
        self.cap.release()

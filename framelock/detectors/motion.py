"""Model-free motion detector (background subtraction).

A zero-setup local fallback and the detector used by the simulated demo. It is
not semantic AI; for real scenes prefer the cloud or ONNX backend.
"""

from __future__ import annotations

import cv2
import numpy as np

from framelock.types import BBox, Detection


class MotionDetector:
    name = "motion"

    def __init__(self, min_area: float = 400.0, history: int = 200) -> None:
        self.min_area = min_area
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=25, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = self._sub.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            out.append(Detection(BBox(float(x), float(y), float(w), float(h)), 0.99, "motion"))
        return out

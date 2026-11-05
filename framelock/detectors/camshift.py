"""Click-to-lock single-subject tracker using CamShift (color histogram).

Works on static or moving subjects with no model download: the operator clicks a
region, we model its color, and CamShift follows it frame to frame. A back-
projection confidence check reports loss so the lock manager can coast then stop.
"""

from __future__ import annotations

import cv2
import numpy as np

from framelock.types import BBox


class CamShiftLock:
    def __init__(self, min_area: float = 200.0, max_area_frac: float = 0.5) -> None:
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self._hist: np.ndarray | None = None
        self._window: tuple[int, int, int, int] | None = None
        self._ref_mean: float = 0.0
        self._term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    @property
    def active(self) -> bool:
        return self._hist is not None and self._window is not None

    @staticmethod
    def _backproject(hsv: np.ndarray, hist: np.ndarray) -> np.ndarray:
        bp = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        # Ignore near-gray / dark pixels so the window cannot balloon into a
        # low-saturation background that shares the subject's hue bin.
        valid = cv2.inRange(hsv, (0, 60, 32), (180, 255, 255))
        bp &= valid
        return bp

    def init(self, frame: np.ndarray, roi: BBox) -> bool:
        h, w = frame.shape[:2]
        roi = roi.clipped(w, h)
        x, y, bw, bh = int(roi.x), int(roi.y), int(roi.w), int(roi.h)
        if bw < 4 or bh < 4:
            return False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        region = hsv[y : y + bh, x : x + bw]
        mask = cv2.inRange(region, (0, 60, 32), (180, 255, 255))
        hist = cv2.calcHist([region], [0], mask, [16], [0, 180])
        if hist.sum() <= 0:
            return False
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self._hist = hist
        self._window = (x, y, bw, bh)
        bp = self._backproject(hsv, hist)
        self._ref_mean = float(bp[y : y + bh, x : x + bw].mean())
        return True

    def update(self, frame: np.ndarray) -> BBox | None:
        if not self.active:
            return None
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bp = self._backproject(hsv, self._hist)
        _, window = cv2.CamShift(bp, self._window, self._term)
        x, y, bw, bh = window
        if bw < 2 or bh < 2 or x < 0 or y < 0 or x + bw > w or y + bh > h:
            return None
        area = bw * bh
        if area < self.min_area or area > self.max_area_frac * w * h:
            return None
        mean = float(bp[y : y + bh, x : x + bw].mean())
        if self._ref_mean > 0 and mean < 0.25 * self._ref_mean:
            return None  # color match lost
        self._window = window
        return BBox(float(x), float(y), float(bw), float(bh))

    def reset(self) -> None:
        self._hist = None
        self._window = None
        self._ref_mean = 0.0

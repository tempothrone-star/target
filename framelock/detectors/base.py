"""Detector protocol shared by cloud and local backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from framelock.types import Detection


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for a BGR image (HxWx3 uint8)."""
        ...

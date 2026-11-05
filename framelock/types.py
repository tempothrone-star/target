"""Shared value types for the tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in image pixels: top-left (x, y) plus width/height."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w < 0 or self.h < 0:
            raise ValueError("bbox width/height must be non-negative")

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @classmethod
    def from_center(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        return cls(cx - w / 2.0, cy - h / 2.0, w, h)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def iou(self, other: BBox) -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clipped(self, width: float, height: float) -> BBox:
        x = min(max(self.x, 0.0), width)
        y = min(max(self.y, 0.0), height)
        x2 = min(max(self.x2, 0.0), width)
        y2 = min(max(self.y2, 0.0), height)
        return BBox(x, y, max(0.0, x2 - x), max(0.0, y2 - y))


@dataclass(frozen=True)
class Detection:
    """A single detector output for one frame."""

    bbox: BBox
    score: float
    label: str = "object"


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


@dataclass
class Track:
    """A tracked subject with a stable id and a predicted box."""

    track_id: int
    bbox: BBox
    label: str
    score: float
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    misses: int = 0
    age: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)


class LockState(str, Enum):
    IDLE = "idle"
    LOCKED = "locked"
    COASTING = "coasting"
    LOST = "lost"


@dataclass(frozen=True)
class GimbalCommand:
    """Bounded angular-rate command for a framing gimbal (degrees/second)."""

    pan_rate: float
    tilt_rate: float
    valid: bool = True
    meta: dict = field(default_factory=dict)

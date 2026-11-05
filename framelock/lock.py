"""Operator-designated target lock with bounded coast-then-stop loss handling.

Safety properties:
- A lock only ever begins from an explicit operator designation.
- On target loss the system coasts on a short linear prediction, then stops and
  drops the lock. It never autonomously searches or re-acquires a new subject.
"""

from __future__ import annotations

from dataclasses import dataclass

from framelock.tracker import Tracker
from framelock.types import BBox, LockState, Track, TrackState


@dataclass(frozen=True)
class LockResult:
    state: LockState
    track_id: int | None
    bbox: BBox | None
    coast_elapsed: float = 0.0


class LockManager:
    def __init__(self, max_coast_seconds: float = 1.5) -> None:
        if max_coast_seconds < 0:
            raise ValueError("max_coast_seconds must be >= 0")
        self.max_coast_seconds = max_coast_seconds
        self._track_id: int | None = None
        self._state = LockState.IDLE
        self._last_bbox: BBox | None = None
        self._velocity: tuple[float, float] = (0.0, 0.0)
        self._coast_elapsed = 0.0

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def track_id(self) -> int | None:
        return self._track_id

    def designate(self, tracks: list[Track], x: float, y: float) -> LockResult:
        """Lock the operator-chosen subject nearest the clicked image point."""
        candidates = [t for t in tracks if t.state != TrackState.COASTING] or tracks
        target = Tracker.nearest_to_point(candidates, x, y)
        if target is None:
            return LockResult(LockState.IDLE, None, None)
        self._track_id = target.track_id
        self._state = LockState.LOCKED
        self._last_bbox = target.bbox
        self._velocity = target.velocity
        self._coast_elapsed = 0.0
        return LockResult(self._state, self._track_id, self._last_bbox)

    def release(self) -> LockResult:
        self._track_id = None
        self._state = LockState.IDLE
        self._last_bbox = None
        self._velocity = (0.0, 0.0)
        self._coast_elapsed = 0.0
        return LockResult(self._state, None, None)

    def update(self, tracks: list[Track], dt: float) -> LockResult:
        if self._track_id is None:
            # Preserve a sticky LOST state until the operator designates again.
            return LockResult(self._state, None, None)

        track = next((t for t in tracks if t.track_id == self._track_id), None)
        if track is not None and track.misses == 0:
            self._state = LockState.LOCKED
            self._last_bbox = track.bbox
            self._velocity = track.velocity
            self._coast_elapsed = 0.0
            return LockResult(self._state, self._track_id, self._last_bbox)

        # Target not freshly matched: coast on a bounded linear prediction.
        self._coast_elapsed += dt
        if self._coast_elapsed > self.max_coast_seconds or self._last_bbox is None:
            return self._lost()

        self._state = LockState.COASTING
        vx, vy = self._velocity
        predicted = BBox.from_center(
            self._last_bbox.cx + vx * self._coast_elapsed,
            self._last_bbox.cy + vy * self._coast_elapsed,
            self._last_bbox.w,
            self._last_bbox.h,
        )
        return LockResult(self._state, self._track_id, predicted, self._coast_elapsed)

    def _lost(self) -> LockResult:
        self._track_id = None
        self._state = LockState.LOST
        self._last_bbox = None
        self._velocity = (0.0, 0.0)
        self._coast_elapsed = 0.0
        return LockResult(LockState.LOST, None, None)

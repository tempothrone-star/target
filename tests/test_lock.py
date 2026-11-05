"""Behavior tests for the operator-designated lock manager."""

import pytest

from framelock.lock import LockManager
from framelock.types import BBox, LockState, Track, TrackState


def _track(tid=1, cx=100, cy=100, misses=0, velocity=(0.0, 0.0)):
    return Track(
        track_id=tid,
        bbox=BBox.from_center(cx, cy, 40, 40),
        label="person",
        score=0.9,
        state=TrackState.CONFIRMED if misses == 0 else TrackState.COASTING,
        misses=misses,
        velocity=velocity,
    )


def test_designate_locks_nearest_track():
    lm = LockManager()
    tracks = [_track(1, 100, 100), _track(2, 400, 300)]
    res = lm.designate(tracks, 390, 310)
    assert res.state == LockState.LOCKED
    assert res.track_id == 2


def test_designate_with_no_tracks_is_idle():
    lm = LockManager()
    assert lm.designate([], 10, 10).state == LockState.IDLE


def test_update_keeps_lock_while_matched():
    lm = LockManager()
    lm.designate([_track(1, 100, 100)], 100, 100)
    res = lm.update([_track(1, 130, 100)], dt=0.1)
    assert res.state == LockState.LOCKED
    assert res.bbox.cx == 130


def test_coast_uses_velocity_then_stops_lost():
    lm = LockManager(max_coast_seconds=0.5)
    lm.designate([_track(1, 100, 100, velocity=(100.0, 0.0))], 100, 100)
    coasting = lm.update([_track(1, 100, 100, misses=1, velocity=(100.0, 0.0))], dt=0.1)
    assert coasting.state == LockState.COASTING
    assert coasting.bbox.cx > 100  # predicted forward

    res = coasting
    for _ in range(10):
        res = lm.update([], dt=0.1)
    assert res.state == LockState.LOST
    assert lm.track_id is None


def test_lost_lock_does_not_autonomously_reacquire():
    lm = LockManager(max_coast_seconds=0.0)
    lm.designate([_track(1, 100, 100)], 100, 100)
    lm.update([], dt=0.1)  # immediately lost
    # A brand-new subject appears; the system must NOT lock it on its own.
    res = lm.update([_track(2, 500, 500)], dt=0.1)
    assert res.state == LockState.LOST  # sticky until the operator acts
    assert res.track_id is None
    assert res.bbox is None


def test_release_returns_to_idle():
    lm = LockManager()
    lm.designate([_track()], 100, 100)
    assert lm.release().state == LockState.IDLE


def test_negative_coast_rejected():
    with pytest.raises(ValueError):
        LockManager(max_coast_seconds=-1)

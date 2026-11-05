"""Behavior tests for the multi-object tracker."""

from framelock.tracker import Tracker
from framelock.types import BBox, Detection, TrackState


def _det(cx, cy, w=40, h=40, label="person"):
    return Detection(BBox.from_center(cx, cy, w, h), score=0.9, label=label)


def test_single_target_keeps_stable_id_and_confirms():
    tr = Tracker(min_hits=3)
    ids = set()
    for i in range(6):
        tracks = tr.update([_det(100 + i * 5, 100)], dt=1 / 30)
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert len(ids) == 1
    assert tracks[0].state == TrackState.CONFIRMED


def test_two_separated_targets_get_distinct_ids():
    tr = Tracker(min_hits=2)
    for _ in range(4):
        tracks = tr.update([_det(100, 100), _det(400, 300)], dt=1 / 30)
    assert len(tracks) == 2
    assert len({t.track_id for t in tracks}) == 2
    assert all(t.state == TrackState.CONFIRMED for t in tracks)


def test_occluded_track_coasts_then_is_removed():
    tr = Tracker(min_hits=1, max_misses=3)
    for _ in range(3):
        tr.update([_det(100, 100)], dt=1 / 30)
    # Detection disappears.
    states = []
    for _ in range(5):
        tracks = tr.update([], dt=1 / 30)
        states.append(tracks[0].state if tracks else None)
    assert states[0] == TrackState.COASTING
    assert states[-1] is None  # removed after max_misses


def test_coasting_track_predicts_forward_motion():
    tr = Tracker(min_hits=1)
    for i in range(20):
        tr.update([_det(100 + i * 10, 100)], dt=0.1)
    last_cx = tr.tracks[0].bbox.cx
    tr.update([], dt=0.1)
    assert tr.tracks[0].bbox.cx > last_cx  # kept moving right while coasting


def test_new_detection_spawns_new_track():
    tr = Tracker(min_hits=1)
    tr.update([_det(100, 100)], dt=1 / 30)
    tracks = tr.update([_det(100, 100), _det(500, 400)], dt=1 / 30)
    assert len(tracks) == 2


def test_nearest_to_point_prefers_containing_box():
    tr = Tracker(min_hits=1)
    tracks = tr.update([_det(100, 100, 80, 80), _det(120, 120, 20, 20)], dt=1 / 30)
    chosen = Tracker.nearest_to_point(tracks, 120, 120)
    assert chosen.bbox.w == 20  # smaller containing box wins


def test_ego_motion_compensation_keeps_track_locked():
    """A world-fixed target must stay associated while the camera pans fast."""
    tr = Tracker(min_hits=1)
    world_x, cam = 300.0, 0.0
    for _ in range(5):
        tr.update([_det(world_x - cam, 100)], dt=1 / 30)
    for _ in range(20):
        cam += 40.0  # 40 px/frame pan: a full box-width, would break naive IoU
        tracks = tr.update([_det(world_x - cam, 100)], dt=1 / 30, ego_shift=(40.0, 0.0))
    t = tracks[0]
    assert t.misses == 0
    assert abs(t.bbox.cx - (world_x - cam)) < 5.0


def test_without_compensation_fast_pan_loses_track():
    tr = Tracker(min_hits=1)
    world_x, cam = 300.0, 0.0
    for _ in range(5):
        tr.update([_det(world_x - cam, 100)], dt=1 / 30)
    lost = False
    for _ in range(20):
        cam += 40.0
        tracks = tr.update([_det(world_x - cam, 100)], dt=1 / 30)  # no ego_shift
        if any(t.misses > 0 for t in tracks):
            lost = True
    assert lost


def test_invalid_min_hits_rejected():
    import pytest

    with pytest.raises(ValueError):
        Tracker(min_hits=0)

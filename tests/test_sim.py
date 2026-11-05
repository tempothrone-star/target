"""Behavior tests for the synthetic world used by the demo."""

from framelock.sim import SimDetector, SimWorld


def test_objects_stay_inside_world():
    world = SimWorld(seed=1)
    for _ in range(2000):
        world.step(1 / 30)
    for o in world.objects:
        assert 0 <= o.x <= world.world_w
        assert 0 <= o.y <= world.world_h


def test_origin_clamped_to_world_bounds():
    world = SimWorld()
    world.camera.pan = 1000.0  # absurd angle
    world.camera.tilt = -1000.0
    ox, oy = world.origin()
    assert 0 <= ox <= world.world_w - world.view_w
    assert 0 <= oy <= world.world_h - world.view_h


def test_render_shape_and_detections_visible():
    world = SimWorld(seed=3)
    frame = world.render()
    assert frame.shape == (world.view_h, world.view_w, 3)
    dets = SimDetector(world).detect(frame)
    assert len(dets) >= 1
    for d in dets:
        assert 0 <= d.bbox.cx <= world.view_w
        assert 0 <= d.bbox.cy <= world.view_h


def test_dropout_can_suppress_all_detections():
    world = SimWorld(seed=3, dropout_prob=1.0)
    assert world.detections() == []

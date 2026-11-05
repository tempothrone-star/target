"""Behavior tests for shared value types."""

import pytest

from framelock.types import BBox


def test_center_and_corners():
    b = BBox(10, 20, 30, 40)
    assert (b.cx, b.cy) == (25, 40)
    assert (b.x2, b.y2) == (40, 60)
    assert b.area == 1200


def test_from_center_roundtrip():
    b = BBox.from_center(100, 50, 20, 10)
    assert (b.x, b.y, b.w, b.h) == (90, 45, 20, 10)


def test_iou_identical_and_disjoint():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BBox(100, 100, 10, 10)) == 0.0


def test_iou_half_overlap():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 0, 10, 10)
    assert a.iou(b) == pytest.approx(50 / 150)


def test_contains():
    b = BBox(0, 0, 10, 10)
    assert b.contains(5, 5)
    assert not b.contains(11, 5)


def test_clipped():
    b = BBox(-5, -5, 20, 20).clipped(10, 10)
    assert (b.x, b.y, b.w, b.h) == (0, 0, 10, 10)


def test_negative_size_rejected():
    with pytest.raises(ValueError):
        BBox(0, 0, -1, 5)

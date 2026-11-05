"""Behavior tests for the digital PTZ viewport."""

import numpy as np
import pytest

from framelock.ptz import DigitalPTZ
from framelock.types import BBox, GimbalCommand


def test_step_moves_center_and_clamps():
    ptz = DigitalPTZ(1000, 800, viewport_scale=0.5, px_per_deg=10.0)
    for _ in range(500):
        ptz.step(GimbalCommand(100.0, 100.0), dt=0.1)
    assert ptz.cx == 1000 - ptz.vw / 2
    assert ptz.cy == 800 - ptz.vh / 2


def test_invalid_command_holds():
    ptz = DigitalPTZ(1000, 800)
    ptz.step(GimbalCommand(50.0, 50.0), dt=1.0)
    cx, cy = ptz.cx, ptz.cy
    ptz.step(GimbalCommand(99.0, 99.0, valid=False), dt=1.0)
    assert (ptz.cx, ptz.cy) == (cx, cy)


def test_crop_shape_and_viewport_mapping():
    ptz = DigitalPTZ(640, 480, viewport_scale=0.5)
    frame = np.zeros((480, 640, 3), np.uint8)
    crop = ptz.crop(frame)
    assert crop.shape == (ptz.vh, ptz.vw, 3)
    vp = ptz.viewport_bbox()
    assert (vp.w, vp.h) == (ptz.vw, ptz.vh)
    # A box at the viewport center maps to the crop center.
    center = BBox.from_center(ptz.cx, ptz.cy, 20, 20)
    mapped = ptz.to_viewport(center)
    assert abs(mapped.cx - ptz.vw / 2) < 1 and abs(mapped.cy - ptz.vh / 2) < 1


def test_invalid_scale_rejected():
    with pytest.raises(ValueError):
        DigitalPTZ(640, 480, viewport_scale=1.0)

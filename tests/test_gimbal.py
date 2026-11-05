"""Behavior tests for gimbal adapters (no hardware required)."""

import json
import math
import socket
from types import SimpleNamespace

import pytest

from framelock.gimbal import MavlinkGimbal, SimulatedGimbal, UdpJsonGimbal
from framelock.types import GimbalCommand


def test_simulated_integrates_and_clamps():
    g = SimulatedGimbal(pan_limit=10.0, tilt_limit=5.0)
    for _ in range(100):
        g.step(GimbalCommand(pan_rate=100.0, tilt_rate=-100.0), dt=0.1)
    assert g.pan == 10.0
    assert g.tilt == -5.0


def test_simulated_invalid_command_holds_position():
    g = SimulatedGimbal()
    g.step(GimbalCommand(30.0, 30.0), dt=1.0)
    pan, tilt = g.pan, g.tilt
    g.step(GimbalCommand(99.0, 99.0, valid=False), dt=1.0)
    assert (g.pan, g.tilt) == (pan, tilt)


def test_udp_json_roundtrip():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.settimeout(2.0)
    g = UdpJsonGimbal("127.0.0.1", port)
    try:
        g.send(GimbalCommand(12.5, -7.25, valid=True))
        data, _ = server.recvfrom(4096)
        msg = json.loads(data)
        assert msg["type"] == "gimbal_rate"
        assert msg["pan_rate_dps"] == 12.5
        assert msg["tilt_rate_dps"] == -7.25
        assert msg["valid"] is True
    finally:
        g.close()
        server.close()


def test_mavlink_message_serializes_and_decodes():
    from pymavlink.dialects.v20 import common as mavlink2

    class Capture:
        def __init__(self):
            self.buf = bytearray()

        def write(self, b):
            self.buf.extend(b)

    cap = Capture()
    mav = mavlink2.MAVLink(cap)
    mav.srcSystem = 255
    mav.srcComponent = 190
    conn = SimpleNamespace(mav=mav)

    adapter = MavlinkGimbal(conn, target_system=1, target_component=1)
    adapter.send(GimbalCommand(pan_rate=60.0, tilt_rate=-30.0, valid=True))

    parser = mavlink2.MAVLink(Capture())
    msgs = parser.parse_buffer(bytes(cap.buf))
    assert msgs and msgs[0].get_type() == "GIMBAL_MANAGER_SET_PITCHYAW"
    m = msgs[0]
    assert m.yaw_rate == pytest.approx(math.radians(60.0), abs=1e-4)
    assert m.pitch_rate == pytest.approx(math.radians(-30.0), abs=1e-4)


def test_mavlink_invalid_command_is_zero_rate_hold():
    calls = []

    class FakeMav:
        def gimbal_manager_set_pitchyaw_send(self, *args):
            calls.append(args)

    adapter = MavlinkGimbal(SimpleNamespace(mav=FakeMav()))
    adapter.send(GimbalCommand(50.0, 50.0, valid=False))
    args = calls[0]
    # (sys, comp, flags, device_id, pitch, yaw, pitch_rate, yaw_rate)
    assert args[2] == 0  # no mode-change flags
    assert args[6] == 0.0 and args[7] == 0.0  # zero rates

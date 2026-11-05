"""Gimbal output adapters.

The pipeline only ever emits bounded angular-rate framing commands. Adapters
translate those to a specific transport. None of them control a payload.

- SimulatedGimbal: integrates rates in software (tests + demo).
- UdpJsonGimbal: sends JSON datagrams to a companion-computer bridge.
- MavlinkGimbal: optional pymavlink adapter for PX4/ArduPilot gimbal managers.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Protocol

from framelock.types import GimbalCommand


class Gimbal(Protocol):
    def send(self, command: GimbalCommand) -> None: ...


class SimulatedGimbal:
    """Integrates angular rates into pan/tilt angles with soft limits."""

    def __init__(
        self,
        pan_limit: float = 180.0,
        tilt_limit: float = 90.0,
        px_per_deg: float = 5.0,
        fixed_dt: float | None = None,
    ) -> None:
        self.pan = 0.0
        self.tilt = 0.0
        self.pan_limit = pan_limit
        self.tilt_limit = tilt_limit
        self.px_per_deg = px_per_deg
        # When set, send() integrates with this fixed step (deterministic demo/tests).
        self.fixed_dt = fixed_dt
        self.last_command = GimbalCommand(0.0, 0.0, valid=False)
        self._last_t: float | None = None

    def send(self, command: GimbalCommand) -> None:
        if self.fixed_dt is not None:
            self.step(command, self.fixed_dt)
            return
        now = time.monotonic()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.1)
        self._last_t = now
        self.last_command = command
        if not command.valid:
            return
        self.pan = _clamp(self.pan + command.pan_rate * dt, -self.pan_limit, self.pan_limit)
        self.tilt = _clamp(self.tilt + command.tilt_rate * dt, -self.tilt_limit, self.tilt_limit)

    def step(self, command: GimbalCommand, dt: float) -> None:
        """Deterministic integration for tests and the fixed-step demo loop."""
        self.last_command = command
        if not command.valid:
            return
        self.pan = _clamp(self.pan + command.pan_rate * dt, -self.pan_limit, self.pan_limit)
        self.tilt = _clamp(self.tilt + command.tilt_rate * dt, -self.tilt_limit, self.tilt_limit)


class UdpJsonGimbal:
    """Emit framing commands as JSON datagrams to a companion-computer bridge."""

    def __init__(self, host: str, port: int) -> None:
        self.address = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: GimbalCommand) -> None:
        payload = json.dumps(
            {
                "type": "gimbal_rate",
                "pan_rate_dps": round(command.pan_rate, 4),
                "tilt_rate_dps": round(command.tilt_rate, 4),
                "valid": command.valid,
            }
        ).encode("utf-8")
        self._sock.sendto(payload, self.address)

    def close(self) -> None:
        self._sock.close()


class MavlinkGimbal:
    """Optional MAVLink gimbal-manager adapter (requires pymavlink).

    Uses GIMBAL_MANAGER_SET_PITCHYAW with angular velocities in rad/s. Construct
    with a pymavlink connection; the connection is owned by the caller.
    """

    def __init__(self, connection, target_system: int = 1, target_component: int = 1) -> None:
        self.connection = connection
        self.target_system = target_system
        self.target_component = target_component

    def send(self, command: GimbalCommand) -> None:
        import math

        # On loss, command a zero-rate hold (never a retract/neutral mode change).
        pan = command.pan_rate if command.valid else 0.0
        tilt = command.tilt_rate if command.valid else 0.0
        self.connection.mav.gimbal_manager_set_pitchyaw_send(
            self.target_system,
            self.target_component,
            0,  # flags: no mode change
            0,  # gimbal_device_id: default
            float("nan"),  # pitch angle: unused (rate control)
            float("nan"),  # yaw angle: unused (rate control)
            math.radians(tilt),
            math.radians(pan),
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

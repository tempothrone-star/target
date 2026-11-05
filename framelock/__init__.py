"""FrameLock: operator-designated camera target lock and auto-framing.

Civilian camera-drone assistance only. A human operator designates the subject;
the system keeps it framed and emits bounded gimbal-framing commands. It has no
payload, firing, or autonomous-interception capability.
"""

from framelock.types import BBox, Detection, GimbalCommand, LockState, Track, TrackState

__all__ = [
    "BBox",
    "Detection",
    "GimbalCommand",
    "LockState",
    "Track",
    "TrackState",
]

__version__ = "0.1.0"

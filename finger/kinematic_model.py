"""Compatibility wrapper for finger_kinematics."""

from __future__ import annotations

try:
    from .finger_kinematics import *  # noqa: F401,F403
except ImportError:
    from finger_kinematics import *  # type: ignore # noqa: F401,F403

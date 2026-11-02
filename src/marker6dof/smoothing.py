"""Timestamp-aware display smoothing of camera-frame rigid poses.

Smooth the reference pose before composing a fixed TCP transform.  This keeps
the reference and TCP overlays on the same rigid body; it does not change the
accepted measurement or calibrate the TCP offset.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .projection import validate_rigid_transform


class PoseSmoother:
    """Adaptive exponential position and shortest-path rotation smoothing.

    Cutoffs are in Hz.  ``motion_gain`` is Hz per m/s and
    ``rotation_motion_gain`` is Hz per rad/s.  Low-pass *velocity vectors*,
    rather than raw inter-frame speed, drive the adaptive cutoffs so alternating
    measurement noise does not immediately open the display bandwidth.

    Positive minimum cutoffs always follow slow real motion; there is no
    deadband or stationary freeze.  Lower cutoffs reduce noise at the cost of
    display lag.  Lost observations and clock/gap discontinuities clear the
    filter, so the next accepted pose initializes a new display segment.
    """

    def __init__(
        self,
        position_cutoff_hz=1.0,
        rotation_cutoff_hz=0.7,
        motion_gain=8.0,
        velocity_cutoff_hz=1.0,
        max_gap_s=0.2,
        rotation_motion_gain=1.0,
    ):
        for name, value in (
            ("position_cutoff_hz", position_cutoff_hz),
            ("rotation_cutoff_hz", rotation_cutoff_hz),
            ("velocity_cutoff_hz", velocity_cutoff_hz),
            ("max_gap_s", max_gap_s),
        ):
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        for name, value in (
            ("motion_gain", motion_gain),
            ("rotation_motion_gain", rotation_motion_gain),
        ):
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, value)
        self.reset()

    def reset(self):
        self._last_time = None
        self._last_position = None
        self._last_rotation = None
        self._position = None
        self._rotation = None
        self._velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)

    @staticmethod
    def _alpha(cutoff_hz, dt):
        return -math.expm1(-2.0 * math.pi * cutoff_hz * dt)

    def _pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self._rotation.as_matrix()
        pose[:3, 3] = self._position
        return pose

    def update(self, pose_or_none, timestamp):
        """Return a smoothed pose copy, or ``None`` for a lost observation.

        ``timestamp`` is finite seconds in the camera's timestamp domain.
        Repeated/backward timestamps and gaps over ``max_gap_s`` initialize
        directly from the current pose instead of interpolating stale state.
        Neither the input matrix nor the returned matrix is retained by alias.
        """
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if pose_or_none is None:
            self.reset()
            return None

        pose = validate_rigid_transform(pose_or_none, "pose")
        position = pose[:3, 3].copy()
        rotation = Rotation.from_matrix(pose[:3, :3])
        dt = timestamp - self._last_time if self._last_time is not None else 0.0
        if self._last_time is None or dt <= 0.0 or dt > self.max_gap_s:
            self.reset()
            self._position = position.copy()
            self._rotation = rotation
        else:
            velocity_alpha = self._alpha(self.velocity_cutoff_hz, dt)
            velocity = (position - self._last_position) / dt
            # Express angular velocity in the camera frame for a consistent
            # vector basis as the object rotates.
            angular_velocity = (rotation * self._last_rotation.inv()).as_rotvec() / dt
            self._velocity += velocity_alpha * (velocity - self._velocity)
            self._angular_velocity += velocity_alpha * (angular_velocity - self._angular_velocity)

            position_cutoff = self.position_cutoff_hz + self.motion_gain * np.linalg.norm(self._velocity)
            rotation_cutoff = (
                self.rotation_cutoff_hz
                + self.rotation_motion_gain * np.linalg.norm(self._angular_velocity)
            )
            self._position += self._alpha(position_cutoff, dt) * (position - self._position)
            increment = (self._rotation.inv() * rotation).as_rotvec()
            self._rotation = self._rotation * Rotation.from_rotvec(
                self._alpha(rotation_cutoff, dt) * increment
            )

        self._last_time = timestamp
        self._last_position = position
        self._last_rotation = rotation
        return self._pose()

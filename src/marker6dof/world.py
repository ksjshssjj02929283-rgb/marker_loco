"""Compose synchronized VIO camera poses with marker-derived object poses.

Every transform follows the repository convention ``T_target_source``.  In
particular, a mobile T265 setup supplies ``T_world_left`` from its VIO pose
stream and the marker tracker supplies ``T_left_tcp``.  Their product is the
metric TCP pose in the VIO world frame:

``T_world_tcp = T_world_left @ T_left_tcp``.

The T265's pose stream and fisheye stream share the device hardware clock, but
they run at different rates.  The helpers below reject a missing, stale or
low-confidence pose rather than silently combining measurements from different
instants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .projection import validate_rigid_transform
from .tracking import RelativeTrace


def transform_from_translation_quaternion(translation, quaternion_xyzw) -> np.ndarray:
    """Build ``T_world_pose`` from RealSense pose translation/quaternion data."""

    position = np.asarray(translation, dtype=np.float64).reshape(-1)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("pose translation must contain three finite values")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("pose quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("pose quaternion must be non-zero")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = position
    return transform


def t265_pose_to_transform(pose_data: Any) -> np.ndarray:
    """Convert ``pyrealsense2.pose_data`` to its documented world pose matrix."""

    return transform_from_translation_quaternion(
        [pose_data.translation.x, pose_data.translation.y, pose_data.translation.z],
        [pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w],
    )


@dataclass(frozen=True)
class VioSample:
    """A fisheye-time camera pose, or an explicit reason it was rejected."""

    timestamp_ms: Optional[float]
    tracker_confidence: Optional[int]
    T_world_camera: Optional[np.ndarray]
    reason: str

    @property
    def valid(self) -> bool:
        return self.T_world_camera is not None


def synchronized_t265_camera_pose(
    frames: Any,
    camera_timestamp_ms: float,
    T_pose_camera: np.ndarray,
    *,
    max_delta_ms: float = 20.0,
    min_tracker_confidence: int = 2,
) -> VioSample:
    """Return ``T_world_camera`` only for a time-aligned confident T265 pose.

    ``T_pose_camera`` is obtained from SDK stream extrinsics, e.g.
    ``left_profile.get_extrinsics_to(pose_profile)``.  This keeps the VIO
    sensor frame distinct from the left fisheye optical frame used by PnP.
    """

    if max_delta_ms < 0:
        raise ValueError("max_delta_ms must be non-negative")
    if min_tracker_confidence < 0:
        raise ValueError("min_tracker_confidence must be non-negative")
    pose_from_camera = validate_rigid_transform(T_pose_camera, "T_pose_camera")
    pose_frame = frames.get_pose_frame()
    if not pose_frame:
        return VioSample(None, None, None, "vio_pose_missing")
    pose_timestamp_ms = float(pose_frame.get_timestamp())
    if not np.isfinite(pose_timestamp_ms):
        return VioSample(None, None, None, "vio_pose_timestamp_invalid")
    if abs(pose_timestamp_ms - float(camera_timestamp_ms)) > max_delta_ms:
        return VioSample(pose_timestamp_ms, None, None, "vio_pose_time_mismatch")
    pose_data = pose_frame.get_pose_data()
    confidence = int(getattr(pose_data, "tracker_confidence", 0))
    if confidence < min_tracker_confidence:
        return VioSample(pose_timestamp_ms, confidence, None, "vio_low_confidence")
    try:
        world_from_pose = t265_pose_to_transform(pose_data)
    except (AttributeError, TypeError, ValueError):
        return VioSample(pose_timestamp_ms, confidence, None, "vio_pose_invalid")
    return VioSample(
        pose_timestamp_ms,
        confidence,
        validate_rigid_transform(world_from_pose @ pose_from_camera, "T_world_camera"),
        "ok",
    )


class WorldTrajectory:
    """Synchronized ego and object traces in one shared initial-VIO frame.

    The output origin is the first accepted VIO camera pose after construction
    or ``reset()``.  Thus both traces use the *same* frame; unlike two
    independently initialised relative tracks, their spatial separation is
    meaningful.  A rejected VIO sample breaks both traces.
    """

    def __init__(self, limit: int = 3000):
        self.ego = RelativeTrace(limit)
        self.object = RelativeTrace(limit)
        self.origin_from_world: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.ego.reset()
        self.object.reset()
        self.origin_from_world = None

    def update(
        self,
        timestamp_s: float,
        T_world_camera: Optional[np.ndarray],
        T_camera_object: Optional[np.ndarray],
        *,
        object_quality: str = "LOST",
        display_world_object: Optional[np.ndarray] = None,
    ) -> tuple[dict, dict]:
        """Append synchronized raw poses and an optional world-frame display pose.

        The display pose must already be smoothed in the world frame; a
        delayed camera-frame pose must not be paired with current VIO.
        """

        ego_input = {"status": "VIO", "reason": "vio_ok", "pose": None}
        object_input = {"status": object_quality, "reason": "marker_pose_missing", "pose": None}
        display_pose = None
        if T_world_camera is None:
            ego_input["reason"] = "vio_pose_unavailable"
            self.ego.broken = self.object.broken = True
            return self.ego.update(ego_input, timestamp_s), self.object.update(object_input, timestamp_s)

        world_from_camera = validate_rigid_transform(T_world_camera, "T_world_camera")
        if self.origin_from_world is None:
            self.origin_from_world = np.linalg.inv(world_from_camera)
        ego_input["pose"] = world_from_camera
        if T_camera_object is not None:
            camera_from_object = validate_rigid_transform(T_camera_object, "T_camera_object")
            object_input.update(reason="ok", pose=world_from_camera @ camera_from_object)
            if display_world_object is not None:
                display_pose = validate_rigid_transform(display_world_object, "display_world_object")
        return (
            self.ego.update(ego_input, timestamp_s, initial_transform=self.origin_from_world),
            self.object.update(
                object_input, timestamp_s, initial_transform=self.origin_from_world,
                display_pose=display_pose,
            ),
        )


__all__ = [
    "VioSample",
    "WorldTrajectory",
    "synchronized_t265_camera_pose",
    "t265_pose_to_transform",
    "transform_from_translation_quaternion",
]

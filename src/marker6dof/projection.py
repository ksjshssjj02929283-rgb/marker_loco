"""Rigid-transform and RGB projection helpers for camera-frame overlays.

The tracking code represents a pose as ``T_camera_object``: a homogeneous
matrix that maps points in the object frame into the camera optical frame.
This module keeps that convention explicit for CAD-derived TCP overlays and
handles the common case where a trajectory briefly goes behind the camera.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .realsense import project_realsense


def validate_rigid_transform(value: Sequence[Sequence[float]], name: str = "transform") -> np.ndarray:
    """Return *value* as a finite 4x4 SE(3) matrix or raise ``ValueError``."""

    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform


def load_transform(path: Path | str, key: str = "T_reference_TCP") -> np.ndarray:
    """Load and validate a named transform from JSON.

    For CAD workflows the file may instead contain ``T_reference_CAD`` and
    ``T_CAD_TCP``; both are point-mapping transforms and are composed in that
    order. This keeps the CAD coordinate frame explicit instead of assuming
    that the Onshape world origin is the ArUco reference frame.
    """

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if "units" in data and data["units"] != "m":
        raise ValueError(f"{source}:units must be 'm'")
    if key in data:
        return validate_rigid_transform(data[key], f"{source}:{key}")
    if key == "T_reference_TCP" and {"T_reference_CAD", "T_CAD_TCP"}.issubset(data):
        reference_from_cad = validate_rigid_transform(data["T_reference_CAD"], f"{source}:T_reference_CAD")
        cad_from_tcp = validate_rigid_transform(data["T_CAD_TCP"], f"{source}:T_CAD_TCP")
        return validate_rigid_transform(reference_from_cad @ cad_from_tcp, f"{source}:T_reference_TCP")
    raise ValueError(f"{source} has no {key}")


def transform_points(transform: np.ndarray, points: Iterable[Sequence[float]]) -> np.ndarray:
    """Map ``N`` object-frame points through a homogeneous transform."""

    matrix = validate_rigid_transform(transform)
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    return (matrix[:3, :3] @ array.T).T + matrix[:3, 3]


def project_realsense_points(intrinsics, camera_points: Iterable[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points with D455 intrinsics.

    Returns ``(pixels, valid)``.  Invalid points include non-finite values and
    points on or behind the optical center; their pixel rows are ``NaN``.  The
    validity mask lets callers draw segmented trajectories without joining a
    visible point to a point behind the camera.
    """

    points = np.asarray(camera_points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("camera_points must have shape (N, 3)")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        pixels[valid] = project_realsense(intrinsics, points[valid])
    return pixels, valid


def draw_realsense_polyline(
    image: np.ndarray,
    intrinsics,
    camera_points: Iterable[Sequence[float]],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> int:
    """Draw visible contiguous segments of a camera-frame trajectory.

    The return value is the number of line segments drawn.  A gap is kept when
    either endpoint is behind the camera, so a temporary tracking loss cannot
    create a misleading line across the RGB image.
    """

    pixels, valid = project_realsense_points(intrinsics, camera_points)
    drawn = 0
    for index in range(1, len(pixels)):
        if not (valid[index - 1] and valid[index]):
            continue
        start = tuple(np.round(pixels[index - 1]).astype(int))
        end = tuple(np.round(pixels[index]).astype(int))
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
        drawn += 1
    return drawn


def draw_realsense_frame(
    image: np.ndarray,
    intrinsics,
    camera_from_frame: np.ndarray,
    axis_length_m: float = 0.05,
    thickness: int = 2,
) -> bool:
    """Draw a frame's XYZ axes on an RGB image using RealSense projection."""

    if axis_length_m <= 0:
        raise ValueError("axis_length_m must be positive")
    transform = validate_rigid_transform(camera_from_frame, "camera_from_frame")
    points = transform_points(
        transform,
        [[0.0, 0.0, 0.0], [axis_length_m, 0.0, 0.0],
         [0.0, axis_length_m, 0.0], [0.0, 0.0, axis_length_m]],
    )
    pixels, valid = project_realsense_points(intrinsics, points)
    if not np.all(valid):
        return False
    origin = tuple(np.round(pixels[0]).astype(int))
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # X/Y/Z in BGR
    for endpoint, color in zip(pixels[1:], colors):
        cv2.line(image, origin, tuple(np.round(endpoint).astype(int)), color, thickness, cv2.LINE_AA)
    return True


__all__ = [
    "draw_realsense_frame",
    "draw_realsense_polyline",
    "load_transform",
    "project_realsense_points",
    "transform_points",
    "validate_rigid_transform",
]

"""Legacy corner-based depth pose checks, retained for API compatibility.

New D455 applications should use ``realsense_depth`` and ``rgbd``: RealSense
alignment alone does not generally convert the native depth-camera Z values
to the color-camera optical Z expected by this deprojection helper.

Depth is treated as a second measurement of the calibrated marker geometry.
Invalid samples (which are common on black ArUco squares) are discarded and
the RGB PnP result remains the source of truth when the two measurements do
not agree.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def sample_depth_m(
    depth_image: np.ndarray,
    pixel: Sequence[float],
    depth_scale_m: float,
    radius_px: int = 2,
    min_samples: int = 3,
) -> float:
    """Return a robust local depth sample in metres, or ``nan``.

    The image must already be aligned to the RGB stream. A small median patch
    is less sensitive to the D455's zero-depth holes than one pixel.
    """

    if depth_scale_m <= 0 or radius_px < 0:
        raise ValueError("depth_scale_m must be positive and radius_px non-negative")
    image = np.asarray(depth_image)
    if image.ndim != 2:
        raise ValueError("depth_image must be a two-dimensional array")
    x, y = np.round(np.asarray(pixel, dtype=float).reshape(2)).astype(int)
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return float("nan")
    x0, x1 = max(0, x - radius_px), min(image.shape[1], x + radius_px + 1)
    y0, y1 = max(0, y - radius_px), min(image.shape[0], y + radius_px + 1)
    samples = image[y0:y1, x0:x1]
    samples = samples[np.isfinite(samples) & (samples > 0)]
    if samples.size < min_samples:
        return float("nan")
    return float(np.median(samples) * depth_scale_m)


def _deproject(intrinsics, pixel: Sequence[float], depth_m: float) -> np.ndarray:
    try:
        import pyrealsense2 as rs
    except ImportError as error:  # pragma: no cover - only live hardware path
        raise RuntimeError("depth fusion requires pyrealsense2") from error
    return np.asarray(
        rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel[0]), float(pixel[1])], depth_m),
        dtype=np.float64,
    )


def _rigid_fit(object_points: np.ndarray, camera_points: np.ndarray) -> np.ndarray:
    """Fit ``camera = R*object + t`` with the Kabsch algorithm."""

    object_center = object_points.mean(axis=0)
    camera_center = camera_points.mean(axis=0)
    covariance = (object_points - object_center).T @ (camera_points - camera_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = camera_center - rotation @ object_center
    return result


def estimate_depth_pose(
    object_points: Iterable[Sequence[float]],
    pixels: Iterable[Sequence[float]],
    depth_image: np.ndarray,
    depth_scale_m: float,
    intrinsics,
    *,
    radius_px: int = 2,
    min_samples: int = 3,
    min_points: int = 6,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Estimate a pose from RGB marker correspondences and aligned depth.

    The returned report records how many valid depth samples were used. A
    planar marker can be fit with three points, but the default requires six
    corners so one invalid or background sample cannot determine the result.
    """

    model = np.asarray(list(object_points), dtype=np.float64).reshape(-1, 3)
    image = np.asarray(list(pixels), dtype=np.float64).reshape(-1, 2)
    if len(model) != len(image):
        raise ValueError("object_points and pixels must have the same length")
    camera = []
    model_used = []
    for point, pixel in zip(model, image):
        depth_m = sample_depth_m(depth_image, pixel, depth_scale_m, radius_px, min_samples)
        if not math.isfinite(depth_m):
            continue
        deprojected = _deproject(intrinsics, pixel, depth_m)
        if np.isfinite(deprojected).all() and deprojected[2] > 0:
            model_used.append(point)
            camera.append(deprojected)
    report = {
        "valid_depth_points": len(camera),
        "requested_points": len(model),
        "status": "INSUFFICIENT_DEPTH",
        "translation_delta_m": None,
        "rotation_delta_deg": None,
    }
    if len(camera) < max(3, min_points):
        return None, report
    model_used = np.asarray(model_used, dtype=np.float64)
    camera = np.asarray(camera, dtype=np.float64)
    if np.linalg.matrix_rank(model_used - model_used.mean(axis=0)) < 2:
        report["status"] = "DEGENERATE_DEPTH_GEOMETRY"
        return None, report
    report["status"] = "DEPTH_POSE"
    return _rigid_fit(model_used, camera), report


def fuse_rgb_depth_pose(
    rgb_pose: np.ndarray,
    depth_pose: np.ndarray | None,
    report: dict[str, object],
    *,
    max_translation_delta_m: float = 0.05,
    max_rotation_delta_deg: float = 10.0,
    depth_weight: float = 0.2,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fuse a compatible depth pose into RGB PnP with a low default weight."""

    rgb = np.asarray(rgb_pose, dtype=np.float64)
    if rgb.shape != (4, 4):
        raise ValueError("rgb_pose must be 4x4")
    if depth_pose is None:
        report["status"] = str(report.get("status", "INSUFFICIENT_DEPTH"))
        return rgb, report
    if not 0.0 <= depth_weight <= 1.0:
        raise ValueError("depth_weight must be in [0, 1]")
    depth = np.asarray(depth_pose, dtype=np.float64)
    translation_delta = float(np.linalg.norm(depth[:3, 3] - rgb[:3, 3]))
    rotation_delta = float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(rgb[:3, :3].T @ depth[:3, :3]).as_rotvec()
    )))
    report["translation_delta_m"] = translation_delta
    report["rotation_delta_deg"] = rotation_delta
    if translation_delta > max_translation_delta_m or rotation_delta > max_rotation_delta_deg:
        report["status"] = "DEPTH_REJECTED_RGB_DISAGREEMENT"
        return rgb, report
    weight = float(depth_weight)
    fused = rgb.copy()
    fused[:3, 3] = (1.0 - weight) * rgb[:3, 3] + weight * depth[:3, 3]
    delta_rotation = Rotation.from_matrix(rgb[:3, :3].T @ depth[:3, :3])
    fused[:3, :3] = rgb[:3, :3] @ Rotation.from_rotvec(weight * delta_rotation.as_rotvec()).as_matrix()
    report["status"] = "RGB_DEPTH_FUSED"
    return fused, report


__all__ = ["estimate_depth_pose", "fuse_rgb_depth_pose", "sample_depth_m"]

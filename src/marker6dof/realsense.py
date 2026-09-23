"""Shared OpenCV/RealSense primitives used by capture and live tracking.

The RealSense dependency is loaded lazily so offline localization remains
usable on machines that do not have the optional D455 extra installed.
"""

from __future__ import annotations

import cv2
import numpy as np


def _realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError(
            "D455 tracking requires the optional 'd455' dependencies; "
            "install with: pip install -e '.[d455]'"
        ) from error
    return rs


def detector_parameters():
    """Create the project-wide ArUco detector configuration."""
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def detect_markers(gray, dictionary, parameters):
    """Detect ArUco markers across OpenCV 4.x API variants."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def marker_object_points(marker_size_m: float) -> np.ndarray:
    """Return ArUco corner coordinates in the marker frame (metres)."""
    if marker_size_m <= 0:
        raise ValueError("marker_size_m must be positive")
    half = float(marker_size_m) / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )


def normalized_image_points(intrinsics, corners: np.ndarray) -> np.ndarray:
    """Convert pixels to normalized rays using the D455 distortion model."""
    rs = _realsense()
    points = []
    for pixel in np.asarray(corners).reshape(4, 2):
        ray = rs.rs2_deproject_pixel_to_point(intrinsics, pixel.tolist(), 1.0)
        if abs(ray[2]) < 1e-12:
            raise ValueError("RealSense returned a zero-depth ray")
        points.append([ray[0] / ray[2], ray[1] / ray[2]])
    return np.asarray(points, dtype=np.float32)


def project_realsense(intrinsics, camera_points: np.ndarray) -> np.ndarray:
    """Project camera-frame points using the D455's reported intrinsics."""
    rs = _realsense()
    return np.asarray(
        [rs.rs2_project_point_to_pixel(intrinsics, point.tolist())
         for point in np.asarray(camera_points)],
        dtype=np.float32,
    )


__all__ = [
    "detect_markers",
    "detector_parameters",
    "marker_object_points",
    "normalized_image_points",
    "project_realsense",
]

"""Shared OpenCV/RealSense primitives used by capture and live tracking.

The RealSense dependency is loaded lazily so offline localization remains
usable on machines that do not have the optional D455 extra installed.
"""

from __future__ import annotations

import cv2
import numpy as np


# OpenCV 4.5 exposes ``drawMarker`` while newer contrib releases expose
# ``generateImageMarker``. Keep the small helper available to callers and
# tests across both APIs.
if hasattr(cv2, "aruco") and not hasattr(cv2.aruco, "generateImageMarker") and hasattr(cv2.aruco, "drawMarker"):
    def _generate_image_marker(dictionary, marker_id, side_pixels, borderBits=1):
        image = np.zeros((int(side_pixels), int(side_pixels)), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, int(marker_id), int(side_pixels), image, int(borderBits))
        return image

    cv2.aruco.generateImageMarker = _generate_image_marker


DEFAULT_MARKER_SIZE_MM = 47.5
ARUCO_DICTIONARIES = tuple(
    f"DICT_{bits}X{bits}_{count}"
    for bits in (4, 5, 6, 7) for count in (50, 100, 250, 1000)
)


def _realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError(
            "D455 tracking requires the optional 'd455' dependencies; "
            "install with: pip install -e '.[d455]'"
        ) from error
    return rs


def intrinsics_from_metadata(data: dict):
    """Rebuild a RealSense intrinsics object from recorder metadata."""
    rs = _realsense()
    intrinsics = rs.intrinsics()
    intrinsics.width = int(data["width"])
    intrinsics.height = int(data["height"])
    intrinsics.fx = float(data["fx"])
    intrinsics.fy = float(data["fy"])
    intrinsics.ppx = float(data["cx"])
    intrinsics.ppy = float(data["cy"])
    model_name = str(data["distortion_model"]).rsplit(".", 1)[-1]
    try:
        intrinsics.model = getattr(rs.distortion, model_name)
    except AttributeError as error:
        raise ValueError(f"Unsupported RealSense distortion model: {model_name}") from error
    intrinsics.coeffs = [float(value) for value in data["coefficients"]]
    return intrinsics


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
    """Convert pixels to normalized rays using the supplied camera model."""
    from .camera import OpenCVCamera
    if isinstance(intrinsics, OpenCVCamera):
        return intrinsics.normalized_points(corners)
    rs = _realsense()
    points = []
    for pixel in np.asarray(corners).reshape(4, 2):
        ray = rs.rs2_deproject_pixel_to_point(intrinsics, pixel.tolist(), 1.0)
        if abs(ray[2]) < 1e-12:
            raise ValueError("RealSense returned a zero-depth ray")
        points.append([ray[0] / ray[2], ray[1] / ray[2]])
    return np.asarray(points, dtype=np.float32)


def project_realsense(intrinsics, camera_points: np.ndarray) -> np.ndarray:
    """Project camera-frame points with SDK or calibrated OpenCV intrinsics."""
    from .camera import OpenCVCamera
    if isinstance(intrinsics, OpenCVCamera):
        return intrinsics.project_points(camera_points)
    rs = _realsense()
    return np.asarray(
        [rs.rs2_project_point_to_pixel(intrinsics, point.tolist())
         for point in np.asarray(camera_points)],
        dtype=np.float32,
    )


__all__ = [
    "DEFAULT_MARKER_SIZE_MM",
    "detect_markers",
    "detector_parameters",
    "intrinsics_from_metadata",
    "marker_object_points",
    "normalized_image_points",
    "project_realsense",
]

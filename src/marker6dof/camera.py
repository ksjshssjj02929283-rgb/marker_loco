"""SDK-independent Kalibr pinhole cameras, including equidistant lenses."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class OpenCVCamera:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray
    distortion_model: str
    topic: str = ""

    def normalized_points(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if self.distortion_model == "equidistant":
            points = cv2.fisheye.undistortPoints(pixels, self.matrix, self.distortion)
        else:
            points = cv2.undistortPoints(pixels, self.matrix, self.distortion)
        if not np.isfinite(points).all() or np.max(np.abs(points), initial=0) > 1e5:
            raise ValueError("camera unprojection failed")
        return points.reshape(-1, 2)

    def project_points(self, camera_points):
        points = np.asarray(camera_points, dtype=np.float64).reshape(-1, 1, 3)
        if self.distortion_model == "equidistant":
            pixels, _ = cv2.fisheye.projectPoints(
                points, np.zeros(3), np.zeros(3), self.matrix, self.distortion)
        else:
            pixels, _ = cv2.projectPoints(
                points, np.zeros(3), np.zeros(3), self.matrix, self.distortion)
        return pixels.reshape(-1, 2)


def load_kalibr_camera(path: Path | str, camera_key="cam0") -> OpenCVCamera:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Kalibr YAML requires pip install -e '.[rosbag]'") from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))[camera_key]
    if data["camera_model"] != "pinhole":
        raise ValueError("only Kalibr pinhole cameras are supported")
    model = data["distortion_model"]
    if model not in {"equidistant", "radtan", "none"}:
        raise ValueError(f"unsupported Kalibr distortion model: {model}")
    fx, fy, cx, cy = np.asarray(data["intrinsics"], dtype=float)
    width, height = map(int, data["resolution"])
    coeffs = np.asarray(data.get("distortion_coeffs", []), dtype=float)
    if (width <= 0 or height <= 0 or fx <= 0 or fy <= 0
            or not np.isfinite([fx, fy, cx, cy]).all()
            or not 0 <= cx < width or not 0 <= cy < height
            or not np.isfinite(coeffs).all()):
        raise ValueError("invalid camera intrinsics/resolution")
    if model == "equidistant" and coeffs.size != 4:
        raise ValueError("equidistant needs four distortion coefficients")
    if model == "radtan" and coeffs.size not in {4, 5, 8}:
        raise ValueError("radtan needs four, five or eight distortion coefficients")
    if model == "none":
        coeffs = np.zeros(4)
    return OpenCVCamera(width, height, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]),
                        coeffs, model, data.get("rostopic", ""))

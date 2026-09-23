"""Calibrated ArUco detection and rigid multi-marker TCP pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


def _array(value, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result


def _transform(value, name: str) -> np.ndarray:
    result = _array(value, (4, 4), name)
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return result


@dataclass(frozen=True)
class QualityConfig:
    min_markers: int = 2
    min_inlier_corners: int = 6
    ransac_threshold_px: float = 3.0
    ransac_confidence: float = 0.999
    ransac_iterations: int = 100
    max_reprojection_rmse_px: float = 2.5
    max_reprojection_error_px: float = 6.0


@dataclass(frozen=True)
class BodyConfig:
    name: str
    tag_corners_gripper_m: Mapping[int, np.ndarray]
    T_gripper_tcp: np.ndarray


@dataclass(frozen=True)
class Calibration:
    image_size: Tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    dictionary_name: str
    bodies: Mapping[str, BodyConfig]
    quality: QualityConfig


@dataclass(frozen=True)
class MarkerDetection:
    tag_id: int
    corners_px: np.ndarray


@dataclass
class PoseEstimate:
    body: str
    status: str
    reason: str
    visible_ids: List[int]
    used_ids: List[int]
    corner_count: int = 0
    inlier_count: int = 0
    reprojection_rmse_px: Optional[float] = None
    max_reprojection_error_px: Optional[float] = None
    T_camera_gripper: Optional[np.ndarray] = None
    T_camera_tcp: Optional[np.ndarray] = None


def load_calibration(path: Path | str) -> Calibration:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("calibration schema_version must be 1")
    if raw.get("units") != "m":
        raise ValueError("calibration units must be 'm'")

    width = int(raw["image_width"])
    height = int(raw["image_height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    camera = _array(raw["camera_matrix"], (3, 3), "camera_matrix")
    if camera[0, 0] <= 0 or camera[1, 1] <= 0 or not np.allclose(camera[2], [0, 0, 1]):
        raise ValueError("camera_matrix must contain positive focal lengths and K[2]=[0,0,1]")
    if not (0 <= camera[0, 2] < width and 0 <= camera[1, 2] < height):
        raise ValueError("camera principal point must lie inside the configured image")
    distortion = np.asarray(raw["distortion"], dtype=np.float64).reshape(-1, 1)
    if distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
        raise ValueError("distortion must contain 4, 5, 8, 12, or 14 finite coefficients")

    aruco = raw.get("aruco", {})
    dictionary_name = str(aruco.get("dictionary", ""))
    if not dictionary_name.startswith("DICT_") or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unsupported OpenCV ArUco dictionary: {dictionary_name!r}")

    bodies: Dict[str, BodyConfig] = {}
    owner_by_tag: Dict[int, str] = {}
    for name, body_raw in raw.get("bodies", {}).items():
        tags: Dict[int, np.ndarray] = {}
        for raw_id, corners_raw in body_raw.get("tag_corners_gripper_m", {}).items():
            tag_id = int(raw_id)
            if tag_id in owner_by_tag:
                raise ValueError(
                    f"tag {tag_id} belongs to both {owner_by_tag[tag_id]!r} and {name!r}"
                )
            corners = _array(corners_raw, (4, 3), f"{name}.tag[{tag_id}]")
            if np.linalg.matrix_rank(corners - corners.mean(axis=0)) != 2:
                raise ValueError(f"{name}.tag[{tag_id}] corners must define one non-degenerate plane")
            edge_lengths = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
            if np.min(edge_lengths) <= 1e-5:
                raise ValueError(f"{name}.tag[{tag_id}] has a degenerate edge")
            tags[tag_id] = corners
            owner_by_tag[tag_id] = str(name)
        if not tags:
            raise ValueError(f"body {name!r} must define at least one tag")
        bodies[str(name)] = BodyConfig(
            name=str(name),
            tag_corners_gripper_m=tags,
            T_gripper_tcp=_transform(body_raw["T_gripper_tcp"], f"{name}.T_gripper_tcp"),
        )
    if not bodies:
        raise ValueError("calibration must define at least one body")

    quality_raw = raw.get("quality", {})
    quality = QualityConfig(
        min_markers=int(quality_raw.get("min_markers", 2)),
        min_inlier_corners=int(quality_raw.get("min_inlier_corners", 6)),
        ransac_threshold_px=float(quality_raw.get("ransac_threshold_px", 3.0)),
        ransac_confidence=float(quality_raw.get("ransac_confidence", 0.999)),
        ransac_iterations=int(quality_raw.get("ransac_iterations", 100)),
        max_reprojection_rmse_px=float(quality_raw.get("max_reprojection_rmse_px", 2.5)),
        max_reprojection_error_px=float(quality_raw.get("max_reprojection_error_px", 6.0)),
    )
    if quality.min_markers < 1 or quality.min_inlier_corners < 4:
        raise ValueError("quality min_markers must be >=1 and min_inlier_corners must be >=4")
    if not 0 < quality.ransac_confidence < 1:
        raise ValueError("quality ransac_confidence must be in (0,1)")

    return Calibration(
        image_size=(width, height),
        camera_matrix=camera,
        distortion=distortion,
        dictionary_name=dictionary_name,
        bodies=bodies,
        quality=quality,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a valid 3x3 rotation matrix to a normalized xyzw quaternion."""
    rotation = _array(rotation, (3, 3), "rotation")
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-15))
            x, y, z, w = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-15))
            x, y, z, w = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = 2.0 * np.sqrt(max(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-15))
            x, y, z, w = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    result[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return result


class MarkerLocalizer:
    """Detect configured markers and estimate each rigid body's camera-frame TCP pose."""

    def __init__(self, calibration: Calibration):
        self.calibration = calibration
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, calibration.dictionary_name)
        )
        self.detector = (
            cv2.aruco.ArucoDetector(dictionary, parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else (dictionary, parameters)
        )

    def detect(self, image: np.ndarray) -> Tuple[List[MarkerDetection], int]:
        if image is None:
            raise ValueError("image must not be None")
        height, width = image.shape[:2]
        if (width, height) != self.calibration.image_size:
            raise ValueError(
                f"image size {(width, height)} does not match calibration {self.calibration.image_size}"
            )
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            dictionary, parameters = self.detector
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters
            )
        if ids is None:
            return [], len(rejected)
        detections = [
            MarkerDetection(int(tag_id), corner.reshape(4, 2).astype(np.float64))
            for tag_id, corner in zip(ids.ravel(), corners)
        ]
        return detections, len(rejected)

    def process(self, image: np.ndarray) -> Tuple[List[MarkerDetection], List[PoseEstimate], int]:
        detections, rejected_count = self.detect(image)
        estimates = [self.estimate_body(detections, body) for body in self.calibration.bodies.values()]
        return detections, estimates, rejected_count

    def estimate_body(
        self,
        detections: Iterable[MarkerDetection],
        body: BodyConfig,
    ) -> PoseEstimate:
        detections = list(detections)
        visible_ids = sorted({item.tag_id for item in detections})
        grouped: Dict[int, List[MarkerDetection]] = {}
        for item in detections:
            grouped.setdefault(item.tag_id, []).append(item)

        usable = {
            tag_id: items[0]
            for tag_id, items in grouped.items()
            if tag_id in body.tag_corners_gripper_m and len(items) == 1
        }
        used_ids = sorted(usable)
        quality = self.calibration.quality
        if len(used_ids) < quality.min_markers:
            duplicate_ids = sorted(tag_id for tag_id, items in grouped.items() if len(items) > 1)
            reason = "duplicate_marker_id" if duplicate_ids else "insufficient_registered_markers"
            return PoseEstimate(body.name, "LOST", reason, visible_ids, used_ids)

        object_points = np.concatenate(
            [body.tag_corners_gripper_m[tag_id] for tag_id in used_ids], axis=0
        ).astype(np.float64)
        image_points = np.concatenate(
            [usable[tag_id].corners_px for tag_id in used_ids], axis=0
        ).astype(np.float64)

        solution = self._solve_pose(object_points, image_points)
        if solution is None:
            return PoseEstimate(
                body.name,
                "LOST",
                "pnp_failed",
                visible_ids,
                used_ids,
                corner_count=len(object_points),
            )
        rvec, tvec, inliers = solution
        inlier_count = int(len(inliers))
        if inlier_count < quality.min_inlier_corners:
            return PoseEstimate(
                body.name,
                "LOST",
                "too_few_inlier_corners",
                visible_ids,
                used_ids,
                corner_count=len(object_points),
                inlier_count=inlier_count,
            )

        rotation = cv2.Rodrigues(rvec)[0]
        camera_points = object_points @ rotation.T + tvec.reshape(1, 3)
        if np.any(camera_points[:, 2] <= 0):
            return PoseEstimate(
                body.name,
                "LOST",
                "non_positive_depth",
                visible_ids,
                used_ids,
                corner_count=len(object_points),
                inlier_count=inlier_count,
            )

        projected = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.calibration.camera_matrix,
            self.calibration.distortion,
        )[0].reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        max_error = float(np.max(errors))
        if rmse > quality.max_reprojection_rmse_px:
            reason = "reprojection_rmse_rejected"
            status = "LOST"
        elif max_error > quality.max_reprojection_error_px:
            reason = "reprojection_max_rejected"
            status = "LOST"
        else:
            reason = "ok"
            status = "MEASURED"

        T_camera_gripper = make_transform(rvec, tvec)
        return PoseEstimate(
            body=body.name,
            status=status,
            reason=reason,
            visible_ids=visible_ids,
            used_ids=used_ids,
            corner_count=len(object_points),
            inlier_count=inlier_count,
            reprojection_rmse_px=rmse,
            max_reprojection_error_px=max_error,
            T_camera_gripper=T_camera_gripper if status == "MEASURED" else None,
            T_camera_tcp=T_camera_gripper @ body.T_gripper_tcp if status == "MEASURED" else None,
        )

    def _solve_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        camera = self.calibration.camera_matrix
        distortion = self.calibration.distortion
        quality = self.calibration.quality

        if len(object_points) == 4:
            result = cv2.solvePnPGeneric(
                object_points,
                image_points,
                camera,
                distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
            candidates = []
            for rvec, tvec in zip(result[1], result[2]):
                rotation = cv2.Rodrigues(rvec)[0]
                camera_points = object_points @ rotation.T + tvec.reshape(1, 3)
                if np.all(camera_points[:, 2] > 0):
                    projected = cv2.projectPoints(
                        object_points, rvec, tvec, camera, distortion
                    )[0].reshape(-1, 2)
                    error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
                    candidates.append((error, rvec, tvec))
            if not candidates:
                return None
            _, rvec, tvec = min(candidates, key=lambda item: item[0])
            return rvec, tvec, np.arange(4, dtype=np.int32).reshape(-1, 1)

        solved, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=quality.ransac_threshold_px,
            confidence=quality.ransac_confidence,
            iterationsCount=quality.ransac_iterations,
        )
        if not solved or inliers is None or len(inliers) < 4:
            return None
        indices = inliers[:, 0]
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[indices],
                image_points[indices],
                camera,
                distortion,
                rvec,
                tvec,
            )
        else:
            solved, rvec, tvec = cv2.solvePnP(
                object_points[indices],
                image_points[indices],
                camera,
                distortion,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not solved:
                return None
        return rvec, tvec, inliers


def estimate_to_row(estimate: PoseEstimate) -> Dict[str, object]:
    row: Dict[str, object] = {
        "body": estimate.body,
        "status": estimate.status,
        "reason": estimate.reason,
        "visible_ids": ";".join(map(str, estimate.visible_ids)),
        "used_ids": ";".join(map(str, estimate.used_ids)),
        "corner_count": estimate.corner_count,
        "inlier_count": estimate.inlier_count,
        "reprojection_rmse_px": "" if estimate.reprojection_rmse_px is None else estimate.reprojection_rmse_px,
        "max_reprojection_error_px": "" if estimate.max_reprojection_error_px is None else estimate.max_reprojection_error_px,
        "tx_m": "",
        "ty_m": "",
        "tz_m": "",
        "qx": "",
        "qy": "",
        "qz": "",
        "qw": "",
    }
    if estimate.T_camera_tcp is not None:
        quaternion = matrix_to_quaternion_xyzw(estimate.T_camera_tcp[:3, :3])
        row.update(
            tx_m=float(estimate.T_camera_tcp[0, 3]),
            ty_m=float(estimate.T_camera_tcp[1, 3]),
            tz_m=float(estimate.T_camera_tcp[2, 3]),
            qx=float(quaternion[0]),
            qy=float(quaternion[1]),
            qz=float(quaternion[2]),
            qw=float(quaternion[3]),
        )
    return row


def configured_tag_ids(calibration: Calibration) -> Sequence[int]:
    return sorted(
        tag_id
        for body in calibration.bodies.values()
        for tag_id in body.tag_corners_gripper_m
    )

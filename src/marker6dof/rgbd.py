"""Conservative RGB-corner / depth-plane pose refinement, without camera SDKs.

All 3-D inputs use metres. Poses map object coordinates to the color optical
frame (``T_color_object``); depth points must already be transformed to that
same frame. ``depth_pixels`` are those points projected into the color image,
not pixel coordinates in the native depth image. The projection callback must
use the color camera's actual intrinsics and distortion model.

A single plane supplies normal and normal-translation constraints only. RGB
corners remain in the objective at every nonzero depth weight, including 1.0,
to constrain the other degrees of freedom. This module does not claim a
particular sensor accuracy or estimate a calibrated pose covariance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .projection import validate_rigid_transform


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _points(value: np.ndarray, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (0,):
        return np.empty((0, columns), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    return array


def _inside_convex(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized inclusive convex-polygon test, independent of winding."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(cross >= -1e-8, axis=1) | np.all(cross <= 1e-8, axis=1)


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - center, full_matrices=False)
    return center, vt[-1], singular


def _ransac_plane(
    points: np.ndarray, threshold: float, iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Deterministic RANSAC followed by two SVD/inlier-set refinements."""
    random = np.random.default_rng(0)
    best = None
    best_score = (-1, -np.inf)
    for _ in range(iterations):
        sample = points[random.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            continue
        distances = np.abs((points - sample[0]) @ (normal / length))
        inliers = distances <= threshold
        score = (int(inliers.sum()), -float(distances[inliers].sum()))
        if score > best_score:
            best, best_score = inliers, score
    if best is None or int(best.sum()) < 3:
        return None
    for _ in range(2):
        center, normal, _ = _fit_plane(points[best])
        best = np.abs((points - center) @ normal) <= threshold
        if int(best.sum()) < 3:
            return None
    center, normal, singular = _fit_plane(points[best])
    return center, normal, singular, best


def refine_pose_rgbd(
    rgb_pose: np.ndarray,
    object_corners: Mapping[int, np.ndarray],
    image_corners: Mapping[int, np.ndarray],
    camera_points: np.ndarray,
    depth_pixels: np.ndarray,
    project_points: Callable[[np.ndarray], np.ndarray],
    *,
    depth_weight: float = 0.2,
    max_translation_delta_m: float = 0.05,
    max_rotation_delta_deg: float = 10.0,
    max_reprojection_error_px: float = 2.5,
    max_reprojection_increase_px: float = 0.5,
    max_reprojection_ratio: float = 1.25,
    pixel_sigma_px: float = 1.0,
    depth_sigma_m: float = 0.003,
    roi_inset_fraction: float = 0.15,
    max_points_per_face: int = 200,
    min_depth_points: int = 20,
    min_plane_inlier_ratio: float = 0.65,
    plane_inlier_threshold_m: float = 0.003,
    max_plane_rms_m: float = 0.0025,
    max_plane_distance_m: float = 0.03,
    max_plane_normal_delta_deg: float = 20.0,
    min_coverage_ratio: float = 0.2,
    ransac_iterations: int = 80,
    max_nfev: int = 60,
) -> tuple[np.ndarray, dict[str, object]]:
    """Refine a PnP pose using robust, interior marker-plane depth samples.

    ``depth_weight`` is the relative squared-residual weight of depth, in
    [0, 1]; it is not an interpolation fraction. RGB always retains weight 1.
    Residuals are divided by their sensor noise scales and normalized by the
    number of corners / planes / samples, so dense faces cannot dominate just
    by supplying more pixels. Each marker is treated as one planar face.

    Before fitting, a centroid-shrunken color polygon excludes marker edges,
    and the RGB pose predicts a ray/plane intersection to reject background.
    Supported planes must pass RANSAC, residual, normal, and coverage checks.
    The final pose must preserve every marker's RGB reprojection quality:
    each per-marker RMS must be below ``max_reprojection_error_px`` and below
    ``baseline * max_reprojection_ratio + max_reprojection_increase_px``.
    The additive allowance makes the check meaningful near zero RGB error.

    Invalid configuration or malformed inputs raise ``ValueError``. Missing,
    nonfinite, degenerate or incompatible depth, failed optimization, and
    rejected proposals return the RGB pose unchanged. The report is strict
    JSON-serializable; ``fused_reprojection_error_px`` describes the returned
    pose, while candidate fields / deltas describe an attempted proposal.
    No inputs are modified.
    """
    rgb = validate_rigid_transform(rgb_pose, "rgb_pose")
    depth_weight = float(depth_weight)
    if not np.isfinite(depth_weight) or not 0.0 <= depth_weight <= 1.0:
        raise ValueError("depth_weight must be finite and in [0, 1]")
    max_translation_delta_m = _positive(max_translation_delta_m, "max_translation_delta_m")
    max_rotation_delta_deg = _positive(max_rotation_delta_deg, "max_rotation_delta_deg")
    max_reprojection_error_px = _positive(max_reprojection_error_px, "max_reprojection_error_px")
    pixel_sigma_px = _positive(pixel_sigma_px, "pixel_sigma_px")
    depth_sigma_m = _positive(depth_sigma_m, "depth_sigma_m")
    plane_inlier_threshold_m = _positive(plane_inlier_threshold_m, "plane_inlier_threshold_m")
    max_plane_rms_m = _positive(max_plane_rms_m, "max_plane_rms_m")
    max_plane_distance_m = _positive(max_plane_distance_m, "max_plane_distance_m")
    max_plane_normal_delta_deg = _positive(max_plane_normal_delta_deg, "max_plane_normal_delta_deg")
    max_reprojection_increase_px = float(max_reprojection_increase_px)
    max_reprojection_ratio = float(max_reprojection_ratio)
    if not np.isfinite(max_reprojection_increase_px) or max_reprojection_increase_px < 0:
        raise ValueError("max_reprojection_increase_px must be finite and nonnegative")
    if not np.isfinite(max_reprojection_ratio) or max_reprojection_ratio < 1:
        raise ValueError("max_reprojection_ratio must be finite and >= 1")
    roi_inset_fraction = float(roi_inset_fraction)
    if not np.isfinite(roi_inset_fraction) or not 0 <= roi_inset_fraction < 0.5:
        raise ValueError("roi_inset_fraction must be finite and in [0, 0.5)")
    min_plane_inlier_ratio = float(min_plane_inlier_ratio)
    min_coverage_ratio = float(min_coverage_ratio)
    for name, value in (("min_plane_inlier_ratio", min_plane_inlier_ratio), ("min_coverage_ratio", min_coverage_ratio)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    if max_rotation_delta_deg > 180 or max_plane_normal_delta_deg > 90:
        raise ValueError("rotation limit must be <= 180 degrees and plane normal limit <= 90 degrees")
    min_depth_points = _integer(min_depth_points, "min_depth_points", 6)
    max_points_per_face = _integer(max_points_per_face, "max_points_per_face", min_depth_points)
    ransac_iterations = _integer(ransac_iterations, "ransac_iterations", 1)
    max_nfev = _integer(max_nfev, "max_nfev", 1)
    if not callable(project_points):
        raise ValueError("project_points must be callable")
    points = _points(camera_points, 3, "camera_points")
    pixels = _points(depth_pixels, 2, "depth_pixels")
    if len(points) != len(pixels):
        raise ValueError("camera_points and depth_pixels must have the same length")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1) & (points[:, 2] > 0)
    points, pixels = points[finite], pixels[finite]
    report: dict[str, object] = {
        "status": "INSUFFICIENT_DEPTH", "valid_depth_points": 0,
        "input_depth_points": int(len(finite)), "finite_depth_points": int(finite.sum()),
        "translation_delta_m": 0.0, "rotation_delta_deg": 0.0,
        "rgb_reprojection_error_px": None, "fused_reprojection_error_px": None,
        "planes": [],
    }
    tag_ids = sorted(set(object_corners) & set(image_corners))
    if not tag_ids:
        report["status"] = "NO_MATCHING_MARKERS"
        return rgb, report
    models, observations, geometry = [], [], []
    for tag_id in tag_ids:
        model = np.asarray(object_corners[tag_id], dtype=np.float64)
        observed = np.asarray(image_corners[tag_id], dtype=np.float64)
        if model.shape != (4, 3) or not np.isfinite(model).all():
            raise ValueError(f"object_corners[{tag_id}] must be finite with shape (4, 3)")
        if observed.shape != (4, 2) or not np.isfinite(observed).all():
            raise ValueError(f"image_corners[{tag_id}] must be finite with shape (4, 2)")
        center, normal, singular = _fit_plane(model)
        if singular[1] <= 1e-8 or singular[2] > max(1e-8, singular[1] * 1e-5):
            raise ValueError(f"object_corners[{tag_id}] must define a nondegenerate plane")
        models.append(model)
        observations.append(observed)
        geometry.append((center, normal, singular[1]))
    model_all = np.concatenate(models)
    observed_all = np.concatenate(observations)

    def project(camera: np.ndarray) -> np.ndarray:
        projected = np.asarray(project_points(camera), dtype=np.float64)
        if projected.shape != (len(camera), 2):
            raise ValueError("project_points must return an array with shape (N, 2)")
        return projected

    def camera_corners(pose: np.ndarray) -> np.ndarray:
        return model_all @ pose[:3, :3].T + pose[:3, 3]

    original_camera = camera_corners(rgb)
    if np.any(original_camera[:, 2] <= 1e-6):
        report["status"] = "RGB_INVALID_GEOMETRY"
        return rgb, report
    try:
        original_pixels = project(original_camera)
    except (RuntimeError, FloatingPointError):
        report["status"] = "RGB_INVALID_PROJECTION"
        return rgb, report
    if not np.isfinite(original_pixels).all():
        report["status"] = "RGB_INVALID_PROJECTION"
        return rgb, report

    def reprojection_errors(projected: np.ndarray) -> np.ndarray:
        squared = np.sum((projected - observed_all) ** 2, axis=1)
        return np.sqrt(squared.reshape(len(tag_ids), 4).mean(axis=1))

    baseline = reprojection_errors(original_pixels)
    baseline_rms = float(np.sqrt(np.mean(baseline ** 2)))
    report["rgb_reprojection_error_px"] = baseline_rms
    report["fused_reprojection_error_px"] = baseline_rms
    report["rgb_per_tag_reprojection_error_px"] = {str(key): float(error) for key, error in zip(tag_ids, baseline)}
    if depth_weight == 0:
        report["status"] = "RGB_DEPTH_DISABLED"
        return rgb, report
    if not len(points):
        return rgb, report

    constraints = []
    plane_reports = report["planes"]
    disagreement = False
    for tag_id, observed, (center, normal, object_spread) in zip(tag_ids, observations, geometry):
        diagnostic = {"tag_id": int(tag_id), "status": "INSUFFICIENT_ROI_DEPTH", "roi_points": 0,
                      "gated_points": 0, "sampled_points": 0, "inlier_points": 0}
        plane_reports.append(diagnostic)
        if not cv2.isContourConvex(observed.astype(np.float32)):
            diagnostic["status"] = "DEGENERATE_IMAGE_POLYGON"
            continue
        inset = observed.mean(axis=0) + (observed - observed.mean(axis=0)) * (1.0 - 2.0 * roi_inset_fraction)
        roi_area = abs(float(cv2.contourArea(inset.astype(np.float32))))
        if roi_area < 1.0:
            diagnostic["status"] = "DEGENERATE_IMAGE_POLYGON"
            continue
        selected = _inside_convex(pixels, inset)
        diagnostic["roi_points"] = int(selected.sum())
        if int(selected.sum()) < min_depth_points:
            continue
        candidates, candidate_pixels = points[selected], pixels[selected]
        rgb_center = rgb[:3, :3] @ center + rgb[:3, 3]
        rgb_normal = rgb[:3, :3] @ normal
        rays = candidates / candidates[:, 2:3]
        denominator = rays @ rgb_normal
        expected_z = np.divide(float(rgb_center @ rgb_normal), denominator,
                               out=np.full(len(candidates), np.nan), where=np.abs(denominator) > 1e-6)
        compatible = (np.isfinite(expected_z) & (expected_z > 0)
                      & (np.abs(candidates[:, 2] - expected_z) <= max_plane_distance_m)
                      & (np.abs((candidates - rgb_center) @ rgb_normal) <= max_plane_distance_m))
        candidates, candidate_pixels = candidates[compatible], candidate_pixels[compatible]
        diagnostic["gated_points"] = int(len(candidates))
        if len(candidates) < min_depth_points:
            diagnostic["status"] = "DEPTH_REJECTED_RGB_DISAGREEMENT"
            disagreement = True
            continue
        if len(candidates) > max_points_per_face:
            indices = np.linspace(0, len(candidates) - 1, max_points_per_face, dtype=int)
            candidates, candidate_pixels = candidates[indices], candidate_pixels[indices]
        diagnostic["sampled_points"] = int(len(candidates))
        fit = _ransac_plane(candidates, plane_inlier_threshold_m, ransac_iterations)
        if fit is None:
            diagnostic["status"] = "DEGENERATE_DEPTH_GEOMETRY"
            continue
        measured_center, measured_normal, spread, inliers = fit
        support, support_pixels = candidates[inliers], candidate_pixels[inliers]
        ratio = float(inliers.mean())
        rms = float(np.sqrt(np.mean(((support - measured_center) @ measured_normal) ** 2)))
        hull = cv2.convexHull(support_pixels.astype(np.float32))
        coverage = abs(float(cv2.contourArea(hull))) / roi_area
        normal_delta = float(np.degrees(np.arccos(np.clip(abs(measured_normal @ rgb_normal), 0.0, 1.0))))
        offset_delta = float(abs((rgb_center - measured_center) @ measured_normal))
        diagnostic.update(inlier_points=int(len(support)), inlier_ratio=ratio, rms_m=rms,
                          coverage_ratio=coverage, normal_delta_deg=normal_delta, offset_delta_m=offset_delta)
        if len(support) < min_depth_points or ratio < min_plane_inlier_ratio:
            diagnostic["status"] = "INSUFFICIENT_PLANE_SUPPORT"
        elif spread[1] / np.sqrt(len(support)) < object_spread * 0.04 or coverage < min_coverage_ratio:
            diagnostic["status"] = "DEGENERATE_DEPTH_GEOMETRY"
        elif rms > max_plane_rms_m:
            diagnostic["status"] = "PLANE_RESIDUAL_TOO_LARGE"
        elif normal_delta > max_plane_normal_delta_deg or offset_delta > max_plane_distance_m:
            diagnostic["status"] = "DEPTH_REJECTED_RGB_DISAGREEMENT"
            disagreement = True
        else:
            diagnostic["status"] = "PLANE_ACCEPTED"
            constraints.append((center, normal, support))
    report["valid_depth_points"] = int(sum(len(points_used) for _, _, points_used in constraints))
    if not constraints:
        report["status"] = "DEPTH_REJECTED_RGB_DISAGREEMENT" if disagreement else "INSUFFICIENT_DEPTH"
        return rgb, report

    rgb_scale = 1.0 / (pixel_sigma_px * np.sqrt(observed_all.size))
    depth_scales = [np.sqrt(depth_weight / (len(constraints) * len(support))) / depth_sigma_m
                    for _, _, support in constraints]

    def pose_from_delta(delta: np.ndarray) -> np.ndarray:
        pose = rgb.copy()
        pose[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ rgb[:3, :3]
        pose[:3, 3] += delta[3:]
        return pose

    def residual(delta: np.ndarray) -> np.ndarray:
        pose = pose_from_delta(delta)
        camera = camera_corners(pose)
        # Keep SDK projection calls on positive-depth points even during
        # rejected intermediate proposals; the explicit penalty below drives
        # those proposals back in front of the camera.
        safe_camera = camera.copy()
        safe_camera[:, 2] = np.maximum(safe_camera[:, 2], 1e-6)
        projected = project(safe_camera)
        # A failed proposal cannot become an accepted pose. Finite penalties
        # let the optimizer retreat if its intermediate projection is invalid.
        image_residual = np.nan_to_num((projected - observed_all).ravel() * rgb_scale,
                                      nan=1e6, posinf=1e6, neginf=-1e6)
        behind = np.maximum(1e-6 - camera[:, 2], 0.0) / depth_sigma_m
        terms = [image_residual, behind]
        for (center, normal, support), scale in zip(constraints, depth_scales):
            current_normal = pose[:3, :3] @ normal
            current_center = pose[:3, :3] @ center + pose[:3, 3]
            terms.append(((support - current_center) @ current_normal) * scale)
        return np.concatenate(terms)

    # Some SDK projectors calculate in float32. scipy's default ~1e-8 step
    # can then produce zero image derivatives, even for float64 output arrays.
    # Absolute central steps give useful changes in both radians and metres.
    derivative_steps = np.asarray([1e-4] * 3 + [1e-5] * 3)

    def jacobian(delta: np.ndarray) -> np.ndarray:
        columns = []
        for index, step in enumerate(derivative_steps):
            offset = np.zeros(6)
            offset[index] = step
            columns.append((residual(delta + offset) - residual(delta - offset)) / (2.0 * step))
        return np.column_stack(columns)

    bounds = np.asarray([np.radians(max_rotation_delta_deg)] * 3 + [max_translation_delta_m] * 3)
    try:
        result = least_squares(residual, np.zeros(6), jac=jacobian, bounds=(-bounds, bounds),
                               x_scale=np.asarray([0.05] * 3 + [0.01] * 3),
                               loss="soft_l1", f_scale=1.0, max_nfev=max_nfev)
        candidate = pose_from_delta(result.x)
        validate_rigid_transform(candidate, "refined_pose")
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        report["status"] = "DEPTH_REJECTED_OPTIMIZATION"
        return rgb, report
    report["optimizer_success"] = bool(result.success)
    report["optimizer_nfev"] = int(result.nfev)
    report["translation_delta_m"] = float(np.linalg.norm(candidate[:3, 3] - rgb[:3, 3]))
    report["rotation_delta_deg"] = float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(candidate[:3, :3] @ rgb[:3, :3].T).as_rotvec())))
    if not result.success or not np.isfinite(result.fun).all():
        report["status"] = "DEPTH_REJECTED_OPTIMIZATION"
        return rgb, report
    if (report["translation_delta_m"] > max_translation_delta_m
            or report["rotation_delta_deg"] > max_rotation_delta_deg):
        report["status"] = "DEPTH_REJECTED_RGB_DISAGREEMENT"
        return rgb, report
    candidate_camera = camera_corners(candidate)
    if np.any(candidate_camera[:, 2] <= 1e-6):
        report["status"] = "DEPTH_REJECTED_GEOMETRY"
        return rgb, report
    try:
        candidate_pixels = project(candidate_camera)
    except (RuntimeError, FloatingPointError):
        report["status"] = "DEPTH_REJECTED_PROJECTION"
        return rgb, report
    if not np.isfinite(candidate_pixels).all():
        report["status"] = "DEPTH_REJECTED_PROJECTION"
        return rgb, report
    candidate_errors = reprojection_errors(candidate_pixels)
    candidate_rms = float(np.sqrt(np.mean(candidate_errors ** 2)))
    report["candidate_reprojection_error_px"] = candidate_rms
    report["candidate_per_tag_reprojection_error_px"] = {
        str(key): float(error) for key, error in zip(tag_ids, candidate_errors)}
    if (np.any(candidate_errors > max_reprojection_error_px)
            or np.any(candidate_errors > baseline * max_reprojection_ratio + max_reprojection_increase_px)):
        report["status"] = "DEPTH_REJECTED_REPROJECTION"
        return rgb, report
    report["status"] = "RGB_DEPTH_REFINED"
    report["fused_reprojection_error_px"] = candidate_rms
    return candidate, report


__all__ = ["refine_pose_rgbd"]

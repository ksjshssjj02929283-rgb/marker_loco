"""Native-depth point clouds expressed in the RGB optical coordinate frame.

``rs.align(color)`` resamples raw depth values onto color pixels; it does not
generally turn depth-camera Z into color-camera Z. Use native depth vertices
and the actual depth-to-color extrinsics for metric geometry instead.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .projection import validate_rigid_transform
from .realsense import _realsense


def depth_points_in_color(
    depth_vertices: np.ndarray,
    texture_uv: np.ndarray,
    color_from_depth: np.ndarray,
    width: int,
    height: int,
    regions: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter native SDK vertices/texture coordinates and apply ``T_color_depth``.

    Vertices are metres in the depth optical frame. Texture coordinates are
    the SDK's normalized RGB pixel coordinates (u/width, v/height), generated
    with ``pointcloud.map_to(color_frame)``. ROI filtering happens before the
    transform to avoid processing an entire megapixel cloud in Python. At
    each rounded RGB pixel only the nearest surface is retained.
    """
    points = np.asarray(depth_vertices)
    uv = np.asarray(texture_uv)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("depth_vertices must have shape (N, 3)")
    if uv.shape != (len(points), 2):
        raise ValueError("texture_uv must have shape (N, 2)")
    if width <= 0 or height <= 0:
        raise ValueError("RGB image dimensions must be positive")
    transform = validate_rigid_transform(color_from_depth, "T_color_depth")
    pixels = uv * np.array([width, height], dtype=np.float64)
    valid = (
        np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    if regions is not None:
        in_region = np.zeros(len(points), dtype=bool)
        for region in regions:
            polygon = np.asarray(region, dtype=float).reshape(-1, 2)
            if len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError("regions must contain finite image polygons")
            lo, hi = polygon.min(axis=0), polygon.max(axis=0)
            in_region |= np.all((pixels >= lo) & (pixels <= hi), axis=1)
        valid &= in_region
    points = points[valid].astype(np.float64, copy=False)
    pixels = pixels[valid]
    points = points @ transform[:3, :3].T + transform[:3, 3]
    front = points[:, 2] > 0
    points, pixels = points[front], pixels[front]
    if len(points):
        cells = np.rint(pixels).astype(np.int64)
        cells[:, 0] = np.clip(cells[:, 0], 0, width - 1)
        cells[:, 1] = np.clip(cells[:, 1], 0, height - 1)
        key = cells[:, 1] * width + cells[:, 0]
        order = np.lexsort((points[:, 2], key))
        _, first = np.unique(key[order], return_index=True)
        selected = order[first]
        points, pixels = points[selected], pixels[selected]
    return points, pixels


def depth_frame_timing(color_frame, depth_frame, max_delta_ms: float = 20.0):
    """Reject missing depth or incomparable/stale RGB-D timestamps.

    Returns a JSON-safe ``(status, absolute_timestamp_delta_ms)`` pair.
    SDK framesets alone do not guarantee identical exposure timestamps.
    """
    if not np.isfinite(max_delta_ms) or max_delta_ms <= 0:
        raise ValueError("max_delta_ms must be finite and positive")
    if not depth_frame:
        return "NO_DEPTH_FRAME", None
    if color_frame.get_frame_timestamp_domain() != depth_frame.get_frame_timestamp_domain():
        return "DEPTH_TIMESTAMP_DOMAIN_MISMATCH", None
    delta = abs(float(color_frame.get_timestamp()) - float(depth_frame.get_timestamp()))
    if not np.isfinite(delta):
        return "DEPTH_INVALID_TIMESTAMP", None
    return ("READY" if delta <= max_delta_ms else "DEPTH_TIME_MISMATCH"), delta


def colorize_depth(
    depth_image: np.ndarray,
    depth_scale_m: float,
    *,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Create a visible BGR depth map and JSON-safe frame statistics.

    The input remains in the native depth-camera image plane. Invalid zero
    samples are black, while the configurable display range is mapped through
    Turbo so nearer and farther surfaces are visually distinct.
    """
    image = np.asarray(depth_image)
    if image.ndim != 2:
        raise ValueError("depth_image must be a 2-D array")
    try:
        scale = float(depth_scale_m)
        near = float(min_depth_m)
        far = float(max_depth_m)
    except (TypeError, ValueError) as error:
        raise ValueError("depth scale and display bounds must be numeric") from error
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale_m must be finite and positive")
    if not np.isfinite(near) or not np.isfinite(far) or near < 0 or far <= near:
        raise ValueError("depth display bounds must be finite and satisfy 0 <= min < max")

    depth_m = image.astype(np.float32, copy=False) * scale
    valid = np.isfinite(depth_m) & (depth_m > 0)
    valid_count = int(np.count_nonzero(valid))
    total = int(depth_m.size)
    report = {
        "depth_view_status": "READY" if valid_count else "NO_VALID_DEPTH",
        "depth_min_m": None,
        "depth_max_m": None,
        "depth_median_m": None,
        "depth_valid_ratio": (valid_count / total) if total else 0.0,
        "depth_view_min_m": near,
        "depth_view_max_m": far,
    }
    if not valid_count:
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8), report

    values = depth_m[valid].astype(np.float64, copy=False)
    report.update(
        depth_min_m=float(values.min()),
        depth_max_m=float(values.max()),
        depth_median_m=float(np.median(values)),
    )
    normalized = np.clip((depth_m - near) / (far - near) * 255.0, 0, 255).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview, report


class RealSenseDepthCloud:
    """Reuse an SDK pointcloud block; input must be the **unaligned** depth frame."""

    def __init__(self):
        self._pointcloud = _realsense().pointcloud()

    def sample(self, depth_frame, color_frame, regions=None, camera=None):
        """Sample native depth using the recorded or an overridden RGB model.

        A calibrated RGB model changes texture projection only. Native depth
        vertices and the recorded depth-to-color rigid transform retain their
        original metric geometry.
        """
        if camera is None:
            self._pointcloud.map_to(color_frame)
        cloud = self._pointcloud.calculate(depth_frame)
        vertices = np.asanyarray(cloud.get_vertices()).view(np.float32).reshape(-1, 3)
        depth_profile = depth_frame.profile.as_video_stream_profile()
        color_profile = color_frame.profile.as_video_stream_profile()
        extrinsics = depth_profile.get_extrinsics_to(color_profile)
        transform = np.eye(4)
        # rs2_extrinsics.rotation is column-major, unlike NumPy's default.
        transform[:3, :3] = np.asarray(extrinsics.rotation).reshape(3, 3, order="F")
        transform[:3, 3] = extrinsics.translation
        intrinsics = color_profile.get_intrinsics()
        if camera is not None:
            if (camera.width, camera.height) != (intrinsics.width, intrinsics.height):
                raise ValueError("RGB camera override resolution differs from the recorded color stream")
            transform = validate_rigid_transform(transform, "T_color_depth")
            valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0)
            points = vertices[valid].astype(np.float64)
            points = points @ transform[:3, :3].T + transform[:3, 3]
            points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
            pixels = np.empty((len(points), 2), dtype=np.float64)
            # OpenCV also allocates a projection Jacobian. Bound its peak size
            # while projecting the whole native cloud before RGB ROI filtering.
            chunk_size = 65536
            for start in range(0, len(points), chunk_size):
                stop = start + chunk_size
                pixels[start:stop] = camera.project_points(points[start:stop])
            uv = pixels / np.array([camera.width, camera.height], dtype=np.float64)
            # Points now use the color optical frame; retain the shared ROI
            # selection and nearest-surface rule with an identity transform.
            return depth_points_in_color(
                points, uv, np.eye(4), camera.width, camera.height, regions,
            )
        uv = np.asanyarray(cloud.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        # The returned arrays own their filtered data, so no SDK buffers escape.
        return depth_points_in_color(
            vertices, uv, transform, intrinsics.width, intrinsics.height, regions,
        )


class RealSensePoseRefiner:
    """Per-frame callback used by RigTracker before its temporal acceptance gate.

    Configuration is validated even when depth is disabled or unavailable.
    SDK ``RuntimeError`` failures preserve the RGB pose; malformed inputs and
    configuration errors remain visible to the caller.
    """

    def __init__(
        self, cloud, color_frame, depth_frame, object_points, image_corners, *,
        depth_weight=0.2, max_translation_delta_m=0.05, max_rotation_delta_deg=10.0,
        max_reprojection_error_px=2.5, max_time_delta_ms=20.0, camera=None,
    ):
        settings = dict(
            depth_weight=depth_weight,
            max_translation_delta_m=max_translation_delta_m,
            max_rotation_delta_deg=max_rotation_delta_deg,
            max_reprojection_error_px=max_reprojection_error_px,
            max_time_delta_ms=max_time_delta_ms,
        )
        for name, value in settings.items():
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "depth_weight":
                if not 0.0 <= value <= 1.0:
                    raise ValueError("depth_weight must be in [0, 1]")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive")
            settings[name] = value
        if settings["max_rotation_delta_deg"] > 180.0:
            raise ValueError("max_rotation_delta_deg must be <= 180 degrees")
        self.cloud = cloud
        self.camera = camera
        self.color_frame, self.depth_frame = color_frame, depth_frame
        self.object_points, self.image_corners = object_points, image_corners
        self.max_time_delta_ms = settings.pop("max_time_delta_ms")
        self.options = settings

    def __call__(self, rgb_pose, used_ids):
        from .rgbd import refine_pose_rgbd

        report = dict(status="DEPTH_DISABLED", valid_depth_points=0,
                      translation_delta_m=None, rotation_delta_deg=None,
                      timestamp_delta_ms=None)
        if self.options["depth_weight"] == 0:
            return rgb_pose, report
        try:
            status, delta_ms = depth_frame_timing(
                self.color_frame, self.depth_frame, self.max_time_delta_ms,
            )
            report.update(status=status, timestamp_delta_ms=delta_ms)
            if status != "READY":
                return rgb_pose, report
            objects = {tag: self.object_points[tag] for tag in used_ids}
            images = {tag: np.asarray(self.image_corners[tag]).reshape(4, 2) for tag in used_ids}
            if self.camera is None:
                points, pixels = self.cloud.sample(self.depth_frame, self.color_frame, images.values())
                intrinsics = self.color_frame.profile.as_video_stream_profile().get_intrinsics()
                rs = _realsense()

                def project(points):
                    return np.asarray([
                        rs.rs2_project_point_to_pixel(intrinsics, point.tolist()) for point in points
                    ], dtype=np.float64)
            else:
                points, pixels = self.cloud.sample(
                    self.depth_frame, self.color_frame, images.values(), camera=self.camera,
                )
                project = self.camera.project_points

            pose, refinement = refine_pose_rgbd(
                rgb_pose, objects, images, points, pixels, project, **self.options,
            )
        except RuntimeError as error:
            report.update(status="DEPTH_SDK_ERROR", error=str(error))
            return rgb_pose, report
        refinement["timestamp_delta_ms"] = delta_ms
        return pose, refinement


__all__ = [
    "RealSenseDepthCloud", "RealSensePoseRefiner", "colorize_depth",
    "depth_frame_timing", "depth_points_in_color",
]

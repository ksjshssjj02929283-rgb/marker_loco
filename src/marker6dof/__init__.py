"""Multi-face ArUco 6DoF tracking for an Intel RealSense D455."""

from .localization import (
    BodyConfig,
    Calibration,
    MarkerDetection,
    MarkerLocalizer,
    PoseEstimate,
    QualityConfig,
    estimate_to_row,
    load_calibration,
)
from .realsense import (
    detect_markers,
    detector_parameters,
    marker_object_points,
    normalized_image_points,
    project_realsense,
)
from .projection import (
    draw_realsense_frame,
    draw_realsense_polyline,
    load_transform,
    project_realsense_points,
    transform_points,
    validate_rigid_transform,
)
from .depth import estimate_depth_pose, fuse_rgb_depth_pose, sample_depth_m
from .rgbd import refine_pose_rgbd
from .world import (
    VioSample,
    WorldTrajectory,
    synchronized_t265_camera_pose,
    t265_pose_to_transform,
    transform_from_translation_quaternion,
)

__all__ = [
    "BodyConfig",
    "Calibration",
    "MarkerDetection",
    "MarkerLocalizer",
    "PoseEstimate",
    "QualityConfig",
    "estimate_to_row",
    "load_calibration",
    "detect_markers",
    "detector_parameters",
    "marker_object_points",
    "normalized_image_points",
    "project_realsense",
    "draw_realsense_frame",
    "draw_realsense_polyline",
    "load_transform",
    "project_realsense_points",
    "transform_points",
    "validate_rigid_transform",
    "estimate_depth_pose",
    "fuse_rgb_depth_pose",
    "sample_depth_m",
    "refine_pose_rgbd",
    "VioSample",
    "WorldTrajectory",
    "synchronized_t265_camera_pose",
    "t265_pose_to_transform",
    "transform_from_translation_quaternion",
]

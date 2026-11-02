#!/usr/bin/env python3
"""Replay a recorded D455 session offline (no camera, no pyrealsense2).

Reads the clean color video (color_raw.mp4) and metadata.json written by
scripts/record_d455.py, re-detects the configured ArUco tags, and re-estimates
each single-tag pose with OpenCV's standard pinhole model using the intrinsics
stored in metadata.json. This is the offline path used to confirm that a real
recording replays deterministically without the live camera.

Note: the online recorder estimates poses with the RealSense SDK
deprojection/projection API, so replay numbers are expected to be close but not
bit-identical.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from marker6dof.realsense import (
    intrinsics_from_metadata,
    marker_object_points,
    normalized_image_points,
    project_realsense,
)


CSV_FIELDS = (
    "frame_index",
    "tag_id",
    "tx_m",
    "ty_m",
    "tz_m",
    "rx_rad",
    "ry_rad",
    "rz_rad",
    "reprojection_error_px",
)


def detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def detect_markers(gray, dictionary, parameters):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def reprojection_error_px(corners, rvec, tvec, object_points, camera_matrix, distortion) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    delta = projected.reshape(4, 2) - corners.reshape(4, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def reprojection_error_realsense(corners, rvec, tvec, object_points, intrinsics) -> float:
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ object_points.T).T + tvec.reshape(1, 3)
    projected = project_realsense(intrinsics, camera_points)
    delta = projected - corners.reshape(4, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Recorded session dir (metadata.json + color_raw.mp4)")
    parser.add_argument("--video", type=Path, help="Clean color video to replay")
    parser.add_argument("--metadata", type=Path, help="metadata.json written by record_d455.py")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/replay"))
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        help="Override the recorded black-square edge for correcting historical sessions",
    )
    parser.add_argument("--min-coverage", type=float, default=0.0, help="Required fraction of frames with a target (0..1)")
    args = parser.parse_args()

    if args.session is not None:
        args.metadata = args.session / "metadata.json"
        args.video = args.session / "color_raw.mp4"
    if args.metadata is None or args.video is None:
        raise SystemExit("provide --session, or both --video and --metadata")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    intrinsics = metadata["color_intrinsics"]
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics["coefficients"], dtype=np.float64).reshape(-1)
    try:
        rs_intrinsics = intrinsics_from_metadata(intrinsics)
    except RuntimeError:
        rs_intrinsics = None
    recorded_marker_size_m = float(metadata["aruco"]["marker_black_square_size_m"])
    if args.marker_size_mm is not None and args.marker_size_mm <= 0:
        raise SystemExit("--marker-size-mm must be positive")
    marker_size_m = (
        args.marker_size_mm / 1000.0
        if args.marker_size_mm is not None
        else recorded_marker_size_m
    )
    accepted_ids = {int(value) for value in metadata["aruco"]["accepted_ids"]}
    object_points = marker_object_points(marker_size_m)
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, metadata["aruco"].get("dictionary", "DICT_4X4_50"))
    )
    parameters = detector_parameters()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "replay_poses.csv"
    errors_px = []
    frames = 0
    frames_with_target = 0
    detections = 0
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray, dictionary, parameters)
            visible = []
            if ids is not None:
                for corner, tag_id in zip(corners, ids.reshape(-1)):
                    tag_id = int(tag_id)
                    if tag_id not in accepted_ids:
                        continue
                    tag_corners = corner.reshape(4, 2).astype(np.float64)
                    if rs_intrinsics is not None:
                        image_points = normalized_image_points(rs_intrinsics, tag_corners)
                        success, rvec, tvec = cv2.solvePnP(
                            object_points,
                            image_points,
                            np.eye(3, dtype=np.float64),
                            np.zeros((1, 5), dtype=np.float64),
                            flags=cv2.SOLVEPNP_IPPE_SQUARE,
                        )
                    else:
                        success, rvec, tvec = cv2.solvePnP(
                            object_points,
                            tag_corners,
                            camera_matrix,
                            distortion,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE,
                        )
                    if not success:
                        continue
                    error = (
                        reprojection_error_realsense(
                            tag_corners, rvec, tvec, object_points, rs_intrinsics
                        )
                        if rs_intrinsics is not None
                        else reprojection_error_px(
                            tag_corners, rvec, tvec, object_points, camera_matrix, distortion
                        )
                    )
                    errors_px.append(error)
                    writer.writerow(
                        {
                            "frame_index": frame_index,
                            "tag_id": tag_id,
                            "tx_m": f"{tvec[0, 0]:.9f}",
                            "ty_m": f"{tvec[1, 0]:.9f}",
                            "tz_m": f"{tvec[2, 0]:.9f}",
                            "rx_rad": f"{rvec[0, 0]:.9f}",
                            "ry_rad": f"{rvec[1, 0]:.9f}",
                            "rz_rad": f"{rvec[2, 0]:.9f}",
                            "reprojection_error_px": f"{error:.6f}",
                        }
                    )
                    detections += 1
                    visible.append(tag_id)
            if visible:
                frames_with_target += 1
            frame_index += 1
    capture.release()

    if frames == 0:
        raise RuntimeError("video produced no frames")

    coverage = frames_with_target / frames
    summary = {
        "video": str(args.video),
        "recorded_marker_size_m": recorded_marker_size_m,
        "marker_size_m": marker_size_m,
        "marker_size_overridden": args.marker_size_mm is not None,
        "camera_model_handling": (
            "RealSense SDK deprojection/projection"
            if rs_intrinsics is not None
            else "OpenCV approximation (pyrealsense2 unavailable)"
        ),
        "frames": frames,
        "frames_with_target": frames_with_target,
        "detection_count": detections,
        "frame_coverage": coverage,
        "median_reprojection_error_px": float(np.median(errors_px)) if errors_px else None,
        "p95_reprojection_error_px": float(np.percentile(errors_px, 95)) if errors_px else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if coverage < args.min_coverage:
        raise SystemExit(f"coverage {coverage:.3f} below min {args.min_coverage:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

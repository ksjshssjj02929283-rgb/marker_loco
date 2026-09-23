#!/usr/bin/env python3
"""Record a RealSense D455 stream and estimate OpenCV ArUco poses online."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from .realsense import (
    detect_markers,
    detector_parameters,
    marker_object_points,
    normalized_image_points,
    project_realsense,
)


CSV_FIELDS = (
    "frame_index",
    "device_timestamp_ms",
    "host_timestamp_s",
    "tag_id",
    "tx_m",
    "ty_m",
    "tz_m",
    "rx_rad",
    "ry_rad",
    "rz_rad",
    "qx",
    "qy",
    "qz",
    "qw",
    "reprojection_error_px",
    "depth_median_m",
    "depth_minus_pnp_z_m",
    "corner_0_x",
    "corner_0_y",
    "corner_1_x",
    "corner_1_y",
    "corner_2_x",
    "corner_2_y",
    "corner_3_x",
    "corner_3_y",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D455 + OpenCV ArUco recording and single-tag pose smoke test"
    )
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        default=50.0,
        help="Measured black-square outer edge, not the white paper cutout (default: 50)",
    )
    parser.add_argument(
        "--ids",
        default="2,3,4,5",
        help="Comma-separated accepted IDs; use the IDs on this gripper",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds; 0 records until Q")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", help="RealSense serial; defaults to the only connected device")
    parser.add_argument("--output-root", type=Path, default=Path("captures/d455_aruco"))
    parser.add_argument("--no-preview", action="store_true", help="Run without an OpenCV window")
    parser.add_argument(
        "--no-raw",
        "--no-bag",
        dest="no_raw",
        action="store_true",
        help="Do not save the large raw RealSense recording",
    )
    return parser.parse_args()


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
    return qx, qy, qz, qw


def camera_matrix(intrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def reprojection_error_px(
    corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_points: np.ndarray,
    intrinsics,
) -> float:
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ object_points.T).T + tvec.reshape(1, 3)
    projected = project_realsense(intrinsics, camera_points)
    delta = projected - corners.reshape(4, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def estimate_pose(intrinsics, corners: np.ndarray, object_points: np.ndarray):
    image_points = normalized_image_points(intrinsics, corners)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.eye(3, dtype=np.float64),
        np.zeros((1, 5), dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError("OpenCV solvePnP failed")
    return rvec.reshape(3), tvec.reshape(3)


def draw_axes_realsense(
    image: np.ndarray,
    intrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length_m: float,
) -> None:
    axes = np.array(
        [[0.0, 0.0, 0.0], [axis_length_m, 0.0, 0.0], [0.0, axis_length_m, 0.0], [0.0, 0.0, axis_length_m]],
        dtype=np.float64,
    )
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ axes.T).T + tvec.reshape(1, 3)
    if np.any(camera_points[:, 2] <= 0):
        return
    pixels = np.round(project_realsense(intrinsics, camera_points)).astype(int)
    origin = tuple(pixels[0])
    cv2.line(image, origin, tuple(pixels[1]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(image, origin, tuple(pixels[2]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(image, origin, tuple(pixels[3]), (255, 0, 0), 2, cv2.LINE_AA)


def median_tag_depth_m(
    depth_image: np.ndarray,
    corners: np.ndarray,
    depth_scale: float,
) -> float:
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    polygon = np.round(corners.reshape(4, 2)).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    samples = depth_image[(mask > 0) & (depth_image > 0)]
    if samples.size < 10:
        return float("nan")
    return float(np.median(samples) * depth_scale)


def make_session_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = root / timestamp
    session.mkdir(parents=True, exist_ok=False)
    return session


def device_details(device) -> dict:
    result = {}
    keys = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "product_line": rs.camera_info.product_line,
    }
    for name, key in keys.items():
        if device.supports(key):
            result[name] = device.get_info(key)
    return result


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    device,
    intrinsics,
    depth_scale: float,
    accepted_ids: set[int],
) -> None:
    data = {
        "created_local": datetime.now().astimezone().isoformat(),
        "device": device_details(device),
        "software": {
            "opencv": cv2.__version__,
            "librealsense": getattr(rs, "__version__", "unknown"),
        },
        "stream": {"width": args.width, "height": args.height, "fps": args.fps},
        "aruco": {
            "dictionary": "DICT_4X4_50",
            "accepted_ids": sorted(accepted_ids),
            "marker_black_square_size_m": args.marker_size_mm / 1000.0,
            "corner_refinement": "CORNER_REFINE_SUBPIX",
            "pose_api": "cv2.solvePnP(SOLVEPNP_IPPE_SQUARE)",
            "camera_model_handling": "RealSense SDK deprojection/projection",
        },
        "color_intrinsics": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "distortion_model": str(intrinsics.model),
            "coefficients": list(intrinsics.coeffs),
        },
        "depth_scale_m_per_unit": depth_scale,
        "coordinate_convention": {
            "pose": "T_camera_tag returned by OpenCV PnP",
            "camera_axes": "+X right, +Y down, +Z forward",
            "rotation": "Rodrigues rvec and quaternion qx,qy,qz,qw",
        },
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.marker_size_mm <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("marker size, resolution, and FPS must be positive")
    accepted_ids = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
    if not accepted_ids:
        raise SystemExit("--ids must contain at least one marker ID")

    session = make_session_dir(args.output_root)
    # RealSense SDK 2.58+ validates the ROS 2 sqlite recording extension.
    recording_path = session / "raw_realsense.db3"
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    if not args.no_raw:
        config.enable_record_to_file(str(recording_path.resolve()))

    profile = None
    video_writer = None
    color_writer = None
    pose_file = None
    frame_file = None
    start_monotonic = time.monotonic()
    frame_index = 0
    frames_with_target = 0
    detection_count = 0
    processing_times_ms = []
    reprojection_errors_px = []
    depth_deltas_m = []
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = detector_parameters()

    interrupted = False
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        depth_scale = device.first_depth_sensor().get_depth_scale()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        marker_size_m = args.marker_size_mm / 1000.0
        object_points = marker_object_points(marker_size_m)
        align = rs.align(rs.stream.color)
        write_metadata(session / "metadata.json", args, device, intrinsics, depth_scale, accepted_ids)

        video_writer = cv2.VideoWriter(
            str(session / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not video_writer.isOpened():
            raise RuntimeError("OpenCV could not create annotated.mp4")

        color_writer = cv2.VideoWriter(
            str(session / "color_raw.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not color_writer.isOpened():
            raise RuntimeError("OpenCV could not create color_raw.mp4")

        pose_file = (session / "poses.csv").open("w", newline="", encoding="utf-8")
        pose_writer = csv.DictWriter(pose_file, fieldnames=CSV_FIELDS)
        pose_writer.writeheader()
        frame_file = (session / "frames.csv").open("w", newline="", encoding="utf-8")
        frame_writer = csv.writer(frame_file)
        frame_writer.writerow(
            ["frame_index", "device_timestamp_ms", "host_timestamp_s", "target_ids", "processing_ms"]
        )

        print(f"Device: {device_details(device)}")
        print(f"Session: {session.resolve()}")
        print(f"Marker: DICT_4X4_50 IDs={sorted(accepted_ids)}, black edge={args.marker_size_mm:.2f} mm")
        gui_available = "GUI:                           NONE" not in cv2.getBuildInformation()
        use_preview = not args.no_preview and gui_available
        if not gui_available and not args.no_preview:
            print("OpenCV has no GUI backend; continuing without preview.")
        print("Press Q or Esc to stop." if use_preview else "Recording without preview.")
        if not args.no_raw:
            print("Raw RGB-D recording is enabled and may use about 3 GB per 30 seconds.")

        while True:
            loop_start = time.perf_counter()
            frames = align.process(pipeline.wait_for_frames(5000))
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            host_timestamp = time.time()
            device_timestamp_ms = color_frame.get_timestamp()
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            color_writer.write(color)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray, dictionary, parameters)
            visible_targets = []

            if ids is not None:
                ids_flat = ids.reshape(-1)
                selected = [index for index, tag_id in enumerate(ids_flat) if int(tag_id) in accepted_ids]
                for index in selected:
                    tag_id = int(ids_flat[index])
                    tag_corners = corners[index]
                    rvec, tvec = estimate_pose(intrinsics, tag_corners, object_points)
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
                    error_px = reprojection_error_px(
                        tag_corners, rvec, tvec, object_points, intrinsics
                    )
                    depth_m = median_tag_depth_m(depth, tag_corners, depth_scale)
                    depth_delta = depth_m - float(tvec[2]) if math.isfinite(depth_m) else float("nan")
                    reprojection_errors_px.append(error_px)
                    if math.isfinite(depth_delta):
                        depth_deltas_m.append(depth_delta)
                    flattened = tag_corners.reshape(4, 2).ravel().tolist()
                    row = {
                        "frame_index": frame_index,
                        "device_timestamp_ms": f"{device_timestamp_ms:.3f}",
                        "host_timestamp_s": f"{host_timestamp:.9f}",
                        "tag_id": tag_id,
                        "tx_m": f"{tvec[0]:.9f}",
                        "ty_m": f"{tvec[1]:.9f}",
                        "tz_m": f"{tvec[2]:.9f}",
                        "rx_rad": f"{rvec[0]:.9f}",
                        "ry_rad": f"{rvec[1]:.9f}",
                        "rz_rad": f"{rvec[2]:.9f}",
                        "qx": f"{quaternion[0]:.9f}",
                        "qy": f"{quaternion[1]:.9f}",
                        "qz": f"{quaternion[2]:.9f}",
                        "qw": f"{quaternion[3]:.9f}",
                        "reprojection_error_px": f"{error_px:.6f}",
                        "depth_median_m": f"{depth_m:.6f}",
                        "depth_minus_pnp_z_m": f"{depth_delta:.6f}",
                    }
                    for corner_index in range(4):
                        row[f"corner_{corner_index}_x"] = f"{flattened[corner_index * 2]:.3f}"
                        row[f"corner_{corner_index}_y"] = f"{flattened[corner_index * 2 + 1]:.3f}"
                    pose_writer.writerow(row)
                    visible_targets.append(tag_id)
                    detection_count += 1
                    draw_axes_realsense(
                        color, intrinsics, rvec, tvec, marker_size_m * 0.6
                    )
                    anchor = np.round(tag_corners.reshape(4, 2)[0]).astype(int)
                    cv2.putText(
                        color,
                        f"ID {tag_id} z={tvec[2]:.3f}m err={error_px:.2f}px",
                        (int(anchor[0]), max(24, int(anchor[1]) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if selected:
                    cv2.aruco.drawDetectedMarkers(
                        color, [corners[index] for index in selected], ids_flat[selected].reshape(-1, 1)
                    )

            if visible_targets:
                frames_with_target += 1
            elapsed = time.monotonic() - start_monotonic
            cv2.putText(
                color,
                f"REC {elapsed:6.1f}s  frame {frame_index}  visible {sorted(visible_targets)}",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 30, 255),
                2,
                cv2.LINE_AA,
            )
            processing_ms = (time.perf_counter() - loop_start) * 1000.0
            processing_times_ms.append(processing_ms)
            frame_writer.writerow(
                [frame_index, f"{device_timestamp_ms:.3f}", f"{host_timestamp:.9f}", ";".join(map(str, sorted(visible_targets))), f"{processing_ms:.3f}"]
            )
            video_writer.write(color)
            frame_index += 1

            if use_preview:
                cv2.imshow("D455 ArUco quick validation", color)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            if args.duration > 0 and elapsed >= args.duration:
                break

    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted by user.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if profile is not None:
            try:
                pipeline.stop()
            except KeyboardInterrupt:
                print("Interrupted while finalizing the RealSense recording.")
        if video_writer is not None:
            video_writer.release()
        if color_writer is not None:
            color_writer.release()
        if pose_file is not None:
            pose_file.close()
        if frame_file is not None:
            frame_file.close()
        if "use_preview" in locals() and use_preview:
            cv2.destroyAllWindows()

    coverage = frames_with_target / frame_index if frame_index else 0.0
    summary = {
        "frames": frame_index,
        "frames_with_target": frames_with_target,
        "detection_count": detection_count,
        "frame_coverage": coverage,
        "capture_duration_s": (frame_index / args.fps) if args.fps else 0.0,
        "median_processing_ms": float(np.median(processing_times_ms)) if processing_times_ms else None,
        "p95_processing_ms": float(np.percentile(processing_times_ms, 95)) if processing_times_ms else None,
        "median_reprojection_error_px": float(np.median(reprojection_errors_px)) if reprojection_errors_px else None,
        "median_depth_minus_pnp_z_m": float(np.median(depth_deltas_m)) if depth_deltas_m else None,
        "completed_local": datetime.now().astimezone().isoformat(),
        "interrupted": interrupted,
    }
    (session / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Done: {frame_index} frames, target visible in {frames_with_target} "
        f"({coverage:.1%}), {detection_count} detections"
    )
    print(f"Output: {session.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

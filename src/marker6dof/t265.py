"""Intel RealSense T265 fisheye capture and stereo pose helpers.

The T265 has no RGB or depth stream.  Its two synchronized 848x800 fisheye
streams are therefore treated as calibrated grayscale cameras.  Marker poses
are solved independently in each fisheye frame and can be fused in the left
fisheye frame using the SDK-provided stereo extrinsics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .realsense import (
    ARUCO_DICTIONARIES,
    DEFAULT_MARKER_SIZE_MM,
    detect_markers,
    detector_parameters,
    marker_object_points,
    normalized_image_points,
    project_realsense,
)


def ensure_t265_sdk(rs) -> None:
    """Reject SDK releases after T265 support was removed from librealsense."""
    version = str(getattr(rs, "__version__", "0.0")).split(".")
    try:
        major, minor = int(version[0]), int(re.match(r"\d+", version[1]).group())
    except (IndexError, AttributeError, ValueError):
        return
    if (major, minor) > (2, 53):
        raise RuntimeError(
            f"pyrealsense2 {getattr(rs, '__version__', 'unknown')} no longer supports T265; "
            "use librealsense/pyrealsense2 2.50-2.53 (the ROS2 workspace uses 2.51.1)"
        )


def extrinsics_to_matrix(extrinsics) -> np.ndarray:
    """Convert a RealSense extrinsics object to ``T_target_source``."""
    rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(extrinsics.translation, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("RealSense extrinsics contain non-finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def fuse_stereo_results(left_result, right_result, T_left_right: np.ndarray):
    """Fuse two ``RigTracker.solve`` results into the left-camera frame.

    A single valid eye is retained as a graceful fallback.  When both eyes are
    valid, translation and rotation are averaged only after transforming the
    right-eye pose into the left-eye frame.  The tracker already gates each
    eye's reprojection error; this function adds a conservative cross-eye gate.
    """
    left_pose = left_result.get("pose")
    right_pose = right_result.get("pose")
    valid_left = left_pose is not None
    valid_right = right_pose is not None
    if not valid_left and not valid_right:
        result = dict(left_result)
        result["visible_ids"] = sorted(set(left_result.get("visible_ids", [])) |
                                        set(right_result.get("visible_ids", [])))
        result["used_ids"] = []
        result["camera_sources"] = []
        return result

    right_in_left = None if not valid_right else T_left_right @ right_pose
    if valid_left and valid_right:
        translation_gap = float(np.linalg.norm(left_pose[:3, 3] - right_in_left[:3, 3]))
        rotation_gap = float(np.linalg.norm(
            Rotation.from_matrix(left_pose[:3, :3].T @ right_in_left[:3, :3]).as_rotvec()
        ))
        if translation_gap > 0.08 or rotation_gap > math.radians(12):
            # Keep the better eye rather than averaging an inconsistent pair.
            choose_left = (left_result.get("reprojection_error_px") or float("inf")) <= (
                right_result.get("reprojection_error_px") or float("inf")
            )
            result = dict(left_result if choose_left else right_result)
            if not choose_left:
                result["pose"] = right_in_left
            result["reason"] = "双目位姿不一致，采用重投影误差较小的一目"
            result["camera_sources"] = ["left" if choose_left else "right"]
            return result
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = (left_pose[:3, 3] + right_in_left[:3, 3]) / 2.0
        pose[:3, :3] = Rotation.from_matrix(
            np.stack([left_pose[:3, :3], right_in_left[:3, :3]])
        ).mean().as_matrix()
        result = dict(left_result)
        result["pose"] = pose
        result["status"] = "STEREO"
        result["reason"] = "左右鱼眼联合定位"
        result["visible_ids"] = sorted(set(left_result.get("visible_ids", [])) |
                                        set(right_result.get("visible_ids", [])))
        result["used_ids"] = sorted(set(left_result.get("used_ids", [])) |
                                     set(right_result.get("used_ids", [])))
        errors = [e for e in (left_result.get("reprojection_error_px"),
                              right_result.get("reprojection_error_px")) if e is not None]
        result["reprojection_error_px"] = float(np.mean(errors)) if errors else None
        result["camera_sources"] = ["left", "right"]
        return result

    result = dict(left_result if valid_left else right_result)
    if not valid_left:
        result["pose"] = right_in_left
    result["camera_sources"] = ["left" if valid_left else "right"]
    result["reason"] = ("左鱼眼单目降级" if valid_left else "右鱼眼单目降级")
    return result


def solve_tag_pose(intrinsics, corners: np.ndarray, object_points: np.ndarray):
    """Estimate one planar marker pose in a T265 fisheye camera frame."""
    normalized = normalized_image_points(intrinsics, corners)
    success, rvec, tvec = cv2.solvePnP(
        object_points, normalized, np.eye(3), np.zeros((1, 5)),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError("OpenCV solvePnP failed")
    rotation = cv2.Rodrigues(rvec)[0]
    camera_points = (rotation @ object_points.T).T + tvec.reshape(1, 3)
    projected = project_realsense(intrinsics, camera_points)
    delta = projected - np.asarray(corners).reshape(4, 2)
    error = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    return rvec.reshape(3), tvec.reshape(3), error


def solve_stereo_tag_pose(left_intrinsics, right_intrinsics, left_corners, right_corners,
                          object_points, T_left_right: np.ndarray):
    """Resolve a planar tag's pose using both T265 eyes in the left optical frame."""
    object_points = np.asarray(object_points, dtype=np.float64).reshape(4, 3)
    observations = {
        "left": normalized_image_points(left_intrinsics, left_corners).astype(np.float64),
        "right": normalized_image_points(right_intrinsics, right_corners).astype(np.float64),
    }
    T_right_left = np.linalg.inv(T_left_right)
    candidates = []
    for eye in ("left", "right"):
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, observations[eye], np.eye(3), np.zeros((1, 5)),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        for rvec, tvec in zip(rvecs, tvecs):
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(rvec)[0]
            pose[:3, 3] = tvec.reshape(3)
            if eye == "right":
                pose = T_left_right @ pose
            candidates.append(np.r_[cv2.Rodrigues(pose[:3, :3])[0].reshape(3), pose[:3, 3]])

    def residual(parameters):
        T_left_tag = np.eye(4)
        T_left_tag[:3, :3] = cv2.Rodrigues(parameters[:3])[0]
        T_left_tag[:3, 3] = parameters[3:]
        errors = []
        for eye in ("left", "right"):
            T_eye_tag = T_left_tag if eye == "left" else T_right_left @ T_left_tag
            points = (T_eye_tag[:3, :3] @ object_points.T).T + T_eye_tag[:3, 3]
            if np.any(points[:, 2] <= 0):
                return None
            errors.extend((points[:, :2] / points[:, 2, None] - observations[eye]).ravel())
        return np.asarray(errors)

    ranked = []
    for candidate in candidates:
        error = residual(candidate)
        if error is not None:
            ranked.append((float(error @ error), candidate))
    if not ranked:
        raise RuntimeError("no positive-depth stereo tag pose")
    ranked.sort(key=lambda item: item[0])
    best = None
    for _, candidate in ranked[:2]:
        def optimization_residual(parameters):
            value = residual(parameters)
            return value if value is not None else np.full(16, 100.0)

        fit = least_squares(
            optimization_residual,
            candidate,
            max_nfev=50,
        )
        error = residual(fit.x)
        if error is not None and (best is None or error @ error < best[0]):
            best = (float(error @ error), fit.x)
    if best is None:
        raise RuntimeError("stereo tag optimization did not return a positive-depth pose")

    rvec, tvec = best[1][:3], best[1][3:]
    T_left_tag = np.eye(4)
    T_left_tag[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_left_tag[:3, 3] = tvec
    pixel_residuals = []
    for eye, intrinsics, corners in (("left", left_intrinsics, left_corners),
                                      ("right", right_intrinsics, right_corners)):
        T_eye_tag = T_left_tag if eye == "left" else T_right_left @ T_left_tag
        points = (T_eye_tag[:3, :3] @ object_points.T).T + T_eye_tag[:3, 3]
        projected = project_realsense(intrinsics, points)
        pixel_residuals.extend((projected - np.asarray(corners).reshape(4, 2)).ravel())
    rmse_px = float(np.sqrt(np.mean(np.square(pixel_residuals)) * 2.0))
    return rvec, tvec, rmse_px


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Intel RealSense T265 stereo fisheye ArUco poses")
    parser.add_argument("--marker-size-mm", type=float, default=DEFAULT_MARKER_SIZE_MM)
    parser.add_argument("--dictionary", choices=ARUCO_DICTIONARIES, default="DICT_4X4_50",
                        help="Dictionary used to print the markers")
    parser.add_argument("--secondary-dictionary", choices=ARUCO_DICTIONARIES,
                        help="Optional second gripper dictionary, recorded from the same frames")
    parser.add_argument("--secondary-marker-size-mm", type=float,
                        help="Measured black-square edge for the second gripper")
    parser.add_argument("--ids", default="2,3,4,5")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds; 0 records until Q")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", help="T265 serial; defaults to the only connected device")
    parser.add_argument("--output-root", type=Path, default=Path("captures/t265_aruco"))
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def _intrinsics_dict(intrinsics) -> dict:
    return {
        "width": int(intrinsics.width), "height": int(intrinsics.height),
        "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx), "cy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coefficients": [float(value) for value in intrinsics.coeffs],
    }


def _device_details(device, rs) -> dict:
    values = {}
    for name, key in (("name", rs.camera_info.name), ("serial", rs.camera_info.serial_number),
                      ("firmware", rs.camera_info.firmware_version),
                      ("product_line", rs.camera_info.product_line)):
        if device.supports(key):
            values[name] = device.get_info(key)
    return values


def main() -> int:
    args = _parse_args()
    if args.marker_size_mm <= 0 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("marker size, resolution, and FPS must be positive")
    if args.secondary_dictionary and (args.secondary_marker_size_mm is None or args.secondary_marker_size_mm <= 0):
        raise SystemExit("--secondary-dictionary requires positive --secondary-marker-size-mm")
    if args.secondary_marker_size_mm is not None and not args.secondary_dictionary:
        raise SystemExit("--secondary-marker-size-mm requires --secondary-dictionary")
    accepted_ids = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
    if not accepted_ids:
        raise SystemExit("--ids must contain at least one marker ID")
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise SystemExit("T265 requires pyrealsense2; install with pip install -e '.[realsense]'") from error
    try:
        ensure_t265_sdk(rs)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    session = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    pipeline, config = rs.pipeline(), rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.fisheye, 1, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.fisheye, 2, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.pose)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    secondary_dictionary = (cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.secondary_dictionary)) if args.secondary_dictionary else None)
    parameters = detector_parameters()
    object_points = marker_object_points(args.marker_size_mm / 1000.0)
    secondary_object_points = (marker_object_points(args.secondary_marker_size_mm / 1000.0)
                               if secondary_dictionary is not None else None)
    fields = ["frame_index", "timestamp_ms", "camera", "tag_id", "tx_m", "ty_m", "tz_m",
              "rx_rad", "ry_rad", "rz_rad", "reprojection_error_px"]
    frame_index = 0
    started = None
    writers = {}
    pose_file = None
    secondary_pose_file = None
    pose_state_file = None
    frames_file = None
    profile = None
    try:
        profile = None
        last_error = None
        for _ in range(5):
            try:
                profile = pipeline.start(config)
                break
            except RuntimeError as error:
                last_error = error
                time.sleep(1.0)
        if profile is None:
            raise RuntimeError(
                f"无法启动 T265（{last_error}）。请关闭其他 T265/ROS2 相机进程，并确认使用 2.50-2.53 SDK。"
            ) from last_error
        left_profile = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
        right_profile = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
        left_intrinsics, right_intrinsics = left_profile.get_intrinsics(), right_profile.get_intrinsics()
        T_left_right = extrinsics_to_matrix(right_profile.get_extrinsics_to(left_profile))
        metadata = {
            "created_local": datetime.now().astimezone().isoformat(),
            "device": _device_details(profile.get_device(), rs),
            "software": {"opencv": cv2.__version__, "librealsense": getattr(rs, "__version__", "unknown")},
            "stream": {"width": args.width, "height": args.height, "fps": args.fps},
            "aruco": {"dictionary": args.dictionary, "accepted_ids": sorted(accepted_ids),
                      "marker_black_square_size_m": args.marker_size_mm / 1000.0},
            "fisheye_intrinsics": {"left": _intrinsics_dict(left_intrinsics),
                                   "right": _intrinsics_dict(right_intrinsics)},
            "T_left_right": T_left_right.tolist(),
            "coordinate_convention": "T_left_body is the fused marker pose in left fisheye optical coordinates",
        }
        if secondary_dictionary is not None:
            metadata["aruco_secondary"] = {
                "dictionary": args.secondary_dictionary,
                "accepted_ids": sorted(accepted_ids),
                "marker_black_square_size_m": args.secondary_marker_size_mm / 1000.0,
            }
        (session / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        for name in ("left", "right"):
            writer = cv2.VideoWriter(str(session / f"fisheye_{name}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not create fisheye_{name}.mp4")
            writers[name] = writer
        pose_file = (session / "poses.csv").open("w", newline="", encoding="utf-8")
        pose_writer = csv.DictWriter(pose_file, fieldnames=fields)
        pose_writer.writeheader()
        secondary_pose_writer = None
        if secondary_dictionary is not None:
            secondary_pose_file = (session / "poses_secondary.csv").open("w", newline="", encoding="utf-8")
            secondary_pose_writer = csv.DictWriter(secondary_pose_file, fieldnames=fields)
            secondary_pose_writer.writeheader()
        frames_file = (session / "frames.csv").open("w", newline="", encoding="utf-8")
        frames_writer = csv.DictWriter(frames_file, fieldnames=[
            "frame_index", "camera_timestamp_ms", "left_frame_number", "right_frame_number",
            "timestamp_domain",
        ])
        frames_writer.writeheader()
        pose_state_file = (session / "t265_pose.csv").open("w", newline="", encoding="utf-8")
        pose_state_writer = csv.DictWriter(
            pose_state_file,
            fieldnames=["frame_index", "timestamp_ms", "tx_m", "ty_m", "tz_m",
                        "qx", "qy", "qz", "qw", "tracker_confidence", "pose_timestamp_ms"],
        )
        pose_state_writer.writeheader()
        print(f"Device: {metadata['device']}\nSession: {session.resolve()}")
        use_preview = not args.no_preview and "GUI:                           NONE" not in cv2.getBuildInformation()
        last_fisheye_numbers = None
        started = time.monotonic()
        while True:
            frames = pipeline.wait_for_frames(5000)
            left_frame = frames.get_fisheye_frame(1)
            right_frame = frames.get_fisheye_frame(2)
            if not left_frame or not right_frame:
                continue
            fisheye_numbers = (left_frame.get_frame_number(), right_frame.get_frame_number())
            if fisheye_numbers[0] != fisheye_numbers[1] or fisheye_numbers == last_fisheye_numbers:
                continue
            last_fisheye_numbers = fisheye_numbers
            timestamp_ms = float(left_frame.get_timestamp())
            frames_writer.writerow({
                "frame_index": frame_index,
                "camera_timestamp_ms": f"{timestamp_ms:.3f}",
                "left_frame_number": fisheye_numbers[0],
                "right_frame_number": fisheye_numbers[1],
                "timestamp_domain": str(left_frame.get_frame_timestamp_domain()),
            })
            pose_frame = frames.get_pose_frame()
            if pose_frame:
                pose = pose_frame.get_pose_data()
                pose_state_writer.writerow({
                    "frame_index": frame_index, "timestamp_ms": f"{timestamp_ms:.3f}",
                    "tx_m": f"{pose.translation.x:.9f}", "ty_m": f"{pose.translation.y:.9f}",
                    "tz_m": f"{pose.translation.z:.9f}", "qx": f"{pose.rotation.x:.9f}",
                    "qy": f"{pose.rotation.y:.9f}", "qz": f"{pose.rotation.z:.9f}",
                    "qw": f"{pose.rotation.w:.9f}",
                    "tracker_confidence": getattr(pose, "tracker_confidence", ""),
                    "pose_timestamp_ms": f"{float(pose_frame.get_timestamp()):.3f}",
                })
            images = {}
            preview_images = {}
            for name, frame, intrinsics in (("left", left_frame, left_intrinsics),
                                             ("right", right_frame, right_intrinsics)):
                gray = np.asanyarray(frame.get_data())
                images[name] = gray
                writers[name].write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                preview_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if use_preview else None
                detectors = [("L", dictionary, object_points, pose_writer, (120, 225, 140))]
                if secondary_dictionary is not None:
                    detectors.append(("R", secondary_dictionary, secondary_object_points,
                                      secondary_pose_writer, (225, 115, 225)))
                for label, active_dictionary, active_points, active_writer, color in detectors:
                    corners, ids, _ = detect_markers(gray, active_dictionary, parameters)
                    if ids is None:
                        continue
                    for corner, raw_id in zip(corners, ids.reshape(-1)):
                        tag_id = int(raw_id)
                        if tag_id not in accepted_ids:
                            continue
                        if preview_image is not None:
                            points = np.rint(corner.reshape(4, 2)).astype(np.int32)
                            cv2.polylines(preview_image, [points], True, color, 2, cv2.LINE_AA)
                            cv2.putText(preview_image, f"{label}{tag_id}", tuple(points[0]),
                                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
                        try:
                            rvec, tvec, error = solve_tag_pose(intrinsics, corner, active_points)
                        except (RuntimeError, ValueError):
                            continue
                        active_writer.writerow({"frame_index": frame_index, "timestamp_ms": f"{timestamp_ms:.3f}",
                                                "camera": name, "tag_id": tag_id,
                                                "tx_m": f"{tvec[0]:.9f}", "ty_m": f"{tvec[1]:.9f}", "tz_m": f"{tvec[2]:.9f}",
                                                "rx_rad": f"{rvec[0]:.9f}", "ry_rad": f"{rvec[1]:.9f}", "rz_rad": f"{rvec[2]:.9f}",
                                                "reprojection_error_px": f"{error:.6f}"})
                if preview_image is not None:
                    preview_images[name] = preview_image
            if use_preview and images:
                preview = [preview_images[name] for name in ("left", "right") if name in preview_images]
                cv2.imshow("T265 stereo fisheye ArUco", np.hstack(preview))
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            frame_index += 1
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if profile is not None:
            pipeline.stop()
        for writer in writers.values():
            writer.release()
        if pose_file is not None:
            pose_file.close()
        if secondary_pose_file is not None:
            secondary_pose_file.close()
        if pose_state_file is not None:
            pose_state_file.close()
        if frames_file is not None:
            frames_file.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Headless OpenCV builds raise when no GUI window was ever created;
            # all recording data is already flushed to disk above.
            pass
    summary = {"frames": frame_index, "duration_s": time.monotonic() - started if started is not None else 0,
               "session": str(session.resolve()), "camera": "T265 stereo fisheye",
               "secondary_dictionary": args.secondary_dictionary}
    (session / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


__all__ = ["ensure_t265_sdk", "extrinsics_to_matrix", "fuse_stereo_results", "solve_tag_pose",
           "solve_stereo_tag_pose", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

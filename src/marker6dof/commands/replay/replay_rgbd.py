#!/usr/bin/env python3
"""Compare RGB and RGB-D rig tracking on a native RealSense recording.

Playback uses the stored color/depth streams, calibration and device timestamps.
Every distinct color frame is written, including frames without any marker. The
reported residuals measure internal consistency, not accuracy against ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from marker6dof.realsense import ARUCO_DICTIONARIES, DEFAULT_MARKER_SIZE_MM, detect_markers, detector_parameters
from marker6dof.tracking import RigTracker, load_rig
from marker6dof.camera import load_kalibr_camera


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True, help="Native .db3/.bag RGB-D recording")
    parser.add_argument("--rig", type=Path, required=True, help="JSON containing T_reference_tag")
    parser.add_argument("--dictionary", choices=ARUCO_DICTIONARIES,
                        help="ArUco dictionary; defaults to rig metadata, then DICT_4X4_50")
    parser.add_argument("--camchain", type=Path,
                        help="Optional calibrated RGB K+D; also reprojects native depth into this RGB model")
    parser.add_argument("--marker-size-mm", type=float, default=DEFAULT_MARKER_SIZE_MM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0, help="Distinct color frames; 0 replays the full recording")
    parser.add_argument("--depth-weight", type=float, default=0.2)
    parser.add_argument("--min-edge-px", type=float, default=24.0,
                        help="Minimum observed tag edge for both estimators; lowering it requires accuracy validation")
    parser.add_argument("--depth-max-delta-mm", type=float, default=50.0)
    parser.add_argument("--depth-max-angle-deg", type=float, default=10.0)
    parser.add_argument("--depth-max-time-delta-ms", type=float, default=20.0)
    args = parser.parse_args()
    for key in ("marker_size_mm", "min_edge_px", "depth_max_delta_mm", "depth_max_angle_deg", "depth_max_time_delta_ms"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.depth_weight) or not 0 <= args.depth_weight <= 1:
        parser.error("--depth-weight must be in [0, 1]")
    if args.depth_max_angle_deg > 180:
        parser.error("--depth-max-angle-deg must not exceed 180")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if not args.recording.is_file():
        parser.error(f"recording does not exist: {args.recording}")
    if not args.rig.is_file():
        parser.error(f"rig does not exist: {args.rig}")
    if args.camchain is not None and not args.camchain.is_file():
        parser.error(f"camchain does not exist: {args.camchain}")
    return args


def json_safe(value):
    """Preserve nested diagnostics while preventing NaN/Infinity JSON tokens."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def distribution(values):
    samples = np.asarray(values, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    return {
        "count": int(len(samples)),
        "median": float(np.median(samples)) if len(samples) else None,
        "p95": float(np.percentile(samples, 95)) if len(samples) else None,
        "max": float(np.max(samples)) if len(samples) else None,
    }


def main():
    args = parse_args()
    # Keep --help usable without the optional RealSense SDK installed.
    import pyrealsense2 as rs
    from marker6dof.realsense_depth import RealSenseDepthCloud, RealSensePoseRefiner, depth_frame_timing

    raw_rig = json.loads(args.rig.read_text(encoding="utf-8"))
    calibrated_camera = load_kalibr_camera(args.camchain) if args.camchain else None
    marker_size_m = args.marker_size_mm / 1000.0
    rig = load_rig(raw_rig, marker_size_m)
    rgb_tracker = RigTracker(rig, marker_size_m, min_edge_px=args.min_edge_px)
    rgbd_tracker = RigTracker(rig, marker_size_m, min_edge_px=args.min_edge_px)
    cloud = RealSenseDepthCloud()
    dictionary_name = args.dictionary or raw_rig.get("dictionary", "DICT_4X4_50")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = detector_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = args.output_dir / "frames.jsonl"
    summary_path = args.output_dir / "summary.json"
    for path in (frames_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing replay output: {path}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(args.recording.resolve()), repeat_playback=False)
    # Preserve the actual recorded stream profiles. Partial format-only
    # requests can fail to resolve on SDK playback even when the format exists.
    config.enable_all_streams()
    # Configure offline delivery before starting, so startup also avoids playback
    # dropping frames to keep up with wall time.
    resolved = config.resolve(rs.pipeline_wrapper(pipeline))
    resolved.get_device().as_playback().set_real_time(False)
    profile = None
    counters = Counter({key: 0 for key in (
        "frames", "framesets_without_color", "duplicate_color_frames_skipped",
        "timestamp_discontinuities", "frames_with_registered_marker",
        "frames_without_any_marker", "rgb_accepted_frames", "rgbd_accepted_frames",
        "depth_refinement_accepted_candidates", "depth_fused_accepted_frames",
        "depth_fused_temporally_rejected_frames", "rgb_fallback_frames", "rgbd_no_pose_frames",
    )})
    rgb_statuses, rgbd_statuses, depth_statuses, timing_statuses = (Counter() for _ in range(4))
    rgb_errors, rgbd_errors = [], []
    timing_deltas, frame_intervals = [], []
    detection_times, rgb_times, rgbd_times, processing_times = [], [], [], []
    seen_timestamps = set()
    first_timestamp = last_timestamp = None
    last_domain = None
    first_frame_number = last_frame_number = None
    timestamp_domains = set()
    completion = "eof"
    started = time.perf_counter()
    last_progress = started
    camera_metadata = {}
    recorded_camera_metadata = {}
    try:
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        if color_profile.format() not in (rs.format.bgr8, rs.format.rgb8):
            raise ValueError(f"Unsupported recorded color format: {color_profile.format()}")
        if profile.get_stream(rs.stream.depth).format() != rs.format.z16:
            raise ValueError("Recording must contain native Z16 depth")
        intrinsics = color_profile.get_intrinsics()
        camera_metadata = {
            "width": intrinsics.width, "height": intrinsics.height,
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "cx": intrinsics.ppx, "cy": intrinsics.ppy,
            "distortion_model": str(intrinsics.model), "coefficients": list(intrinsics.coeffs),
        }
        recorded_camera_metadata = dict(camera_metadata)
        if calibrated_camera is not None:
            if (calibrated_camera.width, calibrated_camera.height) != (intrinsics.width, intrinsics.height):
                raise ValueError("Calibrated RGB resolution must match the recorded color stream")
            camera_metadata = {
                "width": calibrated_camera.width, "height": calibrated_camera.height,
                "fx": float(calibrated_camera.matrix[0, 0]), "fy": float(calibrated_camera.matrix[1, 1]),
                "cx": float(calibrated_camera.matrix[0, 2]), "cy": float(calibrated_camera.matrix[1, 2]),
                "distortion_model": calibrated_camera.distortion_model,
                "coefficients": calibrated_camera.distortion.tolist(),
            }
        with frames_path.open("x", encoding="utf-8") as output:
            while True:
                available, frames = pipeline.try_wait_for_frames(5000)
                if not available:
                    if playback.current_status() == rs.playback_status.stopped:
                        break
                    raise RuntimeError(f"Playback timed out before EOF (status={playback.current_status()})")
                color_frame = frames.get_color_frame()
                if not color_frame:
                    counters["framesets_without_color"] += 1
                    continue
                timestamp_ms = float(color_frame.get_timestamp())
                domain = str(color_frame.get_frame_timestamp_domain())
                if not math.isfinite(timestamp_ms):
                    raise ValueError("Recording contains a non-finite color device timestamp")
                timestamp_key = (domain, timestamp_ms)
                if timestamp_key in seen_timestamps:
                    counters["duplicate_color_frames_skipped"] += 1
                    continue
                seen_timestamps.add(timestamp_key)
                frame_number = int(color_frame.get_frame_number())
                if first_timestamp is None:
                    first_timestamp, first_frame_number = timestamp_ms, frame_number
                if last_timestamp is not None:
                    if domain == last_domain and timestamp_ms > last_timestamp:
                        frame_intervals.append(timestamp_ms - last_timestamp)
                    else:
                        counters["timestamp_discontinuities"] += 1
                        rgb_tracker.reset()
                        rgbd_tracker.reset()
                last_timestamp, last_domain, last_frame_number = timestamp_ms, domain, frame_number
                timestamp_domains.add(domain)

                frame_started = time.perf_counter()
                depth_frame = frames.get_depth_frame()
                timing_status, delta_ms = depth_frame_timing(color_frame, depth_frame, args.depth_max_time_delta_ms)
                timing_statuses[timing_status] += 1
                if delta_ms is not None:
                    timing_deltas.append(delta_ms)
                image = np.asanyarray(color_frame.get_data())
                intrinsics = (calibrated_camera if calibrated_camera is not None else
                              color_frame.profile.as_video_stream_profile().get_intrinsics())
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if color_profile.format() == rs.format.bgr8 else cv2.COLOR_RGB2GRAY)
                corners, ids, _ = detect_markers(gray, dictionary, parameters)
                detections = {}
                duplicate_ids = []
                if ids is not None:
                    counts = Counter(int(tag) for tag in ids.flatten())
                    duplicate_ids = sorted(tag for tag, count in counts.items() if count > 1)
                    detections = {int(tag): corner for corner, tag in zip(corners, ids.flatten()) if counts[int(tag)] == 1}
                detection_ms = (time.perf_counter() - frame_started) * 1000.0
                detected_ids = sorted(int(tag) for tag in ids.flatten()) if ids is not None else []
                if any(tag in rig for tag in detections):
                    counters["frames_with_registered_marker"] += 1
                if not detected_ids:
                    counters["frames_without_any_marker"] += 1

                rgb_started = time.perf_counter()
                rgb_result = rgb_tracker.solve(detections, intrinsics, timestamp_ms / 1000.0)
                rgb_ms = (time.perf_counter() - rgb_started) * 1000.0
                refiner = RealSensePoseRefiner(
                    cloud, color_frame, depth_frame, rgbd_tracker.points, detections,
                    depth_weight=args.depth_weight,
                    max_translation_delta_m=args.depth_max_delta_mm / 1000.0,
                    max_rotation_delta_deg=args.depth_max_angle_deg,
                    max_reprojection_error_px=rgbd_tracker.max_error,
                    max_time_delta_ms=args.depth_max_time_delta_ms,
                    camera=calibrated_camera,
                )
                rgbd_started = time.perf_counter()
                rgbd_result = rgbd_tracker.solve(detections, intrinsics, timestamp_ms / 1000.0, pose_refiner=refiner)
                rgbd_ms = (time.perf_counter() - rgbd_started) * 1000.0
                depth_report = rgbd_result.setdefault("depth_report", {
                    "status": "NO_RGB_POSE", "valid_depth_points": 0, "timestamp_delta_ms": delta_ms,
                })
                depth_status = str(depth_report["status"])
                fused = depth_status in {"RGB_DEPTH_REFINED", "RGB_DEPTH_FUSED"}
                rgb_valid = rgb_result.get("pose") is not None
                rgbd_valid = rgbd_result.get("pose") is not None
                counters["rgb_accepted_frames"] += int(rgb_valid)
                counters["rgbd_accepted_frames"] += int(rgbd_valid)
                counters["depth_refinement_accepted_candidates"] += int(fused)
                counters["depth_fused_accepted_frames"] += int(fused and rgbd_valid)
                counters["depth_fused_temporally_rejected_frames"] += int(fused and not rgbd_valid)
                counters["rgb_fallback_frames"] += int(rgbd_valid and not fused)
                counters["rgbd_no_pose_frames"] += int(not rgbd_valid)
                rgb_statuses[rgb_result["status"]] += 1
                rgbd_statuses[rgbd_result["status"]] += 1
                depth_statuses[depth_status] += 1
                if rgb_valid and rgb_result.get("reprojection_error_px") is not None:
                    rgb_errors.append(rgb_result["reprojection_error_px"])
                if rgbd_valid and rgbd_result.get("reprojection_error_px") is not None:
                    rgbd_errors.append(rgbd_result["reprojection_error_px"])
                processing_ms = (time.perf_counter() - frame_started) * 1000.0
                detection_times.append(detection_ms)
                rgb_times.append(rgb_ms)
                rgbd_times.append(rgbd_ms)
                processing_times.append(processing_ms)
                record = {
                    "frame_index": counters["frames"], "color_frame_number": frame_number,
                    "color_timestamp_ms": timestamp_ms, "color_timestamp_domain": domain,
                    "depth_frame_number": int(depth_frame.get_frame_number()) if depth_frame else None,
                    "depth_timestamp_ms": float(depth_frame.get_timestamp()) if depth_frame else None,
                    "depth_timestamp_domain": str(depth_frame.get_frame_timestamp_domain()) if depth_frame else None,
                    "depth_timing_status": timing_status, "depth_timestamp_delta_ms": delta_ms,
                    "detected_ids": detected_ids, "duplicate_marker_ids_rejected": duplicate_ids,
                    "rgb": rgb_result, "rgbd": rgbd_result,
                    "runtime_ms": {"detection": detection_ms, "rgb": rgb_ms, "rgbd": rgbd_ms, "total": processing_ms},
                }
                output.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
                counters["frames"] += 1
                now = time.perf_counter()
                if now - last_progress >= 10:
                    output.flush()
                    print(f"Processed {counters['frames']} color frames; RGB-D accepted={counters['rgbd_accepted_frames']}, fused={counters['depth_fused_accepted_frames']}", flush=True)
                    last_progress = now
                if args.max_frames and counters["frames"] >= args.max_frames:
                    completion = "max_frames"
                    break
    finally:
        if profile is not None:
            pipeline.stop()

    count = counters["frames"]
    if not count:
        raise RuntimeError("Recording ended without delivering a color frame")
    summary = {
        "recording": str(args.recording.resolve()), "rig": str(args.rig.resolve()),
        "frames_jsonl": str(frames_path.resolve()), "completion": completion,
        "complete_recording": completion == "eof", "max_frames": args.max_frames,
        "marker_size_m": marker_size_m, "reference_id": int(raw_rig["reference_id"]),
        "calibrated_ids": sorted(rig), "dictionary": dictionary_name, "color_intrinsics": camera_metadata,
        "recorded_color_intrinsics": recorded_camera_metadata,
        "rgb_calibration_override": {
            "camchain": str(args.camchain.resolve()),
            "sha256": hashlib.sha256(args.camchain.read_bytes()).hexdigest(),
            "scope": "RGB solve, native depth-to-RGB pixel mapping, and RGB-D residual projection",
            "unchanged": "Recorded native depth intrinsics and physical depth-to-color extrinsics",
        } if args.camchain else None,
        "tracking_parameters": {"min_edge_px": args.min_edge_px, "max_reprojection_error_px": rgb_tracker.max_error},
        "software": {"opencv": cv2.__version__, "librealsense": getattr(rs, "__version__", "unknown")},
        "depth_parameters": {
            "weight": args.depth_weight, "max_translation_delta_m": args.depth_max_delta_mm / 1000.0,
            "max_rotation_delta_deg": args.depth_max_angle_deg, "max_time_delta_ms": args.depth_max_time_delta_ms,
        },
        "counts": dict(counters),
        "coverage": {
            "registered_marker": counters["frames_with_registered_marker"] / count,
            "rgb_pose": counters["rgb_accepted_frames"] / count,
            "rgbd_pose": counters["rgbd_accepted_frames"] / count,
            "depth_fused_pose": counters["depth_fused_accepted_frames"] / count,
        },
        "rgb_status_counts": dict(rgb_statuses), "rgbd_status_counts": dict(rgbd_statuses),
        "depth_status_counts": dict(depth_statuses), "depth_timing_status_counts": dict(timing_statuses),
        "first_color_frame_number": first_frame_number, "last_color_frame_number": last_frame_number,
        "first_color_timestamp_ms": first_timestamp, "last_color_timestamp_ms": last_timestamp,
        "color_timestamp_domains": sorted(timestamp_domains),
        "capture_span_s": (last_timestamp - first_timestamp) / 1000.0 if len(timestamp_domains) == 1 and not counters["timestamp_discontinuities"] else None,
        "color_frame_interval_ms": distribution(frame_intervals), "depth_timestamp_delta_ms": distribution(timing_deltas),
        "rgb_reprojection_error_px": distribution(rgb_errors), "rgbd_reprojection_error_px": distribution(rgbd_errors),
        "runtime_ms": {
            "detection": distribution(detection_times), "rgb": distribution(rgb_times),
            "rgbd": distribution(rgbd_times), "total": distribution(processing_times),
        },
        "wall_time_s": time.perf_counter() - started,
        "coordinate_convention": "pose is T_camera_reference: reference marker coordinates to color optical coordinates (+X right, +Y down, +Z forward), metres",
        "time_convention": "Tracker input uses recorded color device timestamp / 1000; domains are recorded and discontinuities reset both trackers",
        "metric_note": "Reprojection statistics include accepted poses only. Runtime excludes frame acquisition and JSON writing; total includes both independent estimators.",
        "accuracy_validated": False,
        "validation_note": "No independent ground truth is supplied. Coverage, reprojection residuals and depth agreement do not establish metric pose accuracy.",
    }
    safe_summary = json_safe(summary)
    summary_path.write_text(json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

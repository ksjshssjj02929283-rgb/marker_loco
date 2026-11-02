"""Replay ROS2 CompressedImage bags with Kalibr intrinsics and RigTracker."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from marker6dof.camera import load_kalibr_camera
from marker6dof.realsense import detect_markers, detector_parameters
from marker6dof.tracking import RigTracker, load_rig


def select_detections(corners, ids, accepted_ids, camera, min_edge_px, exclude_center_rect=None):
    """Select a unique viable contour per ID, with order-independent duplicates.

    One physical marker per ID is assumed. A contour with >=4x the area of
    the runner-up dominates a small false decode. Comparable viable contours
    remain ambiguous and are rejected, rather than silently overwriting.
    """
    groups = {}
    if ids is not None:
        for tag, corner in zip(ids.ravel(), corners):
            tag = int(tag)
            if tag in accepted_ids:
                candidate = np.asarray(corner, dtype=float).reshape(4, 2)
                if exclude_center_rect is not None:
                    x, y = candidate.mean(axis=0)
                    x0, y0, x1, y1 = exclude_center_rect
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        continue
                groups.setdefault(tag, []).append(candidate)
    selected, duplicates = {}, {}
    for tag, candidates in groups.items():
        viable = []
        for c in candidates:
            edge = np.min(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1))
            if (np.isfinite(c).all() and edge >= min_edge_px and c[:, 0].min() > 3
                    and c[:, 1].min() > 3 and c[:, 0].max() < camera.width - 4
                    and c[:, 1].max() < camera.height - 4):
                viable.append(c)
        viable.sort(key=lambda c: abs(cv2.contourArea(c.astype(np.float32))), reverse=True)
        policy = "no_viable_contour"
        if len(viable) == 1:
            selected[tag] = viable[0]; policy = "unique_viable_contour"
        elif len(viable) > 1:
            areas = [abs(cv2.contourArea(c.astype(np.float32))) for c in viable]
            if areas[0] >= 4 * areas[1]:
                selected[tag] = viable[0]; policy = "dominant_area_contour"
            else:
                policy = "ambiguous_duplicate_rejected"
        if len(candidates) > 1:
            duplicates[str(tag)] = dict(policy=policy, corners=[c.tolist() for c in candidates])
    return selected, sorted(groups), duplicates


def read_images(session, topic):
    """Yield decoded original frames; preserve both header and storage times."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError("ROS2 bag replay requires pip install -e '.[rosbag]'") from exc
    with AnyReader([Path(session)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections or not sum(c.msgcount for c in connections):
            raise ValueError(f"no messages on image topic {topic}")
        if any(c.msgtype != "sensor_msgs/msg/CompressedImage" for c in connections):
            raise ValueError("image topic must contain sensor_msgs/msg/CompressedImage")
        for connection, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, connection.msgtype)
            image = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode image at bag time {timestamp}")
            header_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            yield timestamp, header_ns, image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--camchain", type=Path, required=True)
    parser.add_argument("--camera-key", default="cam0")
    parser.add_argument("--topic", help="default: selected camera's Kalibr rostopic")
    parser.add_argument("--marker-size-mm", type=float, required=True)
    parser.add_argument("--ids", default="2,3,4,5")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--rig", type=Path, help="optional matching multi-face calibration")
    parser.add_argument("--min-edge-px", type=float, default=12.0,
                        help="minimum shortest edge; default 12 px for range exploration")
    parser.add_argument("--max-error-px", type=float, default=2.5)
    parser.add_argument("--exclude-center-rect", type=float, nargs=4,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="ignore marker candidates whose centre falls in this image rectangle")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="write annotated video at nominal 30 Hz")
    args = parser.parse_args()
    if (not np.isfinite(args.marker_size_mm) or args.marker_size_mm <= 0
            or args.min_edge_px <= 0 or args.max_error_px <= 0 or args.max_frames < 0):
        parser.error("marker size and quality gates must be positive; max-frames must be nonnegative")
    if args.exclude_center_rect is not None:
        x0, y0, x1, y1 = args.exclude_center_rect
        if not (np.isfinite(args.exclude_center_rect).all() and x0 < x1 and y0 < y1):
            parser.error("exclude-center-rect requires finite X0 < X1 and Y0 < Y1")
    accepted_ids = {int(t) for t in args.ids.split(",")}
    camera = load_kalibr_camera(args.camchain, args.camera_key)
    topic = args.topic or camera.topic
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    parameters = detector_parameters()
    cv2.setNumThreads(2)
    size = args.marker_size_mm / 1000
    trackers = {tag: RigTracker({tag: np.eye(4)}, size, args.max_error_px, args.min_edge_px)
                for tag in accepted_ids}
    rig_tracker = None
    if args.rig:
        rig_tracker = RigTracker(load_rig(json.loads(args.rig.read_text()), size),
                                 size, args.max_error_px, args.min_edge_px)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ["poses.csv", "frames.csv", "frames.jsonl"]):
        raise ValueError("output already contains replay results; choose a fresh directory")
    fields = ["frame_index", "host_timestamp_s", "timestamp_s", "bag_timestamp_ns", "header_timestamp_ns",
              "tag_id", "status", "tx_m", "ty_m", "tz_m", "rx_rad", "ry_rad", "rz_rad",
              "reprojection_error_px", "min_edge_px"]
    fields += [f"corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")]
    counts, detected, duplicate_counts, errors, z_values = Counter(), Counter(), Counter(), [], []
    frame_count = covered = strict_covered = 0
    first_header = first_bag = previous_header = None
    video = None
    if args.video:
        video = cv2.VideoWriter(str(args.output_dir / "annotated.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), 30., (camera.width, camera.height))
        if not video.isOpened():
            raise RuntimeError("cannot open annotated video writer")
    try:
        with (args.output_dir / "poses.csv").open("w", newline="") as poses, \
                (args.output_dir / "frames.csv").open("w", newline="") as frames, \
                (args.output_dir / "frames.jsonl").open("w") as jsonl:
            pw = csv.DictWriter(poses, fieldnames=fields)
            fw = csv.DictWriter(frames, fieldnames=["frame_index", "timestamp_s", "bag_timestamp_ns",
                                                   "header_timestamp_ns", "detected_ids", "accepted_ids", "strict_ids", "duplicate_ids"])
            pw.writeheader(); fw.writeheader()
            for index, (bag_ns, header_ns, image) in enumerate(read_images(args.session, topic)):
                if args.max_frames and index >= args.max_frames:
                    break
                if image.shape[:2] != (camera.height, camera.width):
                    raise ValueError(f"image resolution {image.shape[1::-1]} != calibration {(camera.width, camera.height)}")
                if first_header is None:
                    first_header, first_bag = header_ns, bag_ns
                timestamp = (header_ns - first_header) / 1e9
                if previous_header is not None and header_ns <= previous_header:
                    raise ValueError("image header timestamps are not strictly increasing")
                previous_header = header_ns
                corners, ids, _ = detect_markers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dictionary, parameters)
                detections, raw_ids, duplicates = select_detections(
                    corners, ids, accepted_ids, camera, args.min_edge_px,
                    args.exclude_center_rect)
                duplicate_counts.update(d["policy"] for d in duplicates.values())
                frame_record = dict(frame_index=index, timestamp_s=timestamp,
                                    bag_timestamp_ns=bag_ns, header_timestamp_ns=header_ns, tags={},
                                    detected_ids=raw_ids, duplicate_ids=duplicates,
                                    selected_corners={str(t): c.tolist() for t, c in detections.items()})
                valid, strict = [], []
                for tag, tracker in trackers.items():
                    result = tracker.solve({tag: detections[tag]} if tag in detections else {}, camera, timestamp)
                    if tag in raw_ids:
                        detected[tag] += 1
                    pose = result.pop("pose")
                    frame_record["tags"][str(tag)] = result
                    if pose is None:
                        continue
                    valid.append(tag); counts[tag] += 1
                    c = detections[tag]
                    edge = float(np.min(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
                    if edge >= 24:
                        strict.append(tag)
                    error = result["reprojection_error_px"]
                    errors.append(error); z_values.append(float(pose[2, 3]))
                    frame_record["tags"][str(tag)]["T_camera_tag"] = pose.tolist()
                    row = dict(frame_index=index, host_timestamp_s=header_ns / 1e9,
                               timestamp_s=timestamp, bag_timestamp_ns=bag_ns, header_timestamp_ns=header_ns,
                               tag_id=tag, status=result["status"], reprojection_error_px=error, min_edge_px=edge)
                    row.update(zip(["tx_m", "ty_m", "tz_m"], pose[:3, 3]))
                    row.update(zip(["rx_rad", "ry_rad", "rz_rad"], cv2.Rodrigues(pose[:3, :3])[0].ravel()))
                    row.update(zip(fields[-8:], c.ravel()))
                    pw.writerow(row)
                    if video is not None:
                        cv2.polylines(image, [np.round(c).astype(np.int32)], True, (0, 220, 0), 2)
                        cv2.putText(image, f"ID {tag}: Z={pose[2,3]:.3f}m edge={edge:.1f}px",
                                    tuple(np.round(c[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 0), 2)
                if rig_tracker:
                    body = rig_tracker.solve(detections, camera, timestamp)
                    body_pose = body.pop("pose")
                    body["T_camera_body"] = None if body_pose is None else body_pose.tolist()
                    frame_record["body"] = body
                frame_count += 1; covered += bool(valid); strict_covered += bool(strict)
                fw.writerow(dict(frame_index=index, timestamp_s=timestamp, bag_timestamp_ns=bag_ns,
                                 header_timestamp_ns=header_ns, detected_ids=json.dumps(raw_ids),
                                 accepted_ids=json.dumps(valid), strict_ids=json.dumps(strict),
                                 duplicate_ids=json.dumps(sorted(duplicates))))
                jsonl.write(json.dumps(frame_record, ensure_ascii=False, allow_nan=False) + "\n")
                if video is not None:
                    cv2.putText(image, f"t={timestamp:.2f}s accepted={valid}", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 255), 2)
                    video.write(image)
                if index % 300 == 0:
                    print(f"{index}: t={timestamp:.2f}s detected={sorted(detections)} accepted={valid}", flush=True)
    finally:
        if video is not None:
            video.release()
    if not frame_count:
        raise ValueError("no image frames processed")
    summary = dict(session=str(args.session.resolve()), camchain=str(args.camchain.resolve()),
                   camera_key=args.camera_key, topic=topic, camera_model="pinhole",
                   distortion_model=camera.distortion_model, marker_size_mm=args.marker_size_mm,
                   dictionary=args.dictionary, min_edge_px=args.min_edge_px, max_error_px=args.max_error_px,
                   exclude_center_rect=args.exclude_center_rect,
                   frames=frame_count, duration_s=timestamp, frames_with_pose=covered,
                   frame_coverage=covered / frame_count, strict_24px_coverage=strict_covered / frame_count,
                   detections_by_id=dict(detected), accepted_by_id=dict(counts),
                   duplicate_selection_counts=dict(duplicate_counts),
                   duplicate_policy="one physical marker per ID; unique viable or dominant >=4x area contour; comparable duplicates rejected",
                   reprojection_p50_px=float(np.median(errors)) if errors else None,
                   reprojection_p95_px=float(np.percentile(errors, 95)) if errors else None,
                   accepted_z_range_m=[min(z_values), max(z_values)] if z_values else None,
                   timestamp_source="image header; storage timestamps also retained",
                   interpretation="Single-camera visual depth from known black-square size. No stereo/depth sensor or IMU fusion. 12 px exploratory gate is weaker than the usual 24 px gate. Different tag centres are not interchangeable.")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if video is not None:
        from marker6dof.commands.reporting.video_encoding import encode_comparison_videos
        summary["video_encoding"] = encode_comparison_videos(args.output_dir, ("annotated.mp4",))
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

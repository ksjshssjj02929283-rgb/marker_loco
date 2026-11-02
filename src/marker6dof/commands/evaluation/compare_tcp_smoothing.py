"""Compare both TCP display pipelines on identical stored D455 observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from marker6dof.projection import draw_realsense_frame, draw_realsense_polyline, load_transform
from marker6dof.realsense import intrinsics_from_metadata
from marker6dof.smoothing import PoseSmoother
from marker6dof.tracking import RelativeTrace, RigTracker, load_rig
from marker6dof.commands.reporting.video_encoding import encode_comparison_videos


@dataclass
class FrameObservation:
    frame_index: int
    timestamp_s: float
    elapsed_s: float
    detections: dict


POSE_FIELDS = (
    "raw_reference", "raw_tcp", "old_camera_tcp", "new_camera_tcp",
    "raw_relative", "old_relative", "new_relative",
)
CSV_FIELDS = (
    "frame_index", "timestamp_s", "elapsed_s", "timestamp_discontinuity", "status", "reason",
    "visible_ids", "used_ids", "rejected_ids", "reprojection_error_px", "segment",
    *(f"{prefix}_{axis}_m" for prefix in POSE_FIELDS for axis in "xyz"),
    *(f"{prefix}_q{axis}" for prefix in POSE_FIELDS for axis in "xyzw"),
)


def read_observations(session):
    """Read every frame and its saved corners; reject ambiguous alignment."""
    session = Path(session)
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    with (session / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("frames.csv contains no frames")
    frames = []
    for index, row in enumerate(rows):
        frame_index = int(row["frame_index"])
        if frame_index != index:
            raise ValueError(f"frames.csv must have unique consecutive indices from 0; row {index} has {frame_index}")
        timestamp = float(row["device_timestamp_ms"]) / 1000.0
        if not math.isfinite(timestamp):
            raise ValueError(f"Frame {index}: timestamp must be finite")
        if frames and timestamp < frames[-1].timestamp_s:
            raise ValueError(f"Frame {index}: device timestamps must not go backwards")
        frames.append(FrameObservation(index, timestamp, timestamp - frames[0].timestamp_s if frames else 0.0, {}))
    with (session / "poses.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame_index, tag_id = int(row["frame_index"]), int(row["tag_id"])
            if not 0 <= frame_index < len(frames):
                raise ValueError(f"poses.csv references missing frame {frame_index}")
            frame = frames[frame_index]
            if tag_id in frame.detections:
                raise ValueError(f"Frame {frame_index}: duplicate corners for Tag {tag_id}")
            timestamp = float(row["device_timestamp_ms"]) / 1000.0
            if not math.isclose(timestamp, frame.timestamp_s, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Frame {frame_index}: corner and frame timestamps differ")
            corners = np.array([
                [float(row[f"corner_{index}_{axis}"]) for axis in "xy"] for index in range(4)
            ])
            if not np.isfinite(corners).all():
                raise ValueError(f"Frame {frame_index}: corners must be finite")
            frame.detections[tag_id] = corners
    return metadata, frames


def _payload_pose(payload, smoothed):
    prefix = "smooth_" if smoothed else ""
    position = payload[f"{prefix}position_m"]
    if position is None:
        return None
    pose = np.eye(4)
    pose[:3, 3] = position
    pose[:3, :3] = Rotation.from_quat(payload[f"{prefix}quaternion_xyzw"]).as_matrix()
    return pose


def compare_observations(frames, tracker, intrinsics, tcp_transform, smoother):
    """Solve each observation once and apply only the display changes twice."""
    old_trace, new_trace = RelativeTrace(max(3000, len(frames))), RelativeTrace(max(3000, len(frames)))
    initial_tcp, previous_time, previous_segment = None, None, None
    records = []
    for frame in frames:
        clock_break = previous_time is not None and frame.timestamp_s <= previous_time
        gap = clock_break or (
            previous_time is not None and frame.timestamp_s - previous_time > smoother.max_gap_s
        )
        if gap:
            tracker.reset()
            smoother.reset()
            old_trace.broken = new_trace.broken = True
        result = tracker.solve(frame.detections, intrinsics, frame.timestamp_s)
        raw_reference = result.get("pose")
        raw_tcp = raw_reference @ tcp_transform if raw_reference is not None else None
        display_reference = smoother.update(raw_reference, frame.timestamp_s)
        new_camera_tcp = display_reference @ tcp_transform if display_reference is not None else None
        if initial_tcp is None and raw_tcp is not None:
            initial_tcp = raw_tcp.copy()
        old_data = old_trace.update(dict(result, pose=raw_tcp), frame.timestamp_s)
        new_data = new_trace.update(dict(result, pose=raw_tcp), frame.timestamp_s, display_pose=new_camera_tcp)
        if old_data["segment"] != new_data["segment"]:
            raise RuntimeError("The two pipelines unexpectedly produced different segment boundaries")
        old_relative = _payload_pose(old_data, True)
        record = {
            "frame_index": frame.frame_index, "timestamp_s": frame.timestamp_s, "elapsed_s": frame.elapsed_s,
            "timestamp_discontinuity": clock_break,
            "status": result["status"], "reason": result.get("reason", ""),
            "visible_ids": result.get("visible_ids", []), "used_ids": result.get("used_ids", []),
            "rejected_ids": result.get("rejected_ids", []),
            "reprojection_error_px": result.get("reprojection_error_px"), "segment": old_data["segment"],
            "raw_reference": raw_reference.copy() if raw_reference is not None else None,
            "raw_tcp": raw_tcp,
            "old_camera_tcp": initial_tcp @ old_relative if old_relative is not None else None,
            "new_camera_tcp": new_camera_tcp,
            "raw_relative": _payload_pose(new_data, False),
            "old_relative": old_relative,
            "new_relative": _payload_pose(new_data, True),
            "break_before": gap or (previous_segment is not None and old_data["segment"] != previous_segment),
        }
        records.append(record)
        previous_time, previous_segment = frame.timestamp_s, old_data["segment"]
    if initial_tcp is None:
        raise ValueError("No accepted joint pose in the recording")
    return records


def write_comparison_csv(records, path):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {name: record[name] for name in CSV_FIELDS if name in record}
            for name in ("visible_ids", "used_ids", "rejected_ids"):
                row[name] = ";".join(map(str, row[name]))
            for name in POSE_FIELDS:
                pose = record[name]
                if pose is None:
                    continue
                row.update({f"{name}_{axis}_m": value for axis, value in zip("xyz", pose[:3, 3])})
                row.update({f"{name}_q{axis}": value for axis, value in zip("xyzw", Rotation.from_matrix(pose[:3, :3]).as_quat())})
            writer.writerow(row)


def _position_statistics(poses):
    points = np.array([pose[:3, 3] for pose in poses])
    deviations_mm = (points - points.mean(axis=0)) * 1000.0
    distances = np.linalg.norm(deviations_mm, axis=1)
    return {
        "rms_3d_mm": float(np.sqrt(np.mean(distances ** 2))),
        "radial_p95_mm": float(np.percentile(distances, 95)),
        "std_xyz_mm": deviations_mm.std(axis=0).tolist(),
        "peak_to_peak_xyz_mm": np.ptp(points * 1000.0, axis=0).tolist(),
        "mean_camera_position_m": points.mean(axis=0).tolist(),
    }


def motion_variation(records):
    """Measure short-term path variation on the same contiguous frame triples.

    This includes physical acceleration as well as measurement/filter effects;
    it must not be described as static noise, accuracy, or a motion truth error.
    """
    values = {name: [] for name in ('raw', 'old', 'new')}
    fields = {'raw': 'raw_tcp', 'old': 'old_camera_tcp', 'new': 'new_camera_tcp'}
    for index in range(2, len(records)):
        triple = records[index - 2:index + 1]
        if len({record['segment'] for record in triple}) != 1:
            continue
        if any(record['raw_tcp'] is None for record in triple):
            continue
        if any(record.get('break_before') for record in triple[1:]):
            continue
        dt1 = triple[1]['timestamp_s'] - triple[0]['timestamp_s']
        dt2 = triple[2]['timestamp_s'] - triple[1]['timestamp_s']
        if not (0 < dt1 <= .2 and 0 < dt2 <= .2):
            continue
        for name, field in fields.items():
            points = [record[field][:3, 3] for record in triple]
            first_velocity = (points[1] - points[0]) / dt1
            second_velocity = (points[2] - points[1]) / dt2
            acceleration = (second_velocity - first_velocity) / ((dt1 + dt2) / 2)
            values[name].append(float(np.linalg.norm(acceleration) * 1000))
    result = {
        'eligible_frame_triples': len(values['raw']),
        'units': 'mm/s^2',
        'note': 'Finite-difference acceleration variation includes real motion and filter lag. Lower values describe a smoother displayed path, not verified static noise or absolute accuracy.',
    }
    for name, samples in values.items():
        result[name] = ({
            'rms': float(np.sqrt(np.mean(np.square(samples)))),
            'median': float(np.median(samples)),
            'p95': float(np.percentile(samples, 95)),
        } if samples else None)
    return result


def summarize(records, windows, smoother):
    duration = records[-1]["elapsed_s"]
    accepted = [record for record in records if record["raw_tcp"] is not None]
    windows_summary = []
    for start, end in windows:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError(f"Static window [{start:g}, {end:g}] must fit actual device-time range [0, {duration:g}] s")
        selected = [record for record in records if start <= record["elapsed_s"] <= end]
        usable = [record for record in selected if record["raw_tcp"] is not None]
        if len(usable) < 3:
            raise ValueError(f"Static window [{start:g}, {end:g}] has fewer than 3 accepted poses")
        rotation = Rotation.from_matrix(np.array([record["raw_reference"][:3, :3] for record in usable]))
        angles = (rotation.mean().inv() * rotation).magnitude()
        windows_summary.append({
            "start_s": start, "end_s": end, "frames": len(selected), "accepted_frames": len(usable),
            "classification": "specified_window_not_independently_confirmed_static",
            "note": "The requested time window is not independently confirmed stationary. Any real motion contributes to these statistics; do not interpret moving-window RMS as static jitter.",
            "pose_coverage": len(usable) / len(selected),
            "raw": _position_statistics([record["raw_tcp"] for record in usable]),
            "old": _position_statistics([record["old_camera_tcp"] for record in usable]),
            "new": _position_statistics([record["new_camera_tcp"] for record in usable]),
            "raw_reference_rotation_rms_deg": float(np.rad2deg(np.sqrt(np.mean(angles ** 2)))),
        })
    differences = {}
    for name, field in (("old", "old_camera_tcp"), ("new", "new_camera_tcp")):
        values = [np.linalg.norm(record["raw_tcp"][:3, 3] - record[field][:3, 3]) * 1000 for record in accepted]
        differences[name] = dict(zip(("median", "p95", "max"), map(float, np.percentile(values, [50, 95, 100]))))
    return {
        "frames": len(records), "accepted_frames": len(accepted), "pose_coverage": len(accepted) / len(records),
        "device_time_duration_s": duration,
        "status_counts": dict(Counter(record["status"] for record in records)),
        "used_id_counts": dict(Counter(";".join(map(str, record["used_ids"])) for record in accepted)),
        "segment_count": max(record["segment"] for record in records),
        "timestamp_discontinuities": {
            "frame_indices": [record["frame_index"] for record in records
                              if record.get("timestamp_discontinuity")],
            "policy": "Repeated device timestamps retain their original frames and timestamps. Both pipelines reset and disconnect at the event; differential statistics exclude triples crossing it. No timestamps are synthesized.",
        },
        "static_windows": windows_summary,
        "all_frames_raw_to_display_distance_mm": differences,
        "motion_variation": motion_variation(records),
        "filter_time_constants_s": {
            "old_at_rest": 1 / (2 * math.pi * 2.0),
            "new_position_at_rest": 1 / (2 * math.pi * smoother.position_cutoff_hz),
            "new_rotation_at_rest": 1 / (2 * math.pi * smoother.rotation_cutoff_hz),
        },
        "lag_note": "Time constants describe filter response at rest, not measured end-to-end latency. Adaptive bandwidth increases during motion. Raw-to-display distance includes motion lag and measurement noise; no ground truth supplied.",
        "statistics_note": "RMS and radial P95 are deviations around each pipeline's mean in specified time windows, not absolute TCP accuracy. Stationarity is not independently confirmed; real motion contributes to these values and prevents interpreting them as static jitter or static repeatability.",
    }


def _xy_limits(records):
    points = [pose[:2, 3] for record in records for field in ("raw_relative", "old_relative", "new_relative") if (pose := record[field]) is not None]
    points.append(np.zeros(2))
    points = np.array(points)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    span = max(float(np.max(high - low)) * 1.2, 0.04)
    return center, span


def _draw_xy(image, records, index, field, limits):
    height, width = image.shape[:2]
    panel_width, panel_height = min(320, width // 3), min(260, height // 2)
    left, top = width - panel_width - 12, height - panel_height - 12
    cv2.rectangle(image, (left, top), (left + panel_width, top + panel_height), (22, 25, 30), -1)
    cv2.putText(image, "TCP XY / first pose frame", (left + 10, top + 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1, cv2.LINE_AA)
    center, span = limits
    plot_size = min(panel_width - 28, panel_height - 60)
    plot_center = np.array([left + panel_width / 2, top + panel_height / 2 + 10])

    def project(pose):
        point = (pose[:2, 3] - center) * plot_size / span
        return tuple(np.round(plot_center + point * [1, -1]).astype(int))

    start = max(0, index - 599)
    previous = None
    for record in records[start:index + 1]:
        pose = record[field]
        if pose is None or record["break_before"]:
            previous = None
        if pose is not None:
            current = project(pose)
            if previous is not None:
                cv2.line(image, previous, current, (80, 220, 120), 2, cv2.LINE_AA)
            previous = current
    if previous is not None:
        cv2.circle(image, previous, 4, (0, 240, 255), -1, cv2.LINE_AA)
    cv2.putText(image, f"Same XY span: {span * 1000:.1f} mm", (left + 10, top + panel_height - 12), cv2.FONT_HERSHEY_SIMPLEX, .43, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_history(image, intrinsics, history):
    segment = []
    for point in history:
        if point is None:
            if len(segment) > 1:
                draw_realsense_polyline(image, intrinsics, segment, (0, 230, 255), 2)
            segment = []
        else:
            segment.append(point)
    if len(segment) > 1:
        draw_realsense_polyline(image, intrinsics, segment, (0, 230, 255), 2)


def render_videos(session, output_dir, records, intrinsics, fps, marker_size_m):
    """Render saved images in exact CSV order, without detecting corners again."""
    capture = cv2.VideoCapture(str(Path(session) / "color_raw.mp4"))
    if not capture.isOpened():
        raise RuntimeError("Cannot open color_raw.mp4")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if declared_frames and declared_frames != len(records):
        capture.release()
        raise ValueError(f"Video/CSV frame count differs: video {declared_frames}, CSV {len(records)}")
    writers = []
    histories = [deque(maxlen=600), deque(maxlen=600)]
    limits = _xy_limits(records)
    try:
        for index, record in enumerate(records):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ends before CSV frame {record['frame_index']}")
            if not writers:
                height, width = frame.shape[:2]
                for name, size in (("before.mp4", (width, height)), ("after.mp4", (width, height)), ("side_by_side.mp4", (width * 2, height))):
                    writer = cv2.VideoWriter(str(Path(output_dir) / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                    writers.append(writer)
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create {name}")
            images = [frame.copy(), frame.copy()]
            for branch, (image, camera_field, relative_field, label) in enumerate((
                (images[0], "raw_tcp", "old_relative", "BEFORE: raw RGB TCP / legacy 3D filter"),
                (images[1], "new_camera_tcp", "new_relative", "AFTER: smooth reference then TCP"),
            )):
                pose = record[camera_field]
                if record["break_before"]:
                    histories[branch].append(None)
                histories[branch].append(pose[:3, 3].copy() if pose is not None else None)
                _draw_history(image, intrinsics, histories[branch])
                if pose is not None:
                    draw_realsense_frame(image, intrinsics, pose, marker_size_m)
                cv2.rectangle(image, (0, 0), (image.shape[1], 112), (20, 23, 29), -1)
                cv2.putText(image, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 2, cv2.LINE_AA)
                ids = ",".join(map(str, record["used_ids"])) or "-"
                cv2.putText(image, f"t={record['elapsed_s']:.2f}s  frame={record['frame_index']}  {record['status']}  IDs={ids}  segment={record['segment']}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, .52, (200, 220, 230), 1, cv2.LINE_AA)
                relative = record[relative_field]
                coordinates = ", ".join(f"{value * 1000:+.2f}" for value in relative[:3, 3]) if relative is not None else "unavailable"
                cv2.putText(image, f"3D display XYZ [mm]: {coordinates}", (16, 86), cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 240, 240), 1, cv2.LINE_AA)
                _draw_xy(image, records, index, relative_field, limits)
            writers[0].write(images[0])
            writers[1].write(images[1])
            writers[2].write(np.hstack(images))
        extra, _ = capture.read()
        if extra:
            raise ValueError("Video contains more frames than frames.csv")
    finally:
        capture.release()
        for writer in writers:
            writer.release()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--tcp-transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing-cutoff-hz", type=float, default=.8)
    parser.add_argument("--rotation-cutoff-hz", type=float, default=.5)
    parser.add_argument("--smoothing-motion-gain", type=float, default=8.0)
    parser.add_argument("--rotation-motion-gain", type=float, default=1.0)
    parser.add_argument("--max-error-px", type=float, default=2.5)
    parser.add_argument("--min-edge-px", type=float, default=24.0)
    parser.add_argument("--static-window", nargs=2, type=float, action="append", metavar=("START", "END"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    smoother = PoseSmoother(
        position_cutoff_hz=args.smoothing_cutoff_hz, rotation_cutoff_hz=args.rotation_cutoff_hz,
        motion_gain=args.smoothing_motion_gain, rotation_motion_gain=args.rotation_motion_gain,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    metadata, frames = read_observations(args.session)
    marker_size_m = float(metadata["aruco"]["marker_black_square_size_m"])
    rig_bytes = args.rig.read_bytes()
    tcp_bytes = args.tcp_transform.read_bytes()
    rig = load_rig(json.loads(rig_bytes), marker_size_m)
    tcp_transform = load_transform(args.tcp_transform)
    intrinsics = intrinsics_from_metadata(metadata["color_intrinsics"])
    tracker = RigTracker(rig, marker_size_m, args.max_error_px, args.min_edge_px)
    records = compare_observations(frames, tracker, intrinsics, tcp_transform, smoother)
    summary = summarize(records, args.static_window or [(3.0, 9.0), (23.0, 29.0)], smoother)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(records, args.output_dir / "comparison.csv")
    (args.output_dir / "rig_snapshot.json").write_bytes(rig_bytes)
    (args.output_dir / "tcp_transform_snapshot.json").write_bytes(tcp_bytes)
    render_videos(args.session, args.output_dir, records, intrinsics, float(metadata["stream"]["fps"]), marker_size_m)
    video_encoding = encode_comparison_videos(args.output_dir)
    summary.update({
        "session": str(args.session.resolve()), "rig": str(args.rig.resolve()), "tcp_transform": str(args.tcp_transform.resolve()),
        "rig_sha256": hashlib.sha256(rig_bytes).hexdigest(), "tcp_transform_sha256": hashlib.sha256(tcp_bytes).hexdigest(),
        "source_metadata": metadata,
        "pipelines": {
            "shared": "Saved corner observations, actual frame timestamps, one shared RigTracker solve, one first-accepted raw TCP origin; no MP4 redetection.",
            "before": "RGB axes/history use raw TCP. Main trajectory uses legacy RelativeTrace on raw TCP (cutoff=2+8*raw_speed).",
            "after": "Camera-frame reference pose is smoothed before fixed TCP composition. RGB and main trajectory use the same filtered TCP; trace internal filter is bypassed.",
        },
        "new_filter": {
            "position_cutoff_hz": smoother.position_cutoff_hz, "rotation_cutoff_hz": smoother.rotation_cutoff_hz,
            "motion_gain": smoother.motion_gain, "rotation_motion_gain": smoother.rotation_motion_gain,
            "velocity_cutoff_hz": smoother.velocity_cutoff_hz, "max_gap_s": smoother.max_gap_s,
        },
        "tracker": {"max_error_px": args.max_error_px, "min_edge_px": args.min_edge_px},
        "csv": "comparison.csv", "videos": ["before.mp4", "after.mp4", "side_by_side.mp4"],
        "video_encoding": video_encoding,
        "video_timing": {
            "configured_fps": float(metadata['stream']['fps']),
            "video_duration_s": len(records) / float(metadata['stream']['fps']),
            "device_duration_s": summary['device_time_duration_s'],
            "note": 'Video preserves source frame order and nominal playback FPS; filtering and statistics use actual device timestamps.',
        },
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "frames": summary["frames"], "pose_coverage": summary["pose_coverage"], "static_windows": summary["static_windows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

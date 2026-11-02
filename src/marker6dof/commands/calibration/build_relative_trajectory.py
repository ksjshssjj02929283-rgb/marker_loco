#!/usr/bin/env python3
"""Build a conservative relative gripper trajectory from recorded ArUco poses.

This is an initialization tool. It estimates fixed tag-to-reference transforms
from overlapping observations, rejects weak overlap graphs, then emits the
reference tag pose in its initial frame. If a measured tag-to-TCP transform is
supplied via --tcp-transform, it also emits the metric TCP pose in the camera
frame. It never invents a TCP transform.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def transform_from_row(row):
    rotation, _ = cv2.Rodrigues(np.array([float(row["rx_rad"]), float(row["ry_rad"]), float(row["rz_rad"])]))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(row["tx_m"]), float(row["ty_m"]), float(row["tz_m"])]
    return transform


def rotation_angle(rotation):
    return float(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec()))


def average_rotation(rotations):
    quaternions = Rotation.from_matrix(np.asarray(rotations)).as_quat()
    reference = quaternions[0]
    for index in range(len(quaternions)):
        if np.dot(quaternions[index], reference) < 0:
            quaternions[index] *= -1
    mean = np.mean(quaternions, axis=0)
    return Rotation.from_quat(mean / np.linalg.norm(mean)).as_matrix()


def robust_average(transforms, translation_threshold_m=0.015, rotation_threshold_deg=8.0):
    transforms = list(transforms)
    if not transforms:
        raise ValueError("cannot average an empty transform set")
    translation = np.median([item[:3, 3] for item in transforms], axis=0)
    rotation = average_rotation([item[:3, :3] for item in transforms])
    keep = []
    for index, item in enumerate(transforms):
        translation_error = float(np.linalg.norm(item[:3, 3] - translation))
        rotation_error = np.degrees(rotation_angle(rotation.T @ item[:3, :3]))
        if translation_error <= translation_threshold_m and rotation_error <= rotation_threshold_deg:
            keep.append(index)
    if len(keep) < max(3, int(np.ceil(len(transforms) * 0.5))):
        return None, keep
    selected = [transforms[index] for index in keep]
    result = np.eye(4)
    result[:3, 3] = np.median([item[:3, 3] for item in selected], axis=0)
    result[:3, :3] = average_rotation([item[:3, :3] for item in selected])
    return result, keep


def load_rows(path: Path):
    """Load per-frame tag poses, fusing left+right fisheye into the left frame.

    T265 sessions write one row per ``(frame, tag, camera)``. When a tag is seen
    in both eyes, its right-eye pose is mapped into the left-eye frame via the
    sibling ``metadata.json`` ``T_left_right`` and averaged with the left pose,
    so the calibration consumes both eyes instead of silently keeping the last
    row (right eye). D455 sessions carry no ``camera`` column and pass through.
    """
    raw_by_frame = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_by_frame[int(row["frame_index"])][int(row["tag_id"])].append(row)

    t_left_right = None
    metadata_path = path.with_name("metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidate = metadata.get("T_left_right")
        if candidate is not None:
            matrix = np.asarray(candidate, dtype=float)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                t_left_right = matrix

    by_frame = defaultdict(dict)
    for frame, tags in raw_by_frame.items():
        for tag_id, rows in tags.items():
            left = next((row for row in rows if row.get("camera") == "left"), None)
            right = next((row for row in rows if row.get("camera") == "right"), None)
            if left is None:
                by_frame[frame][tag_id] = right or rows[0]
                continue
            if right is None or t_left_right is None:
                by_frame[frame][tag_id] = left
                continue
            left_transform = transform_from_row(left)
            right_in_left = t_left_right @ transform_from_row(right)
            fused = dict(left)
            fused["camera"] = "stereo"
            fused_transform = np.eye(4)
            fused_transform[:3, :3] = average_rotation([left_transform[:3, :3], right_in_left[:3, :3]])
            fused_transform[:3, 3] = (left_transform[:3, 3] + right_in_left[:3, 3]) / 2.0
            rvec = cv2.Rodrigues(fused_transform[:3, :3])[0].reshape(3)
            fused["tx_m"], fused["ty_m"], fused["tz_m"] = fused_transform[:3, 3].tolist()
            fused["rx_rad"], fused["ry_rad"], fused["rz_rad"] = rvec.tolist()
            fused["reprojection_error_px"] = float(np.mean(
                [float(left["reprojection_error_px"]), float(right["reprojection_error_px"])]
            ))
            by_frame[frame][tag_id] = fused
    return by_frame


def resolve_marker_size_m(poses_path: Path, marker_size_mm):
    if marker_size_mm is not None:
        if marker_size_mm <= 0:
            raise ValueError("--marker-size-mm must be positive")
        return marker_size_mm / 1000.0
    metadata_path = poses_path.with_name("metadata.json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = metadata.get("aruco", {}).get("marker_black_square_size_m")
    return float(value) if value is not None else None


def load_tcp_transform(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    transform = np.asarray(data["T_reference_TCP"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid T_reference_TCP in {path}: must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
        raise ValueError(f"Non-rigid T_reference_TCP in {path}: rotation must be orthonormal with det>0 and last row [0,0,0,1]")
    return transform


def estimate_tag_graph(by_frame, min_overlap):
    observations = defaultdict(list)
    overlap_counts = defaultdict(int)
    for frame, tags in by_frame.items():
        for left in tags:
            observations[left].append(frame)
        ids = sorted(tags)
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                overlap_counts[(left, right)] += 1

    edges = defaultdict(dict)
    edge_report = {}
    for (left, right), count in overlap_counts.items():
        samples_left_to_right = []
        samples_right_to_left = []
        for frame, tags in by_frame.items():
            if left not in tags or right not in tags:
                continue
            t_camera_left = transform_from_row(tags[left])
            t_camera_right = transform_from_row(tags[right])
            samples_left_to_right.append(np.linalg.inv(t_camera_left) @ t_camera_right)
            samples_right_to_left.append(np.linalg.inv(t_camera_right) @ t_camera_left)
        if count < min_overlap:
            edge_report[f"{left}-{right}"] = {"overlap_frames": count, "accepted": False}
            continue
        left_to_right, inliers = robust_average(samples_left_to_right)
        if left_to_right is None:
            edge_report[f"{left}-{right}"] = {
                "overlap_frames": count, "accepted": False, "inlier_frames": len(inliers)
            }
            continue
        right_to_left = np.linalg.inv(left_to_right)
        edges[left][right] = left_to_right
        edges[right][left] = right_to_left
        edge_report[f"{left}-{right}"] = {
            "overlap_frames": count,
            "accepted": True,
            "inlier_frames": len(inliers),
            "translation_m": left_to_right[:3, 3].tolist(),
        }
    return edges, edge_report, {tag: len(frames) for tag, frames in observations.items()}


def reference_transforms(edges, reference):
    result = {reference: np.eye(4)}
    queue = deque([reference])
    while queue:
        current = queue.popleft()
        for neighbor, current_to_neighbor in edges.get(current, {}).items():
            if neighbor in result:
                continue
            # T_ref_neighbor = T_ref_current * T_current_neighbor.
            result[neighbor] = result[current] @ current_to_neighbor
            queue.append(neighbor)
    return result


def average_transform(transforms):
    """Average a list of ``T_ref_tag`` candidates (quaternion-mean rotation)."""
    result = np.eye(4)
    result[:3, :3] = average_rotation([item[:3, :3] for item in transforms])
    result[:3, 3] = np.median([item[:3, 3] for item in transforms], axis=0)
    return result


def refine_transforms(edges, initial, reference, iterations=100):
    """Globally refine tag transforms using every accepted overlap edge.

    BFS alone gives each direct neighbour of the reference a single-hop
    transform and discards loop-closure edges (e.g. 3-4, 4-5). Here each tag is
    repeatedly re-estimated from the mean of its neighbours' transforms composed
    with the measured edge, so redundant edges also constrain the result and
    error no longer rests on one weak direct edge.
    """
    transforms = {tag: transform.copy() for tag, transform in initial.items()}
    order = [tag for tag in transforms if tag != reference]
    for _ in range(iterations):
        for tag in order:
            candidates = [
                transforms[neighbor] @ np.linalg.inv(tag_to_neighbor)
                for neighbor, tag_to_neighbor in edges.get(tag, {}).items()
                if neighbor in transforms
            ]
            if candidates:
                transforms[tag] = average_transform(candidates)
    return transforms


def quaternion_xyzw(rotation):
    return Rotation.from_matrix(rotation).as_quat().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-id", type=int, default=2)
    parser.add_argument("--min-overlap", type=int, default=30)
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        help="Black-square edge used to produce the input poses; inferred from sibling metadata.json when omitted",
    )
    parser.add_argument(
        "--rig-output",
        type=Path,
        help="Optional JSON file containing fixed T_reference_tag transforms for live tracking",
    )
    parser.add_argument(
        "--tcp-transform",
        type=Path,
        help="Optional JSON with a measured T_reference_TCP 4x4 matrix, mapping TCP points into the reference tag frame",
    )
    parser.add_argument(
        "--tcp-output",
        type=Path,
        help="CSV to write the metric TCP pose (camera frame). Defaults to <output stem>_tcp.csv when --tcp-transform is given",
    )
    args = parser.parse_args()

    tcp_transform = load_tcp_transform(args.tcp_transform) if args.tcp_transform else None
    marker_size_m = resolve_marker_size_m(args.poses, args.marker_size_mm)
    if args.rig_output and marker_size_m is None:
        raise SystemExit(
            "rig output requires --marker-size-mm or a sibling metadata.json with marker size"
        )

    by_frame = load_rows(args.poses)
    edges, edge_report, detections = estimate_tag_graph(by_frame, args.min_overlap)
    transforms = reference_transforms(edges, args.reference_id)
    transforms = refine_transforms(edges, transforms, args.reference_id)
    observed_ids = set(detections)
    reachable_ids = set(transforms)
    connected_overlap_graph = observed_ids.issubset(reachable_ids)
    output_rows = []
    tcp_rows = []
    first_pose = None
    usable = 0
    warnings = 0

    for frame in sorted(by_frame):
        candidates = []
        for tag_id, row in by_frame[frame].items():
            if tag_id not in transforms:
                continue
            t_camera_tag = transform_from_row(row)
            t_reference_tag = transforms[tag_id]
            candidates.append((float(row["reprojection_error_px"]), t_camera_tag @ np.linalg.inv(t_reference_tag), tag_id))
        if not candidates:
            output_rows.append({"frame_index": frame, "status": "LOST", "visible_ids": ",".join(map(str, sorted(by_frame[frame])))});
            continue
        candidates.sort(key=lambda item: item[0])
        best_error, t_camera_reference, best_tag = candidates[0]
        translation_spread = max(float(np.linalg.norm(item[1][:3, 3] - t_camera_reference[:3, 3])) for item in candidates)
        rotation_spread = max(np.degrees(rotation_angle(t_camera_reference[:3, :3].T @ item[1][:3, :3])) for item in candidates)
        status = "GOOD" if translation_spread <= 0.015 and rotation_spread <= 8.0 else "CONSISTENCY_WARN"
        if status != "GOOD":
            warnings += 1
        if first_pose is None:
            first_pose = t_camera_reference.copy()
        relative = np.linalg.inv(first_pose) @ t_camera_reference
        q = quaternion_xyzw(relative[:3, :3])
        if tcp_transform is not None:
            t_camera_tcp = t_camera_reference @ tcp_transform
            tcp_q = quaternion_xyzw(t_camera_tcp[:3, :3])
            tcp_rows.append({
                "frame_index": frame,
                "status": status,
                "selected_tag_id": best_tag,
                "tx_m": t_camera_tcp[0, 3],
                "ty_m": t_camera_tcp[1, 3],
                "tz_m": t_camera_tcp[2, 3],
                "qx": tcp_q[0],
                "qy": tcp_q[1],
                "qz": tcp_q[2],
                "qw": tcp_q[3],
                "reprojection_error_px": best_error,
            })
        output_rows.append({
            "frame_index": frame,
            "status": status,
            "visible_ids": ",".join(map(str, sorted(by_frame[frame]))),
            "used_ids": ",".join(map(str, sorted(item[2] for item in candidates))),
            "selected_tag_id": best_tag,
            "translation_spread_m": translation_spread,
            "rotation_spread_deg": rotation_spread,
            "tx_m": relative[0, 3], "ty_m": relative[1, 3], "tz_m": relative[2, 3],
            "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
            "reprojection_error_px": best_error,
        })
        usable += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_index", "status", "visible_ids", "used_ids", "selected_tag_id", "translation_spread_m", "rotation_spread_deg", "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw", "reprojection_error_px"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    tcp_output = None
    if tcp_transform is not None:
        tcp_output = args.tcp_output or args.output.with_name(args.output.stem + "_tcp.csv")
        tcp_output.parent.mkdir(parents=True, exist_ok=True)
        tcp_fields = ["frame_index", "status", "selected_tag_id", "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw", "reprojection_error_px"]
        with tcp_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tcp_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tcp_rows)

    blockers = []
    if tcp_transform is None:
        blockers.append("No measured tag-to-gripper/TCP 3D geometry was supplied.")
    blockers.append("No independent ground-truth reference was supplied.")
    blockers.append("Use accepted chain edges; direct non-overlap pairs are not required.")
    report = {
        "reference_id": args.reference_id,
        "min_overlap": args.min_overlap,
        "detections_by_id": detections,
        "accepted_edges": edge_report,
        "reachable_ids": sorted(transforms),
        "total_frames": len(by_frame),
        "usable_relative_frames": usable,
        "consistency_warning_frames": warnings,
        "connected_overlap_graph": connected_overlap_graph,
        "relative_trajectory_ready": connected_overlap_graph and usable > 0,
        "metric_tcp_ready": connected_overlap_graph and usable > 0 and tcp_transform is not None,
        "high_reliability_ready": False,
        "high_reliability_blockers": blockers,
        "note": "Trajectory is relative to the first accepted reference pose; it is not TCP/world pose.",
    }
    if marker_size_m is not None:
        report["marker_size_m"] = marker_size_m
    if tcp_transform is not None:
        report["tcp_transform_source"] = str(args.tcp_transform)
        report["tcp_output"] = str(tcp_output)
        report["tcp_convention"] = "T_camera_TCP maps TCP points into the D455 color-camera frame (+X right, +Y down, +Z forward); quaternion is xyzw (scipy)."
    if args.rig_output:
        args.rig_output.parent.mkdir(parents=True, exist_ok=True)
        rig = {
            "reference_id": args.reference_id,
            "min_overlap": args.min_overlap,
            "marker_size_m": marker_size_m,
            "reachable_ids": sorted(transforms),
            "T_reference_tag": {str(tag_id): transform.tolist() for tag_id, transform in transforms.items()},
            "accepted_edges": edge_report,
            "coordinate_convention": "T_reference_tag maps points from each tag frame into the ID reference frame.",
            "source": str(args.poses),
        }
        metadata_path = args.poses.with_name("metadata.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rig["dictionary"] = metadata.get("aruco", {}).get("dictionary", "DICT_4X4_50")
        args.rig_output.write_text(json.dumps(rig, indent=2, ensure_ascii=False), encoding="utf-8")
        report["rig_calibration"] = str(args.rig_output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

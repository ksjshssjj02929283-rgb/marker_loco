"""Recover per-tag T265 poses from recorded fisheye videos with a chosen ArUco dictionary."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from marker6dof.realsense import (
    ARUCO_DICTIONARIES,
    detect_markers,
    detector_parameters,
    intrinsics_from_metadata,
    marker_object_points,
)
from marker6dof.t265 import solve_stereo_tag_pose, solve_tag_pose


FIELDS = [
    "frame_index", "timestamp_ms", "camera", "tag_id", "tx_m", "ty_m", "tz_m",
    "rx_rad", "ry_rad", "rz_rad", "reprojection_error_px",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dictionary", choices=ARUCO_DICTIONARIES, required=True)
    parser.add_argument("--marker-size-mm", type=float,
                        help="Measured black-square edge; defaults to source session metadata")
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    source = json.loads((args.session / "metadata.json").read_text(encoding="utf-8"))
    marker_size_m = (args.marker_size_mm / 1000.0 if args.marker_size_mm is not None
                     else float(source["aruco"]["marker_black_square_size_m"]))
    if marker_size_m <= 0:
        parser.error("marker size must be positive")
    accepted_ids = set(map(int, source["aruco"]["accepted_ids"]))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    parameters = detector_parameters()
    object_points = marker_object_points(marker_size_m)
    intrinsics = {name: intrinsics_from_metadata(source["fisheye_intrinsics"][name])
                  for name in ("left", "right")}
    T_left_right = np.asarray(source["T_left_right"], dtype=np.float64)

    timestamps = {}
    with (args.session / "t265_pose.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamps[int(row["frame_index"])] = row["timestamp_ms"]

    captures = {name: cv2.VideoCapture(str(args.session / f"fisheye_{name}.mp4"))
                for name in ("left", "right")}
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise RuntimeError("both T265 fisheye videos must be readable")

    args.output_dir.mkdir(parents=True)
    metadata = dict(source)
    metadata["aruco"] = dict(source["aruco"], dictionary=args.dictionary,
                             marker_black_square_size_m=marker_size_m)
    metadata["reprocessed_from"] = str(args.session.resolve())
    metadata["recorded_dictionary"] = source["aruco"]["dictionary"]
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts = Counter()
    stereo_counts = Counter()
    frame_index = 0
    try:
        with (args.output_dir / "poses.csv").open("w", newline="", encoding="utf-8") as stream, \
             (args.output_dir / "stereo_poses.csv").open("w", newline="", encoding="utf-8") as stereo_stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            stereo_writer = csv.DictWriter(stereo_stream, fieldnames=FIELDS)
            stereo_writer.writeheader()
            while True:
                frames = {name: capture.read() for name, capture in captures.items()}
                if not any(ok for ok, _ in frames.values()):
                    break
                if not all(ok for ok, _ in frames.values()):
                    raise RuntimeError(f"T265 fisheye videos end at different frames: {frame_index}")
                corners_by_tag = {}
                for name, (_, image) in frames.items():
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = detect_markers(gray, dictionary, parameters)
                    if ids is None:
                        continue
                    for corner, raw_id in zip(corners, ids.reshape(-1)):
                        tag_id = int(raw_id)
                        if tag_id not in accepted_ids:
                            continue
                        corners_by_tag.setdefault(tag_id, {})[name] = corner
                        try:
                            rvec, tvec, error = solve_tag_pose(
                                intrinsics[name], corner, object_points
                            )
                        except (RuntimeError, ValueError):
                            continue
                        writer.writerow({
                            "frame_index": frame_index,
                            "timestamp_ms": timestamps.get(frame_index, ""),
                            "camera": name,
                            "tag_id": tag_id,
                            "tx_m": f"{tvec[0]:.9f}", "ty_m": f"{tvec[1]:.9f}",
                            "tz_m": f"{tvec[2]:.9f}",
                            "rx_rad": f"{rvec[0]:.9f}", "ry_rad": f"{rvec[1]:.9f}",
                            "rz_rad": f"{rvec[2]:.9f}",
                            "reprojection_error_px": f"{error:.6f}",
                        })
                        counts[tag_id, name] += 1
                for tag_id, eyes in corners_by_tag.items():
                    if "left" not in eyes or "right" not in eyes:
                        continue
                    try:
                        rvec, tvec, error = solve_stereo_tag_pose(
                            intrinsics["left"], intrinsics["right"],
                            eyes["left"], eyes["right"], object_points,
                            T_left_right,
                        )
                    except (RuntimeError, ValueError):
                        continue
                    stereo_writer.writerow({
                        "frame_index": frame_index,
                        "timestamp_ms": timestamps.get(frame_index, ""),
                        "camera": "stereo", "tag_id": tag_id,
                        "tx_m": f"{tvec[0]:.9f}", "ty_m": f"{tvec[1]:.9f}",
                        "tz_m": f"{tvec[2]:.9f}",
                        "rx_rad": f"{rvec[0]:.9f}", "ry_rad": f"{rvec[1]:.9f}",
                        "rz_rad": f"{rvec[2]:.9f}",
                        "reprojection_error_px": f"{error:.6f}",
                    })
                    stereo_counts[tag_id] += 1
                frame_index += 1
    finally:
        for capture in captures.values():
            capture.release()

    summary = {
        "source_session": str(args.session.resolve()),
        "dictionary": args.dictionary,
        "marker_size_m": marker_size_m,
        "frames": frame_index,
        "detections_by_id_and_eye": {f"{tag_id}:{eye}": count
                                     for (tag_id, eye), count in sorted(counts.items())},
        "stereo_poses_by_id": {str(tag_id): count for tag_id, count in sorted(stereo_counts.items())},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

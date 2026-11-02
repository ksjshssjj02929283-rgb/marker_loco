"""Replay D455 color videos with factory and candidate RGB intrinsics.

The detector runs once per frame and both camera models solve the same corner
measurements.  This keeps the A/B comparison focused on calibration rather
than changing detection, frame selection, or marker geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from marker6dof.camera import load_kalibr_camera
from marker6dof.realsense import (
    detect_markers,
    detector_parameters,
    intrinsics_from_metadata,
    marker_object_points,
    normalized_image_points,
    project_realsense,
)


FIELDS = [
    "frame_index", "device_timestamp_ms", "host_timestamp_s", "tag_id",
    "tx_m", "ty_m", "tz_m", "rx_rad", "ry_rad", "rz_rad",
    "qx", "qy", "qz", "qw", "reprojection_error_px", "min_edge_px",
    *[f"corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")],
]


def _quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    q = np.empty(4, dtype=float)
    trace = float(np.trace(rotation))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (rotation[2, 1] - rotation[1, 2]) / s
        q[1] = (rotation[0, 2] - rotation[2, 0]) / s
        q[2] = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (1, 2) if i == 0 else ((2, 0) if i == 1 else (0, 1))
        s = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        q[i] = 0.25 * s
        q[3] = (rotation[k, j] - rotation[j, k]) / s
        q[j] = (rotation[j, i] + rotation[i, j]) / s
        q[k] = (rotation[k, i] + rotation[i, k]) / s
    return tuple(float(x) for x in q)


def _solve(camera, corners: np.ndarray, object_points: np.ndarray):
    normalized = normalized_image_points(camera, corners).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, normalized, np.eye(3), np.zeros((1, 5)), flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    points = (rotation @ object_points.T).T + tvec.reshape(1, 3)
    pixels = project_realsense(camera, points)
    delta = pixels - corners.reshape(4, 2)
    error = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    return rvec.reshape(3), tvec.reshape(3), rotation, error


def _write_model(path: Path, rows: list[dict], frames: int, model: str, errors: list[float], metadata: dict):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "poses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    source_frames = Path(metadata["_source_session"]) / "frames.csv"
    if source_frames.exists():
        shutil.copyfile(source_frames, path / "frames.csv")
    source_metadata = Path(metadata["_source_session"]) / "metadata.json"
    if source_metadata.exists():
        shutil.copyfile(source_metadata, path / "metadata.json")
    covered = len({int(row["frame_index"]) for row in rows})
    summary = {
        "session": metadata["_source_session"], "model": model,
        "frames": frames, "frames_with_target": covered,
        "frame_coverage": covered / frames if frames else 0.0,
        "detection_count": len(rows),
        "median_reprojection_error_px": float(np.median(errors)) if errors else None,
        "p95_reprojection_error_px": float(np.percentile(errors, 95)) if errors else None,
        "comparison_contract": "same decoded video, detector, marker size and frame index for factory/candidate",
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--camchain", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marker-size-mm", type=float)
    parser.add_argument("--ids", default="", help="comma-separated IDs; default uses metadata accepted_ids")
    args = parser.parse_args()
    metadata = json.loads((args.session / "metadata.json").read_text(encoding="utf-8"))
    video = args.session / "color_raw.mp4"
    if not video.exists():
        raise SystemExit(f"missing color_raw.mp4: {video}")
    recorded_size = float(metadata["aruco"]["marker_black_square_size_m"])
    marker_size = args.marker_size_mm / 1000.0 if args.marker_size_mm else recorded_size
    accepted = {int(x) for x in (args.ids.split(",") if args.ids else metadata["aruco"]["accepted_ids"]) if str(x).strip()}
    if not accepted or marker_size <= 0:
        raise SystemExit("marker size and IDs must be valid")
    factory = intrinsics_from_metadata(metadata["color_intrinsics"])
    candidate = load_kalibr_camera(args.camchain)
    if (candidate.width, candidate.height) != (factory.width, factory.height):
        raise SystemExit("candidate resolution does not match recorded D455 video")
    cameras = {"factory": factory, "candidate": candidate}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = detector_parameters()
    object_points = marker_object_points(marker_size).astype(np.float64)
    frame_times: dict[int, dict] = {}
    frames_csv = args.session / "frames.csv"
    if frames_csv.exists():
        for row in csv.DictReader(frames_csv.open(newline="", encoding="utf-8")):
            frame_times[int(row["frame_index"])] = row
    output_rows = defaultdict(list)
    output_errors = defaultdict(list)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray, dictionary, params)
        if ids is not None:
            for corner, tag_id in zip(corners, ids.reshape(-1)):
                tag = int(tag_id)
                if tag not in accepted:
                    continue
                points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
                frame = frame_times.get(frame_index, {})
                base = {"frame_index": frame_index, "device_timestamp_ms": frame.get("device_timestamp_ms", ""),
                        "host_timestamp_s": frame.get("host_timestamp_s", f"{frame_index / 30.0:.9f}"), "tag_id": tag}
                flat = points.reshape(-1)
                for i in range(4):
                    base[f"corner_{i}_x"], base[f"corner_{i}_y"] = f"{flat[2*i]:.4f}", f"{flat[2*i+1]:.4f}"
                base["min_edge_px"] = f"{np.min(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)):.4f}"
                for name, camera in cameras.items():
                    solved = _solve(camera, points, object_points)
                    if solved is None:
                        continue
                    rvec, tvec, rotation, error = solved
                    qx, qy, qz, qw = _quat(rotation)
                    row = dict(base)
                    row.update({"tx_m": f"{tvec[0]:.9f}", "ty_m": f"{tvec[1]:.9f}", "tz_m": f"{tvec[2]:.9f}",
                               "rx_rad": f"{rvec[0]:.9f}", "ry_rad": f"{rvec[1]:.9f}", "rz_rad": f"{rvec[2]:.9f}",
                               "qx": f"{qx:.9f}", "qy": f"{qy:.9f}", "qz": f"{qz:.9f}", "qw": f"{qw:.9f}",
                               "reprojection_error_px": f"{error:.6f}"})
                    output_rows[name].append(row)
                    output_errors[name].append(error)
        frame_index += 1
    capture.release()
    if not frame_index:
        raise SystemExit("video produced no frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata["_source_session"] = str(args.session.resolve())
    for name in cameras:
        _write_model(args.output_dir / name, output_rows[name], frame_index, name, output_errors[name], metadata)
    result = {"session": str(args.session.resolve()), "frames": frame_index, "accepted_ids": sorted(accepted),
              "marker_size_m": marker_size, "models": {name: json.loads((args.output_dir / name / "summary.json").read_text()) for name in cameras}}
    (args.output_dir / "ab_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: result["models"][name] for name in cameras}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

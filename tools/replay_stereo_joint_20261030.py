#!/usr/bin/env python3
"""Fit a common tag SE(3) to synchronized left and right detected corners.

Uses existing detector outputs, original image header timestamps, and the
independent Kalibr stereo extrinsics. No laser observations enter the fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from marker6dof.camera import load_kalibr_camera


ROOT = Path(__file__).resolve().parents[1]
CAMCHAIN = ROOT / "configs/calibration/uvc_20261030_camchain.yaml"
EXTRINSICS = ROOT / "configs/calibration/uvc_20261030_stereo_extrinsics.yaml"
SIZES_MM = {"marker50": 37.5, "marker60": 50.0, "marker70": 60.0}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corners(row: dict) -> np.ndarray:
    return np.array([[float(row[f"corner_{i}_{axis}"]) for axis in "xy"]
                     for i in range(4)], dtype=np.float64)


def object_corners(size_m: float) -> np.ndarray:
    h = size_m / 2
    # OpenCV ArUco corners: top-left, top-right, bottom-right, bottom-left.
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def transform(rotvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(rotvec)[0]
    matrix[:3, 3] = tvec
    return matrix


def solve_pair(left: dict, right: dict, cameras: dict, T_1_0: np.ndarray,
               points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    observed = {"left": corners(left), "right": corners(right)}
    normalized = {eye: cameras[eye].normalized_points(value)
                  for eye, value in observed.items()}
    candidates = []
    for eye, row in (("left", left), ("right", right)):
        rot = np.array([float(row[f"r{axis}_rad"]) for axis in "xyz"])
        pos = np.array([float(row[f"t{axis}_m"]) for axis in "xyz"])
        pose = transform(rot, pos)
        if eye == "right":
            pose = np.linalg.inv(T_1_0) @ pose
        candidates.append(np.r_[cv2.Rodrigues(pose[:3, :3])[0].ravel(), pose[:3, 3]])

    def normalized_residual(param: np.ndarray) -> np.ndarray:
        left_points = (cv2.Rodrigues(param[:3])[0] @ points.T).T + param[3:]
        right_points = (T_1_0[:3, :3] @ left_points.T).T + T_1_0[:3, 3]
        if min(left_points[:, 2].min(), right_points[:, 2].min()) <= 0:
            return np.full(16, 100.)
        return np.r_[(left_points[:, :2] / left_points[:, 2, None] - normalized["left"]).ravel(),
                     (right_points[:, :2] / right_points[:, 2, None] - normalized["right"]).ravel()]

    seed = min(candidates, key=lambda c: np.linalg.norm(normalized_residual(c)))
    fit = least_squares(normalized_residual, seed, method="lm", max_nfev=35)
    param = fit.x
    left_points = (cv2.Rodrigues(param[:3])[0] @ points.T).T + param[3:]
    right_points = (T_1_0[:3, :3] @ left_points.T).T + T_1_0[:3, 3]
    if min(left_points[:, 2].min(), right_points[:, 2].min()) <= 0:
        raise ValueError("negative depth")
    errors = [np.sqrt(np.mean(np.sum((cameras[eye].project_points(xyz) - observed[eye]) ** 2, axis=1)))
              for eye, xyz in (("left", left_points), ("right", right_points))]
    return param, float(np.mean(errors)), float(errors[0]), float(errors[1])


def replay_case(marker: str, name: str, left_root: Path, right_root: Path,
                output_root: Path, cameras: dict, T_1_0: np.ndarray,
                max_error_px: float) -> dict:
    left_dir, right_dir = left_root / marker / name, right_root / marker / name
    left_summary = json.loads((left_dir / "summary.json").read_text())
    right_summary = json.loads((right_dir / "summary.json").read_text())
    if left_summary["session"] != right_summary["session"]:
        raise ValueError(f"source sessions differ: {marker}/{name}")
    left_frames, right_frames = read_csv(left_dir / "frames.csv"), read_csv(right_dir / "frames.csv")
    right_timestamps = {r["header_timestamp_ns"] for r in right_frames}
    shared_frame_count = sum(r["header_timestamp_ns"] in right_timestamps for r in left_frames)
    right_rows = {(r["header_timestamp_ns"], r["tag_id"]): r
                  for r in read_csv(right_dir / "poses.csv")}
    left_rows = read_csv(left_dir / "poses.csv")
    points = object_corners(SIZES_MM[marker] / 1000)
    out = output_root / marker / name
    out.mkdir(parents=True, exist_ok=True)
    fields = ["frame_index", "host_timestamp_s", "timestamp_s", "bag_timestamp_ns",
              "header_timestamp_ns", "tag_id", "status", "tx_m", "ty_m", "tz_m",
              "rx_rad", "ry_rad", "rz_rad", "reprojection_error_px", "left_error_px", "right_error_px"]
    accepted, rejected = [], []
    for row in left_rows:
        other = right_rows.get((row["header_timestamp_ns"], row["tag_id"]))
        if other is None:
            continue
        try:
            pose, error, left_error, right_error = solve_pair(row, other, cameras, T_1_0, points)
        except (ValueError, cv2.error) as exc:
            rejected.append((row["frame_index"], str(exc)))
            continue
        if left_error > max_error_px or right_error > max_error_px:
            rejected.append((row["frame_index"], f"pixel error {left_error:.2f}/{right_error:.2f}"))
            continue
        result = {key: row[key] for key in fields[:6]}
        result.update(status="STEREO", **{f"t{axis}_m": pose[i + 3] for i, axis in enumerate("xyz")},
                      **{f"r{axis}_rad": pose[i] for i, axis in enumerate("xyz")},
                      reprojection_error_px=error, left_error_px=left_error, right_error_px=right_error)
        accepted.append(result)
    with (out / "poses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)
    with (out / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(left_frames[0]))
        writer.writeheader()
        writer.writerows(left_frames)
    summary = dict(session=left_summary["session"], camera_key="stereo", camchain=str(CAMCHAIN),
                   stereo_extrinsics=str(EXTRINSICS), marker_size_mm=SIZES_MM[marker],
                   frames=len(left_frames), frames_with_pose=len(accepted),
                   frame_coverage=len(accepted) / len(left_frames),
                   shared_header_timestamps=shared_frame_count,
                   unmatched_left_frames=len(left_frames) - shared_frame_count,
                   unmatched_right_frames=len(right_frames) - shared_frame_count,
                   max_error_px=max_error_px,
                   left_source=str(left_dir), right_source=str(right_dir),
                   left_pose_sha256=sha256(left_dir / "poses.csv"),
                   right_pose_sha256=sha256(right_dir / "poses.csv"),
                   camchain_sha256=sha256(CAMCHAIN), extrinsics_sha256=sha256(EXTRINSICS),
                   rejected_pairs=len(rejected), rejected_examples=rejected[:8],
                   reprojection_p50_px=float(np.median([r["reprojection_error_px"] for r in accepted])) if accepted else None,
                   reprojection_p95_px=float(np.percentile([r["reprojection_error_px"] for r in accepted], 95)) if accepted else None)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return dict(case=f"{marker}/{name}", pairs=len(accepted), coverage=summary["frame_coverage"],
                rejected=len(rejected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", type=Path, default=ROOT / "outputs/accuracy_20261030_kalibr/replay")
    parser.add_argument("--right-root", type=Path, default=ROOT / "outputs/accuracy_20261030_kalibr/replay_right")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-error-px", type=float, default=2.5)
    args = parser.parse_args()
    cameras = {"left": load_kalibr_camera(CAMCHAIN, "cam0"),
               "right": load_kalibr_camera(CAMCHAIN, "cam1")}
    calib = yaml.safe_load(EXTRINSICS.read_text())["T_1_0"]
    T_1_0 = np.eye(4)
    T_1_0[:3, :3] = Rotation.from_quat(calib["quaternion_xyzw"]).as_matrix()
    T_1_0[:3, 3] = calib["translation_m"]
    results = []
    for marker in SIZES_MM:
        for left_dir in sorted((args.left_root / marker).glob("*")):
            if not (left_dir.name.startswith("dynamic_track_") or left_dir.name == "continuous_static_20_100"):
                continue
            if marker == "marker60" and left_dir.name == "continuous_static_20_100":
                continue  # user-identified raw-data issue
            result = replay_case(marker, left_dir.name, args.left_root, args.right_root,
                                 args.output_root, cameras, T_1_0, args.max_error_px)
            results.append(result)
            print(json.dumps(result), flush=True)
    (args.output_root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

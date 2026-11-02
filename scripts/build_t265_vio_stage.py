"""Build time aligned T265 VIO world trajectories for the three task recordings."""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

import numpy as np

from marker6dof.commands.replay.replay_dual_t265 import FIELDS
from marker6dof.smoothing import PoseSmoother
from marker6dof.world import transform_from_translation_quaternion


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("01_open_lid_153305", "01_open_lid", "20261102_153305"),
    ("02_fold_item_153420", "02_fold_item", "20261102_153420"),
    ("03_tidy_table_153506", "03_tidy_table", "20261102_153506"),
)
MAX_BRACKET_MS = 80.0
MIN_CONFIDENCE = 2
POSITION_FIELDS = ("x_m", "y_m", "z_m")
QUAT_FIELDS = ("qx", "qy", "qz", "qw")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def matrix_from_row(row: dict) -> np.ndarray | None:
    if not row["x_m"]:
        return None
    return transform_from_translation_quaternion(
        [float(row[key]) for key in POSITION_FIELDS],
        [float(row[key]) for key in QUAT_FIELDS],
    )


def write_pose(row: dict, pose: np.ndarray | None, *, prefix: str = "") -> None:
    values = (None if pose is None else
              [*pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_quat()])
    for index, key in enumerate((*POSITION_FIELDS, *QUAT_FIELDS)):
        row[prefix + key] = "" if values is None else f"{values[index]:.9f}"


def interpolated_pose(samples: list[dict], stamps: list[float], camera_ms: float):
    following = bisect.bisect_left(stamps, camera_ms)
    if following == 0 or following == len(stamps):
        return None, "NO_BRACKET", None
    earlier, later = samples[following - 1], samples[following]
    span = stamps[following] - stamps[following - 1]
    if span <= 0 or span > MAX_BRACKET_MS:
        return None, "VIO_TIME_GAP", span
    if min(int(earlier["tracker_confidence"]), int(later["tracker_confidence"])) < MIN_CONFIDENCE:
        return None, "VIO_LOW_CONFIDENCE", span
    first = transform_from_translation_quaternion(
        [float(earlier[key]) for key in ("tx_m", "ty_m", "tz_m")],
        [float(earlier[key]) for key in QUAT_FIELDS],
    )
    second = transform_from_translation_quaternion(
        [float(later[key]) for key in ("tx_m", "ty_m", "tz_m")],
        [float(later[key]) for key in QUAT_FIELDS],
    )
    fraction = (camera_ms - stamps[following - 1]) / span
    world_from_pose = np.eye(4)
    world_from_pose[:3, 3] = first[:3, 3] * (1 - fraction) + second[:3, 3] * fraction
    world_from_pose[:3, :3] = Slerp(
        [0, 1], Rotation.from_matrix([first[:3, :3], second[:3, :3]])
    )([fraction]).as_matrix()[0]
    return world_from_pose, "OK", span


def build_task(name: str, task: str, stamp: str, T_pose_left: np.ndarray) -> dict:
    source = ROOT / "captures/t265_tasks" / task / stamp
    static = ROOT / "outputs/t265_tasks" / name
    output = ROOT / "outputs/t265_stages/03_stereo_vio" / name
    output.mkdir(parents=True, exist_ok=True)
    old_summary = json.loads((static / "summary.json").read_text(encoding="utf-8"))
    frames = old_summary["frames"]
    camera_rows = read_csv(source / "frames.csv")
    pose_rows = read_csv(source / "t265_pose.csv")
    stereo_rows = read_csv(static / "dual_center_poses.csv")
    if len(camera_rows) < frames or len(stereo_rows) != frames * 2:
        raise ValueError(f"source rows do not match {frames} frames for {name}")
    if any(int(row["frame_index"]) != index for index, row in enumerate(camera_rows[:frames])):
        raise ValueError(f"camera frames are not contiguous for {name}")
    sample_stamps = [float(row["pose_timestamp_ms"]) for row in pose_rows]
    if any(b <= a for a, b in zip(sample_stamps, sample_stamps[1:])):
        raise ValueError(f"VIO samples are not time ordered for {name}")
    by_frame = {int(row["frame_index"]): {} for row in stereo_rows}
    for row in stereo_rows:
        by_frame[int(row["frame_index"])][row["body"]] = row
    if any(set(value) != {"left", "right"} for value in by_frame.values()):
        raise ValueError(f"one body is missing for {name}")
    smoothers = {body: PoseSmoother() for body in ("left", "right")}
    previous_valid = {body: False for body in smoothers}
    previous_segment = {body: None for body in smoothers}
    segments = {body: 0 for body in smoothers}
    counts = {body: 0 for body in smoothers}
    stereo_counts = {body: 0 for body in smoothers}
    both = 0
    vio_ok = 0
    gaps = []
    origin_from_world = None
    ego_rows = []
    out_fields = [*FIELDS, "vio_status", "vio_bracket_ms"]
    with (output / "dual_center_poses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=out_fields)
        writer.writeheader()
        for index in range(frames):
            camera_ms = float(camera_rows[index]["camera_timestamp_ms"])
            world_from_pose, vio_status, span = interpolated_pose(pose_rows, sample_stamps, camera_ms)
            if world_from_pose is not None:
                world_from_left = world_from_pose @ T_pose_left
                if origin_from_world is None:
                    origin_from_world = np.linalg.inv(world_from_left)
                initial_from_left = origin_from_world @ world_from_left
                vio_ok += 1
                gaps.append(span)
            else:
                initial_from_left = None
            ego_row = {"frame_index": index, "time_s": by_frame[index]["left"]["time_s"],
                       "status": vio_status}
            write_pose(ego_row, initial_from_left)
            ego_rows.append(ego_row)
            valid_this_frame = 0
            for body in ("left", "right"):
                row = dict(by_frame[index][body])
                camera_from_center = matrix_from_row(row)
                initial_from_center = (None if initial_from_left is None or camera_from_center is None
                                       else initial_from_left @ camera_from_center)
                smooth = smoothers[body].update(initial_from_center, float(row["time_s"]))
                valid = initial_from_center is not None
                if valid:
                    if not previous_valid[body] or row["segment"] != previous_segment[body]:
                        segments[body] += 1
                    counts[body] += 1
                    stereo_counts[body] += int(row["status"] == "STEREO")
                    valid_this_frame += 1
                elif camera_from_center is not None:
                    row["status"] = "VIO_GAP"
                previous_valid[body] = valid
                previous_segment[body] = row["segment"]
                row["segment"] = segments[body]
                row["frame"] = "vio_initial_left_camera"
                row["vio_status"] = vio_status
                row["vio_bracket_ms"] = "" if span is None else f"{span:.3f}"
                write_pose(row, initial_from_center)
                write_pose(row, smooth, prefix="smooth_")
                writer.writerow(row)
            both += int(valid_this_frame == 2)
    with (output / "ego_pose.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_index", "time_s", "status",
                                                        *POSITION_FIELDS, *QUAT_FIELDS])
        writer.writeheader()
        writer.writerows(ego_rows)
    summary = dict(old_summary)
    summary.update({
        "coordinate_mode": "vio", "frame": "vio_initial_left_camera",
        "vio_interpolation_max_gap_ms": MAX_BRACKET_MS,
        "vio_min_tracker_confidence": MIN_CONFIDENCE,
        "vio_valid_frames": vio_ok,
        "vio_valid_fraction": vio_ok / frames,
        "vio_median_bracket_ms": float(np.median(gaps)) if gaps else None,
        "left_valid_frames": counts["left"], "right_valid_frames": counts["right"],
        "left_stereo_frames": stereo_counts["left"],
        "right_stereo_frames": stereo_counts["right"],
        "both_valid_frames": both,
        "left_valid_fraction": counts["left"] / frames,
        "right_valid_fraction": counts["right"] / frames,
        "both_valid_fraction": both / frames,
        "note": "Interpolated recorded T265 VIO to camera timestamps; poses are marker cube centers in the initial VIO left-camera frame, not TCP.",
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return {"task": name, "frames": frames, "vio_valid": vio_ok,
            "left_valid": counts["left"], "right_valid": counts["right"], "both_valid": both}


def main() -> None:
    extrinsics = json.loads((ROOT / "outputs/t265_stages/vio_extrinsics.json").read_text())
    T_pose_left = np.asarray(extrinsics["T_pose_left"], dtype=np.float64)
    for args in TASKS:
        print(json.dumps(build_task(*args, T_pose_left), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

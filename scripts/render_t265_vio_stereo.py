"""Render ego-compensated marker trails in both recorded T265 fisheye eyes."""

from __future__ import annotations

import csv
from collections import deque
import json
import os
from pathlib import Path

import cv2
import numpy as np

from marker6dof.commands.replay.replay_dual_t265 import _draw_frame_hud, _draw_recent_trail
from marker6dof.commands.tracking.live_dual_t265 import _draw_body_pose, _draw_detections, load_body
from marker6dof.realsense import detect_markers, detector_parameters, intrinsics_from_metadata
from marker6dof.world import transform_from_translation_quaternion


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("01_open_lid_153305", "01_open_lid", "20261102_153305"),
    ("02_fold_item_153420", "02_fold_item", "20261102_153420"),
    ("03_tidy_table_153506", "03_tidy_table", "20261102_153506"),
)


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pose_from_row(row: dict) -> np.ndarray | None:
    if not row["x_m"]:
        return None
    return transform_from_translation_quaternion(
        [float(row[f"{axis}_m"]) for axis in "xyz"],
        [float(row[f"q{axis}"]) for axis in "xyzw"],
    )


def render(name: str, task: str, stamp: str, bodies: dict) -> None:
    source = ROOT / "captures/t265_tasks" / task / stamp
    result = ROOT / "outputs/t265_stages/03_stereo_vio" / name
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    frames = int(summary["frames"])
    intrinsics = {eye: intrinsics_from_metadata(metadata["fisheye_intrinsics"][eye])
                  for eye in ("left", "right")}
    T_left_right = np.asarray(metadata["T_left_right"], dtype=float)
    T_right_left = np.linalg.inv(T_left_right)
    rows = csv_rows(result / "dual_center_poses.csv")
    ego_rows = csv_rows(result / "ego_pose.csv")
    if len(rows) != 2 * frames or len(ego_rows) != frames:
        raise ValueError(f"CSV frame count mismatch: {name}")
    captures = {eye: cv2.VideoCapture(str(source / f"fisheye_{eye}.mp4"))
                for eye in ("left", "right")}
    if not all(cap.isOpened() for cap in captures.values()):
        raise RuntimeError(f"cannot read both raw fisheye videos for {name}")
    temp = result / "annotated_vio_work.mp4"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(summary["annotated_video_fps"]), (1696, 800))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {temp}")
    recent = {body: deque(maxlen=180) for body in bodies}
    parameters = detector_parameters()
    try:
        for index in range(frames):
            read = {eye: cap.read() for eye, cap in captures.items()}
            if not all(ok for ok, _ in read.values()):
                raise RuntimeError(f"raw video ended at frame {index} for {name}")
            images = {eye: image for eye, (_, image) in read.items()}
            gray = {eye: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    for eye, image in images.items()}
            ego = pose_from_row(ego_rows[index])
            left_from_initial = None if ego is None else np.linalg.inv(ego)
            results = {}
            for body_name, body in bodies.items():
                row = rows[2 * index + (body_name == "right")]
                if row["body"] != body_name or int(row["frame_index"]) != index:
                    raise ValueError(f"body CSV order mismatch: {name} frame {index}")
                world_center = pose_from_row(row)
                if world_center is not None:
                    recent[body_name].append((row["segment"], world_center[:3, 3].copy()))
                for eye in ("left", "right"):
                    corners, ids, _ = detect_markers(gray[eye], body["dictionary"], parameters)
                    if ids is not None:
                        selected = [j for j, marker_id in enumerate(ids.ravel())
                                    if int(marker_id) in body["rig"]]
                        corners = [corners[j] for j in selected]
                        ids = ids[selected] if selected else None
                    _draw_detections(images[eye], corners, ids, body_name)
                    camera_from_initial = (left_from_initial if eye == "left" else
                                           None if left_from_initial is None else
                                           T_right_left @ left_from_initial)
                    if camera_from_initial is not None and world_center is not None:
                        _draw_body_pose(images[eye], intrinsics[eye],
                                        camera_from_initial @ world_center,
                                        body_name, body["marker_size_m"])
                results[body_name] = {"status": row["status"],
                                      "used_ids": [part for part in row["used_ids"].split(",") if part]}
            if left_from_initial is not None:
                for body_name in bodies:
                    for eye in ("left", "right"):
                        camera_from_initial = (left_from_initial if eye == "left" else
                                               T_right_left @ left_from_initial)
                        from marker6dof.commands.tracking.live_dual_t265 import COLORS
                        _draw_recent_trail(images[eye], intrinsics[eye], recent[body_name],
                                           camera_from_initial, COLORS[body_name])
            for eye in ("left", "right"):
                _draw_frame_hud(images[eye], eye, float(ego_rows[index]["time_s"]), results)
                cv2.putText(images[eye], "VIO WORLD", (690, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (130, 218, 241), 2, cv2.LINE_AA)
            writer.write(np.hstack((images["left"], images["right"])))
            if index % 200 == 0:
                print(f"{name}: {index + 1}/{frames}", flush=True)
    finally:
        writer.release()
        for capture in captures.values():
            capture.release()
    output = result / "annotated_dual.mp4"
    os.replace(temp, output)
    print(f"rendered {output}", flush=True)


def main() -> None:
    bodies = {
        "left": load_body(ROOT / "configs/local/t265_left_5x5_54p6_rig.json", "left"),
        "right": load_body(ROOT / "configs/local/t265_right_6x6_56_rig.json", "right"),
    }
    for args in TASKS:
        render(*args, bodies)


if __name__ == "__main__":
    main()

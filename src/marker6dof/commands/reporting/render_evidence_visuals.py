#!/usr/bin/env python3
"""Render evidence figures for the TCP and T265 documentation from recorded sessions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections import deque
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marker6dof.d455 import detect_markers, detector_parameters, draw_axes_realsense
from marker6dof.projection import draw_realsense_frame, draw_realsense_polyline, load_transform
from marker6dof.realsense import intrinsics_from_metadata
from marker6dof.tracking import RigTracker, load_rig

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcp-session", type=Path, required=True)
    parser.add_argument("--t265-session", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--tcp-transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_tcp_rows(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    segments: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] not in {"MULTI", "SINGLE"}:
                continue
            segments[int(row["segment"])].append(
                tuple(float(row[field]) * 1000 for field in ("position_x_m", "position_y_m", "position_z_m"))
            )
    return segments


def render_tcp_trajectory(session: Path, output: Path) -> None:
    segments = load_tcp_rows(session / "live_poses.csv")
    if not segments:
        raise RuntimeError("No accepted poses in TCP session")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=170, constrained_layout=True)
    projections = ((0, 1, "X", "Y", "Front / XY"), (0, 2, "X", "Z", "Top / XZ"), (1, 2, "Y", "Z", "Side / YZ"))
    for axis, (first, second, first_name, second_name, title) in zip(axes, projections):
        for index, (_, points) in enumerate(sorted(segments.items())):
            data = np.asarray(points)
            axis.plot(data[:, first], data[:, second], color=COLORS[index % len(COLORS)], lw=1.25, alpha=.82)
        initial = np.asarray(next(iter(sorted(segments.items())))[1][0])
        axis.scatter(initial[first], initial[second], marker="*", s=90, c="#111827", zorder=4, label="First accepted pose")
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(f"{first_name} / mm")
        axis.set_ylabel(f"{second_name} / mm")
        axis.grid(True, alpha=.25)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Relative TCP trajectory (recorded session; loss-separated segments)", fontsize=14, fontweight="bold")
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selected_tcp_frame(path: Path) -> int:
    best_index, best_distance = None, -1.0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] not in {"MULTI", "SINGLE"}:
                continue
            distance = sum(float(row[field]) ** 2 for field in ("position_x_m", "position_y_m", "position_z_m"))
            if distance > best_distance:
                best_index, best_distance = int(row["frame_index"]), distance
    if best_index is None:
        raise RuntimeError("No accepted TCP frames in session")
    return best_index


def render_tcp_projection(session: Path, rig_path: Path, transform_path: Path, output: Path) -> None:
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    marker_size_m = float(metadata["aruco"]["marker_black_square_size_m"])
    intrinsics = intrinsics_from_metadata(metadata["color_intrinsics"])
    rig = load_rig(json.loads(rig_path.read_text(encoding="utf-8")), marker_size_m)
    tcp_transform = load_transform(transform_path)
    capture = cv2.VideoCapture(str(session / "color_raw.mp4"))
    if not capture.isOpened():
        raise RuntimeError("Cannot open D455 color video")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    target = selected_tcp_frame(session / "live_poses.csv")
    tracker = RigTracker(rig, marker_size_m, max_error_px=2.5, min_edge_px=24.0)
    detector = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = detector_parameters()
    history: deque[np.ndarray] = deque(maxlen=180)
    current, result = None, None
    for frame_index in range(target + 1):
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read D455 frame {frame_index}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray, detector, parameters)
        detections = {} if ids is None else {int(tag): corner for corner, tag in zip(corners, ids.flatten())}
        result = tracker.solve(detections, intrinsics, frame_index / fps)
        if result.get("pose") is not None:
            history.append((result["pose"] @ tcp_transform)[:3, 3].copy())
        if frame_index == target:
            current = image
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(current, corners, ids)
    capture.release()
    if current is None or result is None or result.get("pose") is None:
        raise RuntimeError("Selected TCP frame no longer has an accepted pose")
    pose = result["pose"]
    tcp_pose = pose @ tcp_transform
    for tag_id in result["used_ids"]:
        tag_pose = pose @ rig[tag_id]
        rvec = cv2.Rodrigues(tag_pose[:3, :3])[0].reshape(3)
        draw_axes_realsense(current, intrinsics, rvec, tag_pose[:3, 3], marker_size_m * .52)
    if len(history) > 1:
        draw_realsense_polyline(current, intrinsics, list(history), (0, 230, 255), 3)
    draw_realsense_frame(current, intrinsics, tcp_pose, marker_size_m * 1.15, thickness=3)
    overlay = current.copy()
    cv2.rectangle(overlay, (16, 14), (current.shape[1] - 16, 112), (14, 20, 28), -1)
    cv2.addWeighted(overlay, .84, current, .16, 0, current)
    error = result.get("reprojection_error_px")
    error_text = "-" if error is None else f"{error:.2f}px"
    cv2.putText(current, "TCP PROJECTION | D455 RGB | RECORDED SESSION", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, .70, (245, 248, 250), 2, cv2.LINE_AA)
    cv2.putText(current, f"{result['status']}  used IDs: {','.join(map(str, result['used_ids'])) or '-'}  reprojection: {error_text}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, .50, (150, 235, 175), 1, cv2.LINE_AA)
    cv2.putText(current, "yellow: TCP camera-frame history | axes: TCP frame", (30, 99), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 230, 238), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), current, [cv2.IMWRITE_JPEG_QUALITY, 95])


def marker_ids(image: np.ndarray, detector: cv2.aruco.ArucoDetector) -> list[int]:
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return []
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return [int(value) for value in ids.flatten()]


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read video frame {frame_index}")
    return frame


def render_t265_keyframes(session: Path, output: Path) -> None:
    left = cv2.VideoCapture(str(session / "fisheye_left.mp4"))
    right = cv2.VideoCapture(str(session / "fisheye_right.mp4"))
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError("Cannot open T265 fisheye videos")
    fps = left.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(min(left.get(cv2.CAP_PROP_FRAME_COUNT), right.get(cv2.CAP_PROP_FRAME_COUNT)))
    positions = [int(frame_count * fraction) for fraction in (.15, .50, .85)]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    tiles = []
    for frame_index in positions:
        left_frame = read_frame(left, frame_index)
        right_frame = read_frame(right, frame_index)
        left_ids = marker_ids(left_frame, detector)
        right_ids = marker_ids(right_frame, detector)
        pair = np.hstack([left_frame, right_frame])
        cv2.rectangle(pair, (0, 0), (pair.shape[1], 43), (18, 24, 35), -1)
        stamp = f"t={frame_index / fps:05.1f}s  |  Left IDs: {left_ids or '-'}  |  Right IDs: {right_ids or '-'}"
        cv2.putText(pair, stamp, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, .68, (238, 244, 250), 2, cv2.LINE_AA)
        tiles.append(pair)
    left.release()
    right.release()
    montage = np.vstack(tiles)
    cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])


def render_t265_observations(session: Path, output: Path) -> None:
    totals: dict[str, Counter[int]] = defaultdict(Counter)
    errors: dict[str, list[float]] = defaultdict(list)
    with (session / "poses.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            camera, tag_id = row["camera"], int(row["tag_id"])
            totals[camera][tag_id] += 1
            errors[camera].append(float(row["reprojection_error_px"]))
    ids = sorted(set(totals["left"]) | set(totals["right"]))
    positions = np.arange(len(ids))
    fig, (bar, box) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=170, constrained_layout=True)
    bar.bar(positions - .18, [totals["left"][tag] for tag in ids], .36, label="Left fisheye", color="#2563eb")
    bar.bar(positions + .18, [totals["right"][tag] for tag in ids], .36, label="Right fisheye", color="#ea580c")
    bar.set_xticks(positions, [f"ID {tag}" for tag in ids])
    bar.set_ylabel("Detected monocular poses")
    bar.set_title("Marker observation coverage")
    bar.grid(axis="y", alpha=.25)
    bar.legend()
    box.boxplot([errors["left"], errors["right"]], tick_labels=["Left fisheye", "Right fisheye"], showfliers=False, patch_artist=True,
                boxprops={"facecolor": "#bfdbfe"}, medianprops={"color": "#b91c1c", "linewidth": 1.8})
    box.set_ylabel("Reprojection RMS / px")
    box.set_title("Per-eye reprojection error (outliers hidden)")
    box.grid(axis="y", alpha=.25)
    fig.suptitle("T265 stereo fisheye Mark observation quality (recorded session)", fontsize=14, fontweight="bold")
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_tcp_trajectory(args.tcp_session, args.output_dir / "tcp_relative_trajectory.png")
    render_tcp_projection(args.tcp_session, args.rig, args.tcp_transform, args.output_dir / "tcp_projection_evidence.jpg")
    render_t265_keyframes(args.t265_session, args.output_dir / "t265_fisheye_keyframes.jpg")
    render_t265_observations(args.t265_session, args.output_dir / "t265_observation_quality.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()

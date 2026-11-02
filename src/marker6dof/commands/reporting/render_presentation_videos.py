#!/usr/bin/env python3
"""Render compact, presentation-ready Marker/TCP projection videos from D455 RGB."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from marker6dof.d455 import detect_markers, detector_parameters, draw_axes_realsense
from marker6dof.projection import draw_realsense_frame, draw_realsense_polyline, load_transform
from marker6dof.realsense import intrinsics_from_metadata
from marker6dof.tracking import RigTracker, load_rig


def draw_trail(image, intrinsics, history, color, thickness=3):
    segment = []
    for point in history:
        if point is None:
            if len(segment) > 1:
                draw_realsense_polyline(image, intrinsics, segment, color, thickness)
            segment = []
        else:
            segment.append(point)
    if len(segment) > 1:
        draw_realsense_polyline(image, intrinsics, segment, color, thickness)


def alpha_panel(image, top, bottom, left=18, right=None, alpha=0.82):
    right = image.shape[1] - 18 if right is None else right
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (14, 20, 28), -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)


def put_text(image, text, xy, scale, color, thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--mode", choices=("marker", "tcp"), required=True)
    parser.add_argument("--tcp-transform", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trail-frames", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "tcp" and args.tcp_transform is None:
        parser.error("--tcp-transform is required in tcp mode")
    if args.trail_frames < 2:
        parser.error("--trail-frames must be at least 2")

    metadata = json.loads((args.session / "metadata.json").read_text(encoding="utf-8"))
    marker_size_m = float(metadata["aruco"]["marker_black_square_size_m"])
    intrinsics = intrinsics_from_metadata(metadata["color_intrinsics"])
    rig = load_rig(json.loads(args.rig.read_text(encoding="utf-8")), marker_size_m)
    tcp_transform = load_transform(args.tcp_transform) if args.tcp_transform else None
    tracker = RigTracker(rig, marker_size_m, max_error_px=2.5, min_edge_px=24.0)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = detector_parameters()

    capture = cv2.VideoCapture(str(args.session / "color_raw.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.session / 'color_raw.mp4'}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open {args.output}")

    reference_history = deque(maxlen=args.trail_frames)
    tcp_history = deque(maxlen=args.trail_frames)
    frame_count = accepted = detection_count = 0
    reprojection_errors = []

    while frame_count < total:
        ok, image = capture.read()
        if not ok:
            break
        frame_count += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray, dictionary, parameters)
        detections = {}
        if ids is not None:
            detection_count += len(ids)
            detections = {int(tag): corner for corner, tag in zip(corners, ids.flatten())}

        result = tracker.solve(detections, intrinsics, frame_count / fps)
        pose = result.get("pose")
        tcp_pose = None
        if pose is None:
            reference_history.append(None)
            tcp_history.append(None)
        else:
            accepted += 1
            reference_history.append(pose[:3, 3].copy())
            if tcp_transform is not None:
                tcp_pose = pose @ tcp_transform
                tcp_history.append(tcp_pose[:3, 3].copy())
            else:
                tcp_history.append(None)
            if result.get("reprojection_error_px") is not None:
                reprojection_errors.append(float(result["reprojection_error_px"]))
            for tag_id in result["used_ids"]:
                tag_pose = pose @ rig[tag_id]
                rvec = cv2.Rodrigues(tag_pose[:3, :3])[0].reshape(3)
                draw_axes_realsense(image, intrinsics, rvec, tag_pose[:3, 3], marker_size_m * 0.52)
            if tcp_pose is not None:
                draw_realsense_frame(image, intrinsics, tcp_pose, marker_size_m * 1.15, thickness=3)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        # OpenCV colors are BGR: this tuple renders as a readable orange trail.
        draw_trail(image, intrinsics, reference_history, (30, 150, 255), thickness=3)
        if args.mode == "tcp":
            draw_trail(image, intrinsics, tcp_history, (0, 235, 255), thickness=3)

        alpha_panel(image, 14, 102, right=width - 14)
        title = "MARKER TRAJECTORY | D455 RGB" if args.mode == "marker" else "TCP PROJECTION | D455 RGB"
        put_text(image, title, (30, 43), .72, (245, 248, 250), 2)
        used = ",".join(map(str, result["used_ids"])) or "-"
        error = result.get("reprojection_error_px")
        error_text = "-" if error is None else f"{error:.2f}px"
        put_text(image, f"{result['status']}   used IDs: {used}   reprojection: {error_text}", (30, 72), .48, (150, 235, 175), 1)
        if args.mode == "marker":
            legend = "orange trail: reference Marker origin   |   RGB axes: X red / Y green / Z blue"
        else:
            legend = "orange: reference trail   yellow: TCP trail   |   TCP axes: X red / Y green / Z blue"
        put_text(image, legend, (30, 94), .39, (222, 230, 238), 1)

        # A quiet bottom progress bar gives viewers temporal context without hiding the image.
        bar_y = height - 18
        cv2.rectangle(image, (24, bar_y), (width - 24, bar_y + 5), (55, 65, 78), -1)
        progress = frame_count / max(total, 1)
        cv2.rectangle(image, (24, bar_y), (24 + int((width - 48) * progress), bar_y + 5), (80, 220, 150), -1)
        put_text(image, f"{frame_count}/{total}  {frame_count / fps:05.1f}s", (width - 165, height - 28), .4, (215, 224, 232), 1)
        writer.write(image)

    capture.release()
    writer.release()
    summary = {
        "mode": args.mode,
        "source_video": str(args.session / "color_raw.mp4"),
        "output_video": str(args.output),
        "frames": frame_count,
        "fps": fps,
        "duration_s": frame_count / fps if fps else None,
        "accepted_pose_frames": accepted,
        "accepted_fraction": accepted / frame_count if frame_count else 0.0,
        "marker_detections": detection_count,
        "median_reprojection_error_px": float(np.median(reprojection_errors)) if reprojection_errors else None,
        "trail_frames": args.trail_frames,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

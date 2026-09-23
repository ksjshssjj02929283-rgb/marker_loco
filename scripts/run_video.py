#!/usr/bin/env python3
"""Run calibrated multi-marker TCP localization on a video or camera."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from marker6dof.localization import estimate_to_row, load_calibration, MarkerLocalizer


CSV_FIELDS = [
    "frame_index",
    "timestamp_s",
    "body",
    "status",
    "reason",
    "visible_ids",
    "used_ids",
    "corner_count",
    "inlier_count",
    "reprojection_rmse_px",
    "max_reprojection_error_px",
    "tx_m",
    "ty_m",
    "tz_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "runtime_ms",
]


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def draw_result(frame, detections, estimates, calibration):
    output = frame.copy()
    if detections:
        corners = [item.corners_px.astype(np.float32).reshape(1, 4, 2) for item in detections]
        ids = np.asarray([[item.tag_id] for item in detections], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(output, corners, ids)
    y = 28
    for estimate in estimates:
        color = (60, 210, 90) if estimate.status == "MEASURED" else (30, 70, 230)
        text = f"{estimate.body}: {estimate.status} ({estimate.reason}) ids={estimate.used_ids}"
        cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 25
        if estimate.T_camera_tcp is not None:
            rvec = cv2.Rodrigues(estimate.T_camera_tcp[:3, :3])[0]
            tvec = estimate.T_camera_tcp[:3, 3]
            cv2.drawFrameAxes(
                output,
                calibration.camera_matrix,
                calibration.distortion,
                rvec,
                tvec,
                0.06,
                2,
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video", required=True, help="Video path or camera index such as 0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-annotated-video", action="store_true")
    args = parser.parse_args()

    calibration = load_calibration(args.config)
    localizer = MarkerLocalizer(calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(parse_source(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != calibration.image_size:
        capture.release()
        raise ValueError(
            f"video size {(width, height)} does not match calibration {calibration.image_size}; "
            "do not resize images without transforming intrinsics"
        )

    writer = None
    annotated_path = args.output_dir / "annotated.avi"
    if not args.no_annotated_video:
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            calibration.image_size,
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"cannot create annotated video: {annotated_path}")

    rows = []
    frame_index = 0
    while args.max_frames is None or frame_index < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        start = time.perf_counter()
        detections, estimates, _ = localizer.process(frame)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        timestamp_s = frame_index / fps
        for estimate in estimates:
            row = estimate_to_row(estimate)
            row.update(frame_index=frame_index, timestamp_s=timestamp_s, runtime_ms=runtime_ms)
            rows.append(row)
        if writer is not None:
            writer.write(draw_result(frame, detections, estimates, calibration))
        frame_index += 1

    capture.release()
    if writer is not None:
        writer.release()
    if not rows:
        raise RuntimeError("video produced no frames")

    csv_path = args.output_dir / "tcp_poses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    measured = sum(row["status"] == "MEASURED" for row in rows)
    summary = {
        "input": str(args.video),
        "config": str(args.config),
        "frames": frame_index,
        "body_frame_records": len(rows),
        "measured_records": measured,
        "measured_fraction": measured / len(rows),
        "coordinate_convention": "T_camera_tcp maps TCP-frame points into the camera optical frame",
        "translation_unit": "m",
        "quaternion_order": "xyzw",
        "csv": str(csv_path),
        "annotated_video": "" if writer is None else str(annotated_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

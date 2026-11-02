#!/usr/bin/env python3
"""Generate a deterministic synthetic ArUco video and ground-truth poses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from marker6dof.localization import load_calibration, make_transform, matrix_to_quaternion_xyzw


def project(corners, rvec, tvec, calibration):
    return cv2.projectPoints(
        corners,
        rvec,
        tvec,
        calibration.camera_matrix,
        calibration.distortion,
    )[0].reshape(4, 2).astype(np.float32)


def paste_marker(image, marker, destination):
    height, width = marker.shape
    source = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        marker,
        homography,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=255,
    )
    mask_source = np.full(marker.shape, 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        mask_source,
        homography,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    image[mask > 0] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[mask > 0]


def render_frame(calibration, body, dictionary, frame_index, frame_count):
    width, height = calibration.image_size
    image = np.full((height, width, 3), 238, dtype=np.uint8)
    phase = 2.0 * np.pi * frame_index / max(frame_count, 1)
    rvec = np.asarray(
        [0.06 * np.sin(phase), 0.20 + 0.08 * np.cos(phase), 0.04 * np.sin(phase * 0.5)],
        dtype=np.float64,
    )
    tvec = np.asarray(
        [0.025 * np.sin(phase), 0.015 * np.cos(phase), 0.62 + 0.025 * np.sin(phase * 0.7)],
        dtype=np.float64,
    )

    projected_by_tag = {
        tag_id: project(corners, rvec, tvec, calibration)
        for tag_id, corners in body.tag_corners_gripper_m.items()
    }
    all_points = np.concatenate(list(projected_by_tag.values()), axis=0).astype(np.int32)
    hull = cv2.convexHull(all_points)
    cv2.fillConvexPoly(image, hull, (250, 250, 250), cv2.LINE_AA)
    for tag_id, destination in projected_by_tag.items():
        marker = np.zeros((240, 240), dtype=np.uint8)
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(
                dictionary, tag_id, 240, borderBits=1
            )
        else:
            cv2.aruco.drawMarker(dictionary, tag_id, 240, marker, borderBits=1)
        paste_marker(image, marker, destination)

    cv2.putText(
        image,
        "SYNTHETIC CALIBRATED MULTI-MARKER DEMO",
        (15, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    return image, make_transform(rvec, tvec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/demo_calibration.json"),
        help="Calibration JSON; pass an explicit path when running outside the checkout",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("example_output"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if args.frames <= 0 or args.fps <= 0:
        raise ValueError("frames and fps must be positive")

    calibration = load_calibration(args.config)
    if len(calibration.bodies) != 1:
        raise ValueError("demo generator expects exactly one configured body")
    body = next(iter(calibration.bodies.values()))
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, calibration.dictionary_name)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "demo_input.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        calibration.image_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create demo video: {video_path}")

    rows = []
    for frame_index in range(args.frames):
        image, T_camera_gripper = render_frame(
            calibration, body, dictionary, frame_index, args.frames
        )
        writer.write(image)
        T_camera_tcp = T_camera_gripper @ body.T_gripper_tcp
        quaternion = matrix_to_quaternion_xyzw(T_camera_tcp[:3, :3])
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index / args.fps,
                "tx_m": T_camera_tcp[0, 3],
                "ty_m": T_camera_tcp[1, 3],
                "tz_m": T_camera_tcp[2, 3],
                "qx": quaternion[0],
                "qy": quaternion[1],
                "qz": quaternion[2],
                "qw": quaternion[3],
            }
        )
    writer.release()

    with (args.output_dir / "ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)
    print(f"generated {args.frames} frames: {video_path}")


if __name__ == "__main__":
    main()

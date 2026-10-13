#!/usr/bin/env python3
"""D455 multi-face ArUco relative 6DoF tracking, one JSON object per frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from marker6dof.realsense import detect_markers, detector_parameters
from marker6dof.tracking import RelativeTrace, RigTracker, load_rig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--marker-size-mm", type=float, default=48.0)
    parser.add_argument("--serial", help="D455 serial; omit when only one is connected")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-error-px", type=float, default=2.5)
    parser.add_argument("--min-edge-px", type=float, default=24.0)
    args = parser.parse_args()
    if not np.isfinite(args.marker_size_mm) or args.marker_size_mm <= 0:
        parser.error("--marker-size-mm must be finite and positive")
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("--width, --height and --fps must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError("Install D455 dependencies: pip install -e '.[d455]'") from error

    rig = load_rig(json.loads(args.rig.read_text(encoding="utf-8")), args.marker_size_mm / 1000.0)
    tracker = RigTracker(rig, args.marker_size_mm / 1000.0, args.max_error_px, args.min_edge_px)
    trace = RelativeTrace(limit=1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = detector_parameters()
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    index = 0
    try:
        while args.max_frames is None or index < args.max_frames:
            frame = pipeline.wait_for_frames().get_color_frame()
            if not frame:
                continue
            gray = cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray, dictionary, parameters)
            detections = {} if ids is None else {
                int(tag_id): corner for corner, tag_id in zip(corners, ids.flatten())
            }
            timestamp = time.monotonic()
            result = tracker.solve(detections, intrinsics, timestamp)
            row = trace.update(result, timestamp)
            row["frame_index"] = index
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            index += 1
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()

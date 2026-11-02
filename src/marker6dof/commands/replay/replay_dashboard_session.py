#!/usr/bin/env python3
"""Serve a completed D455 dashboard session as a static, full-trace browser panel."""

from __future__ import annotations

import argparse
import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from marker6dof.tracking import RelativeTrace
from marker6dof.web import ASSET_ROOT


PAGE_PATH = ASSET_ROOT / "live_dashboard.html"
REVIEW_PAGE_PATH = ASSET_ROOT / "session_review.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path, help="Completed D455 dashboard recording directory")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def last_video_frame(path: Path) -> bytes:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frame = None
    while True:
        ok, candidate = capture.read()
        if not ok:
            break
        frame = candidate
    capture.release()
    if frame is None:
        raise RuntimeError(f"No frames in {path}")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Cannot encode final annotated frame")
    return encoded.tobytes()


def parse_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(";") if item]


def build_state(session: Path) -> dict:
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    trace_builder = RelativeTrace(limit=100000)
    last, last_data, frame_count = None, None, 0
    with (session / "live_poses.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame_index = int(row["frame_index"])
            frame_count = max(frame_count, frame_index + 1)
            timestamp = float(row["host_timestamp_s"])
            if not row["position_x_m"]:
                trace_builder.update({"status": row["status"]}, timestamp)
                continue
            point = [float(row[field]) for field in ("position_x_m", "position_y_m", "position_z_m")]
            quaternion = [float(row[field]) for field in ("quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")]
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            pose[:3, 3] = point
            last_data = trace_builder.update({"pose": pose, "status": row["status"]}, timestamp)
            trace_builder.trace[-1]["frame_index"] = frame_index
            last = row
    if last is None:
        raise RuntimeError("Session contains no accepted TCP poses")
    trace = list(trace_builder.trace)
    trace_start_s = trace[0]["t"]
    trace_end_s = trace[-1]["t"]
    return {
        "status": last["status"],
        "reason": "离线回放：已加载整段录制轨迹（保留原始分段）",
        "visible_ids": parse_ids(last["used_ids"]),
        "used_ids": parse_ids(last["used_ids"]),
        "rejected_ids": parse_ids(last["rejected_ids"]),
        "calibrated_ids": metadata["aruco"]["accepted_ids"],
        "position_m": last_data["position_m"],
        "smooth_position_m": last_data["smooth_position_m"],
        "quaternion_xyzw": last_data["quaternion_xyzw"],
        "smooth_quaternion_xyzw": last_data["smooth_quaternion_xyzw"],
        "reprojection_error_px": float(last["reprojection_error_px"]),
        "tcp_camera_position_m": None,
        "tcp_transform_loaded": True,
        "frame_rate_hz": metadata["stream"]["fps"],
        "frame_count": frame_count,
        "segment": trace[-1]["segment"],
        "trace": trace,
        "trace_start_s": trace_start_s,
        "trace_duration_s": trace_end_s - trace_start_s,
        "trajectory_name": "夹爪末端 TCP 相对轨迹（整段离线回放）",
        "camera": "D455 recorded session",
    }


def main() -> None:
    args = parse_args()
    if not PAGE_PATH.exists():
        raise FileNotFoundError(PAGE_PATH)
    if not REVIEW_PAGE_PATH.exists():
        raise FileNotFoundError(REVIEW_PAGE_PATH)
    state = build_state(args.session)
    page = PAGE_PATH.read_bytes()
    review_page = REVIEW_PAGE_PATH.read_bytes()
    image = last_video_frame(args.session / "annotated.mp4")
    video_path = args.session / "annotated.mp4"

    def serve_video(handler: BaseHTTPRequestHandler) -> None:
        size = video_path.stat().st_size
        start, end = 0, size - 1
        requested = handler.headers.get("Range")
        if requested and requested.startswith("bytes="):
            value = requested.removeprefix("bytes=").split("-", 1)
            start = int(value[0]) if value[0] else 0
            end = int(value[1]) if len(value) > 1 and value[1] else end
            start, end = max(0, start), min(size - 1, end)
            if start > end:
                handler.send_error(416)
                return
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            handler.send_response(200)
        handler.send_header("Content-Type", "video/mp4")
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(end - start + 1))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        with video_path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    handler.wfile.write(chunk)
                except BrokenPipeError:
                    return
                remaining -= len(chunk)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body, content = review_page, "text/html; charset=utf-8"
            elif path in ("/dashboard", "/dashboard/"):
                body, content = page, "text/html; charset=utf-8"
            elif path == "/state":
                body, content = json.dumps(state, separators=(",", ":"), allow_nan=False).encode(), "application/json"
            elif path == "/image.jpg":
                body, content = image, "image/jpeg"
            elif path == "/annotated.mp4":
                serve_video(self)
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/reset":
                self.send_error(404)
                return
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port} for synchronized video replay | full trace points: {len(state['trace'])}")
    print(f"Open http://127.0.0.1:{args.port}/dashboard for the static full-trace dashboard")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

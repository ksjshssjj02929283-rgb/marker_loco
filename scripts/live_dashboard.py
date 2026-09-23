#!/usr/bin/env python3
"""Live D455 multi-ArUco tracking with a segmented browser trajectory."""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from marker6dof.d455 import detect_markers, detector_parameters, draw_axes_realsense
from marker6dof.tracking import RelativeTrace, RigTracker, load_rig


PAGE_PATH = Path(__file__).with_name('live_dashboard.html')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig', type=Path, required=True, help='JSON from relative_trajectory.py --rig-output')
    parser.add_argument('--marker-size-mm', type=float, default=48.0)
    parser.add_argument('--serial', default='353522302024')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--trace-limit', type=int, default=1800)
    parser.add_argument('--max-error-px', type=float, default=2.5)
    parser.add_argument('--min-edge-px', type=float, default=24.0)
    parser.add_argument('--jpeg-quality', type=int, default=82)
    return parser.parse_args()


class SharedState:
    def __init__(self, calibrated_ids=()):
        self.lock = threading.Lock()
        self.image = b''
        self.reset_requested = False
        self.calibrated_ids = list(calibrated_ids)
        self.data = {
            'status': 'STARTING',
            'reason': '正在启动相机',
            'visible_ids': [],
            'used_ids': [],
            'rejected_ids': [],
            'calibrated_ids': self.calibrated_ids,
            'position_m': None,
            'smooth_position_m': None,
            'quaternion_xyzw': None,
            'smooth_quaternion_xyzw': None,
            'reprojection_error_px': None,
            'frame_rate_hz': 0.0,
            'segment': 0,
            'trace': [],
        }

    def publish(self, image, data, trace):
        payload = data.copy()
        payload['calibrated_ids'] = self.calibrated_ids
        payload['trace'] = list(trace)
        payload['trace_points'] = len(payload['trace'])
        payload['segment_count'] = max((point['segment'] for point in payload['trace']), default=0)
        with self.lock:
            self.image, self.data = image, payload

    def get(self):
        with self.lock:
            return self.image, self.data.copy()

    def request_reset(self):
        with self.lock:
            self.reset_requested = True

    def consume_reset(self):
        with self.lock:
            requested = self.reset_requested
            self.reset_requested = False
            return requested

    def fail(self, reason):
        with self.lock:
            self.data.update(
                status='ERROR', reason=reason, used_ids=[], position_m=None,
                smooth_position_m=None, quaternion_xyzw=None,
                smooth_quaternion_xyzw=None, reprojection_error_px=None,
            )


def overlay(image, data):
    colors = {
        'MULTI': (82, 205, 122),
        'SINGLE': (67, 184, 230),
        'WARN': (48, 154, 245),
        'LOST': (72, 76, 235),
    }
    width = image.shape[1]
    panel_width = min(780, width - 28)
    cv2.rectangle(image, (14, 12), (panel_width, 112), (17, 23, 31), -1)
    status = data['status']
    ids = ','.join(map(str, data['used_ids'])) or '-'
    cv2.putText(
        image, f'{status}  |  used IDs: {ids}', (26, 42), cv2.FONT_HERSHEY_SIMPLEX,
        .65, colors.get(status, (210, 210, 210)), 2, cv2.LINE_AA,
    )
    position = data.get('smooth_position_m')
    text = ('relative xyz [mm]: waiting for an accepted pose' if position is None else
            'relative xyz [mm]: ' + ', '.join(f'{value * 1000:+.1f}' for value in position))
    cv2.putText(image, text, (26, 72), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1, cv2.LINE_AA)
    error = data.get('reprojection_error_px')
    quality = 'reprojection: -' if error is None else f'reprojection: {error:.2f}px'
    cv2.putText(
        image, f"{quality}  |  rate: {data['frame_rate_hz']:.1f}Hz  |  segment: {data['segment']}",
        (26, 98), cv2.FONT_HERSHEY_SIMPLEX, .48, (205, 213, 221), 1, cv2.LINE_AA,
    )


def tracking(args, state, rig, stop_event):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = detector_parameters()
    tracker = RigTracker(rig, args.marker_size_mm / 1000, args.max_error_px, args.min_edge_px)
    trace = RelativeTrace(args.trace_limit)
    frame_times = deque(maxlen=max(2, args.fps))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    try:
        while not stop_event.is_set():
            frame = pipeline.wait_for_frames().get_color_frame()
            if not frame:
                continue
            now = time.monotonic()
            frame_times.append(now)
            frame_rate = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                          if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)
            if state.consume_reset():
                tracker.reset()
                trace.reset()

            image = np.asanyarray(frame.get_data()).copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray, dictionary, parameters)
            detections = {}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                detections = {int(tag_id): corner for corner, tag_id in zip(corners, ids.flatten())}

            result = tracker.solve(detections, intrinsics, now)
            accepted_pose = result.get('pose')
            if accepted_pose is not None:
                for tag_id in result['used_ids']:
                    tag_pose = accepted_pose @ rig[tag_id]
                    rvec = cv2.Rodrigues(tag_pose[:3, :3])[0].reshape(3)
                    draw_axes_realsense(
                        image, intrinsics, rvec, tag_pose[:3, 3], args.marker_size_mm / 2000,
                    )

            data = trace.update(result, now)
            data['frame_rate_hz'] = round(frame_rate, 1)
            overlay(image, data)
            encoded = cv2.imencode(
                '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, max(40, min(args.jpeg_quality, 95))],
            )[1].tobytes()
            state.publish(encoded, data, trace.trace)
    finally:
        pipeline.stop()


def tracking_guard(args, state, rig, stop_event):
    try:
        tracking(args, state, rig, stop_event)
    except Exception as error:  # Keep the HTTP diagnostics available if camera startup fails.
        state.fail(f'跟踪线程异常: {error}')
        traceback.print_exc()


def main():
    args = parse_args()
    if not PAGE_PATH.exists():
        raise FileNotFoundError(f'Dashboard page is missing: {PAGE_PATH}')
    page = {'bytes': PAGE_PATH.read_bytes(), 'mtime': PAGE_PATH.stat().st_mtime_ns}
    raw_rig = json.loads(args.rig.read_text(encoding='utf-8'))
    rig = load_rig(raw_rig, args.marker_size_mm / 1000)
    state = SharedState(sorted(rig))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            image, data = state.get()
            path = self.path.split('?', 1)[0]
            if path in ('/', '/index.html'):
                mtime = PAGE_PATH.stat().st_mtime_ns
                if mtime != page['mtime']:
                    page['bytes'] = PAGE_PATH.read_bytes()
                    page['mtime'] = mtime
                body, content = page['bytes'], 'text/html; charset=utf-8'
            elif path == '/state':
                body = json.dumps(data, separators=(',', ':'), allow_nan=False).encode()
                content = 'application/json'
            elif path == '/image.jpg' and image:
                body, content = image, 'image/jpeg'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.split('?', 1)[0] != '/reset':
                self.send_error(404)
                return
            state.request_reset()
            self.send_response(204)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    stop_event = threading.Event()
    thread = threading.Thread(target=tracking_guard, args=(args, state, rig, stop_event), daemon=True)
    thread.start()
    print(f'Open http://127.0.0.1:{args.port}  |  calibrated IDs: {sorted(rig)}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        stop_event.set()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Live D455 or T265 multi-ArUco tracking with a segmented browser trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from marker6dof.d455 import (
    detect_markers,
    detector_parameters,
    device_details,
    draw_axes_realsense,
    make_session_dir,
    write_metadata,
)
from marker6dof.projection import (
    draw_realsense_frame,
    draw_realsense_polyline,
    load_transform,
)
from marker6dof.realsense import ARUCO_DICTIONARIES, DEFAULT_MARKER_SIZE_MM
from marker6dof.t265 import ensure_t265_sdk, extrinsics_to_matrix, fuse_stereo_results
from marker6dof.tracking import RelativeTrace, RigTracker, load_rig
from marker6dof.smoothing import PoseSmoother
from marker6dof.world import WorldTrajectory, synchronized_t265_camera_pose
from marker6dof.web import ASSET_ROOT


PAGE_PATH = ASSET_ROOT / 'live_dashboard.html'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig', type=Path, required=True, help='JSON from relative_trajectory.py --rig-output')
    parser.add_argument('--camera', choices=('d455', 't265'), default='d455')
    parser.add_argument('--dictionary', choices=ARUCO_DICTIONARIES, default='DICT_4X4_50')
    parser.add_argument(
        '--marker-size-mm', type=float, default=DEFAULT_MARKER_SIZE_MM,
        help=f'Measured black-square outer edge (default: {DEFAULT_MARKER_SIZE_MM:g})',
    )
    parser.add_argument('--serial')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--trace-limit', type=int, default=1800)
    parser.add_argument('--max-error-px', type=float, default=2.5)
    parser.add_argument('--min-edge-px', type=float, default=24.0)
    parser.add_argument('--jpeg-quality', type=int, default=82)
    parser.add_argument(
        '--smoothing-cutoff-hz', type=float, default=1.0,
        help='Position smoothing at rest in Hz; lower values reduce jitter with more lag (default: 1)',
    )
    parser.add_argument(
        '--rotation-cutoff-hz', type=float, default=0.7,
        help='Rotation smoothing at rest in Hz, before TCP conversion (default: 0.7)',
    )
    parser.add_argument(
        '--smoothing-motion-gain', type=float, default=8.0,
        help='Increase position smoothing bandwidth with motion, in Hz per m/s (default: 8)',
    )
    parser.add_argument(
        '--rotation-motion-gain', type=float, default=1.0,
        help='Increase rotation smoothing bandwidth with motion, in Hz per rad/s (default: 1)',
    )
    parser.add_argument(
        '--world-trajectory', action='store_true',
        help='T265 only: combine time-aligned VIO pose with marker/TCP pose and publish '
             'synchronized ego and gripper trajectories in one VIO world frame',
    )
    parser.add_argument(
        '--vio-max-sync-ms', type=float, default=20.0,
        help='T265 only: reject VIO pose samples farther than this from the fisheye timestamp',
    )
    parser.add_argument(
        '--min-vio-confidence', type=int, default=2,
        help='T265 only: minimum native tracker confidence (0-3) accepted for world trajectories',
    )
    parser.add_argument(
        '--world-output', type=Path,
        help='T265 only: new CSV path for synchronized VIO ego and world-frame gripper poses; '
             'requires --world-trajectory',
    )
    parser.add_argument(
        '--tcp-transform', '--cad-transform', dest='tcp_transform', type=Path,
        help='CAD JSON containing T_reference_TCP (TCP points -> reference marker frame)',
    )
    parser.add_argument(
        '--tcp-trace-limit', type=int, default=600,
        help='Number of camera-frame TCP/reference points drawn on the RGB image',
    )
    parser.add_argument(
        '--record-root', type=Path,
        help='When set, record a D455 session (color+depth, videos, joint poses) into a '
             'timestamped dir under this root while the dashboard keeps serving',
    )
    parser.add_argument(
        '--no-raw', '--no-bag', dest='no_raw', action='store_true',
        help='Do not save the raw RealSense .db3 recording (videos and CSVs are still written)',
    )
    args = parser.parse_args()
    for name in ('smoothing_cutoff_hz', 'rotation_cutoff_hz'):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and greater than zero')
    for name in ('smoothing_motion_gain', 'rotation_motion_gain'):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and non-negative')
    return args


def _pose_smoother(args):
    return PoseSmoother(
        position_cutoff_hz=args.smoothing_cutoff_hz,
        rotation_cutoff_hz=args.rotation_cutoff_hz,
        motion_gain=args.smoothing_motion_gain,
        rotation_motion_gain=args.rotation_motion_gain,
    )


def _display_poses(smoother, reference_pose, timestamp, tcp_transform):
    """Compose both displayed frames from one smoothed reference pose.

    The raw TCP pose remains available for recording and raw observations.
    """
    display_reference = smoother.update(reference_pose, timestamp)
    raw_tcp = reference_pose @ tcp_transform if reference_pose is not None and tcp_transform is not None else None
    display_tcp = display_reference @ tcp_transform if display_reference is not None and tcp_transform is not None else None
    return display_reference, raw_tcp, display_tcp


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
            'tcp_camera_position_m': None,
            'smooth_tcp_camera_position_m': None,
            'world_trajectory_enabled': False,
            'vio_status': None,
            'vio_tracker_confidence': None,
            'ego_position_m': None,
            'ego_quaternion_xyzw': None,
            'ego_trace': [],
            'tcp_transform_loaded': False,
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
                tcp_camera_position_m=None,
                smooth_tcp_camera_position_m=None,
                vio_status=None, vio_tracker_confidence=None,
                ego_position_m=None, ego_quaternion_xyzw=None, ego_trace=[],
            )


def overlay(image, data):
    colors = {
        'MULTI': (82, 205, 122),
        'SINGLE': (67, 184, 230),
        'WARN': (48, 154, 245),
        'LOST': (72, 76, 235),
        'STEREO': (194, 126, 255),
    }
    width = image.shape[1]
    panel_width = min(780, width - 28)
    cv2.rectangle(image, (14, 12), (panel_width, 138), (17, 23, 31), -1)
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
        image, f"{quality}  |  segment: {data['segment']}",
        (26, 98), cv2.FONT_HERSHEY_SIMPLEX, .48, (205, 213, 221), 1, cv2.LINE_AA,
    )
    tcp = 'CAD TCP: on' if data.get('tcp_transform_loaded') else 'CAD TCP: off'
    cv2.putText(
        image, tcp, (26, 124), cv2.FONT_HERSHEY_SIMPLEX,
        .46, (205, 213, 221), 1, cv2.LINE_AA,
    )


LIVE_POSE_FIELDS = (
    "frame_index",
    "device_timestamp_ms",
    "host_timestamp_s",
    "status",
    "used_ids",
    "rejected_ids",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
    "reprojection_error_px",
    "segment",
)

WORLD_POSE_FIELDS = (
    'frame_index', 'fisheye_timestamp_ms', 'host_timestamp_s',
    'vio_timestamp_ms', 'vio_tracker_confidence', 'vio_status', 'marker_status',
    'ego_x_m', 'ego_y_m', 'ego_z_m', 'ego_qx', 'ego_qy', 'ego_qz', 'ego_qw',
    'gripper_x_m', 'gripper_y_m', 'gripper_z_m',
    'gripper_qx', 'gripper_qy', 'gripper_qz', 'gripper_qw',
)


class _WorldPoseWriter:
    """Write synchronized T265 VIO and marker trajectories without overwriting data."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=WORLD_POSE_FIELDS)
        self.writer.writeheader()

    @staticmethod
    def _pose(row, prefix, payload):
        position, quaternion = payload.get('position_m'), payload.get('quaternion_xyzw')
        if position is not None:
            row.update({f'{prefix}_{axis}_m': f'{value:.9f}' for axis, value in zip('xyz', position)})
        if quaternion is not None:
            row.update({f'{prefix}_q{axis}': f'{value:.9f}' for axis, value in zip('xyzw', quaternion)})

    def write(self, frame_index, fisheye_timestamp_ms, sample, marker_status, ego, gripper):
        row = {
            'frame_index': frame_index,
            'fisheye_timestamp_ms': f'{fisheye_timestamp_ms:.3f}',
            'host_timestamp_s': f'{time.time():.9f}',
            'vio_timestamp_ms': '' if sample.timestamp_ms is None else f'{sample.timestamp_ms:.3f}',
            'vio_tracker_confidence': '' if sample.tracker_confidence is None else sample.tracker_confidence,
            'vio_status': sample.reason,
            'marker_status': marker_status,
        }
        self._pose(row, 'ego', ego)
        self._pose(row, 'gripper', gripper)
        self.writer.writerow(row)

    def close(self):
        self.file.close()


class _Recorder:
    """Persist a D455 live session while the dashboard keeps serving.

    Writes the same clean color stream the offline D455 replay consumes plus
    the annotated preview, per-frame joint poses and a summary.  The raw .db3
    itself is produced by ``config.enable_record_to_file`` on the pipeline.
    """

    def __init__(self, session: Path, fps: int, width: int, height: int):
        self.session = session
        self.fps = fps
        self.size = (width, height)
        self.frames = 0
        self.frames_with_pose = 0
        self.reprojection_errors_px = []
        self.color_writer = None
        self.annotated_writer = None
        self.pose_file = None
        self.frame_file = None

    def open(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.color_writer = cv2.VideoWriter(
            str(self.session / "color_raw.mp4"), fourcc, self.fps, self.size
        )
        if not self.color_writer.isOpened():
            raise RuntimeError("OpenCV could not create color_raw.mp4")
        self.annotated_writer = cv2.VideoWriter(
            str(self.session / "annotated.mp4"), fourcc, self.fps, self.size
        )
        if not self.annotated_writer.isOpened():
            raise RuntimeError("OpenCV could not create annotated.mp4")
        self.pose_file = (self.session / "live_poses.csv").open("w", newline="", encoding="utf-8")
        self.pose_writer = csv.DictWriter(self.pose_file, fieldnames=LIVE_POSE_FIELDS)
        self.pose_writer.writeheader()
        self.frame_file = (self.session / "frames.csv").open("w", newline="", encoding="utf-8")
        self.frame_writer = csv.writer(self.frame_file)
        self.frame_writer.writerow(
            ["frame_index", "device_timestamp_ms", "host_timestamp_s", "target_ids", "processing_ms"]
        )

    def write_color(self, image):
        self.color_writer.write(image)

    def write_annotated(self, image):
        self.annotated_writer.write(image)

    def write_pose(self, row, has_pose, reprojection_error):
        self.pose_writer.writerow(row)
        if has_pose:
            self.frames_with_pose += 1
        if reprojection_error is not None:
            self.reprojection_errors_px.append(reprojection_error)

    def write_frame(self, row):
        self.frame_writer.writerow(row)
        self.frames += 1

    def close(self, interrupted):
        for writer in (self.color_writer, self.annotated_writer):
            if writer is not None:
                writer.release()
        if self.pose_file is not None:
            self.pose_file.close()
        if self.frame_file is not None:
            self.frame_file.close()
        summary = {
            "frames": self.frames,
            "frames_with_pose": self.frames_with_pose,
            "pose_coverage": (self.frames_with_pose / self.frames) if self.frames else 0.0,
            "median_reprojection_error_px": (
                float(np.median(self.reprojection_errors_px)) if self.reprojection_errors_px else None
            ),
            "capture_duration_s": (self.frames / self.fps) if self.fps else 0.0,
            "interrupted": interrupted,
            "completed_local": datetime.now().astimezone().isoformat(),
        }
        (self.session / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def tracking(args, state, rig, stop_event):
    if args.camera == 't265':
        if args.record_root is not None:
            raise ValueError('--record-root is currently supported by the D455 backend only')
        return tracking_t265(args, state, rig, stop_event)
    import pyrealsense2 as rs

    if args.world_trajectory:
        raise ValueError('--world-trajectory currently requires the T265 VIO pose stream')
    if args.world_output is not None:
        raise ValueError('--world-output currently requires --camera t265 --world-trajectory')
    width, height = args.width or 1280, args.height or 720
    args.width, args.height = width, height
    recording = args.record_root is not None
    session = make_session_dir(args.record_root) if recording else None
    recorder = None
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    parameters = detector_parameters()
    tracker = RigTracker(rig, args.marker_size_mm / 1000, args.max_error_px, args.min_edge_px)
    trace = RelativeTrace(args.trace_limit)
    tcp_trace = RelativeTrace(args.trace_limit)
    frame_times = deque(maxlen=max(2, args.fps))
    tcp_transform = load_transform(args.tcp_transform) if args.tcp_transform else None
    pose_smoother = _pose_smoother(args)
    if args.tcp_trace_limit <= 1:
        raise ValueError('--tcp-trace-limit must be greater than one')
    reference_history = deque(maxlen=args.tcp_trace_limit)
    tcp_history = deque(maxlen=args.tcp_trace_limit)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)
    if recording:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)
        if not args.no_raw:
            config.enable_record_to_file(str((session / "raw_realsense.db3").resolve()))
    profile = pipeline.start(config)
    device = profile.get_device()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    if recording:
        depth_scale = device.first_depth_sensor().get_depth_scale()
        write_metadata(
            session / "metadata.json", args, device, intrinsics, depth_scale, set(rig),
            pose_api="cv2.solvePnPGeneric(SOLVEPNP_IPPE_SQUARE) + joint multi-face solvePnP",
        )
        recorder = _Recorder(session, args.fps, width, height)
        recorder.open()
        print(f'Recording to: {session.resolve()}')
    if tcp_transform is not None:
        state.data['tcp_transform_loaded'] = True
    try:
        previous_frame_number = None
        previous_timestamp = previous_domain = None
        frame_index = 0
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            frame = frames.get_color_frame()
            if not frame:
                continue
            if frame.get_frame_number() == previous_frame_number:
                continue
            previous_frame_number = frame.get_frame_number()
            now = time.monotonic()
            frame_started = time.perf_counter()
            timestamp = float(frame.get_timestamp()) / 1000.0
            device_timestamp_ms = frame.get_timestamp()
            host_timestamp = time.time()
            domain = str(frame.get_frame_timestamp_domain())
            if previous_timestamp is not None and (
                timestamp <= previous_timestamp or domain != previous_domain
                or timestamp - previous_timestamp > pose_smoother.max_gap_s
            ):
                tracker.reset()
                pose_smoother.reset()
                # Preserve the origin and break both RGB and 3D histories at discontinuities.
                trace.broken = True
                tcp_trace.broken = True
                reference_history.append(None)
                tcp_history.append(None)
            previous_timestamp, previous_domain = timestamp, domain
            frame_times.append(now)
            frame_rate = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                          if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)
            if state.consume_reset():
                tracker.reset()
                trace.reset()
                tcp_trace.reset()
                pose_smoother.reset()
                reference_history.clear()
                tcp_history.clear()

            image = np.asanyarray(frame.get_data()).copy()
            if recorder is not None:
                recorder.write_color(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray, dictionary, parameters)
            detections = {}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                detections = {int(tag_id): corner for corner, tag_id in zip(corners, ids.flatten())}

            result = tracker.solve(detections, intrinsics, timestamp)
            accepted_pose = result.get('pose')
            display_pose, tcp_pose, display_tcp_pose = _display_poses(
                pose_smoother, accepted_pose, timestamp, tcp_transform,
            )
            if accepted_pose is not None:
                reference_history.append(display_pose[:3, 3].copy())
                tcp_history.append(display_tcp_pose[:3, 3].copy() if display_tcp_pose is not None else None)
                for tag_id in result['used_ids']:
                    tag_pose = display_pose @ rig[tag_id]
                    rvec = cv2.Rodrigues(tag_pose[:3, :3])[0].reshape(3)
                    draw_axes_realsense(
                        image, intrinsics, rvec, tag_pose[:3, 3], args.marker_size_mm / 2000,
                    )
                if display_tcp_pose is not None:
                    draw_realsense_frame(image, intrinsics, display_tcp_pose, args.marker_size_mm / 1000)
            else:
                reference_history.append(None)
                tcp_history.append(None)
            # Camera-frame histories are projected with the actual D455 model;
            # a lost frame inserts a gap so reacquisition cannot draw a false line.
            for history, color in ((reference_history, (255, 180, 0)), (tcp_history, (0, 255, 255))):
                segment = []
                for point in history:
                    if point is None:
                        if len(segment) > 1:
                            draw_realsense_polyline(image, intrinsics, segment, color, 2)
                        segment = []
                    else:
                        segment.append(point)
                if len(segment) > 1:
                    draw_realsense_polyline(image, intrinsics, segment, color, 2)

            data = trace.update(dict(result), timestamp, display_pose=display_pose)
            if tcp_transform is not None:
                tcp_input = dict(result)
                tcp_input['pose'] = tcp_pose
                tcp_data = tcp_trace.update(tcp_input, timestamp, display_pose=display_tcp_pose)
                # With a CAD transform the dashboard's main coordinate readout and
                # browser trajectory describe the gripper TCP, each relative to its
                # own first accepted pose.
                for field in ('position_m', 'smooth_position_m', 'quaternion_xyzw',
                              'smooth_quaternion_xyzw', 'segment'):
                    data[field] = tcp_data[field]
                data['trajectory_name'] = '夹爪末端 TCP 相对轨迹'
            else:
                data['trajectory_name'] = '参考 Marker 相对轨迹'
            if recorder is not None:
                position = data.get('position_m')
                quaternion = data.get('quaternion_xyzw')
                has_pose = position is not None
                recorder.write_pose({
                    "frame_index": frame_index,
                    "device_timestamp_ms": f"{device_timestamp_ms:.3f}",
                    "host_timestamp_s": f"{host_timestamp:.9f}",
                    "status": data['status'],
                    "used_ids": ";".join(map(str, data['used_ids'])),
                    "rejected_ids": ";".join(map(str, data['rejected_ids'])),
                    "position_x_m": f"{position[0]:.9f}" if has_pose else "",
                    "position_y_m": f"{position[1]:.9f}" if has_pose else "",
                    "position_z_m": f"{position[2]:.9f}" if has_pose else "",
                    "quaternion_x": f"{quaternion[0]:.9f}" if has_pose else "",
                    "quaternion_y": f"{quaternion[1]:.9f}" if has_pose else "",
                    "quaternion_z": f"{quaternion[2]:.9f}" if has_pose else "",
                    "quaternion_w": f"{quaternion[3]:.9f}" if has_pose else "",
                    "reprojection_error_px": (
                        f"{data['reprojection_error_px']:.6f}"
                        if data['reprojection_error_px'] is not None else ""
                    ),
                    "segment": data['segment'],
                }, has_pose, data['reprojection_error_px'])
                processing_ms = (time.perf_counter() - frame_started) * 1000.0
                recorder.write_frame([
                    frame_index,
                    f"{device_timestamp_ms:.3f}",
                    f"{host_timestamp:.9f}",
                    ";".join(map(str, data['used_ids'])),
                    f"{processing_ms:.3f}",
                ])
            data['timestamp_domain'] = domain
            data['frame_rate_hz'] = round(frame_rate, 1)
            data['tcp_transform_loaded'] = tcp_transform is not None
            data['tcp_camera_position_m'] = (
                tcp_pose[:3, 3].tolist() if tcp_pose is not None else None
            )
            data['smooth_tcp_camera_position_m'] = (
                display_tcp_pose[:3, 3].tolist() if display_tcp_pose is not None else None
            )
            overlay(image, data)
            if recorder is not None:
                recorder.write_annotated(image)
            encoded = cv2.imencode(
                '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, max(40, min(args.jpeg_quality, 95))],
            )[1].tobytes()
            state.publish(encoded, data, tcp_trace.trace if tcp_transform is not None else trace.trace)
            frame_index += 1
    finally:
        try:
            if recorder is not None:
                recorder.close(stop_event.is_set())
        finally:
            pipeline.stop()


def tracking_guard(args, state, rig, stop_event):
    try:
        tracking(args, state, rig, stop_event)
    except Exception as error:  # Keep the HTTP diagnostics available if camera startup fails.
        state.fail(f'跟踪线程异常: {error}')
        traceback.print_exc()


def tracking_t265(args, state, rig, stop_event):
    """T265 backend: solve each synchronized fisheye and fuse in left-camera coordinates."""
    import pyrealsense2 as rs

    width, height = args.width or 848, args.height or 800
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    parameters = detector_parameters()
    left_tracker = RigTracker(rig, args.marker_size_mm / 1000, args.max_error_px, args.min_edge_px)
    right_tracker = RigTracker(rig, args.marker_size_mm / 1000, args.max_error_px, args.min_edge_px)
    trace = RelativeTrace(args.trace_limit)
    tcp_trace = RelativeTrace(args.trace_limit)
    world_trace = WorldTrajectory(args.trace_limit) if args.world_trajectory else None
    if args.world_output is not None and world_trace is None:
        raise ValueError('--world-output requires --world-trajectory')
    world_writer = None
    frame_times = deque(maxlen=max(2, args.fps))
    tcp_transform = load_transform(args.tcp_transform) if args.tcp_transform else None
    pose_smoother = _pose_smoother(args)
    world_pose_smoother = _pose_smoother(args) if world_trace is not None else None
    if args.tcp_trace_limit <= 1:
        raise ValueError('--tcp-trace-limit must be greater than one')
    if tcp_transform is not None:
        state.data['tcp_transform_loaded'] = True
    pipeline = rs.pipeline()
    ensure_t265_sdk(rs)
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.fisheye, 1, width, height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.fisheye, 2, width, height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.pose)
    try:
        profile = pipeline.start(config)
    except RuntimeError as error:
        raise RuntimeError(
            f'无法启动 T265（{error}）。请关闭其他 T265/ROS2 相机进程，并确认使用 2.50-2.53 SDK。'
        ) from error
    left_profile = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
    right_profile = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
    pose_profile = profile.get_stream(rs.stream.pose)
    left_intrinsics = left_profile.get_intrinsics()
    right_intrinsics = right_profile.get_intrinsics()
    T_left_right = extrinsics_to_matrix(right_profile.get_extrinsics_to(left_profile))
    # The VIO pose sensor frame is not assumed to be the left optical frame.
    # SDK stream extrinsics yield T_pose_left, so world poses compose correctly.
    T_pose_left = extrinsics_to_matrix(left_profile.get_extrinsics_to(pose_profile))
    if args.world_output is not None:
        world_writer = _WorldPoseWriter(args.world_output)
    try:
        last_fisheye_numbers = None
        frame_index = 0
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            left_frame, right_frame = frames.get_fisheye_frame(1), frames.get_fisheye_frame(2)
            if not left_frame or not right_frame:
                continue
            fisheye_numbers = (left_frame.get_frame_number(), right_frame.get_frame_number())
            if fisheye_numbers[0] != fisheye_numbers[1] or fisheye_numbers == last_fisheye_numbers:
                continue
            last_fisheye_numbers = fisheye_numbers
            now = time.monotonic()
            frame_times.append(now)
            frame_rate = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                          if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)
            if state.consume_reset():
                left_tracker.reset(); right_tracker.reset(); trace.reset(); tcp_trace.reset()
                pose_smoother.reset()
                if world_trace is not None:
                    world_trace.reset()
                    world_pose_smoother.reset()
            images = []
            results = []
            for frame, intrinsics, tracker, label in (
                (left_frame, left_intrinsics, left_tracker, 'left'),
                (right_frame, right_intrinsics, right_tracker, 'right'),
            ):
                gray = np.asanyarray(frame.get_data())
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                corners, ids, _ = detect_markers(gray, dictionary, parameters)
                detections = {}
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)
                    detections = {int(tag_id): corner for corner, tag_id in zip(corners, ids.flatten())}
                result = tracker.solve(detections, intrinsics, now)
                cv2.putText(image, f'T265 {label}', (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2, cv2.LINE_AA)
                images.append(image)
                results.append(result)
            result = fuse_stereo_results(results[0], results[1], T_left_right)
            accepted_pose = result.get('pose')
            display_pose, tcp_pose, display_tcp_pose = _display_poses(
                pose_smoother, accepted_pose, now, tcp_transform,
            )
            if display_pose is not None:
                for image, intrinsics, eye_from_left, eye_result in (
                    (images[0], left_intrinsics, np.eye(4), results[0]),
                    (images[1], right_intrinsics, np.linalg.inv(T_left_right), results[1]),
                ):
                    for tag_id in eye_result['used_ids']:
                        if tag_id not in result['used_ids']:
                            continue
                        tag_pose = eye_from_left @ display_pose @ rig[tag_id]
                        rvec = cv2.Rodrigues(tag_pose[:3, :3])[0].reshape(3)
                        draw_axes_realsense(
                            image, intrinsics, rvec, tag_pose[:3, 3], args.marker_size_mm / 2000,
                        )
            if display_tcp_pose is not None:
                # The fused pose lives in the left fisheye frame.  Draw the same
                # physical TCP axes in both views using the SDK stereo extrinsics.
                for image, intrinsics, pose in (
                    (images[0], left_intrinsics, display_tcp_pose),
                    (images[1], right_intrinsics, np.linalg.inv(T_left_right) @ display_tcp_pose),
                ):
                    rvec = cv2.Rodrigues(pose[:3, :3])[0].reshape(3)
                    draw_axes_realsense(
                        image, intrinsics, rvec, pose[:3, 3], args.marker_size_mm / 1000,
                    )
            data = trace.update(dict(result), now, display_pose=display_pose)
            display_trace = tcp_trace.trace if tcp_transform is not None else trace.trace
            if tcp_transform is not None:
                tcp_input = dict(result)
                tcp_input['pose'] = tcp_pose
                tcp_data = tcp_trace.update(tcp_input, now, display_pose=display_tcp_pose)
                for field in ('position_m', 'smooth_position_m', 'quaternion_xyzw',
                              'smooth_quaternion_xyzw', 'segment'):
                    data[field] = tcp_data[field]
                data['trajectory_name'] = '夹爪末端 TCP 相对轨迹'
            else:
                data['trajectory_name'] = '参考 Marker 相对轨迹'
            if world_trace is not None:
                sample = synchronized_t265_camera_pose(
                    frames,
                    float(left_frame.get_timestamp()),
                    T_pose_left,
                    max_delta_ms=args.vio_max_sync_ms,
                    min_tracker_confidence=args.min_vio_confidence,
                )
                # Without a CAD TCP transform the rigid reference marker is the
                # only observable gripper frame.  Its label remains explicit.
                tracked_object = tcp_pose if tcp_transform is not None else accepted_pose
                # Smooth in the world frame after synchronized VIO composition.
                # Combining today's VIO with a delayed camera-frame pose would
                # make a stationary world object move when the camera moves.
                world_reference = (
                    sample.T_world_camera @ accepted_pose
                    if sample.valid and accepted_pose is not None else None
                )
                display_world_reference = world_pose_smoother.update(world_reference, now)
                display_world_object = (
                    display_world_reference @ tcp_transform
                    if display_world_reference is not None and tcp_transform is not None
                    else display_world_reference
                )
                ego_data, object_data = world_trace.update(
                    now, sample.T_world_camera, tracked_object, object_quality=result['status'],
                    display_world_object=display_world_object,
                )
                for field in ('position_m', 'smooth_position_m', 'quaternion_xyzw',
                              'smooth_quaternion_xyzw', 'segment'):
                    data[field] = object_data[field]
                data.update(
                    world_trajectory_enabled=True,
                    vio_status=sample.reason,
                    vio_tracker_confidence=sample.tracker_confidence,
                    ego_position_m=ego_data['position_m'],
                    ego_quaternion_xyzw=ego_data['quaternion_xyzw'],
                    ego_trace=list(world_trace.ego.trace),
                )
                data['trajectory_name'] = (
                    'VIO 世界系夹爪 TCP 轨迹' if tcp_transform is not None
                    else 'VIO 世界系参考 Marker 轨迹（未提供 TCP 外参）'
                )
                display_trace = world_trace.object.trace
                if world_writer is not None:
                    world_writer.write(
                        frame_index, float(left_frame.get_timestamp()), sample, result['status'],
                        ego_data, object_data,
                    )
            data['frame_rate_hz'] = round(frame_rate, 1)
            data['camera'] = 'T265 stereo fisheye'
            data['tcp_transform_loaded'] = tcp_transform is not None
            data['tcp_camera_position_m'] = tcp_pose[:3, 3].tolist() if tcp_pose is not None else None
            data['smooth_tcp_camera_position_m'] = (
                display_tcp_pose[:3, 3].tolist() if display_tcp_pose is not None else None
            )
            preview = np.hstack(images)
            overlay(preview, data)
            encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, max(40, min(args.jpeg_quality, 95))])[1].tobytes()
            state.publish(encoded, data, display_trace)
            frame_index += 1
    finally:
        try:
            if world_writer is not None:
                world_writer.close()
        finally:
            pipeline.stop()


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
    recording = ' | recording to ' + str(args.record_root.resolve()) if args.record_root else ''
    print(f'Open http://127.0.0.1:{args.port}  |  calibrated IDs: {sorted(rig)}{recording}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        stop_event.set()
        if args.record_root is not None:
            # Block until the recorder finalizes the .db3; a second Ctrl-C here
            # must not abort the finalize and corrupt the raw recording.
            deadline = time.monotonic() + 30.0
            while thread.is_alive() and time.monotonic() < deadline:
                try:
                    thread.join(timeout=1.0)
                except KeyboardInterrupt:
                    print('  Finalizing the recording (.db3); keep waiting…')
            if thread.is_alive():
                print('  Warning: recorder did not stop in time; .db3 may be incomplete.')
        else:
            try:
                thread.join(timeout=2)
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()

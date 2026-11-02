"""Live dual-gripper T265 dashboard, assuming a stationary camera by default."""

from __future__ import annotations

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np

from marker6dof.projection import draw_realsense_frame
from marker6dof.realsense import ARUCO_DICTIONARIES, detect_markers, detector_parameters, project_realsense
from marker6dof.smoothing import PoseSmoother
from marker6dof.t265 import ensure_t265_sdk, extrinsics_to_matrix, fuse_stereo_results
from marker6dof.tracking import RelativeTrace, RigTracker, load_rig
from marker6dof.world import synchronized_t265_camera_pose
from marker6dof.web import ASSET_ROOT


PAGE = ASSET_ROOT / "live_dual_t265.html"
COLORS = {"left": (120, 225, 140), "right": (225, 115, 225)}


def load_body(path: Path, name: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    dictionary_name = data.get("dictionary")
    if dictionary_name not in ARUCO_DICTIONARIES:
        raise ValueError(f"{name} rig requires a supported dictionary, got {dictionary_name!r}")
    marker_size_m = float(data["marker_size_m"])
    rig = load_rig(data, marker_size_m)
    center = np.asarray(data.get("cube_center_reference_m"), dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError(f"{name} rig requires a finite cube_center_reference_m [x, y, z]")
    return {
        "name": name,
        "dictionary_name": dictionary_name,
        "dictionary": cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name)),
        "marker_size_m": marker_size_m,
        "cube_center_reference_m": center,
        "rig": rig,
    }


def validate_bodies(left: dict, right: dict) -> None:
    if left["dictionary_name"] == right["dictionary_name"]:
        overlapping = set(left["rig"]) & set(right["rig"])
        if overlapping:
            raise ValueError(
                f"left and right rigs share dictionary and IDs {sorted(overlapping)}; "
                "their detections cannot be separated"
            )


class SharedState:
    def __init__(self, bodies: dict, ego_mode: str):
        self.lock = threading.Lock()
        self.reset_requested = False
        self.image = b""
        self.data = {
            "status": "STARTING", "reason": "正在启动 T265",
            "ego_mode": ego_mode,
            "frame": "static_left_camera" if ego_mode == "static" else "vio_world",
            "frame_rate_hz": 0.0, "vio_status": "waiting" if ego_mode == "vio" else None,
            "vio_confidence": None,
            "ego_trace": [], "left": {}, "right": {},
            "rigs": {name: {"dictionary": body["dictionary_name"],
                             "marker_size_mm": body["marker_size_m"] * 1000,
                             "cube_center_reference_mm": (body["cube_center_reference_m"] * 1000).tolist(),
                             "ids": sorted(body["rig"])} for name, body in bodies.items()},
        }

    def publish(self, image: bytes, data: dict) -> None:
        with self.lock:
            self.image, self.data = image, data

    def snapshot(self) -> tuple[bytes, dict]:
        with self.lock:
            return self.image, self.data.copy()

    def request_reset(self) -> None:
        with self.lock:
            self.reset_requested = True

    def consume_reset(self) -> bool:
        with self.lock:
            requested = self.reset_requested
            self.reset_requested = False
            return requested

    def fail(self, reason: str) -> None:
        with self.lock:
            self.data = dict(self.data, status="ERROR", reason=reason)


def _draw_detections(image, corners, ids, name):
    if ids is None:
        return {}
    detections = {}
    for corner, raw_id in zip(corners, ids.reshape(-1)):
        tag_id = int(raw_id)
        points = np.rint(corner.reshape(4, 2)).astype(np.int32)
        cv2.polylines(image, [points], True, COLORS[name], 2, cv2.LINE_AA)
        cv2.putText(image, f"{name[0].upper()}{tag_id}", tuple(points[0] + [0, -7]),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, COLORS[name], 2, cv2.LINE_AA)
        detections[tag_id] = corner
    return detections


def _draw_body_pose(image, intrinsics, pose, name, marker_size_m):
    if pose is None:
        return
    draw_realsense_frame(image, intrinsics, pose, marker_size_m * .65, 2)
    if pose[2, 3] > 0:
        pixel = project_realsense(intrinsics, pose[:3, 3].reshape(1, 3))[0]
        center = tuple(np.rint(pixel).astype(int))
        cv2.circle(image, center, 6, COLORS[name], -1, cv2.LINE_AA)
        cv2.putText(image, f"{name[0].upper()} CENTER", (center[0] + 9, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, COLORS[name], 2, cv2.LINE_AA)


def _nearest_pose_frame(recent_frames, camera_frame):
    """Select the closest buffered pose with the same RealSense timestamp domain."""
    domain = camera_frame.get_frame_timestamp_domain()
    candidates = [frame for frame in recent_frames
                  if frame.get_frame_timestamp_domain() == domain]
    if not candidates:
        return None
    camera_time = float(camera_frame.get_timestamp())
    return min(candidates, key=lambda frame: abs(float(frame.get_timestamp()) - camera_time))


def _display_trace(trace):
    """Send only the fields used by the live trajectory canvas."""
    return [{"smooth": point["smooth"], "segment": point["segment"]}
            for point in trace]


def _cube_center_pose(reference_pose, center_reference_m):
    """Move a solved ID-2 reference pose to the fixed cube-center point."""
    if reference_pose is None:
        return None
    pose = reference_pose.copy()
    pose[:3, 3] += pose[:3, :3] @ center_reference_m
    return pose


def track(args, bodies: dict, state: SharedState, stop: threading.Event) -> None:
    import pyrealsense2 as rs

    ensure_t265_sdk(rs)
    parameters = detector_parameters()
    trackers = {name: {eye: RigTracker(body["rig"], body["marker_size_m"],
                                      args.max_error_px, args.min_edge_px)
                       for eye in ("left", "right")}
                for name, body in bodies.items()}
    traces = {name: RelativeTrace(args.trace_limit) for name in bodies}
    smoothers = {name: PoseSmoother() for name in bodies}
    ego_trace = RelativeTrace(args.trace_limit)
    origin_from_world = None
    static_origin = np.eye(4)
    frame_times = deque(maxlen=30)

    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
    config.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)
    if args.ego_mode == "vio":
        config.enable_stream(rs.stream.pose)
    profile = pipeline.start(config)
    left_profile = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
    right_profile = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
    pose_profile = profile.get_stream(rs.stream.pose) if args.ego_mode == "vio" else None
    intrinsics = {"left": left_profile.get_intrinsics(), "right": right_profile.get_intrinsics()}
    T_left_right = extrinsics_to_matrix(right_profile.get_extrinsics_to(left_profile))
    T_right_left = np.linalg.inv(T_left_right)
    T_pose_left = (extrinsics_to_matrix(left_profile.get_extrinsics_to(pose_profile))
                   if pose_profile is not None else None)
    last_numbers = None
    recent_pose_frames = deque(maxlen=32)
    try:
        while not stop.is_set():
            frames = pipeline.wait_for_frames(5000)
            if args.ego_mode == "vio":
                pose_frame = frames.get_pose_frame()
                if pose_frame:
                    recent_pose_frames.append(pose_frame)
            cameras = {"left": frames.get_fisheye_frame(1), "right": frames.get_fisheye_frame(2)}
            if not all(cameras.values()):
                continue
            numbers = tuple(cameras[eye].get_frame_number() for eye in ("left", "right"))
            if numbers[0] != numbers[1] or numbers == last_numbers:
                continue
            last_numbers = numbers
            timestamp_s = float(cameras["left"].get_timestamp()) / 1000.0
            now = time.monotonic()
            frame_times.append(now)
            frame_rate = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                          if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0)
            if state.consume_reset():
                origin_from_world = None
                ego_trace.reset()
                for name in bodies:
                    traces[name].reset()
                    smoothers[name].reset()
                    for tracker in trackers[name].values():
                        tracker.reset()

            grays = {eye: np.asanyarray(frame.get_data()) for eye, frame in cameras.items()}
            images = {eye: cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                      for eye, gray in grays.items()}
            results = {}
            for name, body in bodies.items():
                eye_results = {}
                for eye in ("left", "right"):
                    corners, ids, _ = detect_markers(grays[eye], body["dictionary"], parameters)
                    detections = _draw_detections(images[eye], corners, ids, name)
                    eye_results[eye] = trackers[name][eye].solve(
                        detections, intrinsics[eye], timestamp_s,
                    )
                results[name] = fuse_stereo_results(
                    eye_results["left"], eye_results["right"], T_left_right,
                )
                pose = _cube_center_pose(results[name].get("pose"),
                                         body["cube_center_reference_m"])
                for eye in ("left", "right"):
                    eye_pose = pose if eye == "left" or pose is None else T_right_left @ pose
                    _draw_body_pose(images[eye], intrinsics[eye], eye_pose, name,
                                    body["marker_size_m"])

            sample = None
            if args.ego_mode == "vio":
                nearest_pose = _nearest_pose_frame(recent_pose_frames, cameras["left"])
                pose_source = SimpleNamespace(get_pose_frame=lambda: nearest_pose)
                sample = synchronized_t265_camera_pose(
                    pose_source, float(cameras["left"].get_timestamp()), T_pose_left,
                    max_delta_ms=args.vio_max_sync_ms,
                    min_tracker_confidence=args.min_vio_confidence,
                )
                if sample.valid and origin_from_world is None:
                    origin_from_world = np.linalg.inv(sample.T_world_camera)
                ego_input = {"status": "VIO", "pose": sample.T_world_camera if sample.valid else None}
                ego_data = ego_trace.update(ego_input, timestamp_s,
                                            initial_transform=origin_from_world)
                world_from_camera = sample.T_world_camera if sample.valid else None
                trace_origin = origin_from_world
            else:
                ego_data = None
                world_from_camera = static_origin
                trace_origin = static_origin
            body_data = {}
            for name, result in results.items():
                camera_pose = _cube_center_pose(result.get("pose"),
                                                bodies[name]["cube_center_reference_m"])
                world_pose = (world_from_camera @ camera_pose
                              if world_from_camera is not None and camera_pose is not None else None)
                display_pose = smoothers[name].update(world_pose, timestamp_s)
                trajectory_input = dict(result, pose=world_pose)
                data = traces[name].update(trajectory_input, timestamp_s,
                                           initial_transform=trace_origin,
                                           display_pose=display_pose)
                data["trace"] = _display_trace(traces[name].trace)
                data["camera_sources"] = result.get("camera_sources", [])
                body_data[name] = data

            for eye in ("left", "right"):
                cv2.putText(images[eye], f"T265 {eye.upper()}  |  L 5x5  R 6x6", (18, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, .68, (230, 235, 235), 2, cv2.LINE_AA)
            preview = np.hstack((images["left"], images["right"]))
            encoded = cv2.imencode(".jpg", preview,
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])[1].tobytes()
            state.publish(encoded, {
                "status": "RUNNING", "reason": "ok",
                "ego_mode": args.ego_mode,
                "frame": "static_left_camera" if args.ego_mode == "static" else "vio_world",
                "frame_rate_hz": round(frame_rate, 1),
                "vio_status": sample.reason if sample is not None else None,
                "vio_confidence": sample.tracker_confidence if sample is not None else None,
                "ego_position_m": ego_data["position_m"] if ego_data is not None else None,
                "ego_trace": _display_trace(ego_trace.trace) if ego_data is not None else [],
                "left": body_data["left"], "right": body_data["right"],
                "rigs": state.data["rigs"],
            })
    finally:
        pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-rig", type=Path, required=True)
    parser.add_argument("--right-rig", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--ego-mode", choices=("static", "vio"), default="static",
                        help="static: left camera is fixed (default); vio: use T265 pose stream")
    parser.add_argument("--max-error-px", type=float, default=3.5)
    parser.add_argument("--min-edge-px", type=float, default=24.0)
    parser.add_argument("--vio-max-sync-ms", type=float, default=20.0)
    parser.add_argument("--min-vio-confidence", type=int, default=2)
    parser.add_argument("--trace-limit", type=int, default=1200)
    parser.add_argument("--jpeg-quality", type=int, default=78)
    args = parser.parse_args()
    bodies = {"left": load_body(args.left_rig, "left"),
              "right": load_body(args.right_rig, "right")}
    validate_bodies(bodies["left"], bodies["right"])
    if not PAGE.exists():
        raise FileNotFoundError(PAGE)
    state = SharedState(bodies, args.ego_mode)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            image, data = state.snapshot()
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body, content = PAGE.read_bytes(), "text/html; charset=utf-8"
            elif path == "/state":
                body = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
                content = "application/json"
            elif path == "/image.jpg" and image:
                body, content = image, "image/jpeg"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/reset":
                self.send_error(404)
                return
            state.request_reset()
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_):
            pass

    def guarded_track():
        try:
            track(args, bodies, state, stop)
        except Exception as error:
            state.fail(f"T265 跟踪线程异常: {error}")
            traceback.print_exc()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    stop = threading.Event()
    worker = threading.Thread(target=guarded_track, daemon=True)
    worker.start()
    print(f"Open http://127.0.0.1:{args.port}  |  dual T265 L/R", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        stop.set()
        worker.join(timeout=6)


if __name__ == "__main__":
    main()

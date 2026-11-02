"""Calibrate a D455 RGB camera from ROS2 raw Image AprilGrid recordings.

This is a small, SDK-independent fallback for the official Kalibr workflow.
It consumes ROS2 sqlite bags, detects a 6x6 ``DICT_APRILTAG_36h11`` board,
fits an OpenCV pinhole+radtan model, and writes a Kalibr-compatible camchain
entry.  A deterministic held-out split and per-view MAD rejection are written
alongside the calibration so the result is auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class View:
    session: str
    frame_index: int
    bag_timestamp_ns: int
    corners: dict[int, np.ndarray]
    width: int
    height: int


def _target_points(cols: int, rows: int, tag_size: float, spacing: float) -> dict[int, np.ndarray]:
    """Return board-frame corners in the same order as OpenCV detections."""
    pitch = tag_size * (1.0 + spacing)
    result = {}
    # The printed board uses the IDs supplied by the AprilGrid generator in
    # row-major order. The camera pose absorbs the board's global orientation.
    for row in range(rows):
        for col in range(cols):
            tag = row * cols + col
            x, y = col * pitch, row * pitch
            # OpenCV's AprilTag detector starts at the upper-right corner and
            # walks counter-clockwise for this board orientation.
            result[tag] = np.array(
                [[x + tag_size, y, 0.0], [x, y, 0.0],
                 [x, y + tag_size, 0.0], [x + tag_size, y + tag_size, 0.0]],
                dtype=np.float64,
            )
    return result


def _detector(border_bits: int = 2):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # The Kalibr-style printed grid has a two-cell black border. Decoding it
    # as one cell can accept an inner adaptive-threshold contour. A small
    # subpixel window then leaves the corners inside the physical black edge,
    # falsely increasing the apparent tag spacing. Refine on the actual edge.
    params.markerBorderBits = border_bits
    params.cornerRefinementWinSize = 7
    params.relativeCornerRefinmentWinSize = 1.5
    params.cornerRefinementMaxIterations = 60
    params.cornerRefinementMinAccuracy = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 83
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 10.0
    params.polygonalApproxAccuracyRate = 0.02
    params.minCornerDistanceRate = 0.01
    params.minDistanceToBorder = 1
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = False
    return cv2.aruco.ArucoDetector(dictionary, params)


def _gray_image(msg) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    raw = np.asarray(msg.data, dtype=np.uint8)
    if encoding in {"rgb8", "bgr8"}:
        row = raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(row, cv2.COLOR_RGB2GRAY if encoding == "rgb8" else cv2.COLOR_BGR2GRAY)
    if encoding in {"mono8", "8uc1"}:
        return raw.reshape(height, step)[:, :width].copy()
    raise ValueError(f"unsupported image encoding: {encoding}")


def _read_bag(path: Path, topic: str, detector, target: dict[int, np.ndarray], stride: int, min_tags: int) -> tuple[list[View], dict]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError("install the rosbag extra: pip install -e '.[rosbag]'") from exc

    views: list[View] = []
    info = {"session": str(path), "frames": 0, "sampled_frames": 0, "detected_views": 0}
    with AnyReader([path], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f"no image topic {topic!r} in {path}")
        if any(c.msgtype != "sensor_msgs/msg/Image" for c in connections):
            raise ValueError(f"{path} contains a non-raw image topic")
        connection = connections[0]
        for frame_index, (_, bag_ns, raw) in enumerate(reader.messages(connections=[connection])):
            info["frames"] += 1
            if frame_index % stride:
                continue
            info["sampled_frames"] += 1
            msg = reader.deserialize(raw, connection.msgtype)
            gray = _gray_image(msg)
            corners, ids, _ = detector.detectMarkers(gray)
            selected: dict[int, np.ndarray] = {}
            if ids is not None:
                for corner, tag_id in zip(corners, ids.reshape(-1)):
                    tag = int(tag_id)
                    if tag in target and tag not in selected:
                        selected[tag] = np.asarray(corner, dtype=np.float64).reshape(4, 2)
            if len(selected) >= min_tags:
                views.append(View(str(path), frame_index, int(bag_ns), selected, int(msg.width), int(msg.height)))
                info["detected_views"] += 1
    return views, info


def _uniform(views: list[View], maximum: int) -> list[View]:
    if len(views) <= maximum:
        return views
    indices = np.unique(np.rint(np.linspace(0, len(views) - 1, maximum)).astype(int))
    return [views[int(i)] for i in indices]


def _arrays(views: list[View], target: dict[int, np.ndarray]):
    objects, images = [], []
    for view in views:
        tags = sorted(view.corners)
        objects.append(np.concatenate([target[tag] for tag in tags]).astype(np.float32))
        images.append(np.concatenate([view.corners[tag] for tag in tags]).astype(np.float32))
    return objects, images


def _fit(views: list[View], target: dict[int, np.ndarray], K0: np.ndarray, D0: np.ndarray):
    objects, images = _arrays(views, target)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objects, images, (views[0].width, views[0].height), K0.copy(), D0.copy(),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 150, 1e-10),
    )
    errors = []
    for obj, img, rvec, tvec in zip(objects, images, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        delta = projected.reshape(-1, 2) - img
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return float(rms), K, D.reshape(-1), errors


def _validate(views: list[View], target: dict[int, np.ndarray], K: np.ndarray, D: np.ndarray) -> list[float]:
    errors = []
    for view in views:
        tags = sorted(view.corners)
        obj, img = _arrays([view], target)
        ok, rvec, tvec = cv2.solvePnP(obj[0], img[0], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            errors.append(float("inf"))
            continue
        projected, _ = cv2.projectPoints(obj[0], rvec, tvec, K, D)
        delta = projected.reshape(-1, 2) - img[0]
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


def _stats(errors: list[float]) -> dict:
    finite = np.asarray([x for x in errors if np.isfinite(x)], dtype=float)
    if not finite.size:
        return {"count": 0, "p50_px": None, "p95_px": None, "max_px": None}
    return {"count": int(finite.size), "p50_px": float(np.percentile(finite, 50)),
            "p95_px": float(np.percentile(finite, 95)), "max_px": float(np.max(finite))}


def _write_yaml(path: Path, K: np.ndarray, D: np.ndarray, width: int, height: int, topic: str):
    values = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
    coeffs = [float(x) for x in D[:5]]
    text = """cam0:\n  camera_model: pinhole\n  distortion_model: radtan\n  intrinsics: [{fx:.12g}, {fy:.12g}, {cx:.12g}, {cy:.12g}]\n  distortion_coeffs: [{d}]\n  resolution: [{w}, {h}]\n  rostopic: {topic}\n""".format(fx=values[0], fy=values[1], cx=values[2], cy=values[3],
                                        d=", ".join(f"{x:.12g}" for x in coeffs), w=width, h=height, topic=topic)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topic", default="/d455/color/image_raw")
    parser.add_argument("--camera-info-topic", default=None,
                        help="CameraInfo topic; defaults to the image topic's sibling camera_info topic")
    parser.add_argument("--tag-cols", type=int, default=6)
    parser.add_argument("--tag-rows", type=int, default=6)
    parser.add_argument("--tag-size", type=float, default=0.10)
    parser.add_argument("--tag-spacing", type=float, default=0.3)
    parser.add_argument("--black-border-bits", type=int, default=2,
                        help="printed black border width in code cells (Kalibr grid: 2)")
    parser.add_argument("--camera-info-bag", type=Path,
                        help="bag containing the factory CameraInfo; defaults to the first session")
    parser.add_argument("--holdout-session", action="append", type=Path, default=[],
                        help="reserve all selected views of this input session for independent validation")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-views", type=int, default=300)
    parser.add_argument("--min-tags", type=int, default=8)
    args = parser.parse_args()
    if args.tag_cols <= 0 or args.tag_rows <= 0 or args.tag_size <= 0 or args.tag_spacing < 0 or args.stride <= 0 or args.max_views < 20:
        parser.error("invalid board or sampling parameters")

    target = _target_points(args.tag_cols, args.tag_rows, args.tag_size, args.tag_spacing)
    camera_info_topic = args.camera_info_topic or args.topic.rsplit("/", 1)[0] + "/camera_info"
    if args.black_border_bits < 1:
        parser.error("black border width must be positive")
    detector = _detector(args.black_border_bits)
    all_views, bag_info = [], []
    for session in args.sessions:
        views, info = _read_bag(session, args.topic, detector, target, args.stride, args.min_tags)
        all_views.extend(views)
        bag_info.append(info)
        print(f"{session}: {info['frames']} frames, {info['detected_views']} usable sampled views", flush=True)
    if len(all_views) < 20:
        raise SystemExit(f"only {len(all_views)} usable views; need at least 20")
    selected = _uniform(all_views, args.max_views)
    width, height = selected[0].width, selected[0].height

    # CameraInfo in both bags is the factory prior and is intentionally only a starting guess.
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
        with AnyReader([args.camera_info_bag or args.sessions[0]], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
            ci = next(c for c in reader.connections if c.topic == camera_info_topic)
            _, _, raw = next(reader.messages(connections=[ci]))
            msg = reader.deserialize(raw, ci.msgtype)
            K0 = np.asarray(msg.k, dtype=float).reshape(3, 3)
            D0 = np.asarray(msg.d, dtype=float).reshape(-1)[:5]
    except Exception as exc:
        raise RuntimeError(f"could not read camera_info topic {camera_info_topic!r} from the prior bag") from exc
    prior_valid = bool(D0.size == 5 and np.isfinite(K0).all() and K0[0, 0] > 0 and K0[1, 1] > 0)
    if not prior_valid:
        objects, images = _arrays(selected, target)
        K0 = cv2.initCameraMatrix2D(objects, images, (width, height), 0)
        D0 = np.zeros(5, dtype=float)
        print("CameraInfo has no valid calibration; initialized intrinsics from planar observations", flush=True)

    if args.holdout_session:
        reserved = {p.resolve() for p in args.holdout_session}
        if not reserved.issubset({p.resolve() for p in args.sessions}):
            parser.error("every holdout session must also be listed as an input session")
        train = [view for view in selected if Path(view.session).resolve() not in reserved]
        holdout = [view for view in selected if Path(view.session).resolve() in reserved]
    else:
        train = [view for i, view in enumerate(selected) if i % 5]
        holdout = [view for i, view in enumerate(selected) if not i % 5]
    if len(train) < 20 or not holdout:
        raise ValueError("need at least 20 training views and a nonempty holdout set")
    _, K_train, D_train, train_errors = _fit(train, target, K0, D0)
    median = float(np.median(train_errors))
    mad = float(np.median(np.abs(np.asarray(train_errors) - median)))
    threshold = max(0.6, median + 4.0 * max(mad, 0.01))
    inliers = [view for view, error in zip(train, train_errors) if error <= threshold]
    if len(inliers) < max(20, int(len(train) * 0.6)):
        inliers = train
    final_rms, K, D, final_errors = _fit(inliers, target, K_train, D_train)
    factory_holdout = _validate(holdout, target, K0, D0) if prior_valid else []
    candidate_holdout = _validate(holdout, target, K, D)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(args.output_dir / "camchain.yaml", K, D, width, height, args.topic)
    with (args.output_dir / "views.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["session", "frame_index", "bag_timestamp_ns", "tags", "split", "used_for_fit"])
        writer.writeheader()
        inlier_ids = {id(view) for view in inliers}
        holdout_ids = {id(view) for view in holdout}
        for i, view in enumerate(selected):
            writer.writerow({"session": view.session, "frame_index": view.frame_index, "bag_timestamp_ns": view.bag_timestamp_ns,
                             "tags": len(view.corners), "split": "holdout" if id(view) in holdout_ids else "train",
                             "used_for_fit": id(view) in inlier_ids})
    summary = {
        "model": "pinhole-radtan (OpenCV; Kalibr-compatible camchain)",
        "target": {"tagCols": args.tag_cols, "tagRows": args.tag_rows, "tagSize_m": args.tag_size, "tagSpacing": args.tag_spacing,
                   "dictionary": "DICT_APRILTAG_36h11", "tag_count": len(target), "black_border_bits": args.black_border_bits},
        "detector": {"inverted_markers": False, "corner_refinement": "subpix",
                     "window_size_px": 7, "relative_window_size": 1.5},
        "image": {"width": width, "height": height, "topic": args.topic},
        "bags": bag_info, "sampling": {"stride": args.stride, "selected_views": len(selected), "train_views": len(train),
                                          "holdout_views": len(holdout), "fit_inlier_views": len(inliers), "outlier_threshold_px": threshold,
                                          "holdout_sessions": [str(p) for p in args.holdout_session]},
        "factory_camera_info": {"valid": prior_valid, "camera_matrix": K0.tolist() if prior_valid else None, "distortion_coeffs": D0.tolist() if prior_valid else None, "distortion_model": "plumb_bob" if prior_valid else None},
        "candidate": {"camera_matrix": K.tolist(), "distortion_coeffs": D.tolist(), "distortion_model": "radtan",
                      "fit_rms_px": final_rms, "fit_view_rms_px": float(np.sqrt(np.mean(np.asarray(final_errors) ** 2))),
                      "fit_view_error": _stats(final_errors)},
        "heldout_validation": {"factory_camera_info": _stats(factory_holdout), "candidate": _stats(candidate_holdout),
                                "improvement_p50_px": float(np.median(factory_holdout) - np.median(candidate_holdout)) if prior_valid else None,
                                "improvement_p95_px": float(np.percentile(factory_holdout, 95) - np.percentile(candidate_holdout, 95)) if prior_valid else None},
        "notes": ["Single-camera RGB intrinsics only; no depth/IMU extrinsics are estimated.",
                  "The YAML follows the cam0 fields consumed by marker6dof.camera.load_kalibr_camera.",
                  "Held-out views are whole reserved sessions when requested; otherwise every fifth selected view. No holdout views are removed.",
                  "Old evaluation videos are independent of these views."],
    }
    (args.output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["candidate"], indent=2))
    print(json.dumps(summary["heldout_validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

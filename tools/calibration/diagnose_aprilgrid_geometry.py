"""Measure tag-side/pitch ratios from images without fitting camera intrinsics.

A homography fitted to tag centres defines pitch units. Tag corners then give
an independent image-space check of spacing and corner correspondence. This is
diagnostic evidence, not a replacement for physical board measurements.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from marker6dof.commands.calibration.calibrate_aprilgrid import _detector, _gray_image


def geometry(corners, matrix, distortion, cols=6):
    tags = sorted(corners)
    pixels = np.concatenate([corners[tag] for tag in tags]).reshape(-1, 1, 2)
    points = cv2.undistortPoints(pixels, matrix, distortion).reshape(-1, 4, 2)
    centres = []
    for p in points:
        # Diagonal intersection is the projective image of the square centre.
        lines = [np.cross(np.r_[p[a], 1], np.r_[p[b], 1]) for a, b in [(0, 2), (1, 3)]]
        centre = np.cross(*lines)
        centres.append(centre[:2] / centre[2])
    centres = np.asarray(centres)
    grid = np.asarray([[tag % cols, tag // cols] for tag in tags], dtype=float)
    homography, _ = cv2.findHomography(centres, grid, 0)
    if homography is None:
        raise ValueError("degenerate tag-centre grid")
    mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 4, 2)
    offsets = mapped - grid[:, None, :]
    # Mapping used by the calibration module: lower-right, lower-left,
    # upper-left, upper-right in the board's increasing-row coordinate system.
    signs = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]])
    side_per_pitch = float(np.median(np.linalg.norm(mapped - np.roll(mapped, 1, axis=1), axis=2)))
    expected = signs * side_per_pitch / 2
    return {
        "side_per_pitch": side_per_pitch,
        "inferred_spacing": 1 / side_per_pitch - 1,
        "corner_mapping_rms_pitch": float(np.sqrt(np.mean(np.sum((offsets - expected) ** 2, axis=2)))),
        "median_corner_offsets_pitch": np.median(offsets, axis=0).tolist(),
        "tag_count": len(tags),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--refinement", choices=["subpix", "apriltag", "none", "pupil"], default="subpix")
    args = parser.parse_args()
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with AnyReader([args.sessions[0]], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connection = next(c for c in reader.connections if c.topic == "/d455/color/camera_info")
        _, _, raw = next(reader.messages(connections=[connection]))
        msg = reader.deserialize(raw, connection.msgtype)
        matrix, distortion = np.asarray(msg.k).reshape(3, 3), np.asarray(msg.d)
    detector = _detector()
    params = detector.getDetectorParameters()
    params.cornerRefinementMethod = {
        "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
        "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
        "none": cv2.aruco.CORNER_REFINE_NONE,
        "pupil": cv2.aruco.CORNER_REFINE_SUBPIX,
    }[args.refinement]
    detector.setDetectorParameters(params)
    native = None
    if args.refinement == "pupil":
        from pupil_apriltags import Detector
        native = Detector(families="tag36h11", nthreads=2, quad_decimate=1, refine_edges=1)
    records, observations = [], []
    for session in args.sessions:
        session_records = []
        with AnyReader([session], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
            connection = next(c for c in reader.connections if c.topic == "/d455/color/image_raw")
            for index, (_, timestamp, raw) in enumerate(reader.messages(connections=[connection])):
                if index % args.stride:
                    continue
                msg = reader.deserialize(raw, connection.msgtype)
                gray = _gray_image(msg)
                corners, ids, _ = detector.detectMarkers(gray)
                if native is not None:
                    detections = native.detect(gray)
                    # Native and OpenCV corner orders differ. Match the native
                    # corners to the decoded OpenCV tag orientation for this
                    # diagnostic only; positions still come from native edges.
                    opencv = {} if ids is None else {int(t): c.reshape(4, 2) for c, t in zip(corners, ids.ravel())}
                    corners, tag_ids = [], []
                    for detection in detections:
                        tag = int(detection.tag_id)
                        if tag not in opencv or detection.hamming > 1:
                            continue
                        native_corners = detection.corners
                        order = np.argmin(np.linalg.norm(opencv[tag][:, None] - native_corners[None], axis=2), axis=1)
                        if len(set(order.tolist())) != 4:
                            continue
                        corners.append(native_corners[order])
                        tag_ids.append(tag)
                    ids = np.asarray(tag_ids, dtype=int)
                selected = {} if ids is None else {
                    int(tag): c.reshape(4, 2).astype(float)
                    for c, tag in zip(corners, ids.ravel()) if 0 <= int(tag) < 36
                }
                if len(selected) < 12:
                    continue
                try:
                    result = geometry(selected, matrix, distortion)
                except ValueError:
                    continue
                result.update(session=str(session), frame_index=index)
                session_records.append(result)
                observations.append({"session": str(session), "frame_index": index,
                                     "bag_timestamp_ns": int(timestamp), "width": int(msg.width),
                                     "height": int(msg.height),
                                     "corners": {str(k): v.tolist() for k, v in selected.items()}})
                if len(session_records) == 1:
                    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    for tag, c in selected.items():
                        cv2.polylines(preview, [c.astype(np.int32)], True, (0, 255, 0), 1)
                        cv2.putText(preview, str(tag), tuple(c.mean(axis=0).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.imwrite(str(args.output_dir / f"{session.name}_preview.png"), preview)
        records.extend(session_records)
        values = [r["inferred_spacing"] for r in session_records]
        print(json.dumps({"session": str(session), "views": len(values),
                          "spacing_p05_p50_p95": np.percentile(values, [5, 50, 95]).tolist() if values else None}, ensure_ascii=False), flush=True)
    (args.output_dir / "geometry.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.output_dir / "observations.json").write_text(json.dumps(observations) + "\n")


if __name__ == "__main__":
    main()

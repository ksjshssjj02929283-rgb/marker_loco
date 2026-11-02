"""Compare edge-refinement settings on uniformly sampled calibration images."""
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from marker6dof.commands.calibration.calibrate_aprilgrid import (
    View, _detector, _gray_image, _target_points, _validate,
)
from diagnose_aprilgrid_geometry import geometry


def main():
    root = Path(__file__).resolve().parents[2]
    summary = json.loads((root / "outputs/d455_aprilgrid_1020_spacing03_verified/calibration_summary.json").read_text())
    camera = summary["factory_camera_info"]
    matrix, distortion = np.asarray(camera["camera_matrix"]), np.asarray(camera["distortion_coeffs"])
    target = _target_points(6, 6, .10, .3)
    detectors = {}
    for relative in [.8, 1., 1.2, 1.5]:
        detector = _detector()
        params = detector.getDetectorParameters()
        params.relativeCornerRefinmentWinSize = relative
        detector.setDetectorParameters(params)
        detectors[str(relative)] = detector
    records = []
    for name in ["d455_1019_1", "d455_1019_2", "d455_aprilgrid_1020_retry2"]:
        with AnyReader([root / "captures/calibration/d455" / name], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
            connection = next(c for c in reader.connections if c.topic == "/d455/color/image_raw")
            for index, (_, stamp, raw) in enumerate(reader.messages(connections=[connection])):
                if index % 60:
                    continue
                msg = reader.deserialize(raw, connection.msgtype)
                gray = _gray_image(msg)
                for label, detector in detectors.items():
                    corners, ids, _ = detector.detectMarkers(gray)
                    selected = {} if ids is None else {int(t): c.reshape(4, 2).astype(float)
                                                       for c, t in zip(corners, ids.ravel()) if int(t) in target}
                    if len(selected) < 12:
                        continue
                    view = View(name, index, int(stamp), selected, int(msg.width), int(msg.height))
                    result = geometry(selected, matrix, distortion)
                    records.append({"session": name, "frame": index, "relative_window": label,
                                    "rms_px": _validate([view], target, matrix, distortion)[0], **result})
        print(name, "done", flush=True)
    for label in detectors:
        rows = [r for r in records if r["relative_window"] == label]
        print(label, "views", len(rows), "RMS p50/p95", np.percentile([r["rms_px"] for r in rows], [50, 95]),
              "spacing p05/p50/p95", np.percentile([r["inferred_spacing"] for r in rows], [5, 50, 95]))
    output = root / "outputs/aprilgrid_geometry_corrected/refinement_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

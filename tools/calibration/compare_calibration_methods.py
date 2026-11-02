"""Compare actual Kalibr and OpenCV models on common unseen retry2 corners."""

import json
from pathlib import Path

import numpy as np

from marker6dof.camera import load_kalibr_camera
from marker6dof.commands.calibration.calibrate_aprilgrid import View, _stats, _target_points, _validate


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "outputs/d455_calibration_methods_spacing03"
    output.mkdir(parents=True, exist_ok=True)
    source = root / "outputs/aprilgrid_geometry_final/observations.json"
    records = json.loads(source.read_text())
    # Neither first-two-bag calibration uses this bag. Keep all sampled views.
    views = [View(r["session"], r["frame_index"], r["bag_timestamp_ns"],
                  {int(t): np.asarray(c) for t, c in r["corners"].items()}, r["width"], r["height"])
             for r in records if r["session"] == "d455_aprilgrid_1020_retry2"]
    target = _target_points(6, 6, .10, .3)
    factory = json.loads((root / "outputs/d455_aprilgrid_1020_spacing03_refined/calibration_summary.json").read_text())["factory_camera_info"]
    models = {"factory": (np.asarray(factory["camera_matrix"]), np.asarray(factory["distortion_coeffs"]))}
    files = {"opencv_first_two_bags": root / "outputs/d455_aprilgrid_1020_spacing03_crossbag/camchain.yaml",
             "kalibr_first_two_bags": root / "kalibr_d455/camchain-spacing03.yaml"}
    for name, path in files.items():
        camera = load_kalibr_camera(path)
        models[name] = camera.matrix, camera.distortion
    errors = {name: _validate(views, target, *camera) for name, camera in models.items()}
    result = {"protocol": "Same 30 stride-60 retry2 views/corners and board, independent of first-two-bag calibration; no validation views removed.",
              "metric": "Percentiles of per-view 2D Euclidean RMS after independently fitting board pose; pixels.",
              "target": {"tagCols": 6, "tagRows": 6, "tagSize": .10, "tagSpacing": .3},
              "sources": {name: str(path.relative_to(root)) for name, path in files.items()},
              "models": {name: {"K": k.tolist(), "D": d.tolist(), "validation": _stats(errors[name])}
                         for name, (k, d) in models.items()},
              "views": [{"session": v.session, "frame_index": v.frame_index,
                         "error_px": {name: value[i] for name, value in errors.items()}}
                        for i, v in enumerate(views)]}
    (output / "common_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["models"], indent=2))


if __name__ == "__main__":
    main()

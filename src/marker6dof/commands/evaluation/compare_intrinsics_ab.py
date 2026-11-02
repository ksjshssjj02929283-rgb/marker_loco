"""Summarize factory-vs-candidate D455 intrinsics replay outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _rows(path: Path, tag: int = 2):
    return [r for r in csv.DictReader((path / "poses.csv").open(newline="", encoding="utf-8")) if int(r["tag_id"]) == tag]


def _zdisp(root: Path, scenes: list[str], model: str, tag: int):
    means = []
    for scene in scenes:
        rows = _rows(root / scene / model, tag)
        if not rows:
            raise ValueError(f"no tag {tag} rows in {scene}/{model}")
        means.append(np.mean([[float(r[k]) for k in ("tx_m", "ty_m", "tz_m")] for r in rows], axis=0) * 1000.0)
    points = np.asarray(means)
    centre = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - centre, full_matrices=False)
    axis = vectors[0]
    if axis[2] < 0:
        axis = -axis
    projection = (points - centre) @ axis
    cumulative = (projection - projection[0])
    nominal = np.arange(len(points), dtype=float) * 50.0
    residual = cumulative - nominal
    step = np.diff(cumulative)
    return {
        "mean_position_mm": points.tolist(), "direction_unit": axis.tolist(),
        "axis_tilt_deg": float(np.degrees(np.arccos(np.clip(axis[2], -1, 1)))),
        "projected_cumulative_mm": cumulative.tolist(), "cumulative_residual_mm": residual.tolist(),
        "endpoint_projected_error_mm": float(residual[-1]),
        "endpoint_projected_relative_error_percent": float(residual[-1] / nominal[-1] * 100),
        "cumulative_mae_mm": float(np.mean(np.abs(residual[1:]))),
        "cumulative_rmse_mm": float(np.sqrt(np.mean(residual[1:] ** 2))),
        "step_mae_mm": float(np.mean(np.abs(step - 50))),
        "step_rmse_mm": float(np.sqrt(np.mean((step - 50) ** 2))),
        "point_to_line_rms_mm": float(np.sqrt(np.mean(np.sum(((points - centre) - np.outer(projection, axis)) ** 2, axis=1)))),
        "explained_variance_percent": float(singular[0] ** 2 / np.sum(singular ** 2) * 100),
    }


def _static(root: Path, scenes: list[str], model: str):
    result = {}
    for scene in scenes:
        summary = json.loads((root / scene / model / "summary.json").read_text())
        result[scene] = {"coverage": summary["frame_coverage"], "p50_px": summary["median_reprojection_error_px"],
                         "p95_px": summary["p95_reprojection_error_px"]}
        rows = _rows(root / scene / model, 2)
        if rows:
            pos = np.asarray([[float(r[k]) for k in ("tx_m", "ty_m", "tz_m")] for r in rows]) * 1000
            result[scene]["tag2_std_mm"] = np.std(pos, axis=0).tolist()
    return result


def _tape(root: Path, name: str, model: str):
    path = root / f"{name}_{model}.json"
    data = json.loads(path.read_text())
    s = data["summary"]
    return {"mae_mm": s["baseline_corrected_mae_nonbaseline_mm"],
            "rmse_mm": s["baseline_corrected_rmse_nonbaseline_mm"],
            "max_mm": s["baseline_corrected_max_abs_error_mm"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag-id", type=int, default=2)
    args = parser.parse_args()
    static_scenes = ["static_d300", "static_d500", "static_d800", "static_d1000"]
    z_scenes = ["zdisp_0mm", "zdisp_50mm", "zdisp_100mm", "zdisp_150mm", "zdisp_200mm"]
    result = {"root": str(args.root.resolve()), "models": {}}
    for model in ("factory", "candidate"):
        result["models"][model] = {
            "static": _static(args.root, static_scenes, model),
            "zdisp": _zdisp(args.root, z_scenes, model, args.tag_id),
            "tape_single_face": _tape(args.root, "tape_single", model),
            "tape_20cm": _tape(args.root, "tape_multi", model),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate static PnP distance against tape-measure reference points.

Record one stationary session per point, then pass each as
``--point LABEL=TAPE_DISTANCE_CM=SESSION_DIR``.  The default metric is PnP
optical-axis depth (``tz_m``), appropriate when the marker is kept parallel to
the camera and moved straight along the optical axis.  Use ``--metric range``
only when the tape measures the straight camera-to-marker-centre distance.

Example (20, 40, 60 and 80 cm):
    python scripts/eval_tape_static_accuracy.py \\
      --point 20=20=captures/d455_aruco/tape_20cm/20261010_100000 \\
      --point 40=40=captures/d455_aruco/tape_40cm/20261010_100100 \\
      --point 60=60=captures/d455_aruco/tape_60cm/20261010_100200 \\
      --point 80=80=captures/d455_aruco/tape_80cm/20261010_100300

The raw error includes the unknown offset between the tape's camera reference
point and the RGB optical centre.  The "baseline-corrected" error subtracts
the raw error at one point; it therefore assesses distance change without
silently fitting a scale factor or claiming an absolute calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


AXES = ("tx_m", "ty_m", "tz_m")


def parse_point(value: str) -> tuple[str, float, Path]:
    """Parse LABEL=DISTANCE_CM=SESSION, allowing '=' inside neither field."""
    try:
        label, distance_cm, session = value.split("=", 2)
        distance_cm = float(distance_cm)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "point must be LABEL=TAPE_DISTANCE_CM=SESSION_DIR"
        ) from exc
    if not label:
        raise argparse.ArgumentTypeError("point label must not be empty")
    if distance_cm <= 0:
        raise argparse.ArgumentTypeError("tape distance must be positive")
    if not session:
        raise argparse.ArgumentTypeError("session directory must not be empty")
    return label, distance_cm * 10.0, Path(session)  # cm -> mm


def load_static_point(label: str, tape_mm: float, session: Path, tag_id: int, metric: str) -> dict:
    poses_path = session / "poses.csv"
    if not poses_path.exists():
        raise ValueError(f"{label}: missing poses file: {poses_path}")
    rows = list(csv.DictReader(poses_path.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if int(row["tag_id"]) == tag_id]
    if not rows:
        raise ValueError(f"{label}: tag {tag_id} was not observed")

    tvec_m = np.asarray([[float(row[axis]) for axis in AXES] for row in rows], dtype=np.float64)
    reproj_px = np.asarray([float(row["reprojection_error_px"]) for row in rows], dtype=np.float64)
    values_m = tvec_m[:, 2] if metric == "z" else np.linalg.norm(tvec_m, axis=1)
    return {
        "label": label,
        "session": str(session),
        "tape_distance_mm": tape_mm,
        "frames": len(rows),
        "mean_tvec_mm": {axis: float(tvec_m[:, i].mean() * 1000) for i, axis in enumerate(AXES)},
        "std_tvec_mm": {axis: float(tvec_m[:, i].std() * 1000) for i, axis in enumerate(AXES)},
        "metric_mean_mm": float(values_m.mean() * 1000),
        "metric_std_mm": float(values_m.std() * 1000),
        "reprojection_p50_px": float(np.median(reproj_px)),
        "reprojection_p95_px": float(np.percentile(reproj_px, 95)),
    }


def summarize(points: list[dict], baseline_label: str | None) -> tuple[list[dict], dict]:
    if len(points) < 2:
        raise ValueError("at least two --point values are required")
    labels = [point["label"] for point in points]
    if len(set(labels)) != len(labels):
        raise ValueError("point labels must be unique")
    baseline = next((point for point in points if point["label"] == baseline_label), None)
    if baseline is None:
        raise ValueError(f"baseline label {baseline_label!r} is not one of: {', '.join(labels)}")

    baseline_raw_error = baseline["metric_mean_mm"] - baseline["tape_distance_mm"]
    adjusted_errors = []
    for point in points:
        point["raw_error_mm"] = point["metric_mean_mm"] - point["tape_distance_mm"]
        point["baseline_corrected_error_mm"] = point["raw_error_mm"] - baseline_raw_error
        point["tape_delta_from_baseline_mm"] = point["tape_distance_mm"] - baseline["tape_distance_mm"]
        point["measured_delta_from_baseline_mm"] = point["metric_mean_mm"] - baseline["metric_mean_mm"]
        adjusted_errors.append(point["baseline_corrected_error_mm"])

    adjusted = np.asarray(adjusted_errors, dtype=np.float64)
    nonbaseline = np.asarray(
        [point["baseline_corrected_error_mm"] for point in points if point is not baseline],
        dtype=np.float64,
    )
    metrics = {
        "baseline_label": baseline["label"],
        "baseline_raw_error_mm": float(baseline_raw_error),
        "raw_error_mae_mm": float(np.mean(np.abs([p["raw_error_mm"] for p in points]))),
        "baseline_corrected_mae_mm": float(np.mean(np.abs(adjusted))),
        "baseline_corrected_mae_nonbaseline_mm": float(np.mean(np.abs(nonbaseline))),
        "baseline_corrected_max_abs_error_mm": float(np.max(np.abs(adjusted))),
        "baseline_corrected_rmse_mm": float(np.sqrt(np.mean(adjusted**2))),
    }
    return points, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", action="append", required=True, type=parse_point, metavar="LABEL=CM=DIR")
    parser.add_argument("--tag-id", type=int, default=2, help="single reference marker face (default: 2)")
    parser.add_argument("--metric", choices=("z", "range"), default="z")
    parser.add_argument("--baseline", help="point label used to remove the fixed tape/reference offset; default: first point")
    parser.add_argument("--max-static-std-mm", type=float, default=2.0, help="warn if any tvec-axis std exceeds this")
    parser.add_argument("--output", type=Path, default=Path("outputs/accuracy_test/tape_static_accuracy.json"))
    args = parser.parse_args()

    points = [load_static_point(label, tape_mm, session, args.tag_id, args.metric) for label, tape_mm, session in args.point]
    baseline_label = args.baseline or points[0]["label"]
    points, summary = summarize(points, baseline_label)
    warnings = []
    for point in points:
        bad_axes = [axis for axis, value in point["std_tvec_mm"].items() if value > args.max_static_std_mm]
        if bad_axes:
            warnings.append(f"{point['label']}: not static enough on {', '.join(bad_axes)}")

    print(f"metric: {'PnP Z (optical-axis depth)' if args.metric == 'z' else 'PnP camera-to-marker-centre range'}")
    print(f"tag: {args.tag_id}; baseline: {summary['baseline_label']} (raw offset {summary['baseline_raw_error_mm']:+.3f} mm)")
    print(f"{'point':8} {'tape mm':>9} {'pnp mm':>9} {'raw err':>10} {'zeroed err':>12} {'sigma xyz (mm)':>23} {'reproj p50/p95':>16}")
    for point in points:
        std = point["std_tvec_mm"]
        print(
            f"{point['label']:8} {point['tape_distance_mm']:9.2f} {point['metric_mean_mm']:9.2f} "
            f"{point['raw_error_mm']:+10.2f} {point['baseline_corrected_error_mm']:+12.2f} "
            f"{std['tx_m']:.2f}/{std['ty_m']:.2f}/{std['tz_m']:.2f} "
            f"{point['reprojection_p50_px']:.2f}/{point['reprojection_p95_px']:.2f}"
        )
    print(
        "\nBaseline-corrected MAE (non-baseline) / RMSE / max: "
        f"{summary['baseline_corrected_mae_nonbaseline_mm']:.3f} / "
        f"{summary['baseline_corrected_rmse_mm']:.3f} / "
        f"{summary['baseline_corrected_max_abs_error_mm']:.3f} mm"
    )
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    result = {
        "metric": args.metric,
        "tag_id": args.tag_id,
        "points": points,
        "summary": summary,
        "warnings": warnings,
        "interpretation": (
            "Raw error includes the physical offset between tape reference and RGB optical centre. "
            "Baseline-corrected error evaluates relative distance accuracy only; no scale fitting was applied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

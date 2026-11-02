#!/usr/bin/env python3
"""Compare synthetic demo estimates with generated ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-position-mm", type=float, default=15.0)
    parser.add_argument("--max-rotation-deg", type=float, default=8.0)
    args = parser.parse_args()

    with args.ground_truth.open(newline="", encoding="utf-8") as stream:
        truth = {int(row["frame_index"]): row for row in csv.DictReader(stream)}
    with args.estimates.open(newline="", encoding="utf-8") as stream:
        estimates = list(csv.DictReader(stream))
    if len(estimates) != len(truth):
        raise RuntimeError(f"row mismatch: {len(estimates)} estimates versus {len(truth)} truth rows")

    position_errors_mm = []
    rotation_errors_deg = []
    for estimate in estimates:
        frame_index = int(estimate["frame_index"])
        if estimate["status"] != "MEASURED":
            raise RuntimeError(f"frame {frame_index} is not MEASURED: {estimate['reason']}")
        expected = truth[frame_index]
        expected_position = np.asarray([float(expected[key]) for key in ("tx_m", "ty_m", "tz_m")])
        actual_position = np.asarray([float(estimate[key]) for key in ("tx_m", "ty_m", "tz_m")])
        expected_quaternion = np.asarray([float(expected[key]) for key in ("qx", "qy", "qz", "qw")])
        actual_quaternion = np.asarray([float(estimate[key]) for key in ("qx", "qy", "qz", "qw")])
        position_errors_mm.append(1000.0 * np.linalg.norm(actual_position - expected_position))
        cosine = float(np.clip(abs(np.dot(expected_quaternion, actual_quaternion)), -1.0, 1.0))
        rotation_errors_deg.append(float(np.degrees(2.0 * np.arccos(cosine))))

    result = {
        "frames": len(estimates),
        "position_error_mm": percentile_summary(position_errors_mm),
        "rotation_error_deg": percentile_summary(rotation_errors_deg),
        "thresholds": {
            "max_position_mm": args.max_position_mm,
            "max_rotation_deg": args.max_rotation_deg,
        },
    }
    result["passed"] = (
        result["position_error_mm"]["max"] <= args.max_position_mm
        and result["rotation_error_deg"]["max"] <= args.max_rotation_deg
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

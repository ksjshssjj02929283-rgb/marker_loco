#!/usr/bin/env python3
"""Absolute-accuracy check from a known rigid-body displacement.

Reads two *static* segments (A -> B), computes the camera-frame displacement of
each visible face from poses.csv, then compares the aggregate displacement
against a true displacement you provide. The output is the absolute error
(magnitude + per-axis) and a relative error.

Convention: measured displacement = mean(B) - mean(A), in the camera optical
frame (+X right, +Y down, +Z forward). If you move the marker 100 mm *toward*
the camera along the optical axis, the true vector is dz = -0.100.

Example:
    python scripts/eval_known_displacement.py \
      --seg1 captures/d455_aruco/static/A \
      --seg2 captures/d455_aruco/static/B \
      --true-dz -0.100

Each segment must actually be static: faces whose position std is large are
flagged, and the inter-face spread of the displacement is reported as an
internal-consistency check.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


AXES = ("tx_m", "ty_m", "tz_m")


def load_poses(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise SystemExit(f"missing poses file: {path}")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    by_tag: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_tag[int(row["tag_id"])].append(row)

    poses = {}
    for tag_id, tag_rows in by_tag.items():
        tvec = np.array([[float(r[a]) for a in AXES] for r in tag_rows])  # m
        quat = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in tag_rows])
        poses[tag_id] = {
            "frames": len(tag_rows),
            "mean_tvec_m": tvec.mean(axis=0),
            "std_tvec_m": tvec.std(axis=0),
            "mean_quat": quat.mean(axis=0),
        }
    return poses


def quat_angle_deg(q1, q2) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    cosine = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(cosine)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg1", type=Path, required=True, help="static segment A (dir with poses.csv)")
    parser.add_argument("--seg2", type=Path, required=True, help="static segment B (dir with poses.csv)")
    parser.add_argument("--true-dx", type=float, default=0.0, help="true displacement X (m, camera frame)")
    parser.add_argument("--true-dy", type=float, default=0.0, help="true displacement Y (m, camera frame)")
    parser.add_argument("--true-dz", type=float, default=0.0, help="true displacement Z (m, camera frame)")
    parser.add_argument("--max-seg-std-mm", type=float, default=2.0, help="flag a face if any axis std exceeds this")
    parser.add_argument("--output", type=Path, default=Path("outputs/accuracy_test/displacement.json"))
    args = parser.parse_args()

    seg1 = load_poses(args.seg1 / "poses.csv")
    seg2 = load_poses(args.seg2 / "poses.csv")
    true = np.array([args.true_dx, args.true_dy, args.true_dz], dtype=np.float64)

    common = sorted(set(seg1) & set(seg2))
    if not common:
        raise SystemExit("no tag observed in both segments")

    print(f"seg1: {args.seg1}")
    print(f"seg2: {args.seg2}")
    print(f"{'tag':4} {'f1':5} {'f2':5} {'mean1 (mm)':>26} {'mean2 (mm)':>26} {'disp (mm)':>26} {'|disp| mm':>9}")
    per_tag_disp = []
    flags = []
    for tag_id in common:
        p1, p2 = seg1[tag_id], seg2[tag_id]
        m1 = p1["mean_tvec_m"] * 1000
        m2 = p2["mean_tvec_m"] * 1000
        disp = m2 - m1
        per_tag_disp.append(disp)
        if np.any(p1["std_tvec_m"] * 1000 > args.max_seg_std_mm):
            flags.append(f"tag {tag_id}: seg1 not static (std {np.round(p1['std_tvec_m']*1000, 2)} mm)")
        if np.any(p2["std_tvec_m"] * 1000 > args.max_seg_std_mm):
            flags.append(f"tag {tag_id}: seg2 not static (std {np.round(p2['std_tvec_m']*1000, 2)} mm)")
        print(
            f"{tag_id:4} {p1['frames']:5} {p2['frames']:5} "
            f"({m1[0]:8.2f},{m1[1]:8.2f},{m1[2]:8.2f}) "
            f"({m2[0]:8.2f},{m2[1]:8.2f},{m2[2]:8.2f}) "
            f"({disp[0]:8.2f},{disp[1]:8.2f},{disp[2]:8.2f}) "
            f"{np.linalg.norm(disp):9.2f}"
        )

    disp = np.vstack(per_tag_disp)
    measured = disp.mean(axis=0)  # mm
    spread = float(np.ptp(np.linalg.norm(disp, axis=1)))  # inter-face spread, mm

    error = measured - true * 1000  # mm
    abs_error = float(np.linalg.norm(error))
    true_norm = float(np.linalg.norm(true))
    relative = abs_error / (true_norm * 1000) if true_norm > 1e-12 else float("nan")

    # Rotation change between segments (should be ~0 for a pure translation).
    rot_deg = float(np.mean([
        quat_angle_deg(seg1[t]["mean_quat"], seg2[t]["mean_quat"]) for t in common
    ]))

    print(f"\nmeasured displacement: ({measured[0]:.3f}, {measured[1]:.3f}, {measured[2]:.3f}) mm  |.|={np.linalg.norm(measured):.3f} mm")
    print(f"true displacement:     ({true[0]*1000:.3f}, {true[1]*1000:.3f}, {true[2]*1000:.3f}) mm  |.|={true_norm*1000:.3f} mm")
    print(f"inter-face spread:     {spread:.3f} mm  (faces should agree for a rigid translation)")
    print(f"rotation change:       {rot_deg:.3f} deg  (should be ~0 for pure translation)")
    print(f"\nABSOLUTE error:        {abs_error:.3f} mm   per-axis (dx,dy,dz) = ({error[0]:.3f}, {error[1]:.3f}, {error[2]:.3f}) mm")
    if np.isfinite(relative):
        print(f"relative error:        {relative*100:.2f} % of |true|")

    if flags:
        print("\nWARNINGS:")
        for f in flags:
            print(f"  - {f}")

    result = {
        "seg1": str(args.seg1),
        "seg2": str(args.seg2),
        "tags_in_both": common,
        "measured_displacement_mm": {a: float(measured[i]) for i, a in enumerate(AXES)},
        "true_displacement_mm": {a: float(true[i] * 1000) for i, a in enumerate(AXES)},
        "absolute_error_mm": abs_error,
        "per_axis_error_mm": {a: float(error[i]) for i, a in enumerate(AXES)},
        "relative_error_percent": relative * 100 if np.isfinite(relative) else None,
        "inter_face_spread_mm": spread,
        "rotation_change_deg": rot_deg,
        "warnings": flags,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

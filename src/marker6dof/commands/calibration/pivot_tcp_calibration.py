#!/usr/bin/env python3
"""Estimate the fixed gripper TCP point from a pivot (tool-tip) recording.

Hold the gripper's TCP point (closed-fingertip midpoint, or a pointed tip held
in the jaws) pressed against one fixed point in space while rotating the
gripper through at least two distinct axes (pitch + yaw, or a cone).  Keep the
camera static.  Each frame's reference-marker pose ``T_camera_ref(t)`` maps the
fixed point to the same camera-frame location, so

    T_camera_ref(t) @ p_ref = p_camera   (constant over t)

which is linear in the 6 unknowns (p_ref, p_camera) and is solved by least
squares over all frames.  This recovers the TCP *position* in the reference
marker (ID 2) frame; the TCP *axes* (rotation) must still be measured from CAD
or physically, so the emitted transform keeps identity rotation.

Example:
    python scripts/pivot_tcp_calibration.py \
      --poses captures/d455_aruco/pivot/20261013_120000/poses.csv \
      --tag-id 2 \
      --tcp-output configs/tcp_transform_pivot.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def transform_from_row(row) -> np.ndarray:
    rvec = np.asarray([float(row["rx_rad"]), float(row["ry_rad"]), float(row["rz_rad"])])
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(row["tx_m"]), float(row["ty_m"]), float(row["tz_m"])]
    return transform


def load_rig(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): np.asarray(v, dtype=float) for k, v in data["T_reference_tag"].items()}


def load_reference_transforms(poses: Path, tag_id: int, rig: dict[int, np.ndarray] | None = None) -> list[np.ndarray]:
    """Collect ``T_camera_ref`` observations.

    Without ``--rig`` only the single ``tag_id`` face is used.  With ``--rig``
    every detected face is mapped into the reference frame via the calibrated
    ``T_reference_tag``, so the pivot can consume frames where any one face is
    visible instead of requiring the reference face to stay in view.
    """
    if not poses.exists():
        raise SystemExit(f"missing poses file: {poses}")
    transforms = []
    with poses.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tag = int(row["tag_id"])
            if rig is not None:
                if tag not in rig:
                    continue
                t_camera_ref = transform_from_row(row) @ np.linalg.inv(rig[tag])
            else:
                if tag != tag_id:
                    continue
                t_camera_ref = transform_from_row(row)
            transforms.append(t_camera_ref)
    return transforms


def solve_pivot(transforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Least-squares solve for (p_ref, p_camera); drop gross outlier frames once."""
    working = list(transforms)

    def fit(items):
        a = np.vstack([np.hstack([t[:3, :3], -np.eye(3)]) for t in items])
        b = np.concatenate([-t[:3, 3] for t in items])
        solution, _, rank, singular_values = np.linalg.lstsq(a, b, rcond=None)
        return solution[:3], solution[3:], rank, singular_values

    p_ref, p_camera, rank, singular_values = fit(working)

    # One outlier-rejection pass: drop frames whose TCP point strays far from
    # the consensus (motion blur, a slipped fingertip).
    residuals_mm = np.asarray([
        np.linalg.norm(t[:3, :3] @ p_ref + t[:3, 3] - p_camera) * 1000
        for t in working
    ])
    threshold = max(3.0, 5.0 * float(np.median(residuals_mm)))
    kept = [t for t, r in zip(working, residuals_mm) if r <= threshold]
    if len(kept) >= 10 and len(kept) < len(working):
        p_ref, p_camera, rank, singular_values = fit(kept)

    return p_ref, p_camera, rank, singular_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", type=Path, required=True, help="recorded session poses.csv")
    parser.add_argument("--rig", type=Path, help="rig_calibration JSON; when given, all visible faces feed the pivot (recommended)")
    parser.add_argument("--tag-id", type=int, default=2, help="reference marker face used when --rig is not given (default: 2)")
    parser.add_argument("--min-frames", type=int, default=100, help="require at least this many reference poses")
    parser.add_argument("--tcp-output", type=Path, help="write a tcp_transform JSON (identity rotation + solved point)")
    parser.add_argument("--output", type=Path, default=Path("outputs/pivot_calibration/tcp_point.json"))
    args = parser.parse_args()

    rig = load_rig(args.rig) if args.rig else None
    transforms = load_reference_transforms(args.poses, args.tag_id, rig)
    if len(transforms) < args.min_frames:
        raise SystemExit(
            f"only {len(transforms)} reference poses; need >= {args.min_frames}. "
            "Record a longer / more diverse pivot motion."
        )

    p_ref, p_camera, rank, singular_values = solve_pivot(transforms)

    residuals_mm = np.asarray([
        np.linalg.norm(t[:3, :3] @ p_ref + t[:3, 3] - p_camera) * 1000
        for t in transforms
    ])
    singular_values = np.asarray(singular_values)
    condition = (
        float(singular_values.max() / singular_values.min())
        if singular_values.size and singular_values.min() > 1e-12
        else float("inf")
    )

    if rank < 6:
        warning = (
            "motion is degenerate (rank<6): the point is under-constrained along the "
            "missing rotation axis(es). Rotate about at least two non-parallel axes "
            "(add roll/yaw, not just pitch), then re-record."
        )
    else:
        warning = None

    result = {
        "tag_id": args.tag_id,
        "frames": len(transforms),
        "rank": int(rank),
        "p_ref_mm": (p_ref * 1000).tolist(),
        "arm_length_mm": float(np.linalg.norm(p_ref) * 1000),
        "p_camera_mm": (p_camera * 1000).tolist(),
        "residual_rmse_mm": float(np.sqrt(np.mean(residuals_mm ** 2))),
        "residual_p95_mm": float(np.percentile(residuals_mm, 95)),
        "residual_max_mm": float(residuals_mm.max()),
        "condition_number": condition,
        "warning": warning,
        "note": (
            "p_ref is the TCP point in the reference tag (ID 2) frame, metres. "
            "Pivot recovers only the point; TCP axes (rotation) still need CAD/physical "
            "measurement, so --tcp-output keeps identity rotation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    if args.tcp_output:
        transform = np.eye(4)
        transform[:3, 3] = p_ref
        payload = {
            "schema_version": 1,
            "units": "m",
            "source": f"pivot calibration from {args.poses} (tag {args.tag_id}, {len(transforms)} frames)",
            "reference_tag_id": args.tag_id,
            "T_reference_TCP": transform.tolist(),
            "convention": "T_reference_TCP maps points from the TCP frame into the reference-tag (ID 2) frame: p_ref = T_reference_TCP @ p_tcp.",
            "tcp_definition": "TCP point from pivot calibration; rotation is identity placeholder pending CAD/physical axis measurement.",
            "residual_rmse_mm": float(np.sqrt(np.mean(residuals_mm ** 2))),
        }
        args.tcp_output.parent.mkdir(parents=True, exist_ok=True)
        args.tcp_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.tcp_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

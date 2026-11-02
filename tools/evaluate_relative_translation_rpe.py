#!/usr/bin/env python3
"""Score held-out relative translation over a fixed laser sampling interval.

The aligned camera points already use a time offset and fixed-scale SE(3)
estimated on the opposite chronological half. Pairs never cross fold borders.
Laser provides position only, so this is the translation component of RPE,
not a full 6DoF relative-pose accuracy measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def score(case: Path, *, interval_s: float = 0.1,
          tolerance_s: float = 0.035) -> dict:
    source = case / "cross_validated_trajectories.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no cross-validated trajectory rows: {source}")
    errors_mm: list[float] = []
    truth_steps_mm: list[float] = []
    eligible_pairs = 0
    for fold in sorted({row["fold"] for row in rows}):
        section = sorted((row for row in rows if row["fold"] == fold),
                         key=lambda row: float(row["gt_time_s"]))
        times = np.array([float(row["gt_time_s"]) for row in section])
        paired = np.array([row["paired"].lower() == "true" for row in section])
        camera = np.array([
            [float(row[f"aligned_{axis}_m"]) if is_paired else np.nan
             for axis in "xyz"]
            for row, is_paired in zip(section, paired)])
        laser = np.array([
            [float(row[f"gt_{axis}_m"]) for axis in "xyz"]
            for row in section])
        for first, time in enumerate(times):
            insertion = int(np.searchsorted(times, time + interval_s))
            candidates = [index for index in (insertion - 1, insertion)
                          if first < index < len(section)]
            if not candidates:
                continue
            last = min(candidates,
                       key=lambda index: abs(times[index] - time - interval_s))
            if abs(times[last] - time - interval_s) > tolerance_s:
                continue
            eligible_pairs += 1
            if not (paired[first] and paired[last]):
                continue
            camera_step = camera[last] - camera[first]
            laser_step = laser[last] - laser[first]
            errors_mm.append(float(np.linalg.norm(camera_step - laser_step) * 1000))
            truth_steps_mm.append(float(np.linalg.norm(laser_step) * 1000))
    if not errors_mm:
        raise ValueError(f"no valid {interval_s} s pairs: {source}")
    values = np.asarray(errors_mm)
    return {
        "interval_s": interval_s,
        "tolerance_s": tolerance_s,
        "pairs": len(values),
        "eligible_pairs": eligible_pairs,
        "pair_coverage": len(values) / eligible_pairs,
        "mae_mm": float(np.mean(values)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(values)))),
        "p50_mm": float(np.median(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "median_laser_step_mm": float(np.median(truth_steps_mm)),
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "definition": "norm of held-out camera and laser 3D displacement-vector difference; "
                      "both endpoints in the same validation half; opposite half fitted "
                      "the time offset and fixed-scale SE(3)",
        "limitation": "laser position only; no orientation ground truth or measured tag-to-laser-target offset",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, action="append")
    parser.add_argument("--all", type=Path,
                        help="evaluation root containing camera/marker/sequence cases")
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    cases = list(args.case_dir or [])
    if args.all:
        cases.extend(path.parent for path in sorted(args.all.glob(
            "*/marker*/*/cross_validated_trajectories.csv")))
    if not cases:
        parser.error("provide --case-dir or --all")
    for case in dict.fromkeys(cases):
        result = score(case, interval_s=args.interval)
        destination = case / f"relative_translation_rpe_{args.interval:g}s.json"
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"{case}: MAE {result['mae_mm']:.2f} mm, {result['pairs']} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare held-out 0.1 s stereo and monocular RPE on identical GT intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def pair_errors(path: Path, interval_s: float = .1, tolerance_s: float = .035) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for fold in sorted({r["fold"] for r in rows}):
        part = sorted((r for r in rows if r["fold"] == fold), key=lambda r: float(r["gt_time_s"]))
        times = np.array([float(r["gt_time_s"]) for r in part])
        for i, time in enumerate(times):
            j = int(np.searchsorted(times, time + interval_s))
            choices = [k for k in (j-1, j) if i < k < len(part)]
            if not choices:
                continue
            j = min(choices, key=lambda k: abs(times[k] - time - interval_s))
            if abs(times[j] - time - interval_s) > tolerance_s:
                continue
            if not (part[i]["paired"] == "True" and part[j]["paired"] == "True"):
                continue
            def vector(row, prefix):
                return np.array([float(row[f"{prefix}_{a}_m"]) for a in "xyz"])
            camera_step = vector(part[j], "aligned") - vector(part[i], "aligned")
            truth_step = vector(part[j], "gt") - vector(part[i], "gt")
            key = (fold, part[i]["gt_time_s"], part[j]["gt_time_s"])
            result[key] = float(np.linalg.norm(camera_step-truth_step)*1000)
    return result


def point_errors(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["fold"], r["gt_time_s"]): float(r["residual_mm"])
            for r in rows if r["paired"] == "True"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-root", type=Path, required=True)
    parser.add_argument("--mono-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in sorted(args.stereo_root.glob("marker*/*/cross_validated_trajectories.csv")):
        marker, name = case.parts[-3:-1]
        paths = {"left": args.mono_root / "left" / marker / name / case.name,
                 "right": args.mono_root / "right" / marker / name / case.name,
                 "stereo": case}
        errors = {label: pair_errors(path) for label, path in paths.items()}
        keys = set.intersection(*(set(value) for value in errors.values()))
        if not keys:
            raise ValueError(f"no common intervals: {marker}/{name}")
        means = {label: float(np.mean([value[key] for key in keys]))
                 for label, value in errors.items()}
        best = min(means["left"], means["right"])
        positions = {label: point_errors(path) for label, path in paths.items()}
        point_keys = set.intersection(*(set(value) for value in positions.values()))
        position_rmse = {label: float(np.sqrt(np.mean([value[key]**2 for key in point_keys])))
                         for label, value in positions.items()}
        replay = json.loads((args.stereo_root.parent / "replay" / marker / name / "summary.json").read_text())
        row = dict(marker=marker, sequence=name, common_pairs=len(keys),
                   left_mae_mm=means["left"], right_mae_mm=means["right"],
                   best_mono_mae_mm=best, stereo_mae_mm=means["stereo"],
                   improvement_vs_best_pct=100*(best-means["stereo"])/best,
                   stereo_pose_coverage=replay["frame_coverage"],
                   stereo_rejected_pairs=replay["rejected_pairs"],
                   stereo_rpe_pairs_all=len(errors["stereo"]),
                   common_position_points=len(point_keys),
                   best_mono_position_rmse_mm=min(position_rmse["left"], position_rmse["right"]),
                   stereo_position_rmse_mm=position_rmse["stereo"])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("summary", json.dumps(dict(cases=len(rows), improved=sum(r["improvement_vs_best_pct"]>0 for r in rows),
                                     median_improvement_pct=float(np.median([r["improvement_vs_best_pct"] for r in rows])),
                                     under_2mm=sum(r["stereo_mae_mm"] <= 2 for r in rows)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

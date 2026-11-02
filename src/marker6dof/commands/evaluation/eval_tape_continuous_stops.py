#!/usr/bin/env python3
"""Evaluate tape-measure accuracy from one recording with stationary stops.

The marker must be moved monotonically and held still at each known tape
distance.  Stops are found from the PnP distance speed, then matched in time
order to ``--distances-cm``.  This is intended for a quick 30,40,...,80 cm
test; use ``eval_tape_static_accuracy.py`` when every point is separately
recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


AXES = ("tx_m", "ty_m", "tz_m")


def parse_distances(value: str) -> list[float]:
    try:
        distances = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("distances must be comma-separated centimetres") from exc
    if len(distances) < 2 or any(distance <= 0 for distance in distances):
        raise argparse.ArgumentTypeError("provide at least two positive distances")
    return distances


def load_observations(session: Path, tag_id: int, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = session / "poses.csv"
    if not path.exists():
        raise ValueError(f"missing poses file: {path}")
    rows = [row for row in csv.DictReader(path.open(newline="", encoding="utf-8")) if int(row["tag_id"]) == tag_id]
    if not rows:
        raise ValueError(f"tag {tag_id} was not observed")
    times = np.asarray([float(row["host_timestamp_s"]) for row in rows])
    frames = np.asarray([int(row["frame_index"]) for row in rows])
    tvec = np.asarray([[float(row[axis]) for axis in AXES] for row in rows])
    reproj = np.asarray([float(row["reprojection_error_px"]) for row in rows])
    # Platform detection needs a scalar; a fitted 3D path still uses Z for its
    # speed gate, then projects the accepted static positions onto that path.
    values = tvec[:, 2] if metric in ("z", "line") else np.linalg.norm(tvec, axis=1)
    return times, frames, tvec, reproj, values


def find_stops(times: np.ndarray, values: np.ndarray, *, speed_mm_s: float, min_stop_s: float, merge_mm: float) -> list[tuple[int, int]]:
    """Find flat runs using a centred ~1 s difference, then merge same-level runs."""
    half_window = 15
    speed = np.full(len(values), np.inf)
    for index in range(half_window, len(values) - half_window):
        dt = times[index + half_window] - times[index - half_window]
        # Do not bridge a detection dropout: it is not evidence of stillness.
        if 0 < dt <= 1.5:
            speed[index] = abs(values[index + half_window] - values[index - half_window]) / dt * 1000
    stationary = speed <= speed_mm_s
    candidates: list[tuple[int, int]] = []
    index = 0
    while index < len(stationary):
        if not stationary[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(stationary) and stationary[end + 1]:
            end += 1
        if times[end] - times[index] >= min_stop_s:
            candidates.append((index, end))
        index = end + 1

    merged: list[tuple[int, int]] = []
    for start, end in candidates:
        if merged:
            previous_start, previous_end = merged[-1]
            previous_mean = values[previous_start : previous_end + 1].mean()
            current_mean = values[start : end + 1].mean()
            if abs(current_mean - previous_mean) * 1000 <= merge_mm:
                # A hand often makes a short preliminary pause, then settles at
                # the same tape mark.  Keep the longest individual flat run;
                # concatenating the intervening movement would inflate noise.
                if times[end] - times[start] > times[previous_end] - times[previous_start]:
                    merged[-1] = (start, end)
                continue
        merged.append((start, end))
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path, help="recording directory containing poses.csv")
    parser.add_argument("--distances-cm", required=True, type=parse_distances, help="tape stops in recording order, e.g. 30,40,50,60,70,80")
    parser.add_argument("--tag-id", type=int, default=3, help="face visible throughout the recording (default: 3)")
    parser.add_argument("--metric", choices=("z", "range", "line"), default="z", help="Z for an optical-axis test; line for tape marks along a straight but angled travel path")
    parser.add_argument("--max-speed-mm-s", type=float, default=10.0, help="maximum PnP-distance speed considered stationary")
    parser.add_argument("--min-stop-s", type=float, default=2.0, help="minimum stationary duration")
    parser.add_argument("--merge-mm", type=float, default=30.0, help="merge adjacent candidate stops closer than this")
    parser.add_argument("--output", type=Path, default=Path("outputs/accuracy_test/tape_continuous_stops.json"))
    args = parser.parse_args()

    times, frames, tvec, reproj, values = load_observations(args.session, args.tag_id, args.metric)
    stops = find_stops(
        times, values,
        speed_mm_s=args.max_speed_mm_s,
        min_stop_s=args.min_stop_s,
        merge_mm=args.merge_mm,
    )
    if len(stops) != len(args.distances_cm):
        raise SystemExit(
            f"found {len(stops)} stationary platforms, but received {len(args.distances_cm)} distances. "
            f"Adjust --max-speed-mm-s / --min-stop-s / --merge-mm; candidate frame ranges: "
            + ", ".join(f"{frames[a]}-{frames[b]}" for a, b in stops)
        )

    points = []
    for distance_cm, (start, end) in zip(args.distances_cm, stops):
        positions = tvec[start : end + 1]
        distance_values = values[start : end + 1]
        point = {
            "tape_distance_mm": distance_cm * 10,
            "frame_range": [int(frames[start]), int(frames[end])],
            "duration_s": float(times[end] - times[start]),
            "frames": int(end - start + 1),
            "mean_tvec_mm": {axis: float(positions[:, i].mean() * 1000) for i, axis in enumerate(AXES)},
            "std_tvec_mm": {axis: float(positions[:, i].std() * 1000) for i, axis in enumerate(AXES)},
            "reprojection_p50_px": float(np.median(reproj[start : end + 1])),
            "reprojection_p95_px": float(np.percentile(reproj[start : end + 1], 95)),
        }
        point["_positions_m"] = positions
        if args.metric != "line":
            point["metric_mean_mm"] = float(distance_values.mean() * 1000)
            point["metric_std_mm"] = float(distance_values.std() * 1000)
        points.append(point)

    line_fit = None
    if args.metric == "line":
        centres = np.asarray([point["_positions_m"].mean(axis=0) for point in points])
        _, singular_values, right_vectors = np.linalg.svd(centres - centres.mean(axis=0), full_matrices=False)
        axis = right_vectors[0]
        if np.dot(centres[-1] - centres[0], axis) < 0:
            axis = -axis
        origin = centres[0]
        for point in points:
            path_coordinates = (point["_positions_m"] - origin) @ axis * 1000
            point["metric_mean_mm"] = float(path_coordinates.mean())
            point["metric_std_mm"] = float(path_coordinates.std())
        line_fit = {
            "origin_first_stop_tvec_mm": {axis_name: float(origin[i] * 1000) for i, axis_name in enumerate(AXES)},
            "unit_direction_camera_xyz": {axis_name: float(axis[i]) for i, axis_name in enumerate(AXES)},
            "explained_variance_percent": float(singular_values[0] ** 2 / np.sum(singular_values ** 2) * 100),
            "angle_from_camera_z_deg": float(np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0)))),
        }

    for point in points:
        point["raw_error_mm"] = point["metric_mean_mm"] - point["tape_distance_mm"]
    baseline_error = points[0]["raw_error_mm"]
    corrected = np.asarray([point["raw_error_mm"] - baseline_error for point in points])
    for point, error in zip(points, corrected):
        point["baseline_corrected_error_mm"] = float(error)
    nonbaseline = corrected[1:]
    summary = {
        "baseline_tape_distance_mm": points[0]["tape_distance_mm"],
        "baseline_raw_error_mm": float(baseline_error),
        "baseline_corrected_mae_nonbaseline_mm": float(np.mean(np.abs(nonbaseline))),
        "baseline_corrected_rmse_nonbaseline_mm": float(np.sqrt(np.mean(nonbaseline**2))),
        "baseline_corrected_max_abs_error_mm": float(np.max(np.abs(nonbaseline))),
    }

    metric_name = {
        "z": "PnP Z (optical-axis depth)",
        "range": "PnP camera-to-marker-centre range",
        "line": "PnP displacement projected onto fitted 3D travel line",
    }[args.metric]
    print(f"metric: {metric_name}; tag {args.tag_id}")
    print(f"{'tape mm':>8} {'frames':>11} {'pnp mm':>9} {'raw err':>10} {'zeroed err':>12} {'sigma z':>9} {'reproj p50/p95':>16}")
    for point in points:
        print(
            f"{point['tape_distance_mm']:8.1f} {point['frame_range'][0]:4}-{point['frame_range'][1]:<4} "
            f"{point['metric_mean_mm']:9.2f} {point['raw_error_mm']:+10.2f} "
            f"{point['baseline_corrected_error_mm']:+12.2f} {point['std_tvec_mm']['tz_m']:9.2f} "
            f"{point['reprojection_p50_px']:.2f}/{point['reprojection_p95_px']:.2f}"
        )
    print(
        "\nBaseline-corrected MAE / RMSE / max (excluding the anchored first stop): "
        f"{summary['baseline_corrected_mae_nonbaseline_mm']:.3f} / "
        f"{summary['baseline_corrected_rmse_nonbaseline_mm']:.3f} / "
        f"{summary['baseline_corrected_max_abs_error_mm']:.3f} mm"
    )

    for point in points:
        point.pop("_positions_m")
    result = {
        "session": str(args.session), "tag_id": args.tag_id, "metric": args.metric,
        "detection": {"max_speed_mm_s": args.max_speed_mm_s, "min_stop_s": args.min_stop_s, "merge_mm": args.merge_mm},
        "points": points, "summary": summary, "line_fit": line_fit,
        "interpretation": "Raw error includes the fixed tape-reference to RGB-optical-centre offset. Baseline-corrected error evaluates relative distance accuracy; no scale was fitted. For metric=line, raw error has an arbitrary line-origin offset and only baseline-corrected values are meaningful.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate October 30 marker replays against laser tracker positions.

Inputs are read only. Replay directories follow marker50/static_020cm and
marker50/dynamic_track_01 naming by default; --manifest can override paths.
Laser positions are in millimetres and use Asia/Shanghai wall clock time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


MARKERS = {
    "marker50": {"id": 2, "black_square_mm": 37.5, "gt_dir": "marker_50mm", "tracks": 4, "static": True, "continuous": "dynamic1.txt"},
    "marker60": {"id": 2, "black_square_mm": 50.0, "gt_dir": "marker_60mm", "tracks": 4, "static": True, "continuous": "dynamic1.txt"},
    "marker70": {"id": 6, "black_square_mm": 60.0, "gt_dir": "marker_70mm", "tracks": 1, "static": False, "continuous": "dynamic.txt"},
}
TZ = ZoneInfo("Asia/Shanghai")
GT_TIME = re.compile(r"^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)\.\s*(\d{1,3})$")


def parse_gt_time(value: str) -> float:
    match = GT_TIME.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid laser timestamp: {value!r}")
    # The export pads milliseconds with spaces on the left: '. 13' means 013 ms.
    stamp = datetime.strptime(match[1], "%Y/%m/%d %H:%M:%S").replace(tzinfo=TZ)
    return stamp.timestamp() + int(match[2]) / 1000.0


def load_gt(path: Path, *, dynamic: bool) -> tuple[list[dict], dict]:
    raw = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 7:
                raise ValueError(f"{path}:{number}: expected seven fields, got {len(row)}")
            raw.append((row[2], np.array([float(v) for v in row[3:6]]) / 1000.0,
                        parse_gt_time(row[6])))
    if not raw:
        raise ValueError(f"empty laser file: {path}")
    if not dynamic:
        return ([dict(label=label, xyz_m=xyz, time_s=time) for label, xyz, time in raw],
                dict(rows=len(raw), time_groups=len(raw)))

    groups: dict[float, list[np.ndarray]] = defaultdict(list)
    for _, xyz, time in raw:
        groups[time].append(xyz)
    result = []
    for time, values in sorted(groups.items()):
        positions = np.asarray(values)
        centre = positions.mean(axis=0)
        spread_mm = float(np.linalg.norm(positions - centre, axis=1).max() * 2000)
        result.append(dict(time_s=time, xyz_m=centre, samples=len(values), spread_mm=spread_mm))
    backwards = sum(b[2] < a[2] for a, b in zip(raw, raw[1:]))
    return result, dict(rows=len(raw), time_groups=len(result), backwards=backwards,
                        group_sizes=dict(Counter(str(item["samples"]) for item in result)),
                        max_spread_mm=max(item["spread_mm"] for item in result))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_replay(directory: Path, tag_id: int, expected_size_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frames = read_csv(directory / "frames.csv")
    poses = read_csv(directory / "poses.csv")
    selected = [row for row in poses if int(row["tag_id"]) == tag_id]
    data = []
    for row in selected:
        try:
            xyz = [float(row[f"t{axis}_m"]) for axis in "xyz"]
            time = float(row["header_timestamp_ns"]) / 1e9
            frame = int(row["frame_index"])
            if np.isfinite(xyz).all() and math.isfinite(time):
                data.append((time, frame, xyz))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid replay pose row in {directory}") from exc
    data.sort(key=lambda item: item[0])
    if len({item[1] for item in data}) != len(data):
        raise ValueError(f"duplicate frame index for tag {tag_id}: {directory}")
    times = np.array([item[0] for item in data], dtype=float)
    indexes = np.array([item[1] for item in data], dtype=int)
    positions = np.array([item[2] for item in data], dtype=float).reshape(-1, 3)
    info = dict(frames=len(frames), accepted=len(data), coverage=len(data) / len(frames) if frames else 0,
                replay_dir=str(directory.resolve()), tag_id=tag_id)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        info.update(camera_key=summary.get("camera_key"), camchain=summary.get("camchain"),
                    marker_size_mm=summary.get("marker_size_mm"), topic=summary.get("topic"),
                    replay_frame_coverage=summary.get("frame_coverage"),
                    strict_24px_coverage=summary.get("strict_24px_coverage"),
                    reprojection_p50_px=summary.get("reprojection_p50_px"),
                    reprojection_p95_px=summary.get("reprojection_p95_px"),
                    duplicate_selection_counts=summary.get("duplicate_selection_counts", {}),
                    accepted_by_id=summary.get("accepted_by_id", {}),
                    detections_by_id=summary.get("detections_by_id", {}))
        actual_size = summary.get("marker_size_mm")
        if actual_size is not None and not math.isclose(float(actual_size), expected_size_mm, abs_tol=0.01):
            raise ValueError(f"{directory}: replay black-square size {actual_size} mm != expected {expected_size_mm} mm")
        camchain = summary.get("camchain")
        if camchain and Path(camchain).is_file():
            info["camchain_sha256"] = sha256(Path(camchain))
        info["replay_summary_sha256"] = sha256(summary_path)
    info["expected_black_square_mm"] = expected_size_mm
    return times, indexes, positions, info


def interpolate_valid(times: np.ndarray, indexes: np.ndarray, xyz: np.ndarray,
                      queries: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    output = np.full((len(queries), 3), np.nan)
    valid = np.zeros(len(queries), dtype=bool)
    if len(times) < 2:
        return output, valid
    right = np.searchsorted(times, queries, side="left")
    exact = (right < len(times)) & (times[np.minimum(right, len(times) - 1)] == queries)
    output[exact] = xyz[right[exact]]
    valid[exact] = True
    between = (right > 0) & (right < len(times)) & ~exact
    loc = np.flatnonzero(between)
    if len(loc):
        left = right[loc] - 1
        span = times[right[loc]] - times[left]
        eligible = (span > 0) & (span <= max_gap_s) & (indexes[right[loc]] - indexes[left] == 1)
        loc, left, span = loc[eligible], left[eligible], span[eligible]
        weight = (queries[loc] - times[left]) / span
        output[loc] = xyz[left] * (1 - weight[:, None]) + xyz[left + 1] * weight[:, None]
        valid[loc] = True
    return output, valid


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, singular, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    diagonal = np.eye(3)
    diagonal[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ diagonal @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation, singular


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def displacement_metrics(camera: np.ndarray, laser: np.ndarray) -> dict:
    errors = []
    for i in range(len(camera) - 1):
        cam_distance = np.linalg.norm(camera[i + 1:] - camera[i], axis=1)
        gt_distance = np.linalg.norm(laser[i + 1:] - laser[i], axis=1)
        errors.extend(((cam_distance - gt_distance) * 1000).tolist())
    array = np.asarray(errors)
    return dict(pairs=len(array), rmse_mm=rms(array), mae_mm=float(np.mean(np.abs(array))),
                bias_mm=float(array.mean()), p95_abs_mm=float(np.percentile(np.abs(array), 95)))


def cross_validated_se3(camera_times: np.ndarray, indexes: np.ndarray, camera_xyz: np.ndarray,
                        gt_times: np.ndarray, gt_xyz: np.ndarray, offsets: np.ndarray,
                        *, max_gap_s: float, min_coverage: float,
                        min_motion_fraction: float) -> tuple[dict, list[dict]] | None:
    """Fit time shift and SE(3) on one continuous half, score the other half.

    The direction is repeated with the halves swapped, so every ground-truth
    matched observation is scored once with a transform learned from a disjoint time
    interval. Scale remains fixed at one. The resulting scores describe
    cross-segment generalization, not absolute calibration accuracy.
    """
    if len(gt_times) < 40:
        return None
    split = len(gt_times) // 2
    sections = (("first_to_second", np.arange(0, split), np.arange(split, len(gt_times))),
                ("second_to_first", np.arange(split, len(gt_times)), np.arange(0, split)))
    folds = []
    output_rows = []
    all_errors = []
    all_displacements = []
    for fold_name, train_ids, test_ids in sections:
        train_gt = gt_xyz[train_ids]
        train_times = gt_times[train_ids]
        full_motion = float(np.linalg.norm(np.ptp(train_gt, axis=0)))
        candidates = []
        def add_candidates(offset_values):
            for offset in offset_values:
                sampled, valid = interpolate_valid(camera_times, indexes, camera_xyz,
                                                   train_times - offset, max_gap_s)
                matched = int(valid.sum())
                coverage = matched / len(train_ids)
                motion = float(np.linalg.norm(np.ptp(train_gt[valid], axis=0))) if matched else 0.0
                motion_fraction = motion / full_motion if full_motion > 0 else 1.0
                if matched < 20 or coverage < min_coverage or motion_fraction < min_motion_fraction:
                    continue
                rotation, translation, singular = rigid_fit(sampled[valid], train_gt[valid])
                residual = np.linalg.norm(
                    (sampled[valid] @ rotation.T + translation - train_gt[valid]) * 1000, axis=1)
                candidates.append((rms(residual), -matched, float(offset), rotation,
                                   translation, singular, matched, coverage, motion_fraction))

        add_candidates(offsets)
        if candidates and len(offsets) > 1:
            coarse_best = min(candidates, key=lambda row: (row[0], row[1]))[2]
            coarse_step = float(np.median(np.diff(np.sort(np.unique(offsets)))))
            fine_step = min(0.005, coarse_step / 10)
            fine_offsets = np.arange(coarse_best - coarse_step, coarse_best + coarse_step + fine_step / 2,
                                     fine_step)
            add_candidates(fine_offsets)
        if not candidates:
            return None
        best = min(candidates, key=lambda row: (row[0], row[1]))
        (fit_rmse, _, offset, rotation, translation, singular,
         train_matched, train_coverage, motion_fraction) = best
        sampled_test, valid_test = interpolate_valid(
            camera_times, indexes, camera_xyz, gt_times[test_ids] - offset, max_gap_s)
        if valid_test.sum() < 20 or valid_test.mean() < min_coverage:
            return None
        test_truth = gt_xyz[test_ids]
        aligned_test = sampled_test @ rotation.T + translation
        errors = np.linalg.norm((aligned_test[valid_test] - test_truth[valid_test]) * 1000, axis=1)
        all_errors.extend(errors.tolist())
        if valid_test.sum() >= 2:
            all_displacements.append(displacement_metrics(sampled_test[valid_test], test_truth[valid_test]))
        ratios = (singular / singular[0]).tolist() if singular[0] > 0 else [0.0, 0.0, 0.0]
        folds.append(dict(
            fold=fold_name,
            fit_section=("first_half" if fold_name == "first_to_second" else "second_half"),
            evaluation_section=("second_half" if fold_name == "first_to_second" else "first_half"),
            fit_gt_time_start_s=float(train_times[0]), fit_gt_time_end_s=float(train_times[-1]),
            evaluation_gt_time_start_s=float(gt_times[test_ids[0]]),
            evaluation_gt_time_end_s=float(gt_times[test_ids[-1]]),
            fit_time_offset_s=offset, fit_matched_groups=train_matched,
            fit_coverage=train_coverage, fit_motion_fraction=motion_fraction,
            fit_rmse_mm=fit_rmse, evaluation_groups=int(len(test_ids)),
            evaluation_matched_groups=int(valid_test.sum()),
            evaluation_coverage=float(valid_test.mean()),
            evaluation_rmse_mm=rms(errors),
            evaluation_mae_mm=float(np.mean(np.abs(errors))) if len(errors) else None,
            evaluation_p95_mm=float(np.percentile(errors, 95)) if len(errors) else None,
            rigid_fit=dict(rotation=rotation, translation_m=translation,
                           singular_values=singular, singular_value_ratios=ratios,
                           geometry_warning="near_collinear_motion" if ratios[1] < 1e-3 else None)))
        for local_index, gt_index in enumerate(test_ids):
            is_valid = bool(valid_test[local_index])
            camera_point = sampled_test[local_index]
            aligned_point = aligned_test[local_index]
            truth = test_truth[local_index]
            output_rows.append(dict(
                fold=fold_name, gt_index=int(gt_index), gt_time_s=float(gt_times[gt_index]),
                camera_time_s=float(gt_times[gt_index] - offset), fit_offset_s=offset,
                paired=is_valid,
                camera_x_m=float(camera_point[0]) if is_valid else "",
                camera_y_m=float(camera_point[1]) if is_valid else "",
                camera_z_m=float(camera_point[2]) if is_valid else "",
                gt_x_m=float(truth[0]), gt_y_m=float(truth[1]), gt_z_m=float(truth[2]),
                aligned_x_m=float(aligned_point[0]) if is_valid else "",
                aligned_y_m=float(aligned_point[1]) if is_valid else "",
                aligned_z_m=float(aligned_point[2]) if is_valid else "",
                residual_mm=float(np.linalg.norm((aligned_point - truth) * 1000))
                            if is_valid else ""))
    if not all_errors:
        return None
    fold_displacements = [fold for fold in all_displacements if fold.get("rmse_mm") is not None]
    return dict(
        method="two chronological halves; estimate offset and fit fixed-scale SE(3) on one half, evaluate on the other, then swap",
        scale=1.0, folds=folds,
        geometry_warning_second_singular_value_ratio_below=1e-3,
        heldout_groups=len(all_errors),
        heldout_coverage=len(all_errors) / len(gt_times),
        heldout_position_rmse_mm=rms(np.asarray(all_errors)),
        heldout_position_mae_mm=float(np.mean(np.abs(all_errors))),
        heldout_position_p95_mm=float(np.percentile(all_errors, 95)),
        heldout_displacement_folds=fold_displacements), output_rows


def interval_metrics(times: np.ndarray, camera: np.ndarray, laser: np.ndarray,
                     intervals=(0.5, 1.0, 2.0, 5.0), tolerance_s=0.035) -> list[dict]:
    rows = []
    for interval in intervals:
        errors = []
        for i, time in enumerate(times):
            j = np.searchsorted(times, time + interval)
            candidates = [index for index in (j - 1, j) if i < index < len(times)]
            if not candidates:
                continue
            end = min(candidates, key=lambda index: abs(times[index] - time - interval))
            if abs(times[end] - time - interval) > tolerance_s:
                continue
            cam_length = np.linalg.norm(camera[end] - camera[i])
            gt_length = np.linalg.norm(laser[end] - laser[i])
            errors.append((cam_length - gt_length) * 1000)
        values = np.asarray(errors)
        rows.append(dict(interval_s=interval, pairs=len(values), rmse_mm=rms(values),
                         mae_mm=float(np.mean(np.abs(values))) if len(values) else None,
                         bias_mm=float(np.mean(values)) if len(values) else None))
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _plot_modules():
    """Load optional plotting dependencies without making them a CLI requirement."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/marker_6dof_mplconfig")))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_static(out: Path) -> bool:
    plt = _plot_modules()
    if plt is None:
        return False
    points = json.loads((out / "static_points.json").read_text(encoding="utf-8"))
    pairs = read_csv(out / "static_pairs.csv")
    distances = [point["distance_cm"] for point in points]
    coverage = [point.get("coverage", 0) * 100 for point in points]
    jitter = [point.get("jitter_rms_mm", np.nan) for point in points]
    adjacent = [pair for pair in pairs if int(pair["to_cm"]) == int(pair["from_cm"]) + 10]
    labels = [f"{int(pair['from_cm'])}–{int(pair['to_cm'])}" for pair in adjacent]
    errors = [float(pair["error_mm"]) for pair in adjacent]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].bar(labels, errors, color=["#ba5345" if abs(value) > 10 else "#46769a" for value in errors])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Distance error (mm)")
    axes[0].set_title("Adjacent laser displacement error")
    axes[1].plot(distances, coverage, "o-", label="pose coverage (%)")
    axes[1].axhline(80, color="#a07035", linestyle="--", linewidth=1, label="80% reference")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Nominal distance (cm)")
    axes[1].set_ylabel("Coverage (%)")
    twin = axes[1].twinx()
    twin.plot(distances, jitter, "s-", color="#8b4c7f", label="jitter RMS (mm)")
    twin.axhline(3, color="#8b4c7f", linestyle=":", linewidth=1)
    twin.set_ylabel("Static jitter RMS (mm)")
    lines = [line for line in axes[1].get_lines() + twin.get_lines()
             if not line.get_label().startswith("_")]
    axes[1].legend(lines, [line.get_label() for line in lines], loc="best")
    fig.savefig(out / "static_accuracy.png", dpi=150)
    plt.close(fig)
    return True


def plot_dynamic(out: Path) -> bool:
    plt = _plot_modules()
    if plt is None:
        return False
    paired_path, scan_path = out / "paired_trajectories.csv", out / "time_offset_scan.csv"
    if not scan_path.exists():
        return False
    made_plot = False
    scan = read_csv(scan_path)
    if not scan:
        return made_plot
    if paired_path.exists():
        pairs = read_csv(paired_path)
        if pairs:
            times = np.array([float(row["gt_time_s"]) for row in pairs])
            elapsed = times - times[0]
            fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
            for axis_index, axis_name in enumerate("xyz"):
                truth = np.array([float(row[f"gt_{axis_name}_m"]) for row in pairs]) * 1000
                aligned = np.array([float(row[f"aligned_{axis_name}_m"]) for row in pairs]) * 1000
                axes[axis_index].plot(elapsed, truth, label="laser GT", linewidth=1.2)
                axes[axis_index].plot(elapsed, aligned, label="camera after fitted SE(3)", linewidth=1.0)
                axes[axis_index].set_ylabel(f"{axis_name.upper()} (mm)")
                if axis_index == 0:
                    axes[axis_index].legend(loc="best")
            residual = np.array([float(row["residual_mm"]) for row in pairs])
            axes[3].plot(elapsed, residual, color="#ba5345", linewidth=1)
            axes[3].axhline(0, color="black", linewidth=0.8)
            axes[3].set_ylabel("3D residual (mm)")
            axes[3].set_xlabel("Elapsed laser time (s)")
            fig.savefig(out / "trajectory_alignment.png", dpi=150)
            plt.close(fig)
            made_plot = True

    offsets = np.array([float(row["offset_s"]) for row in scan])
    coverage = np.array([float(row["coverage"]) * 100 for row in scan])
    rmse_values = np.array([float(row["fit_rmse_mm"]) if row["fit_rmse_mm"] else np.nan for row in scan])
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    axis.plot(offsets, rmse_values, label="SE(3) fit RMSE (mm)", color="#356e96")
    axis.set_xlabel("Laser minus camera time offset (s)")
    axis.set_ylabel("Fit RMSE (mm)")
    twin = axis.twinx()
    twin.plot(offsets, coverage, label="GT pair coverage (%)", color="#d08035", alpha=0.8)
    twin.axhline(80, color="#d08035", linestyle=":", linewidth=1)
    twin.set_ylabel("GT pair coverage (%)")
    lines = [line for line in axis.get_lines() + twin.get_lines()
             if not line.get_label().startswith("_")]
    axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.savefig(out / "time_offset_scan.png", dpi=150)
    plt.close(fig)
    made_plot = True
    if (out / "cross_validated_trajectories.csv").exists():
        made_plot = plot_cross_validated(out) or made_plot
    return made_plot


def plot_cross_validated(out: Path) -> bool:
    """Draw the two disjoint trajectory halves after cross-fitted SE(3)."""
    plt = _plot_modules()
    if plt is None:
        return False
    from matplotlib import font_manager
    for font_path in font_manager.findSystemFonts():
        if "NotoSansCJK-Regular" in font_path or "DroidSansFallback" in font_path:
            font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = read_csv(out / "cross_validated_trajectories.csv")
    rows.sort(key=lambda row: int(row["gt_index"]))
    if not rows:
        return False
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    cv = summary.get("cross_validation", {})
    colors = {"first_to_second": "#D87531", "second_to_first": "#237A6B"}
    truth = np.array([[float(row[f"gt_{axis}_m"]) for axis in "xyz"] for row in rows]) * 1000
    origin = truth[0].copy()
    truth -= origin
    elapsed = np.array([float(row["gt_time_s"]) for row in rows])
    elapsed -= elapsed[0]
    gt_relative = truth
    principal_axis = np.linalg.svd(gt_relative - gt_relative.mean(axis=0), full_matrices=False)[2][0]
    if np.dot(gt_relative[-1] - gt_relative[0], principal_axis) < 0:
        principal_axis *= -1
    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.35, 0.9, 0.9),
                            left=0.08, right=0.97, top=0.88, bottom=0.08,
                            hspace=0.43, wspace=0.25)
    progress_axis = fig.add_subplot(grid[0, 1])
    projections = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]),
                   fig.add_subplot(grid[1, 1])]
    error_axis = fig.add_subplot(grid[2, :])
    labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
    dimensions = [(0, 1), (0, 2), (1, 2)]
    progress_axis.plot(elapsed, gt_relative @ principal_axis,
                       color="#26394A", linewidth=2.0, label="激光真值")
    for axis, (x_dim, y_dim), (x_label, y_label) in zip(projections, dimensions, labels):
        axis.plot(gt_relative[:, x_dim], gt_relative[:, y_dim], color="#26394A",
                  linewidth=1.9, label="激光真值")
        axis.scatter(gt_relative[0, x_dim], gt_relative[0, y_dim], color="#26394A", marker="o", s=24)
        axis.scatter(gt_relative[-1, x_dim], gt_relative[-1, y_dim], color="#26394A", marker="X", s=32)
        axis.set_xlabel(f"{x_label} 位移 (mm)")
        axis.set_ylabel(f"{y_label} 位移 (mm)")
        if (x_dim, y_dim) == (0, 1):
            axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.22)
    for fold, color in colors.items():
        selected = [row for row in rows if row["fold"] == fold and row["paired"].lower() == "true"]
        if not selected:
            continue
        selected.sort(key=lambda row: int(row["gt_index"]))
        valid_ids = [int(row["gt_index"]) for row in selected]
        valid_gt = np.array([[float(row[f"gt_{axis}_m"]) for axis in "xyz"] for row in selected]) * 1000 - origin
        aligned = np.array([[float(row[f"aligned_{axis}_m"]) for axis in "xyz"] for row in selected]) * 1000 - origin
        errors = np.array([float(row["residual_mm"]) for row in selected])
        breaks = np.flatnonzero(np.diff(valid_ids) > 1) + 1
        for segment in np.split(np.arange(len(selected)), breaks):
            if not len(segment):
                continue
            segment_points = aligned[segment]
            progress_axis.plot(elapsed[np.asarray(valid_ids)[segment]],
                               segment_points @ principal_axis, color=color,
                               linewidth=1.4, linestyle="--",
                               label=("相机验证：后半段" if fold == "first_to_second" else "相机验证：前半段")
                               if segment[0] == 0 else None)
            for axis, (x_dim, y_dim) in zip(projections, dimensions):
                axis.plot(segment_points[:, x_dim], segment_points[:, y_dim], color=color,
                          linewidth=1.3, linestyle="--")
        stride = max(1, len(selected) // 24)
        for index in range(0, len(selected), stride):
            progress_axis.plot([elapsed[valid_ids[index]]] * 2,
                               [valid_gt[index] @ principal_axis, aligned[index] @ principal_axis],
                               color=color, alpha=0.2, linewidth=0.75)
            for axis, (x_dim, y_dim) in zip(projections, dimensions):
                axis.plot([valid_gt[index, x_dim], aligned[index, x_dim]],
                          [valid_gt[index, y_dim], aligned[index, y_dim]],
                          color=color, alpha=0.2, linewidth=0.65)
        error_axis.plot(elapsed[valid_ids], errors, color=color, linewidth=1.0,
                        label="后半段" if fold == "first_to_second" else "前半段")
    progress_axis.axvline(elapsed[len(rows) // 2], color="#8996A3", linestyle=":", linewidth=1)
    progress_axis.set_xlabel("激光相对时间 (s)")
    progress_axis.set_ylabel("沿主运动轴位移 (mm)")
    progress_axis.set_title("同一时刻的运动进度对齐", fontsize=11)
    progress_axis.grid(alpha=0.22)
    progress_axis.legend(loc="best", fontsize=8)
    for axis, (x_label, y_label) in zip(projections, labels):
        axis.set_title(f"{x_label}{y_label} 轨迹" + ("（Z 轴放大）" if y_label == "Z" else "（等比例）"))
    error_axis.axhline(0, color="#52606D", linewidth=0.8)
    error_axis.set_xlabel("激光相对时间 (s)")
    error_axis.set_ylabel("跨段 3D 位置误差 (mm)")
    error_axis.set_title("验证段逐点误差")
    error_axis.grid(alpha=0.22)
    error_axis.axvline(elapsed[len(rows) // 2], color="#8996A3", linestyle=":",
                       linewidth=1.0, label="前后半段分界")
    error_axis.legend(frameon=False, fontsize=8)
    rmse = cv.get("heldout_position_rmse_mm")
    p95 = cv.get("heldout_position_p95_mm")
    coverage = cv.get("heldout_coverage")
    subtitle = "尺度固定为 1；每半段只用另一半段拟合的时移与 SE(3) 变换来评估。"
    if rmse is not None:
        subtitle += f"  跨段 RMSE {rmse:.2f} mm · P95 {p95:.2f} mm · 覆盖 {coverage:.1%}。"
    fig.suptitle("连续轨迹：两段交叉对齐验证", x=0.08, y=0.975, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(0.08, 0.935, subtitle, ha="left", fontsize=9, color="#596774")
    fig.savefig(out / "trajectory_cross_validation.png", dpi=180)
    fig.savefig(out / "trajectory_cross_validation.svg", bbox_inches="tight")
    plt.close(fig)
    return True


def find_replay(root: Path, name: str, manifest: dict[str, str]) -> Path | None:
    if name in manifest:
        directory = Path(manifest[name]).expanduser()
        return directory if (directory / "poses.csv").exists() else None
    candidate = root / name
    return candidate if (candidate / "poses.csv").exists() else None


def evaluate_static(marker: str, root: Path, gt_root: Path, out: Path,
                    manifest: dict[str, str]) -> dict:
    gt_path = gt_root / MARKERS[marker]["gt_dir"] / "static.txt"
    laser, _ = load_gt(gt_path, dynamic=False)
    laser_by_label = {item["label"]: item for item in laser}
    points = []
    for cm in range(20, 101, 10):
        key = f"{marker}/static_{cm:03d}cm"
        directory = find_replay(root, key, manifest)
        if directory is None:
            points.append(dict(distance_cm=cm, status="missing_replay", canonical_key=key,
                               coverage=0.0))
            continue
        times, indexes, positions, info = load_replay(directory, MARKERS[marker]["id"],
                                                      MARKERS[marker]["black_square_mm"])
        if not len(positions):
            points.append(dict(distance_cm=cm, status="no_pose", **info))
            continue
        label = f"P_{cm}"
        if label not in laser_by_label:
            raise ValueError(f"missing GT static point {label} in {gt_path}")
        centre = np.median(positions, axis=0)
        radial = np.linalg.norm((positions - centre) * 1000, axis=1)
        points.append(dict(distance_cm=cm, status="ok", estimate_xyz_m=centre,
                           laser_xyz_m=laser_by_label[label]["xyz_m"],
                           jitter_rms_mm=rms(radial), jitter_p95_mm=float(np.percentile(radial, 95)),
                           std_xyz_mm=np.std(positions, axis=0) * 1000,
                           pose_sha256=sha256(directory / "poses.csv"),
                           frames_sha256=sha256(directory / "frames.csv"), **info))
    valid = [point for point in points if point["status"] == "ok"]
    pairs = []
    for i, first in enumerate(valid):
        for second in valid[i + 1:]:
            estimate_mm = float(np.linalg.norm(second["estimate_xyz_m"] - first["estimate_xyz_m"]) * 1000)
            laser_mm = float(np.linalg.norm(second["laser_xyz_m"] - first["laser_xyz_m"]) * 1000)
            pairs.append(dict(from_cm=first["distance_cm"], to_cm=second["distance_cm"],
                              estimate_mm=estimate_mm, laser_mm=laser_mm,
                              error_mm=estimate_mm - laser_mm))
    errors = np.asarray([pair["error_mm"] for pair in pairs])
    summary = dict(marker=marker, tag_id=MARKERS[marker]["id"], gt_file=str(gt_path),
                   gt_sha256=sha256(gt_path),
                   available_points=len(points), valid_points=len(valid), pairs=len(pairs),
                   pair_rmse_mm=rms(errors), pair_mae_mm=float(np.mean(np.abs(errors))) if len(errors) else None,
                   pair_bias_mm=float(np.mean(errors)) if len(errors) else None,
                   interpretation="Pairwise static displacement lengths; no spatial or scale fit. Laser provides position only, not 6DoF orientation. Tag centre versus laser target transform is unknown.")
    summary["quality_flags"] = []
    if any(point.get("status") != "ok" for point in points):
        summary["quality_flags"].append("missing_or_invalid_static_replay")
    if any(point.get("coverage", 0) < 0.8 for point in valid):
        summary["quality_flags"].append("static_pose_coverage_below_80pct")
    if any(point.get("jitter_rms_mm", 0) > 3.0 for point in valid):
        summary["quality_flags"].append("static_jitter_rms_above_3mm")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "static_points.json", points)
    write_csv(out / "static_pairs.csv", pairs,
              ["from_cm", "to_cm", "estimate_mm", "laser_mm", "error_mm"])
    write_json(out / "static_summary.json", summary)
    summary["plot_created"] = plot_static(out)
    write_json(out / "static_summary.json", summary)
    return summary


def evaluate_dynamic(marker: str, name: str, root: Path, gt_path: Path, out: Path,
                     manifest: dict[str, str], *, max_gap_s: float, min_coverage: float,
                     min_motion_fraction: float, scan_step_s: float,
                     scan_margin_s: float, max_group_spread_mm: float,
                     scan_center_override: float | None = None) -> dict | None:
    key = f"{marker}/{name}"
    directory = find_replay(root, key, manifest)
    if directory is None:
        result = dict(name=key, status="missing_replay", expected_gt_file=str(gt_path),
                      expected_replay=str(manifest.get(key, root / key)))
        missing_out = out / marker / name
        missing_out.mkdir(parents=True, exist_ok=True)
        write_json(missing_out / "summary.json", result)
        return result
    camera_times, indexes, camera_xyz, replay_info = load_replay(
        directory, MARKERS[marker]["id"], MARKERS[marker]["black_square_mm"])
    grouped, gt_info = load_gt(gt_path, dynamic=True)
    kept = [item for item in grouped if item["samples"] == 10 and item["spread_mm"] <= max_group_spread_mm]
    kept_times = {item["time_s"] for item in kept}
    if len(kept) < 3 or len(camera_times) < 3:
        result = dict(name=key, status="insufficient_poses_or_gt", replay=replay_info, gt=gt_info)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "summary.json", result)
        result["plot_created"] = plot_dynamic(out)
        write_json(out / "summary.json", result)
        return result
    gt_times = np.array([item["time_s"] for item in kept])
    gt_xyz = np.array([item["xyz_m"] for item in kept])
    full_motion_m = np.linalg.norm(np.ptp(gt_xyz, axis=0))
    center_offset = (gt_times[0] - camera_times[0] if scan_center_override is None
                     else scan_center_override)
    if scan_margin_s > 0 and scan_step_s > 0:
        offsets = center_offset + np.arange(-scan_margin_s, scan_margin_s + scan_step_s / 2,
                                            scan_step_s)
    else:
        offsets = np.asarray([center_offset])
    scan = []
    candidates = []
    for offset in offsets:
        sampled, valid = interpolate_valid(camera_times, indexes, camera_xyz,
                                           gt_times - offset, max_gap_s)
        matched = int(valid.sum())
        coverage = matched / len(gt_times)
        motion_m = np.linalg.norm(np.ptp(gt_xyz[valid], axis=0)) if matched else 0
        motion_fraction = motion_m / full_motion_m if full_motion_m else 1.0
        eligible = matched >= 20 and coverage >= min_coverage and motion_fraction >= min_motion_fraction
        residual_mm = None
        if eligible:
            rotation, translation, _ = rigid_fit(sampled[valid], gt_xyz[valid])
            residual_mm = rms(np.linalg.norm((sampled[valid] @ rotation.T + translation - gt_xyz[valid]) * 1000, axis=1))
            candidates.append((residual_mm, -matched, float(offset), sampled, valid))
        scan.append(dict(offset_s=float(offset), matched=matched, coverage=coverage,
                         motion_fraction=motion_fraction, fit_rmse_mm=residual_mm,
                         eligible=eligible))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "time_offset_scan.csv", scan,
              ["offset_s", "matched", "coverage", "motion_fraction", "fit_rmse_mm", "eligible"])
    write_csv(out / "laser_grouped.csv", [dict(time_s=item["time_s"], x_m=item["xyz_m"][0],
                                                y_m=item["xyz_m"][1], z_m=item["xyz_m"][2],
                                                samples=item["samples"], spread_mm=item["spread_mm"],
                                                eligible=item["time_s"] in kept_times) for item in grouped],
              ["time_s", "x_m", "y_m", "z_m", "samples", "spread_mm", "eligible"])
    if not candidates:
        best_overlap = max(scan, key=lambda item: item["coverage"])
        result = dict(name=key, status="no_valid_time_offset", replay=replay_info, gt=gt_info,
                      excluded_gt_groups=len(grouped) - len(kept),
                      scan_center_offset_s=float(center_offset), scan_margin_s=scan_margin_s,
                      max_overlap_candidate=best_overlap,
                      quality_flags=["no_offset_candidate_met_minimum_overlap_and_motion"])
        write_json(out / "summary.json", result)
        result["plot_created"] = plot_dynamic(out)
        write_json(out / "summary.json", result)
        return result
    _, _, best_offset, sampled, valid = min(candidates, key=lambda item: (item[0], item[1]))
    paired_cam = sampled[valid]
    paired_gt = gt_xyz[valid]
    paired_times = gt_times[valid]
    rotation, translation, singular = rigid_fit(paired_cam, paired_gt)
    aligned = paired_cam @ rotation.T + translation
    residual = np.linalg.norm((aligned - paired_gt) * 1000, axis=1)
    pair_rows = []
    for time, estimate, truth, transformed, error in zip(paired_times, paired_cam, paired_gt, aligned, residual):
        pair_rows.append(dict(gt_time_s=time, camera_time_s=time - best_offset,
                              camera_x_m=estimate[0], camera_y_m=estimate[1], camera_z_m=estimate[2],
                              gt_x_m=truth[0], gt_y_m=truth[1], gt_z_m=truth[2],
                              aligned_x_m=transformed[0], aligned_y_m=transformed[1],
                              aligned_z_m=transformed[2], residual_mm=error))
    write_csv(out / "paired_trajectories.csv", pair_rows, list(pair_rows[0]))
    intervals = interval_metrics(paired_times, paired_cam, paired_gt)
    write_csv(out / "relative_intervals.csv", intervals,
              ["interval_s", "pairs", "rmse_mm", "mae_mm", "bias_mm"])
    cross_validation, cross_validated_rows = cross_validated_se3(
        camera_times, indexes, camera_xyz, gt_times, gt_xyz, offsets,
        max_gap_s=max_gap_s, min_coverage=min_coverage,
        min_motion_fraction=min_motion_fraction)
    if cross_validation is not None:
        write_csv(out / "cross_validated_trajectories.csv", cross_validated_rows,
                  list(cross_validated_rows[0]))
    else:
        for stale in ("cross_validated_trajectories.csv", "trajectory_cross_validation.png",
                      "trajectory_cross_validation.svg"):
            (out / stale).unlink(missing_ok=True)
    relative = displacement_metrics(paired_cam, paired_gt)
    quality_flags = []
    if replay_info.get("replay_frame_coverage") is not None and replay_info["replay_frame_coverage"] < 0.8:
        quality_flags.append("replay_frame_coverage_below_80pct")
    if len(pair_rows) / len(kept) < 0.8:
        quality_flags.append("gt_pair_coverage_below_80pct")
    if (cross_validation is not None and any(
            fold["rigid_fit"].get("geometry_warning") for fold in cross_validation["folds"])):
        quality_flags.append("cross_validation_fit_near_collinear")
    if cross_validation is None:
        quality_flags.append("cross_validation_unavailable_insufficient_two_segment_support")
    if replay_info.get("duplicate_selection_counts"):
        quality_flags.append("duplicate_or_ambiguous_marker_candidates_present")
    summary = dict(name=key, status="ok", quality_flags=quality_flags,
                   replay=replay_info, gt_file=str(gt_path), gt=gt_info,
                   gt_sha256=sha256(gt_path), pose_sha256=sha256(directory / "poses.csv"),
                   frames_sha256=sha256(directory / "frames.csv"),
                   gt_groups_eligible=len(kept), gt_groups_excluded=len(grouped) - len(kept),
                   paired_groups=len(pair_rows), gt_group_coverage=len(pair_rows) / len(kept),
                   time_offset_s=best_offset, time_convention="laser_time = camera_header_time + offset",
                   scan_center_offset_s=float(center_offset), scan_margin_s=scan_margin_s,
                   trajectory_rmse_mm=rms(residual), trajectory_mae_mm=float(np.mean(residual)),
                   trajectory_p95_mm=float(np.percentile(residual, 95)),
                   relative_all_pairs=relative, intervals=intervals,
                   rigid_fit=dict(rotation=rotation, translation_m=translation,
                                  singular_values=singular, scale=1.0),
                   cross_validation=cross_validation,
                   interpretation="SE(3) and time shift fitted on this same sequence: descriptive aligned trajectory error, not independent absolute accuracy. Raw displacement length errors require no spatial fit. Laser target and tag centre transform is unknown; laser has no orientation ground truth, so 6DoF orientation accuracy cannot be assessed.")
    write_json(out / "summary.json", summary)
    summary["plots_created"] = plot_dynamic(out)
    write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--gt-root", type=Path, default=Path("/home/dex/20261030/GT"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="optional JSON: canonical key -> replay directory")
    parser.add_argument("--marker", choices=MARKERS, action="append", help="select markers; default all")
    parser.add_argument("--max-interp-gap-s", type=float, default=0.1)
    parser.add_argument("--min-gt-coverage", type=float, default=0.65,
                        help="minimum matched laser time groups required for time-offset candidate selection")
    parser.add_argument("--min-motion-fraction", type=float, default=0.75)
    parser.add_argument("--scan-step-s", type=float, default=0.05)
    parser.add_argument("--scan-margin-s", type=float, default=20.0)
    parser.add_argument("--scan-center-s", type=float, help="fixed offset center for a camera with a different host clock")
    parser.add_argument("--max-group-spread-mm", type=float, default=60.0)
    args = parser.parse_args()
    if not (0 < args.max_interp_gap_s <= 1 and 0 < args.min_gt_coverage <= 1 and
            0 < args.min_motion_fraction <= 1 and 0 < args.scan_step_s <= 1 and
            args.scan_margin_s >= 0 and args.max_group_spread_mm > 0):
        parser.error("invalid scan, coverage, interpolation or laser group limits")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else {}
    if not isinstance(manifest, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                 for key, value in manifest.items()):
        parser.error("manifest must be a JSON object of canonical key to replay directory")
    markers = args.marker or list(MARKERS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for marker in markers:
        gt_dir = args.gt_root / MARKERS[marker]["gt_dir"]
        if MARKERS[marker]["static"]:
            summaries.append(evaluate_static(marker, args.replay_root, args.gt_root,
                                             args.output_dir / marker, manifest))
        dynamic_names = [("continuous_static_20_100", gt_dir / MARKERS[marker]["continuous"])]
        dynamic_names += [(f"dynamic_track_{index:02d}", gt_dir /
                           (f"track{index}.txt" if marker != "marker70" else "track.txt"))
                          for index in range(1, MARKERS[marker]["tracks"] + 1)]
        for name, gt_path in dynamic_names:
            result = evaluate_dynamic(marker, name, args.replay_root, gt_path,
                                      args.output_dir / marker / name, manifest,
                                      max_gap_s=args.max_interp_gap_s,
                                      min_coverage=args.min_gt_coverage,
                                      min_motion_fraction=args.min_motion_fraction,
                                      scan_step_s=args.scan_step_s, scan_margin_s=args.scan_margin_s,
                                      max_group_spread_mm=args.max_group_spread_mm,
                                      scan_center_override=args.scan_center_s)
            summaries.append(result)
    calibration_sources = {}
    for result in summaries:
        replay_info = result.get("replay", {})
        path = replay_info.get("camchain")
        if path:
            calibration_sources[path] = replay_info.get("camchain_sha256")
    protocol = dict(test_date="2026-10-30", replay_root=args.replay_root.resolve(),
                    ground_truth_root=args.gt_root.resolve(), output_dir=args.output_dir.resolve(),
                    tag_ids={name: spec["id"] for name, spec in MARKERS.items()},
                    black_square_size_mm={name: spec["black_square_mm"] for name, spec in MARKERS.items()},
                    camera_calibration_candidates=calibration_sources,
                    defaults=dict(max_interp_gap_s=args.max_interp_gap_s,
                                  min_gt_coverage=args.min_gt_coverage,
                                  min_motion_fraction=args.min_motion_fraction,
                                  scan_step_s=args.scan_step_s,
                                  scan_margin_s=args.scan_margin_s,
                                  scan_center_s=args.scan_center_s,
                                  max_group_spread_mm=args.max_group_spread_mm),
                    mapping_manifest=manifest)
    write_json(args.output_dir / "evaluation_protocol.json", protocol)
    write_json(args.output_dir / "summary.json", dict(protocol=protocol, results=summaries))
    lines = ["# 2026-10-30 激光精度评估", "",
             f"Marker 黑码边长：marker50=37.5 mm、marker60=50 mm、marker70=60 mm；ID 分别为 2、2、6。回放使用的相机内参及 SHA256 见 `evaluation_protocol.json`：{', '.join(f'`{path}`' for path in calibration_sources) or '回放摘要中未记录'}。",
             "静态指标为各距离中位数的两两位移长度误差，无空间或尺度拟合。逐点覆盖率与抖动见各 Marker 的 `static_points.json`，全点对见 `static_pairs.csv`。",
             "动态位移长度误差不做空间拟合；同段轨迹位置残差会在本段拟合时移和固定尺度 SE(3)，仅作对齐示意。另按时间切分前后连续两段，分别用一段拟合并在另一段评估，再交换两段；该跨段指标比同段残差更能检查过拟合，但仍不等于独立绝对定位精度。",
             "GT 为激光测量点，回放为 Tag 中心；二者的实测固定外参未知，动态残差包含靶点偏移。GT 只有位置，无姿态，不能评估 6DoF 姿态精度。",
             "", "| 数据 | 状态 | 有效/GT | 回放覆盖 | 主指标 | 质量标记 |", "| --- | --- | ---: | ---: | --- | --- |"]
    for result in summaries:
        if "pair_rmse_mm" in result:
            lines.append(f"| {result['marker']} 静态 | ok | {result['valid_points']}/{result['available_points']} 点 | — | 点间 RMSE {result['pair_rmse_mm']:.2f} mm | {', '.join(result['quality_flags']) or '—'} |")
        elif result["status"] == "ok":
            flags = ", ".join(result["quality_flags"]) or "—"
            coverage = result["replay"].get("replay_frame_coverage")
            replay_coverage = f"{coverage:.1%}" if coverage is not None else "—"
            cv = result.get("cross_validation")
            cv_text = (f"；跨段验证 RMSE {cv['heldout_position_rmse_mm']:.2f} mm / "
                       f"P95 {cv['heldout_position_p95_mm']:.2f} mm"
                       if cv and cv.get("heldout_position_rmse_mm") is not None else "")
            lines.append(f"| {result['name']} | ok | {result['paired_groups']}/{result['gt_groups_eligible']} 组 ({result['gt_group_coverage']:.1%}) | {replay_coverage} | 位移 RMSE {result['relative_all_pairs']['rmse_mm']:.2f} mm；同段对齐 RMSE {result['trajectory_rmse_mm']:.2f} mm{cv_text} | {flags} |")
        else:
            replay = result.get("replay", {})
            coverage = replay.get("replay_frame_coverage")
            replay_coverage = f"{coverage:.1%}" if coverage is not None else "—"
            flags = ", ".join(result.get("quality_flags", [])) or "see summary.json"
            lines.append(f"| {result['name']} | {result['status']} | — | {replay_coverage} | — | {flags} |")
    for result in summaries:
        if "pair_rmse_mm" not in result:
            continue
        point_file = args.output_dir / result["marker"] / "static_points.json"
        points = json.loads(point_file.read_text(encoding="utf-8"))
        pair_file = args.output_dir / result["marker"] / "static_pairs.csv"
        pairs = read_csv(pair_file)
        by_cm = {point["distance_cm"]: point for point in points}
        lines += ["", f"#### {result['marker']} 静态点覆盖与相邻间距误差", "",
                  "| 距离 cm | Tag 位姿覆盖率 | 抖动 RMS mm | 相邻估计位移误差 mm |",
                  "| ---: | ---: | ---: | ---: |"]
        adjacent_by_start = {int(pair["from_cm"]): float(pair["error_mm"])
                             for pair in pairs if int(pair["to_cm"]) == int(pair["from_cm"]) + 10}
        for cm in range(20, 101, 10):
            point = by_cm[cm]
            if point["status"] != "ok":
                lines.append(f"| {cm} | — | — | — |")
                continue
            edge_error = adjacent_by_start.get(cm)
            edge_text = f"{edge_error:+.2f}" if edge_error is not None else "—"
            lines.append(f"| {cm} | {point['coverage']:.1%} | {point['jitter_rms_mm']:.2f} | {edge_text} |")
    lines += ["", "每个静态 Marker 目录会生成 `static_accuracy.png`。动态文件夹包含 `time_offset_scan.csv`（所有偏移候选、有效重叠和运动跨度）、`paired_trajectories.csv`、`cross_validated_trajectories.csv`、`relative_intervals.csv`、`trajectory_alignment.png`、`trajectory_cross_validation.png`（PNG/SVG）和 `time_offset_scan.png`，以及带输入哈希和质量字段的 `summary.json`。交叉验证按时间切成两段：一段拟合时间偏移与固定尺度 SE(3)，另一段评估，再交换两段。图中的深蓝色是真值轨迹，虚线是用另一段拟合变换后的相机轨迹，连接线表示配对位置残差；右下角给出跨段误差随时间变化。", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"evaluated {len(summaries)} cases; report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

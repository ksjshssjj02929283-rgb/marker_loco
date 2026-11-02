#!/usr/bin/env python3
"""Create report figures from the frozen 2026-10-30 Kalibr evaluations.

This module only reads evaluation results. It does not rerun estimation, alter
time alignment, or remove samples. ``create_figures`` returns PNG and SVG paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


CAMERAS = (("left", "Left camera", "#2367A6"), ("right", "Right camera", "#D87531"))
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#A6ADB5",
    "axes.labelcolor": "#26313C",
    "text.color": "#26313C",
    "xtick.color": "#42505C",
    "ytick.color": "#42505C",
    "grid.color": "#DFE4E9",
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation(root: Path, camera: str) -> Path:
    return root / f"evaluation_{camera}_filtered"


def _summary(root: Path, camera: str, marker: str, sequence: str) -> dict:
    result = _read_json(_evaluation(root, camera) / marker / sequence / "summary.json")
    if result.get("status") != "ok":
        raise ValueError(f"Cannot plot unsuccessful evaluation: {camera}/{marker}/{sequence}")
    return result


def _interval_rmse(summary: dict, interval: float = 1.0) -> float:
    matches = [entry["rmse_mm"] for entry in summary["intervals"] if entry["interval_s"] == interval]
    if len(matches) != 1:
        raise ValueError(f"Expected one interval metric at {interval} seconds")
    return float(matches[0])


def _save(fig, directory: Path, name: str) -> list[Path]:
    paths = [directory / f"{name}.{extension}" for extension in ("png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def _figure_title(fig, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.08, y=0.99, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.08, 0.935, subtitle, ha="left", fontsize=10, color="#596774")


def _static_figure(root: Path, directory: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for row, marker in enumerate(("marker50", "marker60")):
        for camera, label, color in CAMERAS:
            points = _read_json(_evaluation(root, camera) / marker / "static_points.json")
            points = sorted((point for point in points if point["status"] == "ok"), key=lambda p: p["distance_cm"])
            distances = np.array([point["distance_cm"] for point in points])
            estimates = np.array([point["estimate_xyz_m"] for point in points])
            truth = np.array([point["laser_xyz_m"] for point in points])
            if distances[0] != 20:
                raise ValueError(f"Expected nominal 20 cm reference for {camera}/{marker}")
            error = 1000 * (np.linalg.norm(estimates - estimates[0], axis=1) - np.linalg.norm(truth - truth[0], axis=1))
            axes[row, 0].plot(distances, [point["jitter_rms_mm"] for point in points], "o-", color=color, lw=1.8, ms=5, label=label)
            axes[row, 1].plot(distances, error, "o-", color=color, lw=1.8, ms=5, label=label)
        axes[row, 0].set_title(f"{marker}  |  Static position jitter")
        axes[row, 0].set_ylabel("Position jitter RMS (mm)")
        axes[row, 0].set_ylim(bottom=0)
        axes[row, 1].set_title(f"{marker}  |  Displacement from 20 cm reference")
        axes[row, 1].set_ylabel("Signed displacement-length error (mm)")
        axes[row, 1].axhline(0, color="#77818C", lw=0.8, zorder=0)
        axes[row, 0].axhline(3, color="#5B6470", lw=1.1, ls="--", zorder=0)
        for ax in axes[row]:
            ax.set_xticks(distances)
            ax.set_xlabel("Nominal position label (cm)")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
    static_handles = [
        Line2D([0], [0], color=color, marker="o", lw=1.8, label=label)
        for _, label, color in CAMERAS
    ]
    static_handles.append(Line2D([0], [0], color="#5B6470", lw=1.1, ls="--", label="Requested 3 mm jitter target"))
    axes[0, 0].legend(handles=static_handles, frameon=False, loc="upper left", fontsize=9)
    axes[0, 1].legend(frameon=False, loc="upper left")
    _figure_title(fig, "Static measurements: jitter and displacement error", "Each camera uses its own PnP estimate. Positive signed error means the estimated displacement is too long.")
    fig.text(0.08, 0.018, "Jitter measures within-recording repeatability. Displacement uses each recording's median position; no spatial or scale fit.", fontsize=9, color="#596774")
    fig.subplots_adjust(top=0.88, bottom=0.105, left=0.08, right=0.985, hspace=0.4, wspace=0.27)
    return _save(fig, directory, "static_jitter_and_displacement")


def _grouped_horizontal_bars(ax, labels: list[str], values: dict[str, list[float]]) -> None:
    y = np.arange(len(labels))
    bar_height = 0.31
    for index, (camera, label, color) in enumerate(CAMERAS):
        bars = ax.barh(y + (index - 0.5) * bar_height, values[camera], height=bar_height, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(max(series) for series in values.values()) * 1.2)
    ax.set_xlabel("Displacement-length RMSE (mm)")


def _dynamic_figure(root: Path, directory: Path) -> list[Path]:
    tracks = [(marker, track) for marker, count in (("marker50", 4), ("marker60", 4), ("marker70", 1)) for track in range(1, count + 1)]
    labels = [f"{marker} / DT-{track}" for marker, track in tracks]
    metrics = [{camera: [] for camera, _, _ in CAMERAS} for _ in range(2)]
    for camera, _, _ in CAMERAS:
        for marker, track in tracks:
            summary = _summary(root, camera, marker, f"dynamic_track_{track:02d}")
            metrics[0][camera].append(_interval_rmse(summary))
            metrics[1][camera].append(summary["relative_all_pairs"]["rmse_mm"])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.5), sharey=True)
    for ax, values, title in zip(axes, metrics, ("Displacements over 1 second", "Displacements over all paired times")):
        _grouped_horizontal_bars(ax, labels, values)
        ax.axvline(3, color="#5B6470", lw=1.1, ls="--", zorder=0)
        ax.set_title(title, pad=14)
        ax.legend(handles=[
            Line2D([0], [0], color=color, lw=5, label=label)
            for _, label, color in CAMERAS
        ] + [Line2D([0], [0], color="#5B6470", lw=1.1, ls="--", label="Requested 3 mm target")],
            frameon=False, loc="lower right", fontsize=9)
    _figure_title(fig, "Dynamic trajectories: nine recordings", "The two panels measure different displacement lengths and should not be treated as the same accuracy metric.")
    fig.text(0.08, 0.02, "Time offset is fitted separately for each camera and recording. Neither metric fits a spatial transform or scale. No pooled score.", fontsize=9, color="#596774")
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.16, right=0.985, wspace=0.11)
    return _save(fig, directory, "dynamic_displacement_rmse")


def _continuous_figure(root: Path, directory: Path) -> list[Path]:
    markers = ("marker50", "marker60", "marker70")
    labels = ["marker50", "marker60\n(extended range)", "marker70"]
    metrics = [{camera: [] for camera, _, _ in CAMERAS} for _ in range(2)]
    for camera, _, _ in CAMERAS:
        for marker in markers:
            summary = _summary(root, camera, marker, "continuous_static_20_100")
            metrics[0][camera].append(_interval_rmse(summary))
            metrics[1][camera].append(summary["relative_all_pairs"]["rmse_mm"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    x = np.arange(len(labels))
    width = 0.32
    for ax, values, title in zip(axes, metrics, ("Displacements over 1 second", "Displacements over all paired times")):
        for index, (camera, label, color) in enumerate(CAMERAS):
            bars = ax.bar(x + (index - 0.5) * width, values[camera], width, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
        ax.axhline(3, color="#5B6470", lw=1.1, ls="--", zorder=0)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, max(max(series) for series in values.values()) * 1.22)
        ax.set_ylabel("Displacement-length RMSE (mm)")
        ax.set_title(title, pad=14)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(handles=[
            Line2D([0], [0], color=color, lw=5, label=label)
            for _, label, color in CAMERAS
        ] + [Line2D([0], [0], color="#5B6470", lw=1.1, ls="--", label="Requested 3 mm target")],
            frameon=False, loc="upper left", fontsize=9)
    _figure_title(fig, "Continuous recordings with stationary pauses", "All paired samples, including motion between pauses, are included. Each recording spans a different motion range.")
    fig.text(0.08, 0.01, "marker60 spans about 1.29 m of laser displacement; its filename is not evidence of a 20-100 cm camera-distance limit.", fontsize=9, color="#596774")
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.08, right=0.985, wspace=0.25)
    return _save(fig, directory, "continuous_displacement_rmse")


def _extended_range_figure(root: Path, directory: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for camera, label, color in CAMERAS:
        path = _evaluation(root, camera) / "marker60/continuous_static_20_100/paired_trajectories.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        estimates = np.array([[float(row[f"camera_{axis}_m"]) for axis in "xyz"] for row in rows])
        truth = np.array([[float(row[f"gt_{axis}_m"]) for axis in "xyz"] for row in rows])
        truth_displacement = np.linalg.norm(truth - truth[0], axis=1)
        signed_error = 1000 * (np.linalg.norm(estimates - estimates[0], axis=1) - truth_displacement)
        ax.plot(truth_displacement, signed_error, color=color, lw=1.2, alpha=0.9, label=label)
    ax.axhline(0, color="#77818C", lw=0.8, zorder=0)
    ax.axvline(0.8, color="#7B8793", lw=1.1, ls="--", label="0.8 m displacement reference")
    ax.set_xlabel("Laser displacement from first paired point (m)")
    ax.set_ylabel("Signed displacement-length error (mm)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    _figure_title(fig, "marker60 continuous recording: error over displacement", "Each curve uses that camera's first paired point as its reference. Positive error means estimated displacement is too long.")
    fig.text(0.08, 0.02, "The horizontal axis is displacement from a laser point, not camera-to-marker distance. No spatial transform or scale is fitted.", fontsize=9, color="#596774")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.985)
    return _save(fig, directory, "marker60_extended_range")


def _static_range_figure(root: Path, directory: Path) -> list[Path]:
    endpoints = np.arange(30, 101, 10)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    camera_colors = {camera: color for camera, _, color in CAMERAS}
    line_styles = {"marker50": "-", "marker60": "--"}
    for marker in ("marker50", "marker60"):
        for camera, label, _ in CAMERAS:
            path = _evaluation(root, camera) / marker / "static_pairs.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            points = []
            for endpoint in endpoints:
                errors = np.array([
                    float(row["error_mm"])
                    for row in rows
                    if float(row["from_cm"]) >= 20 and float(row["to_cm"]) <= endpoint
                ])
                if not len(errors):
                    raise ValueError(f"No static point pairs through {endpoint} cm: {camera}/{marker}")
                points.append(float(np.sqrt(np.mean(errors**2))))
            ax.plot(endpoints, points, marker="o", lw=1.8, ms=5,
                    color=camera_colors[camera], linestyle=line_styles[marker],
                    label=f"{marker} / {label}")
    ax.axhline(3, color="#202830", lw=1.2, ls=":", label="Requested 3 mm displacement RMSE target")
    ax.set_xticks(endpoints)
    ax.set_xlabel("Farthest nominal position included (cm)")
    ax.set_ylabel("Cumulative all-pairs displacement error RMSE (mm)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    _figure_title(fig, "Static measurement range against a 3 mm target", "For each endpoint, all pairwise distances among nominal positions 20 cm through that endpoint are included.")
    fig.text(0.08, 0.02, "RMSE combines signed over- and under-estimation as squared errors. The target is a requested reference, not a pass/fail certification.", fontsize=9, color="#596774")
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.08, right=0.985)
    return _save(fig, directory, "static_range_3mm")


def create_figures(root: Path, output: Path) -> list[Path]:
    """Read both filtered evaluations in ``root`` and write ten figure files."""
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with plt.rc_context(STYLE):
        for make_figure in (_static_figure, _dynamic_figure, _continuous_figure, _extended_range_figure, _static_range_figure):
            paths.extend(make_figure(Path(root), directory))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/accuracy_20261030_kalibr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in create_figures(args.root, args.output):
        print(path)


if __name__ == "__main__":
    main()

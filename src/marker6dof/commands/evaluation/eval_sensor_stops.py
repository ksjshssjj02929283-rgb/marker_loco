"""Estimate static depths and visibility from a calibrated sensor-bag replay.

No absolute tape reference is assumed. Static candidates use only observed
motion, never the expected 20 cm step, and remain reviewable in the plot.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def find_static_intervals(times, positions, *, window_s=.8, speed_mm_s=20.,
                          min_stop_s=1.5, merge_mm=35.):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(times) < 3:
        return []
    if np.any(np.diff(times) <= 0):
        raise ValueError("pose timestamps must be strictly increasing")
    # Robust short-window centres reduce single-frame planar-PnP jitter. The
    # unfiltered positions are retained for all final noise/depth statistics.
    smooth = np.array([np.median(positions[(times >= t - .15) & (times <= t + .15)], axis=0)
                       for t in times])
    stationary = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        lo = np.searchsorted(times, t - window_s / 2)
        hi = np.searchsorted(times, t + window_s / 2, side="right") - 1
        dt = times[hi] - times[lo]
        if dt < window_s * .7 or np.max(np.diff(times[lo:hi + 1]), initial=0) > .2:
            continue
        speed = np.linalg.norm(smooth[hi] - smooth[lo]) / dt * 1000
        span = np.max(np.linalg.norm(smooth[lo:hi + 1] - np.median(smooth[lo:hi + 1], axis=0), axis=1)) * 1000
        stationary[i] = speed <= speed_mm_s and span <= max(8., speed_mm_s * window_s / 2)
    candidates = []
    start = None
    for i, valid in enumerate(stationary):
        if valid and (i == 0 or times[i] - times[i - 1] <= .2):
            if start is None:
                start = i
        else:
            if start is not None and times[i - 1] - times[start] >= min_stop_s:
                candidates.append((start, i - 1))
            start = i if valid else None
    if start is not None and times[-1] - times[start] >= min_stop_s:
        candidates.append((start, len(times) - 1))
    merged = []
    for a, b in candidates:
        if merged:
            c, d = merged[-1]
            # Merge only nearby pauses within 2 s. Retain the longer actual
            # static run rather than counting the intervening motion as static.
            same = np.linalg.norm(np.median(positions[a:b+1], axis=0)
                                  - np.median(positions[c:d+1], axis=0)) * 1000 <= merge_mm
            if same and times[a] - times[d] <= 2:
                if times[b] - times[a] > times[d] - times[c]:
                    merged[-1] = (a, b)
                continue
        merged.append((a, b))
    return merged


def analyze(replay_dir, tag_id=2, sequence_end_s=None, speed_mm_s=20., min_stop_s=1.5,
            reviewed_windows=()):
    replay_dir = Path(replay_dir)
    summary = json.loads((replay_dir / "summary.json").read_text())
    frames = list(csv.DictReader((replay_dir / "frames.csv").open()))
    rows = [r for r in csv.DictReader((replay_dir / "poses.csv").open()) if int(r["tag_id"]) == tag_id]
    if not rows:
        raise ValueError(f"tag {tag_id} has no accepted poses")
    times = np.array([float(r["timestamp_s"]) for r in rows])
    positions = np.array([[float(r[k]) for k in ("tx_m", "ty_m", "tz_m")] for r in rows])
    errors = np.array([float(r["reprojection_error_px"]) for r in rows])
    edges = np.array([float(r["min_edge_px"]) for r in rows])
    intervals = [(a, b, "automatic", float(times[a]), float(times[b])) for a, b in
                 find_static_intervals(times, positions, speed_mm_s=speed_mm_s, min_stop_s=min_stop_s)]
    for start, end in reviewed_windows:
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= float(frames[-1]["timestamp_s"]):
            raise ValueError("reviewed windows must lie within the recording")
        indices = np.flatnonzero((times >= start) & (times <= end))
        if len(indices) < 3:
            raise ValueError(f"reviewed window {start}-{end} has fewer than three accepted poses")
        # The explicit window replaces overlapping auto candidates. It does
        # not satisfy automatic stillness checks merely because it was supplied.
        intervals = [item for item in intervals if item[4] < start or item[3] > end]
        intervals.append((int(indices[0]), int(indices[-1]), "visually_reviewed_hold", start, end))
    intervals.sort(key=lambda item: item[3])
    points = []
    for a, b, source, start, end in intervals:
        in_frames = [r for r in frames if start <= float(r["timestamp_s"]) <= end]
        samples = positions[a:b + 1]
        point = dict(index=len(points) + 1, start_s=start, end_s=end, selection=source,
                     frame_range=[int(rows[a]["frame_index"]), int(rows[b]["frame_index"])],
                     duration_s=end - start, accepted_frames=b - a + 1,
                     total_frames=len(in_frames), coverage=(b - a + 1) / len(in_frames),
                     median_xyz_m=np.median(samples, axis=0).tolist(),
                     z_median_m=float(np.median(samples[:, 2])), z_mean_m=float(np.mean(samples[:, 2])),
                     z_std_mm=float(np.std(samples[:, 2]) * 1000),
                     z_p05_p95_m=np.percentile(samples[:, 2], [5, 95]).tolist(),
                     range_median_m=float(np.median(np.linalg.norm(samples, axis=1))),
                     edge_median_px=float(np.median(edges[a:b + 1])),
                     strict_24px_fraction=float(np.mean(edges[a:b + 1] >= 24)),
                     reprojection_p50_px=float(np.median(errors[a:b + 1])),
                     sequence="primary" if sequence_end_s is None or end <= sequence_end_s else "after_view_change")
        if points and point["sequence"] == points[-1]["sequence"]:
            delta = np.asarray(point["median_xyz_m"]) - points[-1]["median_xyz_m"]
            point["previous_delta_z_cm"] = float(delta[2] * 100)
            point["previous_distance_3d_cm"] = float(np.linalg.norm(delta) * 100)
        points.append(point)
    bins = []
    for start in np.arange(0, float(frames[-1]["timestamp_s"]) + .001, 2.):
        fr = [r for r in frames if start <= float(r["timestamp_s"]) < start + 2]
        ix = (times >= start) & (times < start + 2)
        if not fr:
            continue
        bins.append(dict(start_s=float(start), end_s=float(start + 2), frames=len(fr),
                         detected=sum(tag_id in json.loads(r["detected_ids"]) for r in fr),
                         accepted=int(ix.sum()), strict=int(np.sum(ix & (edges >= 24))),
                         z_median_m=float(np.median(positions[ix, 2])) if ix.any() else None))
    primary = [p for p in points if p["sequence"] == "primary"]
    supported = [p for p in points if p["coverage"] >= .8]
    strict_points = [p for p in supported if p["strict_24px_fraction"] >= .8]
    result = dict(replay=summary, tag_id=tag_id,
                  parameters=dict(window_s=.8, max_speed_mm_s=speed_mm_s, min_stop_s=min_stop_s,
                                  merge_mm=35, sequence_end_s=sequence_end_s,
                                  reviewed_windows=list(reviewed_windows), high_coverage_threshold=.8),
                  static_points=points, visibility_bins=bins,
                  observed_max_static_z_m=max((p["z_median_m"] for p in points), default=None),
                  primary_sequence_max_static_z_m=max((p["z_median_m"] for p in primary), default=None),
                  high_coverage_max_static_z_m=max((p["z_median_m"] for p in supported), default=None),
                  strict_24px_high_coverage_max_static_z_m=max((p["z_median_m"] for p in strict_points), default=None),
                  interpretation="Empirical visibility in this recording, not a measured sensor range limit. Static-depth spread includes hand motion and PnP jitter. Visually reviewed holds can fail the automatic stillness gate and include sparse accepted samples; their depth estimates are conditional on successful detection. Depth is camera optical-axis Z at this tag centre, not TCP or Euclidean range. The 20 cm spacing is not used to detect stops or fit a scale. Only accepted frames have a depth; lost-frame distance is unknown. Sequence cutoff is a reviewed camera-occlusion/view-change boundary. No displacement is computed across that boundary.")
    return result, times, positions, edges


def plot_result(result, times, positions, edges, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].scatter(times, positions[:, 2], s=2, alpha=.45, label="accepted raw Z")
    for p in result["static_points"]:
        axes[0].axvspan(p["start_s"], p["end_s"], color="green" if p["sequence"] == "primary" else "gray", alpha=.15)
        axes[0].plot([p["start_s"], p["end_s"]], [p["z_median_m"]] * 2, color="red", lw=2)
        axes[0].text((p["start_s"] + p["end_s"]) / 2, p["z_median_m"] + .025, str(p["index"]), ha="center", fontsize=9)
    axes[0].set_ylabel("Tag-centre Z (m)")
    axes[0].legend(loc="upper left")
    bins = result["visibility_bins"]
    mid = [b["start_s"] + 1 for b in bins]
    axes[1].plot(mid, [b["detected"] / b["frames"] * 100 for b in bins], label="ArUco decoded", marker=".")
    axes[1].plot(mid, [b["accepted"] / b["frames"] * 100 for b in bins],
                 label=f"accepted >={result['replay']['min_edge_px']:g}px", marker=".")
    axes[1].plot(mid, [b["strict"] / b["frames"] * 100 for b in bins], label="accepted >=24px", marker=".")
    axes[1].set_ylabel("2s visibility (%)"); axes[1].set_ylim(-3, 103); axes[1].legend(loc="lower left")
    axes[2].scatter(times, edges, s=2, alpha=.4)
    axes[2].axhline(24, color="red", ls="--", label="usual 24px gate")
    axes[2].axhline(result["replay"]["min_edge_px"], color="orange", ls="--", label="replay gate")
    axes[2].set_ylabel("Shortest edge (px)"); axes[2].set_xlabel("Image-header elapsed time (s)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
        cutoff = result["parameters"]["sequence_end_s"]
        if cutoff is not None:
            ax.axvline(cutoff, color="black", ls=":")
    fig.suptitle(f"{result['replay']['marker_size_mm']:g} mm marker / equidistant / ID {result['tag_id']}")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--tag-id", type=int, default=2)
    parser.add_argument("--sequence-end-s", type=float)
    parser.add_argument("--max-speed-mm-s", type=float, default=20.)
    parser.add_argument("--min-stop-s", type=float, default=1.5)
    parser.add_argument("--static-window", type=float, nargs=2, action="append", default=[],
                        metavar=("START", "END"), help="visually reviewed hold, including sparse far detections")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result, times, positions, edges = analyze(args.replay_dir, args.tag_id, args.sequence_end_s,
                                              args.max_speed_mm_s, args.min_stop_s, args.static_window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "static_depths.json").write_text(json.dumps(result, indent=2) + "\n")
    points = result["static_points"]
    if points:
        fields = ["index", "sequence", "selection", "start_s", "end_s", "duration_s", "z_median_m", "z_std_mm",
                  "range_median_m", "coverage", "edge_median_px", "strict_24px_fraction",
                  "previous_delta_z_cm", "previous_distance_3d_cm"]
        with (args.output_dir / "static_depths.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(points)
    plot_result(result, times, positions, edges, args.output_dir / "depth_visibility.png")
    print(json.dumps({k: result[k] for k in ["observed_max_static_z_m", "high_coverage_max_static_z_m", "strict_24px_high_coverage_max_static_z_m"]}, indent=2))
    for p in points:
        print(f"{p['index']:2} {p['sequence']:12} {p['start_s']:6.1f}-{p['end_s']:6.1f}s Z={p['z_median_m']:.3f}m std={p['z_std_mm']:.1f}mm coverage={p['coverage']:.0%} edge={p['edge_median_px']:.1f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

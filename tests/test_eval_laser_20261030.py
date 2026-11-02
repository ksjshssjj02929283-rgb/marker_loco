import csv
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from tools.eval_laser_20261030 import (
    cross_validated_se3,
    evaluate_dynamic,
    evaluate_static,
    interpolate_valid,
    parse_gt_time,
)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def replay(directory, xyz_by_frame, base_time, *, size_mm=37.5):
    frames = []
    poses = []
    for index, xyz in enumerate(xyz_by_frame):
        time = base_time + index / 30
        frames.append(dict(frame_index=index, header_timestamp_ns=round(time * 1e9)))
        if xyz is not None:
            poses.append(dict(frame_index=index, header_timestamp_ns=round(time * 1e9),
                              tag_id=2, tx_m=xyz[0], ty_m=xyz[1], tz_m=xyz[2]))
    write_csv(directory / "frames.csv", list(frames[0]), frames)
    write_csv(directory / "poses.csv", list(poses[0]), poses)
    (directory / "summary.json").write_text(json.dumps(dict(marker_size_mm=size_mm)))


def test_laser_millisecond_padding_and_no_dropout_interpolation():
    assert parse_gt_time("2026/10/30 11:52:26. 13") == parse_gt_time("2026/10/30 11:52:26.013")
    positions, valid = interpolate_valid(
        np.array([0.0, 0.0667]), np.array([0, 2]),
        np.array([[0., 0., 0.], [1., 0., 0.]]), np.array([0.0333]), 0.1)
    assert not valid[0]
    assert np.isnan(positions[0]).all()


def test_static_pair_length_and_coverage(tmp_path):
    gt = tmp_path / "gt" / "marker_50mm" / "static.txt"
    gt.parent.mkdir(parents=True)
    gt.write_text("50,static,P_20,0,0,0,2026/10/30 11:00:00.000\n"
                  "50,static,P_30,100,0,0,2026/10/30 11:00:10.000\n")
    start = parse_gt_time("2026/10/30 11:00:00.000")
    root = tmp_path / "replay"
    replay(root / "marker50" / "static_020cm", [[0., 0., 0.], [0., 0., 0.]], start)
    replay(root / "marker50" / "static_030cm", [[0.102, 0., 0.], None], start)
    summary = evaluate_static("marker50", root, tmp_path / "gt", tmp_path / "out", {})
    assert summary["valid_points"] == 2
    assert summary["available_points"] == 9
    assert abs(summary["pair_rmse_mm"] - 2) < 1e-8
    points = json.loads((tmp_path / "out" / "static_points.json").read_text())
    assert next(point for point in points if point["distance_cm"] == 30)["coverage"] == 0.5


def test_dynamic_finds_offset_and_preserves_metric_scale(tmp_path):
    gt = tmp_path / "gt.txt"
    start = parse_gt_time("2026/10/30 11:00:00.000")

    def trajectory(t):
        return np.array([0.8 + 0.09 * t, 0.3 + 0.12 * np.sin(0.7 * t),
                         0.2 + 0.07 * np.cos(1.1 * t)])

    lines = []
    for index in range(60):
        time = start + index / 10
        stamp = datetime.fromtimestamp(time, ZoneInfo("Asia/Shanghai")).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        xyz = trajectory(index / 10) * 1000
        for sample in range(10):
            lines.append(f"50,track1,P_{index * 10 + sample + 1},{xyz[0]},{xyz[1]},{xyz[2]},{stamp}\n")
    gt.write_text("".join(lines))
    directory = tmp_path / "replay" / "marker50" / "dynamic_track_01"
    camera = [trajectory(index / 30) + [0.2, -0.1, 0.05] for index in range(180)]
    replay(directory, camera, start + 2)
    result = evaluate_dynamic("marker50", "dynamic_track_01", tmp_path / "replay", gt,
                              tmp_path / "out", {}, max_gap_s=0.1, min_coverage=0.8,
                              min_motion_fraction=0.8, scan_step_s=0.05,
                              scan_margin_s=1.0, max_group_spread_mm=60)
    assert result["status"] == "ok"
    assert abs(result["time_offset_s"] + 2) <= 0.05
    assert result["gt_group_coverage"] > 0.95
    assert result["relative_all_pairs"]["rmse_mm"] < 0.15
    assert result["trajectory_rmse_mm"] < 0.15
    cv = result["cross_validation"]
    assert cv["heldout_coverage"] > 0.95
    assert cv["heldout_position_rmse_mm"] < 0.2
    assert len(cv["folds"]) == 2
    assert cv["folds"][0]["fit_gt_time_end_s"] < cv["folds"][0]["evaluation_gt_time_start_s"]
    assert (tmp_path / "out" / "cross_validated_trajectories.csv").exists()
    assert (tmp_path / "out" / "trajectory_cross_validation.png").exists()


def test_cross_validated_se3_exposes_unseen_segment_bias():
    camera_times = np.arange(0, 12, 1 / 30)
    gt_times = np.arange(0.1, 10, 0.1)

    def motion(t):
        return np.column_stack((0.5 + 0.05 * t, 0.1 * np.sin(t), 0.08 * np.cos(0.7 * t)))

    truth = motion(gt_times)
    camera = motion(camera_times - 1.0) + [0.2, -0.1, 0.05]
    camera[camera_times >= 6.0] += [0.025, 0, 0]
    cv, rows = cross_validated_se3(
        camera_times, np.arange(len(camera_times)), camera, gt_times, truth,
        np.arange(-1.2, -0.8, 0.05), max_gap_s=0.1,
        min_coverage=0.8, min_motion_fraction=0.7)
    assert cv is not None
    assert len(rows) == len(gt_times)
    assert cv["heldout_coverage"] > 0.9
    assert cv["heldout_position_rmse_mm"] > 5
    for fold in cv["folds"]:
        assert abs(fold["fit_time_offset_s"] + 1) < 0.11
        assert fold["rigid_fit"]["rotation"].shape == (3, 3)

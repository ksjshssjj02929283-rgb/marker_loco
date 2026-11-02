import csv
import json

import numpy as np
import pytest

from marker6dof.commands.evaluation.eval_sensor_stops import analyze, find_static_intervals


def test_static_detector_keeps_stops_and_does_not_bridge_missing_data():
    times = np.arange(0, 14, 1 / 30)
    z = np.where(times < 4, .3, np.where(times < 5, .3 + .2 * (times - 4), .5))
    positions = np.column_stack([times * 0, times * 0, z])
    keep = (times < 8) | (times > 10)
    times, positions = times[keep], positions[keep]
    intervals = find_static_intervals(times, positions)
    assert len(intervals) >= 2
    for a, b in intervals:
        assert np.ptp(positions[a:b + 1, 2]) < .02
        assert np.max(np.diff(times[a:b + 1])) < .2
    with pytest.raises(ValueError, match="strictly increasing"):
        find_static_intervals([0, 0, 1], np.zeros((3, 3)))


def test_sparse_review_window_uses_all_frames_for_coverage_and_no_cross_view_delta(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"min_edge_px": 12}))
    with (tmp_path / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "timestamp_s", "detected_ids"])
        writer.writeheader()
        for i in range(120):
            writer.writerow(dict(frame_index=i, timestamp_s=i / 30, detected_ids="[2]" if i % 10 == 0 else "[]"))
    fields = ["tag_id", "frame_index", "timestamp_s", "tx_m", "ty_m", "tz_m", "min_edge_px", "reprojection_error_px"]
    with (tmp_path / "poses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i in range(0, 120, 10):
            writer.writerow(dict(tag_id=2, frame_index=i, timestamp_s=i / 30, tx_m=0, ty_m=0,
                                 tz_m=.5 if i < 60 else 1.5, min_edge_px=15, reprojection_error_px=.2))
    result, *_ = analyze(tmp_path, sequence_end_s=1.5, reviewed_windows=[(0, 1), (2, 3)])
    p, q = result["static_points"]
    assert p["coverage"] == pytest.approx(4 / 31)
    assert p["selection"] == "visually_reviewed_hold"
    assert q["z_median_m"] == 1.5
    assert q["sequence"] == "after_view_change"
    assert "previous_delta_z_cm" not in q
    assert result["high_coverage_max_static_z_m"] is None
    assert result["observed_max_static_z_m"] == 1.5
    with pytest.raises(ValueError, match="within"):
        analyze(tmp_path, reviewed_windows=[(2, 5)])

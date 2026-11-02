import csv

from marker6dof.commands.replay.replay_dual_t265 import _recorded_frame_times


def test_dual_replay_uses_recorded_camera_times(tmp_path):
    with (tmp_path / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_index", "camera_timestamp_ms"])
        writer.writeheader()
        writer.writerows([
            {"frame_index": 0, "camera_timestamp_ms": 1000},
            {"frame_index": 1, "camera_timestamp_ms": 1050},
            {"frame_index": 2, "camera_timestamp_ms": 1100},
        ])

    times, fps, source = _recorded_frame_times(tmp_path, 30)

    assert times == {0: 0.0, 1: 0.05, 2: 0.1}
    assert fps == 20
    assert source == "recorded_left_fisheye_timestamp"


def test_dual_replay_falls_back_for_older_recordings(tmp_path):
    times, fps, source = _recorded_frame_times(tmp_path, 30)

    assert times == {}
    assert fps == 30
    assert source == "frame_index / recorded_fps"

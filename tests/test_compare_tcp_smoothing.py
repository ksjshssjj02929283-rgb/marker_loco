import csv
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from marker6dof.commands.evaluation import compare_tcp_smoothing as comparison
from marker6dof.commands.evaluation.compare_tcp_smoothing import (
    FrameObservation, compare_observations, read_observations, summarize, write_comparison_csv,
)
from marker6dof.smoothing import PoseSmoother


def pose(angle=0.0, x=0.0):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([0.0, angle, 0.0]).as_matrix()
    transform[:3, 3] = [x, 0.0, .5]
    return transform


class RecordedTracker:
    def __init__(self, poses):
        self.poses = iter(poses)
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def solve(self, detections, intrinsics, timestamp):
        self.calls.append((detections, timestamp))
        transform = next(self.poses)
        return {"pose": transform, "status": "MULTI" if transform is not None else "LOST", "reason": "test", "used_ids": [2, 5] if transform is not None else [], "visible_ids": list(detections), "rejected_ids": [], "reprojection_error_px": .1 if transform is not None else None}


def compare(poses, timestamps=None):
    if timestamps is None:
        timestamps = np.arange(len(poses)) / 30.0
    frames = [FrameObservation(index, timestamp, timestamp - timestamps[0], {2: np.ones((4, 2))} if transform is not None else {}) for index, (timestamp, transform) in enumerate(zip(timestamps, poses))]
    tcp = np.eye(4)
    tcp[2, 3] = -.320
    tracker = RecordedTracker(poses)
    smoother = PoseSmoother(position_cutoff_hz=.8, rotation_cutoff_hz=.5)
    return compare_observations(frames, tracker, None, tcp, smoother), tracker, smoother


@pytest.mark.parametrize('acceleration', [0.0, 0.4])
def test_motion_variation_handles_irregular_sampling_without_calling_motion_noise(acceleration):
    times = [0.0, .02, .07, .10, .16]
    records = []
    for timestamp in times:
        transform = pose(x=2 * timestamp + .5 * acceleration * timestamp ** 2)
        records.append({
            'timestamp_s': timestamp, 'segment': 1, 'break_before': False,
            'raw_tcp': transform, 'old_camera_tcp': transform.copy(), 'new_camera_tcp': transform.copy(),
        })
    report = comparison.motion_variation(records)
    assert report['eligible_frame_triples'] == 3
    for name in ('raw', 'old', 'new'):
        assert report[name]['rms'] == pytest.approx(acceleration * 1000, abs=1e-9)
    assert 'real motion' in report['note']
    assert 'absolute accuracy' in report['note']


def test_motion_variation_excludes_lost_frames_and_discontinuities():
    records = []
    for index in range(5):
        transform = pose(x=index * .02)
        records.append({
            'timestamp_s': index / 30, 'segment': 1, 'break_before': False,
            'raw_tcp': transform, 'old_camera_tcp': transform.copy(), 'new_camera_tcp': transform.copy(),
        })
    records[2]['raw_tcp'] = None
    report = comparison.motion_variation(records)
    assert report['eligible_frame_triples'] == 0
    assert report['new'] is None
    records[2]['raw_tcp'] = pose(x=.04)
    records[3]['break_before'] = True
    report = comparison.motion_variation(records)
    assert report['eligible_frame_triples'] == 1


def test_common_tracker_input_origin_raw_output_and_loss_rows(tmp_path):
    poses = [pose(.1), pose(.15, .02), None, pose(.2, .04), pose(.21, .05)]
    records, tracker, _ = compare(poses)
    assert len(tracker.calls) == len(poses) == len(records)
    assert records[2]["status"] == "LOST"
    assert records[2]["new_camera_tcp"] is None
    assert records[3]["break_before"]
    assert records[3]["segment"] == 2
    tcp = np.eye(4)
    tcp[2, 3] = -.320
    initial = poses[0] @ tcp
    for transform, record in zip(poses, records):
        if transform is None:
            continue
        np.testing.assert_allclose(record["raw_tcp"], transform @ tcp)
        np.testing.assert_allclose(record["raw_relative"], np.linalg.inv(initial) @ transform @ tcp, atol=1e-14)
        np.testing.assert_allclose(initial @ record["new_relative"], record["new_camera_tcp"], atol=1e-14)
        np.testing.assert_allclose(initial @ record["old_relative"], record["old_camera_tcp"], atol=1e-14)
    assert not np.allclose(records[1]["old_camera_tcp"], records[1]["new_camera_tcp"])
    path = tmp_path / "comparison.csv"
    write_comparison_csv(records, path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert rows[2]["raw_tcp_x_m"] == ""
    assert rows[3]["used_ids"] == "2;5"


def test_long_gap_resets_display_and_breaks_both_pipelines():
    poses = [pose(), pose(.1, .02), pose(.5, .2)]
    records, tracker, _ = compare(poses, [0.0, .03, .4])
    assert tracker.resets == 1
    assert records[-1]["segment"] == 2
    assert records[-1]["break_before"]
    np.testing.assert_allclose(records[-1]["raw_tcp"], records[-1]["new_camera_tcp"])
    np.testing.assert_allclose(records[-1]["raw_tcp"], records[-1]["old_camera_tcp"])


def test_static_statistics_reduce_noise_without_resetting_window_origins():
    rng = np.random.default_rng(232)
    records, _, smoother = compare([pose(rng.normal(0, .006)) for _ in range(901)])
    initial_relative = records[90]["new_relative"].copy()
    report = summarize(records, [(3.0, 9.0), (23.0, 29.0)], smoother)
    assert report["frames"] == 901
    assert report["pose_coverage"] == 1.0
    for window in report["static_windows"]:
        assert window["new"]["rms_3d_mm"] < window["old"]["rms_3d_mm"] < window["raw"]["rms_3d_mm"]
    np.testing.assert_array_equal(records[90]["new_relative"], initial_relative)
    with pytest.raises(ValueError, match="actual device-time range"):
        summarize(records, [(29.0, 31.0)], smoother)


def test_no_accepted_pose_does_not_report_false_success():
    with pytest.raises(ValueError, match="No accepted"):
        compare([None, None])


def write_session(tmp_path, timestamps=(1000, 1033), frame_indices=(0, 1), pose_frames=(0,)):
    (tmp_path / "metadata.json").write_text(json.dumps({"aruco": {"marker_black_square_size_m": .0375}}))
    with (tmp_path / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "device_timestamp_ms"])
        writer.writerows(zip(frame_indices, timestamps))
    with (tmp_path / "poses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "device_timestamp_ms", "tag_id", *(f"corner_{index}_{axis}" for index in range(4) for axis in "xy")])
        for frame in pose_frames:
            writer.writerow([frame, timestamps[frame], 2, *range(8)])


def test_read_corners_preserves_empty_frames_and_actual_timestamps(tmp_path):
    write_session(tmp_path)
    _, frames = read_observations(tmp_path)
    assert len(frames) == 2
    assert frames[1].detections == {}
    assert frames[1].elapsed_s == pytest.approx(.033)
    np.testing.assert_array_equal(frames[0].detections[2], np.arange(8).reshape(4, 2))


def test_repeated_device_timestamp_preserves_indexed_corners_and_empty_frame(tmp_path):
    write_session(tmp_path, timestamps=(1000, 1000))
    _, frames = read_observations(tmp_path)
    assert len(frames) == 2
    assert frames[0].timestamp_s == frames[1].timestamp_s == 1.0
    assert 2 in frames[0].detections and frames[1].detections == {}


@pytest.mark.parametrize('duplicate_lost', [False, True])
def test_repeated_timestamp_disconnects_both_pipelines_without_inventing_time(duplicate_lost):
    transforms = [pose(), pose(.1), None if duplicate_lost else pose(.2), pose(.3), pose(.4)]
    records, tracker, smoother = compare(transforms, [0, .03, .03, .06, .09])
    assert len(records) == len(tracker.calls) == 5
    assert records[2]['timestamp_s'] == records[1]['timestamp_s']
    assert records[2]['timestamp_discontinuity'] and records[2]['break_before']
    assert tracker.resets == 1
    if duplicate_lost:
        assert records[2]['raw_tcp'] is None
        assert records[3]['segment'] > records[1]['segment']
    else:
        assert records[2]['segment'] > records[1]['segment']
        np.testing.assert_allclose(records[2]['old_camera_tcp'], records[2]['raw_tcp'])
        np.testing.assert_allclose(records[2]['new_camera_tcp'], records[2]['raw_tcp'])
    summary = summarize(records, [], smoother)
    assert summary['timestamp_discontinuities']['frame_indices'] == [2]
    assert summary['motion_variation']['eligible_frame_triples'] == (0 if duplicate_lost else 1)


@pytest.mark.parametrize("timestamps,indices", [((1000, 999), (0, 1)), ((1000, 1033), (0, 0)), ((1000, 1033), (0, 2))])
def test_ambiguous_frame_alignment_is_rejected(tmp_path, timestamps, indices):
    write_session(tmp_path, timestamps, indices)
    with pytest.raises(ValueError):
        read_observations(tmp_path)


def test_duplicate_tag_observation_is_rejected(tmp_path):
    write_session(tmp_path, pose_frames=(0, 0))
    with pytest.raises(ValueError, match="duplicate corners"):
        read_observations(tmp_path)


def test_window_without_accepted_observations_is_rejected():
    records, _, smoother = compare([pose(), None, None, None, pose()], [0.0, .1, .2, .3, .4])
    with pytest.raises(ValueError, match="fewer than 3"):
        summarize(records, [(.1, .3)], smoother)


class FakeCapture:
    def __init__(self, image_count, declared_count):
        self.image_count = image_count
        self.declared_count = declared_count
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, field):
        return self.declared_count

    def read(self):
        if self.index >= self.image_count:
            return False, None
        self.index += 1
        return True, np.zeros((256, 320, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def mock_video(monkeypatch, image_count, declared_count):
    capture = FakeCapture(image_count, declared_count)
    writers = []

    def create_writer(*args):
        writer = FakeWriter()
        writers.append(writer)
        return writer

    monkeypatch.setattr(comparison.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(comparison.cv2, "VideoWriter", create_writer)
    monkeypatch.setattr(comparison, "draw_realsense_frame", lambda *args: None)
    monkeypatch.setattr(comparison, "draw_realsense_polyline", lambda *args: None)
    return capture, writers


def test_videos_keep_every_csv_frame_including_loss(monkeypatch, tmp_path):
    records, _, _ = compare([pose(), None])
    capture, writers = mock_video(monkeypatch, 2, 2)
    comparison.render_videos(tmp_path, tmp_path, records, None, 30.0, .0375)
    assert len(writers) == 3
    assert all(len(writer.frames) == 2 and writer.released for writer in writers)
    assert writers[2].frames[0].shape == (256, 640, 3)
    assert capture.released


@pytest.mark.parametrize("image_count,declared_count,message", [
    (1, 0, "ends before CSV frame"),
    (3, 2, "more frames"),
    (2, 3, "frame count differs"),
])
def test_video_csv_mismatch_is_explicit(monkeypatch, tmp_path, image_count, declared_count, message):
    records, _, _ = compare([pose(), pose(.1)])
    capture, writers = mock_video(monkeypatch, image_count, declared_count)
    with pytest.raises(ValueError, match=message):
        comparison.render_videos(tmp_path, tmp_path, records, None, 30.0, .0375)
    assert capture.released
    assert all(writer.released for writer in writers)

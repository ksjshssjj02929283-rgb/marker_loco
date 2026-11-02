"""Exercise the live display/recording paths without opening a camera."""

import copy
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from marker6dof.commands.tracking import live_dashboard as dashboard
from marker6dof.smoothing import PoseSmoother
from marker6dof.world import WorldTrajectory


def pose(x=0.0, angle=0.0):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('y', angle).as_matrix()
    transform[:3, 3] = [x, 0.0, 0.5]
    return transform


@pytest.mark.parametrize('discontinuity', ['lost', 'gap'])
def test_d455_overlay_and_browser_share_pose_while_recording_stays_raw(
    monkeypatch, tmp_path, discontinuity,
):
    monkeypatch.setattr(sys, 'argv', [
        'marker-live-dashboard', '--rig', 'unused.json', '--record-root', str(tmp_path),
        '--tcp-transform', 'unused_tcp.json',
    ])
    args = dashboard.parse_args()
    observations = [pose(), pose(0.012, 0.015), pose(0.020, 0.025), pose(0.025, 0.03)]
    timestamps = [1000.0, 1033.333, 1066.666, 1099.999]
    if discontinuity == 'lost':
        observations[2] = None
    else:
        timestamps[2:] = [1600.0, 1633.333]
    tcp_transform = np.eye(4)
    tcp_transform[2, 3] = -0.320
    stop = threading.Event()
    intrinsics = SimpleNamespace(width=1280, height=720)

    class Pipeline:
        index = 0
        stopped = False

        def start(self, _config):
            device = SimpleNamespace(first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001))
            return SimpleNamespace(
                get_device=lambda: device,
                get_stream=lambda _stream: SimpleNamespace(
                    as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intrinsics),
                ),
            )

        def wait_for_frames(self):
            index = self.index
            self.index += 1
            if self.index == len(observations):
                stop.set()
            frame = SimpleNamespace(
                get_frame_number=lambda: index + 1,
                get_timestamp=lambda: timestamps[index],
                get_frame_timestamp_domain=lambda: 'device-clock',
                get_data=lambda: np.zeros((60, 80, 3), dtype=np.uint8),
            )
            return SimpleNamespace(get_color_frame=lambda: frame)

        def stop(self):
            self.stopped = True

    pipeline = Pipeline()
    config = SimpleNamespace(
        enable_device=lambda *_: None, enable_stream=lambda *_: None,
        enable_record_to_file=lambda *_: None,
    )
    monkeypatch.setitem(sys.modules, 'pyrealsense2', SimpleNamespace(
        pipeline=lambda: pipeline, config=lambda: config,
        stream=SimpleNamespace(color='color', depth='depth'),
        format=SimpleNamespace(bgr8='bgr8', z16='z16'),
    ))

    class Tracker:
        def __init__(self, *_args):
            self.index = 0

        def reset(self):
            pass

        def solve(self, *_args):
            observation = observations[self.index]
            self.index += 1
            return dict(
                pose=observation, status='MULTI' if observation is not None else 'LOST',
                reason='synthetic observation', visible_ids=[2, 5],
                used_ids=[2, 5] if observation is not None else [], rejected_ids=[],
                reprojection_error_px=0.1 if observation is not None else None,
            )

    class Recorder:
        rows = []

        def __init__(self, *_args):
            pass

        def open(self):
            pass

        def write_color(self, _image):
            pass

        def write_annotated(self, _image):
            pass

        def write_pose(self, row, *_args):
            self.rows.append(row.copy())

        def write_frame(self, _row):
            pass

        def close(self, _interrupted):
            pass

    class State(dashboard.SharedState):
        def __init__(self):
            super().__init__([2, 5])
            self.frames = []

        def publish(self, image, data, trace):
            super().publish(image, data, trace)
            self.frames.append(copy.deepcopy(self.get()[1]))

    overlays, histories = [], []
    monkeypatch.setattr(dashboard, 'RigTracker', Tracker)
    monkeypatch.setattr(dashboard, '_Recorder', Recorder)
    monkeypatch.setattr(dashboard, 'make_session_dir', lambda _root: tmp_path)
    monkeypatch.setattr(dashboard, 'write_metadata', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dashboard, 'load_transform', lambda _path: tcp_transform)
    monkeypatch.setattr(dashboard, 'detect_markers', lambda *_args: ([], None, []))
    monkeypatch.setattr(dashboard, 'draw_axes_realsense', lambda *_args: None)
    monkeypatch.setattr(dashboard, 'draw_realsense_frame',
                        lambda _image, _intrinsics, transform, _size: overlays.append(transform.copy()))
    monkeypatch.setattr(dashboard, 'draw_realsense_polyline',
                        lambda _image, _intrinsics, points, *_args: histories.append(np.array(points)))
    monkeypatch.setattr(dashboard, 'overlay', lambda *_args: None)
    state = State()
    dashboard.tracking(args, state, {2: np.eye(4), 5: np.eye(4)}, stop)

    assert pipeline.stopped
    assert len(state.frames) == len(observations) == len(Recorder.rows)
    first_tcp = observations[0] @ tcp_transform
    origin = np.linalg.inv(first_tcp)
    valid_frames = [frame for frame, observation in zip(state.frames, observations) if observation is not None]
    for frame, overlay_pose in zip(valid_frames, overlays):
        np.testing.assert_allclose(frame['smooth_tcp_camera_position_m'], overlay_pose[:3, 3])
        np.testing.assert_allclose(frame['smooth_position_m'], (origin @ overlay_pose)[:3, 3], atol=1e-14)
    for frame, row, observation in zip(state.frames, Recorder.rows, observations):
        if observation is None:
            assert frame['smooth_tcp_camera_position_m'] is None
            assert row['position_x_m'] == ''
            continue
        raw_tcp = observation @ tcp_transform
        raw_relative = origin @ raw_tcp
        np.testing.assert_allclose(frame['tcp_camera_position_m'], raw_tcp[:3, 3])
        np.testing.assert_allclose(frame['position_m'], raw_relative[:3, 3], atol=1e-14)
        np.testing.assert_allclose(
            [float(row[f'position_{axis}_m']) for axis in 'xyz'], raw_relative[:3, 3], atol=1e-9,
        )
    assert not np.allclose(state.frames[1]['position_m'], state.frames[1]['smooth_position_m'])
    assert state.frames[-1]['segment'] == 2
    assert all(len(history) <= 2 for history in histories)


def test_world_smoothing_preserves_stationary_tcp_while_camera_moves():
    trajectory = WorldTrajectory()
    smoother = PoseSmoother()
    world_reference = pose(1.0, 0.2)
    tcp_transform = np.eye(4)
    tcp_transform[2, 3] = -0.320
    world_tcp = world_reference @ tcp_transform
    for index in range(91):
        timestamp = index / 30.0
        world_camera = np.eye(4)
        world_camera[0, 3] = timestamp * 0.2
        camera_reference = np.linalg.inv(world_camera) @ world_reference
        camera_tcp = camera_reference @ tcp_transform
        display_world_reference = smoother.update(world_camera @ camera_reference, timestamp)
        _, tcp = trajectory.update(
            timestamp, world_camera, camera_tcp, object_quality='MULTI',
            display_world_object=display_world_reference @ tcp_transform,
        )
        np.testing.assert_allclose(tcp['position_m'], world_tcp[:3, 3], atol=1e-14)
        np.testing.assert_allclose(tcp['smooth_position_m'], world_tcp[:3, 3], atol=1e-14)


@pytest.mark.parametrize('flag,value', [
    ('--smoothing-cutoff-hz', '0'), ('--rotation-cutoff-hz', 'nan'),
    ('--smoothing-motion-gain', '-1'), ('--rotation-motion-gain', 'inf'),
])
def test_invalid_smoothing_cli_is_rejected_before_camera_start(monkeypatch, flag, value):
    monkeypatch.setattr(sys, 'argv', ['marker-live-dashboard', '--rig', 'unused.json', flag, value])
    with pytest.raises(SystemExit) as error:
        dashboard.parse_args()
    assert error.value.code == 2

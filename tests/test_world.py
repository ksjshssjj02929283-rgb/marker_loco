from types import SimpleNamespace

import numpy as np

from marker6dof.world import (
    WorldTrajectory,
    synchronized_t265_camera_pose,
    transform_from_translation_quaternion,
)


def _pose_frame(timestamp_ms=100.0, confidence=3):
    data = SimpleNamespace(
        translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        tracker_confidence=confidence,
    )
    return SimpleNamespace(get_timestamp=lambda: timestamp_ms, get_pose_data=lambda: data)


def test_t265_vio_pose_uses_sdk_pose_to_fisheye_extrinsics():
    frame = _pose_frame()
    frames = SimpleNamespace(get_pose_frame=lambda: frame)
    pose_from_left = np.eye(4)
    pose_from_left[0, 3] = 0.063

    sample = synchronized_t265_camera_pose(frames, 100.0, pose_from_left)

    assert sample.valid
    assert sample.reason == "ok"
    assert sample.tracker_confidence == 3
    np.testing.assert_allclose(sample.T_world_camera[:3, 3], [1.063, 2.0, 3.0])


def test_t265_vio_pose_rejects_stale_or_low_confidence_data():
    stale = SimpleNamespace(get_pose_frame=lambda: _pose_frame(timestamp_ms=141.0))
    result = synchronized_t265_camera_pose(stale, 100.0, np.eye(4), max_delta_ms=20.0)
    assert not result.valid
    assert result.reason == "vio_pose_time_mismatch"

    low_confidence = SimpleNamespace(get_pose_frame=lambda: _pose_frame(confidence=1))
    result = synchronized_t265_camera_pose(low_confidence, 100.0, np.eye(4), min_tracker_confidence=2)
    assert not result.valid
    assert result.reason == "vio_low_confidence"


def test_world_trajectory_keeps_ego_and_tcp_in_one_common_origin():
    trajectory = WorldTrajectory()
    world_from_camera_0 = transform_from_translation_quaternion([10, 0, 0], [0, 0, 0, 1])
    camera_from_tcp = transform_from_translation_quaternion([0.2, 0, 0], [0, 0, 0, 1])

    ego, tcp = trajectory.update(1.0, world_from_camera_0, camera_from_tcp, object_quality="MULTI")
    np.testing.assert_allclose(ego["position_m"], [0, 0, 0])
    np.testing.assert_allclose(tcp["position_m"], [0.2, 0, 0])

    world_from_camera_1 = transform_from_translation_quaternion([11, 0, 0], [0, 0, 0, 1])
    ego, tcp = trajectory.update(1.1, world_from_camera_1, camera_from_tcp, object_quality="MULTI")
    np.testing.assert_allclose(ego["position_m"], [1, 0, 0])
    np.testing.assert_allclose(tcp["position_m"], [1.2, 0, 0])
    assert ego["segment"] == tcp["segment"] == 1


def test_world_trajectory_breaks_both_traces_when_vio_is_unavailable():
    trajectory = WorldTrajectory()
    camera = np.eye(4)
    tcp = np.eye(4)
    trajectory.update(1.0, camera, tcp, object_quality="MULTI")
    ego, object_pose = trajectory.update(1.1, None, None)
    assert ego["position_m"] is None
    assert object_pose["position_m"] is None
    ego, object_pose = trajectory.update(1.2, camera, tcp, object_quality="MULTI")
    assert ego["segment"] == object_pose["segment"] == 2

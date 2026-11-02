import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from marker6dof.projection import validate_rigid_transform
from marker6dof.smoothing import PoseSmoother


def make_pose(position=(0.0, 0.0, 0.5), rotvec=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    pose[:3, 3] = position
    return pose


def radial_rms(points):
    points = np.asarray(points)
    return np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))


def test_stationary_rotation_noise_at_320mm_tcp_is_reduced():
    rng = np.random.default_rng(581)
    smoother = PoseSmoother()
    tcp_transform = np.eye(4)
    tcp_transform[2, 3] = -0.320
    raw_tcp, display_tcp = [], []
    for index in range(600):
        pose = make_pose(rotvec=rng.normal(0.0, np.deg2rad(0.25), 3))
        untouched = pose.copy()
        display_pose = smoother.update(pose, index / 30.0)
        np.testing.assert_array_equal(pose, untouched)
        validate_rigid_transform(display_pose)
        raw_tcp.append((pose @ tcp_transform)[:3, 3])
        display_tcp.append((display_pose @ tcp_transform)[:3, 3])
    assert radial_rms(display_tcp[90:]) < 0.4 * radial_rms(raw_tcp[90:])


def test_slow_translation_and_rotation_are_followed_without_deadband():
    smoother = PoseSmoother()
    displays = []
    for index in range(301):
        timestamp = index / 30.0
        pose = make_pose(
            position=(timestamp * 0.0001, 0.0, 0.5),
            rotvec=(0.0, timestamp * 0.0002, 0.0),
        )
        displays.append(smoother.update(pose, timestamp))
    positions = np.array([pose[0, 3] for pose in displays])
    angles = np.array([Rotation.from_matrix(pose[:3, :3]).as_rotvec()[1] for pose in displays])
    assert np.all(np.diff(positions) > 0)
    assert np.all(np.diff(angles) > 0)
    assert positions[-1] > 0.00097
    assert angles[-1] > 0.0019


def test_adaptive_bandwidth_improves_constant_motion_response():
    adaptive = PoseSmoother()
    fixed = PoseSmoother(motion_gain=0.0, rotation_motion_gain=0.0)
    for index in range(91):
        timestamp = index / 30.0
        pose = make_pose(
            position=(timestamp * 0.2, 0.0, 0.5),
            rotvec=(0.0, timestamp * 0.7, 0.0),
        )
        adaptive_pose = adaptive.update(pose, timestamp)
        fixed_pose = fixed.update(pose, timestamp)
    assert 0.0 < pose[0, 3] - adaptive_pose[0, 3] < 0.02
    assert pose[0, 3] - adaptive_pose[0, 3] < 0.6 * (pose[0, 3] - fixed_pose[0, 3])
    adaptive_angle_error = (
        Rotation.from_matrix(adaptive_pose[:3, :3]).inv()
        * Rotation.from_matrix(pose[:3, :3])
    ).magnitude()
    fixed_angle_error = (
        Rotation.from_matrix(fixed_pose[:3, :3]).inv()
        * Rotation.from_matrix(pose[:3, :3])
    ).magnitude()
    assert 0.0 < adaptive_angle_error < 0.1
    assert adaptive_angle_error < 0.7 * fixed_angle_error


@pytest.mark.parametrize("discontinuity", ["lost", "gap", "backward", "duplicate", "reset"])
def test_discontinuities_do_not_reuse_old_display_pose(discontinuity):
    smoother = PoseSmoother()
    smoother.update(make_pose(), 1.0)
    smoother.update(make_pose(position=(0.1, 0.0, 0.5)), 1.03)
    timestamp = 1.06
    if discontinuity == "lost":
        assert smoother.update(None, 1.04) is None
    elif discontinuity == "gap":
        timestamp = 2.0
    elif discontinuity == "backward":
        timestamp = 0.5
    elif discontinuity == "duplicate":
        timestamp = 1.03
    else:
        smoother.reset()
    current = make_pose(position=(0.6, 0.2, 0.4), rotvec=(0.3, -0.2, 0.1))
    np.testing.assert_allclose(smoother.update(current, timestamp), current, atol=1e-14)


def test_rotation_crosses_180_degrees_by_shortest_path():
    smoother = PoseSmoother(rotation_motion_gain=0.0)
    before = make_pose(rotvec=(0.0, np.deg2rad(179.0), 0.0))
    after = make_pose(rotvec=(0.0, np.deg2rad(-179.0), 0.0))
    smoother.update(before, 0.0)
    display = smoother.update(after, 1.0 / 30.0)
    relative = Rotation.from_matrix(before[:3, :3]).inv() * Rotation.from_matrix(display[:3, :3])
    assert 0.0 < relative.as_rotvec()[1] < np.deg2rad(2.0)
    assert Rotation.from_matrix(display[:3, :3]).magnitude() > np.deg2rad(178.0)


def test_exponential_filter_respects_elapsed_time():
    initial = make_pose()
    target = make_pose(position=(0.1, 0.0, 0.5), rotvec=(0.0, 0.5, 0.0))
    outputs = []
    for fps in (30, 60):
        smoother = PoseSmoother(motion_gain=0.0, rotation_motion_gain=0.0)
        smoother.update(initial, 0.0)
        for index in range(1, fps + 1):
            display = smoother.update(target, index / fps)
        outputs.append(display)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-13)


def test_output_cannot_mutate_filter_state():
    smoother = PoseSmoother()
    pose = make_pose()
    display = smoother.update(pose, 0.0)
    display[0, 3] = 200.0
    np.testing.assert_allclose(smoother.update(pose, 1.0 / 30.0), pose, atol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_cutoff_hz": 0.0},
        {"rotation_cutoff_hz": -1.0},
        {"velocity_cutoff_hz": float("inf")},
        {"max_gap_s": float("nan")},
        {"motion_gain": -1.0},
        {"rotation_motion_gain": float("inf")},
    ],
)
def test_invalid_filter_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PoseSmoother(**kwargs)


def test_invalid_timestamp_and_nonrigid_pose_are_rejected():
    smoother = PoseSmoother()
    with pytest.raises(ValueError, match="timestamp"):
        smoother.update(make_pose(), float("nan"))
    nonrigid = make_pose()
    nonrigid[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        smoother.update(nonrigid, 0.0)

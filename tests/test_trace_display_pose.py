import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from marker6dof.tracking import RelativeTrace


def pose(position=(0, 0, 0), angles=(0, 0, 0)):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    transform[:3, 3] = position
    return transform


def output_pose(output, smooth=False):
    prefix = 'smooth_' if smooth else ''
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(output[f'{prefix}quaternion_xyzw']).as_matrix()
    transform[:3, 3] = output[f'{prefix}position_m']
    return transform


def update(trace, raw, timestamp, **kwargs):
    return trace.update(dict(pose=raw, status='MULTI'), timestamp, **kwargs)


def test_raw_and_display_marker_tcp_share_one_rigid_pose_and_origin():
    raw = pose((0.1, -0.2, 0.5), (4, 12, -7))
    displayed = pose((0.101, -0.199, 0.502), (3.8, 11.9, -6.8))
    tcp_offset = pose((0.02, -0.01, -0.32), (10, -3, 5))
    initial = np.linalg.inv(raw @ tcp_offset)
    marker_trace, tcp_trace = RelativeTrace(), RelativeTrace()
    snapshots = [array.copy() for array in (raw, displayed, tcp_offset, initial)]
    raw.setflags(write=False)
    displayed.setflags(write=False)

    marker = update(marker_trace, raw, 0, initial_transform=initial, display_pose=displayed)
    tcp = update(tcp_trace, raw @ tcp_offset, 0, initial_transform=initial,
                 display_pose=displayed @ tcp_offset)

    np.testing.assert_allclose(output_pose(marker), initial @ raw, atol=1e-12)
    np.testing.assert_allclose(output_pose(tcp), initial @ raw @ tcp_offset, atol=1e-12)
    np.testing.assert_allclose(output_pose(marker, smooth=True), initial @ displayed, atol=1e-12)
    np.testing.assert_allclose(output_pose(tcp, smooth=True),
                               output_pose(marker, smooth=True) @ tcp_offset, atol=1e-12)
    for array, snapshot in zip((raw, displayed, tcp_offset, initial), snapshots):
        np.testing.assert_array_equal(array, snapshot)
    np.testing.assert_allclose(tcp_trace.trace[-1]['raw'], tcp['position_m'])
    np.testing.assert_allclose(tcp_trace.trace[-1]['smooth'], tcp['smooth_position_m'])


def test_external_display_pose_bypasses_filter_and_uses_raw_first_pose_origin():
    first = pose((0.1, 0.2, 0.5), (5, 10, -2))
    trace = RelativeTrace()
    initial_display = pose((0.102, 0.2, 0.5), (5, 9.5, -2))
    output = update(trace, first, 0, display_pose=initial_display)
    np.testing.assert_allclose(output_pose(output), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(output_pose(output, smooth=True),
                               np.linalg.inv(first) @ initial_display, atol=1e-12)

    raw = pose((0.12, 0.2, 0.5), (5, 11, -2))
    displayed = pose((0.115, 0.2, 0.5), (5, 10.7, -2))
    output = update(trace, raw, 1 / 30, display_pose=displayed)
    np.testing.assert_allclose(output_pose(output), np.linalg.inv(first) @ raw, atol=1e-12)
    np.testing.assert_allclose(output_pose(output, smooth=True),
                               np.linalg.inv(first) @ displayed, atol=1e-12)
    assert output['segment'] == 1


@pytest.mark.parametrize('break_kind', ['loss', 'long_gap', 'same_time', 'backwards_time'])
def test_external_display_trace_breaks_segments_and_keeps_initial_origin(break_kind):
    trace = RelativeTrace()
    first = pose((0.1, 0, 0.5))
    update(trace, first, 1, display_pose=first)
    update(trace, pose((0.11, 0, 0.5)), 1.03, display_pose=pose((0.105, 0, 0.5)))
    if break_kind == 'loss':
        # A display prediction alone is not an accepted raw observation.
        lost = update(trace, None, 1.06, display_pose=np.full((4, 4), np.nan))
        assert lost['position_m'] is None
        assert lost['smooth_position_m'] is None
        assert len(trace.trace) == 2
        timestamp = 1.09
    else:
        timestamp = {'long_gap': 1.4, 'same_time': 1.03, 'backwards_time': 1.02}[break_kind]
    raw, displayed = pose((0.15, 0, 0.5)), pose((0.145, 0, 0.5))
    output = update(trace, raw, timestamp, display_pose=displayed)
    assert output['segment'] == 2
    assert [sample['segment'] for sample in trace.trace] == [1, 1, 2]
    np.testing.assert_allclose(output['position_m'], [0.05, 0, 0], atol=1e-12)
    np.testing.assert_allclose(output['smooth_position_m'], [0.045, 0, 0], atol=1e-12)


def test_external_display_quaternion_sign_is_continuous_without_changing_rotation():
    trace = RelativeTrace()
    first_display, next_display = pose(angles=(0, 0, 269)), pose(angles=(0, 0, 271))
    first = update(trace, np.eye(4), 0, display_pose=first_display)
    following = update(trace, np.eye(4), 1 / 30, display_pose=next_display)
    assert np.dot(first['smooth_quaternion_xyzw'], following['smooth_quaternion_xyzw']) > 0
    np.testing.assert_allclose(output_pose(following, smooth=True), next_display, atol=1e-12)
    np.testing.assert_allclose(output_pose(following), np.eye(4), atol=1e-12)


@pytest.mark.parametrize('invalid_display', [
    np.eye(3), np.full((4, 4), np.nan), np.diag([2, 1, 1, 1]),
    np.diag([-1, 1, 1, 1]), np.diag([1, 1, 1, 2]),
])
def test_external_display_must_be_a_rigid_transform(invalid_display):
    trace = RelativeTrace()
    with pytest.raises(ValueError, match='display_pose'):
        update(trace, np.eye(4), 0, display_pose=invalid_display)
    assert len(trace.trace) == 0
    assert trace.initial is None


def test_legacy_internal_filter_remains_adaptive_position_and_quaternion_nlerp():
    trace = RelativeTrace()
    update(trace, np.eye(4), 0)
    timestamp = 1 / 30
    raw = pose((0.003, 0, 0), (0, 2, 0))
    output = update(trace, raw, timestamp)
    speed = 0.003 / timestamp
    alpha = 1 - math.exp(-2 * math.pi * (2 + 8 * speed) * timestamp)
    raw_q = Rotation.from_matrix(raw[:3, :3]).as_quat()
    expected_q = np.array([0., 0., 0., 1.]) + alpha * (raw_q - [0, 0, 0, 1])
    expected_q /= np.linalg.norm(expected_q)
    np.testing.assert_allclose(output['position_m'], raw[:3, 3])
    np.testing.assert_allclose(output['smooth_position_m'], alpha * raw[:3, 3])
    np.testing.assert_allclose(output['smooth_quaternion_xyzw'], expected_q)
    assert output['segment'] == 1

    after_gap = update(trace, pose((0.01, 0, 0), (0, 3, 0)), 1)
    np.testing.assert_allclose(after_gap['smooth_position_m'], after_gap['position_m'])
    np.testing.assert_allclose(after_gap['smooth_quaternion_xyzw'], after_gap['quaternion_xyzw'])
    assert after_gap['segment'] == 2

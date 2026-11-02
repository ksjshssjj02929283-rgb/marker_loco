"""Ensure RGB-D refinement participates in the tracker's actual acceptance gates."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from marker6dof import tracking


@pytest.fixture
def scene(monkeypatch):
    intrinsics = SimpleNamespace(width=1280, height=720)

    def project(_intrinsics, points):
        return points[:, :2] / points[:, 2:3] * 650 + [640, 360]

    monkeypatch.setattr(tracking, 'project_realsense', project)
    monkeypatch.setattr(tracking, 'normalized_image_points',
                        lambda _intrinsics, pixels: (pixels - [640, 360]) / 650)
    tracker = tracking.RigTracker({2: np.eye(4)}, .0475)
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler('xyz', [170, 10, 3], degrees=True).as_matrix()
    truth[:3, 3] = [.02, -.01, .6]

    def observations(pose):
        return {2: project(intrinsics, tracker.points[2] @ pose[:3, :3].T + pose[:3, 3])}

    return tracker, intrinsics, truth, observations


def test_refinement_updates_cached_pose_and_reported_reprojection(scene):
    tracker, intrinsics, truth, observations = scene

    def refiner(rgb, ids):
        assert ids == [2]
        rgb[2, 3] += .001
        return rgb, {'status': 'RGB_DEPTH_REFINED'}

    result = tracker.solve(observations(truth), intrinsics, 1.0, pose_refiner=refiner)
    assert result['status'] == 'SINGLE'
    assert result['depth_report']['status'] == 'RGB_DEPTH_REFINED'
    assert result['reprojection_error_px'] > result['rgb_reprojection_error_px']
    np.testing.assert_array_equal(result['pose'], tracker.previous)
    np.testing.assert_allclose(result['pose'][2, 3], .601, atol=1e-6)


def test_bad_refinement_cannot_bypass_rgb_reprojection_gate(scene):
    tracker, intrinsics, truth, observations = scene

    def refiner(rgb, ids):
        rgb[0, 3] += .1
        return rgb, {'status': 'RGB_DEPTH_REFINED', 'fused_reprojection_error_px': 99.0}

    result = tracker.solve(observations(truth), intrinsics, 1.0, pose_refiner=refiner)
    assert result['depth_report']['status'] == 'DEPTH_REJECTED_TRACKER_REPROJECTION'
    np.testing.assert_allclose(result['pose'], truth, atol=1e-6)
    assert result['reprojection_error_px'] == result['rgb_reprojection_error_px']
    assert result['depth_report']['fused_reprojection_error_px'] == result['reprojection_error_px']


def test_temporal_gate_checks_fused_pose(scene):
    tracker, intrinsics, truth, observations = scene
    assert tracker.solve(observations(truth), intrinsics, 1.0)['pose'] is not None
    next_pose = truth.copy()
    next_pose[2, 3] += .0202  # RGB step is just inside the 20.3 mm temporal gate.

    def refiner(rgb, ids):
        rgb[2, 3] += .001  # The final step now exceeds that gate.
        return rgb, {'status': 'RGB_DEPTH_REFINED'}

    result = tracker.solve(observations(next_pose), intrinsics, 1.0001, pose_refiner=refiner)
    assert result['status'] == 'WARN'
    assert result['pose'] is None
    assert tracker.previous is None
    assert '突跳' in result['reason']

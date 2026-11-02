"""Check configurable replay gating without changing the conservative default."""

import sys

import numpy as np
import pytest

from marker6dof.commands.replay import replay_rgbd


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_replay_rejects_invalid_edge_threshold(monkeypatch, value, capsys):
    monkeypatch.setattr(sys, "argv", ["replay", "--recording", "unused.db3",
                                    "--rig", "unused.json", "--output-dir", "unused",
                                    "--min-edge-px", value])
    with pytest.raises(SystemExit) as error:
        replay_rgbd.parse_args()
    assert error.value.code == 2
    assert "--min-edge-px must be finite and positive" in capsys.readouterr().err


def test_replay_preserves_default_and_accepts_explicit_gate(tmp_path, monkeypatch):
    recording = tmp_path / "input.db3"
    rig = tmp_path / "rig.json"
    recording.touch()
    rig.write_text("{}")
    arguments = ["replay", "--recording", str(recording), "--rig", str(rig),
                 "--output-dir", str(tmp_path / "output")]
    monkeypatch.setattr(sys, "argv", arguments)
    assert replay_rgbd.parse_args().min_edge_px == 24
    monkeypatch.setattr(sys, "argv", arguments + ["--min-edge-px", "12"])
    assert replay_rgbd.parse_args().min_edge_px == 12


def test_edge_gate_changes_acceptance_for_small_marker(monkeypatch):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from marker6dof import tracking

    intrinsics = SimpleNamespace(width=1280, height=720)
    project = lambda _, points: points[:, :2] / points[:, 2:3] * 650 + [640, 360]
    monkeypatch.setattr(tracking, "project_realsense", project)
    monkeypatch.setattr(tracking, "normalized_image_points",
                        lambda _, pixels: (pixels - [640, 360]) / 650)
    default = tracking.RigTracker({2: np.eye(4)}, .0375)
    experiment = tracking.RigTracker({2: np.eye(4)}, .0375, min_edge_px=12)
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [175, 5, 2], degrees=True).as_matrix()
    truth[:3, 3] = [.02, -.01, 1.5]
    corners = project(intrinsics, default.points[2] @ truth[:3, :3].T + truth[:3, 3])
    assert default.solve({2: corners}, intrinsics, 1.)["pose"] is None
    result = experiment.solve({2: corners}, intrinsics, 1.)
    assert result["status"] == "SINGLE"
    np.testing.assert_allclose(result["pose"], truth, atol=1e-6)


def test_edge_weighted_refinement_recovers_multi_marker_pose(monkeypatch):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from marker6dof import tracking

    intrinsics = SimpleNamespace(width=1280, height=720)
    project = lambda _, points: points[:, :2] / points[:, 2:3] * 650 + [640, 360]
    monkeypatch.setattr(tracking, "project_realsense", project)
    monkeypatch.setattr(tracking, "normalized_image_points",
                        lambda _, pixels: (pixels - [640, 360]) / 650)

    tag2 = np.eye(4)
    tag2[:3, 3] = [-.06, 0, 0]
    tag3 = np.eye(4)
    tag3[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    tag3[:3, 3] = [.06, 0, 0]
    rig = {2: np.eye(4), 3: np.eye(4)}
    rig[2], rig[3] = tag2, tag3
    tracker = tracking.RigTracker(rig, .0475, min_edge_px=1, corner_weighting="edge_size")
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [175, 5, 2], degrees=True).as_matrix()
    truth[:3, 3] = [.02, -.01, 1.2]
    detections = {
        tag: project(intrinsics, tracker.points[tag] @ truth[:3, :3].T + truth[:3, 3])
        for tag in rig
    }

    result = tracker.solve(detections, intrinsics, 1.)

    assert result["status"] == "MULTI"
    np.testing.assert_allclose(result["pose"], truth, atol=1e-6)


def test_huber_corner_loss_limits_single_corner_outlier(monkeypatch):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from marker6dof import tracking

    intrinsics = SimpleNamespace(width=1280, height=720)
    project = lambda _, points: points[:, :2] / points[:, 2:3] * 650 + [640, 360]
    monkeypatch.setattr(tracking, "project_realsense", project)
    monkeypatch.setattr(tracking, "normalized_image_points",
                        lambda _, pixels: (pixels - [640, 360]) / 650)

    tag2 = np.eye(4)
    tag2[:3, 3] = [-.06, 0, 0]
    tag3 = np.eye(4)
    tag3[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    tag3[:3, 3] = [.06, 0, 0]
    rig = {2: tag2, 3: tag3}
    seed_tracker = tracking.RigTracker(rig, .0475, min_edge_px=1)
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [175, 5, 2], degrees=True).as_matrix()
    truth[:3, 3] = [.02, -.01, 1.2]
    detections = {
        tag: project(intrinsics, seed_tracker.points[tag] @ truth[:3, :3].T + truth[:3, 3])
        for tag in rig
    }
    detections[2][0] += [2., -1.5]

    uniform = tracking.RigTracker(rig, .0475, min_edge_px=1).solve(detections, intrinsics, 1.)
    robust = tracking.RigTracker(
        rig, .0475, min_edge_px=1, corner_loss="huber", corner_loss_scale_px=.5,
    ).solve(detections, intrinsics, 1.)

    assert uniform["status"] == robust["status"] == "MULTI"
    uniform_translation_error = np.linalg.norm(uniform["pose"][:3, 3] - truth[:3, 3])
    robust_translation_error = np.linalg.norm(robust["pose"][:3, 3] - truth[:3, 3])
    uniform_rotation_error = np.linalg.norm(Rotation.from_matrix(
        truth[:3, :3].T @ uniform["pose"][:3, :3]
    ).as_rotvec())
    robust_rotation_error = np.linalg.norm(Rotation.from_matrix(
        truth[:3, :3].T @ robust["pose"][:3, :3]
    ).as_rotvec())
    assert robust_translation_error < uniform_translation_error
    assert robust_rotation_error < uniform_rotation_error

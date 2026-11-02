"""Check offline VIO time alignment used by the task video export."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_t265_vio_stage.py"
SPEC = importlib.util.spec_from_file_location("build_t265_vio_stage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample(time_ms, x, angle_deg=0, confidence=2):
    quat = Rotation.from_euler("z", angle_deg, degrees=True).as_quat()
    return {"pose_timestamp_ms": str(time_ms), "tx_m": str(x), "ty_m": "0", "tz_m": "0",
            "qx": str(quat[0]), "qy": str(quat[1]), "qz": str(quat[2]), "qw": str(quat[3]),
            "tracker_confidence": str(confidence)}


def test_vio_interpolates_at_camera_time():
    samples = [sample(1000, 0, 0), sample(1040, .04, 90)]
    pose, reason, span = MODULE.interpolated_pose(samples, [1000, 1040], 1020)
    assert reason == "OK" and span == 40
    np.testing.assert_allclose(pose[:3, 3], [.02, 0, 0])
    np.testing.assert_allclose(Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)[2],
                               45, atol=1e-6)


def test_vio_rejects_unbracketed_stale_and_low_confidence_pose():
    samples = [sample(1000, 0), sample(1090, .09, confidence=1)]
    for camera_ms, expected in ((999, "NO_BRACKET"), (1030, "VIO_TIME_GAP"),
                                (1100, "NO_BRACKET")):
        pose, reason, _ = MODULE.interpolated_pose(samples, [1000, 1090], camera_ms)
        assert pose is None and reason == expected
    samples[1] = sample(1070, .07, confidence=1)
    pose, reason, _ = MODULE.interpolated_pose(samples, [1000, 1070], 1030)
    assert pose is None and reason == "VIO_LOW_CONFIDENCE"

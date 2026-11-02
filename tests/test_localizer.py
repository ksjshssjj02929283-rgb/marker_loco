from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from marker6dof.localization import MarkerDetection, MarkerLocalizer, load_calibration, make_transform


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "demo_calibration.json"


class MarkerLocalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_calibration(CONFIG)
        cls.localizer = MarkerLocalizer(cls.calibration)
        cls.body = cls.calibration.bodies["demo_gripper"]

    def test_joint_pnp_recovers_tcp_pose(self):
        expected = make_transform(np.asarray([0.08, -0.18, 0.03]), np.asarray([0.02, -0.01, 0.68]))
        rvec = cv2.Rodrigues(expected[:3, :3])[0]
        detections = []
        for tag_id, corners in self.body.tag_corners_gripper_m.items():
            pixels = cv2.projectPoints(
                corners,
                rvec,
                expected[:3, 3],
                self.calibration.camera_matrix,
                self.calibration.distortion,
            )[0].reshape(4, 2)
            detections.append(MarkerDetection(tag_id, pixels))
        estimate = self.localizer.estimate_body(detections, self.body)
        self.assertEqual(estimate.status, "MEASURED")
        np.testing.assert_allclose(estimate.T_camera_gripper, expected, atol=1e-6)
        np.testing.assert_allclose(
            estimate.T_camera_tcp,
            expected @ self.body.T_gripper_tcp,
            atol=1e-6,
        )

    def test_unknown_marker_is_not_used(self):
        detection = MarkerDetection(42, np.zeros((4, 2), dtype=np.float64))
        estimate = self.localizer.estimate_body([detection], self.body)
        self.assertEqual(estimate.status, "LOST")
        self.assertEqual(estimate.reason, "insufficient_registered_markers")

    def test_duplicate_id_is_rejected(self):
        corners = np.asarray([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float64)
        estimate = self.localizer.estimate_body(
            [MarkerDetection(2, corners), MarkerDetection(2, corners)], self.body
        )
        self.assertEqual(estimate.status, "LOST")
        self.assertEqual(estimate.reason, "duplicate_marker_id")

    def test_config_requires_metric_units(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["units"] = "mm"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "units"):
                load_calibration(path)

    def test_image_size_must_match_calibration(self):
        with self.assertRaisesRegex(ValueError, "does not match calibration"):
            self.localizer.detect(np.zeros((100, 100, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

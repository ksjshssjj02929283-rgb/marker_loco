from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from marker6dof.depth import fuse_rgb_depth_pose, sample_depth_m
from marker6dof.projection import load_transform, transform_points, validate_rigid_transform


class ProjectionAndDepthTests(unittest.TestCase):
    def test_transform_points_and_validation(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.1, -0.2, 0.3]
        points = transform_points(transform, [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        np.testing.assert_allclose(points[0], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(points[1], [1.1, 1.8, 3.3])
        bad = transform.copy()
        bad[0, 0] = 2.0
        with self.assertRaises(ValueError):
            validate_rigid_transform(bad)

    def test_load_transform_uses_explicit_cad_key(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "cad.json"
            path.write_text(json.dumps({"T_reference_TCP": np.eye(4).tolist()}), encoding="utf-8")
            np.testing.assert_allclose(load_transform(path), np.eye(4))
            path.write_text(
                json.dumps({"units": "mm", "T_reference_TCP": np.eye(4).tolist()}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_transform(path)

            reference_from_cad = np.eye(4)
            reference_from_cad[0, 3] = 0.1
            cad_from_tcp = np.eye(4)
            cad_from_tcp[1, 3] = 0.2
            path.write_text(
                json.dumps({
                    "units": "m",
                    "T_reference_CAD": reference_from_cad.tolist(),
                    "T_CAD_TCP": cad_from_tcp.tolist(),
                }),
                encoding="utf-8",
            )
            expected = reference_from_cad @ cad_from_tcp
            np.testing.assert_allclose(load_transform(path), expected)

    def test_depth_patch_ignores_zero_holes(self):
        image = np.zeros((9, 9), dtype=np.uint16)
        image[3:6, 3:6] = 1000
        self.assertAlmostEqual(sample_depth_m(image, [4, 4], 0.001), 1.0)
        self.assertTrue(np.isnan(sample_depth_m(image, [0, 0], 0.001)))

    def test_fusion_rejects_inconsistent_depth(self):
        rgb = np.eye(4)
        depth = np.eye(4)
        depth[:3, 3] = [0.2, 0.0, 0.0]
        fused, report = fuse_rgb_depth_pose(
            rgb, depth, {"status": "DEPTH_POSE"}, max_translation_delta_m=0.05
        )
        np.testing.assert_allclose(fused, rgb)
        self.assertEqual(report["status"], "DEPTH_REJECTED_RGB_DISAGREEMENT")

    def test_fusion_low_weight_preserves_rgb_geometry(self):
        rgb = np.eye(4)
        depth = np.eye(4)
        depth[:3, 3] = [0.01, 0.0, 0.0]
        depth[:3, :3] = Rotation.from_euler("z", 5, degrees=True).as_matrix()
        fused, report = fuse_rgb_depth_pose(
            rgb, depth, {"status": "DEPTH_POSE"}, max_translation_delta_m=0.05,
            max_rotation_delta_deg=10.0, depth_weight=0.2,
        )
        self.assertEqual(report["status"], "RGB_DEPTH_FUSED")
        self.assertAlmostEqual(fused[0, 3], 0.002, places=7)
        self.assertLess(np.degrees(np.linalg.norm(Rotation.from_matrix(fused[:3, :3]).as_rotvec())), 2.0)


if __name__ == "__main__":
    unittest.main()

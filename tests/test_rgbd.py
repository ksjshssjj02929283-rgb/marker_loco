"""Synthetic geometry checks; these are not D455 accuracy benchmarks."""

from __future__ import annotations

import json
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from marker6dof.rgbd import refine_pose_rgbd


def project(points):
    points = np.asarray(points, dtype=np.float64)
    return points[:, :2] / points[:, 2:3] * [650.0, 645.0] + [640.0, 360.0]


def pose(rotation=(0.12, -0.18, 0.02), translation=(0.01, -0.01, 0.65)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    value[:3, 3] = translation
    return value


def transform(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def scene(multiple=False, noise=0.00025):
    corners = np.array([[-0.03, 0.03, 0.0], [0.03, 0.03, 0.0],
                        [0.03, -0.03, 0.0], [-0.03, -0.03, 0.0]])
    grid = np.linspace(-0.02, 0.02, 19)
    xx, yy = np.meshgrid(grid, grid)
    surface = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    models = {2: corners}
    samples = [surface]
    if multiple:
        models[2] = corners + [-0.055, 0.0, 0.0]
        samples[0] = surface + [-0.055, 0.0, 0.0]
        side = Rotation.from_euler("y", 55, degrees=True).as_matrix()
        models[5] = corners @ side.T + [0.045, 0.0, -0.01]
        samples.append(surface @ side.T + [0.045, 0.0, -0.01])
    truth = pose()
    images = {key: project(transform(truth, value)) for key, value in models.items()}
    camera = transform(truth, np.concatenate(samples))
    pixels = project(camera)
    random = np.random.default_rng(4)
    # Axial noise follows camera rays, preserving each point's RGB pixel.
    camera *= 1.0 + random.normal(0.0, noise, (len(camera), 1)) / camera[:, 2:3]
    return truth, models, images, camera, pixels


class RGBDRefinementTests(unittest.TestCase):
    def assert_report_json(self, report):
        json.dumps(report, allow_nan=False)
        self.assertIn("rgb_reprojection_error_px", report)
        self.assertIn("fused_reprojection_error_px", report)

    def test_single_face_corrects_small_depth_offset(self):
        truth, models, images, points, pixels = scene()
        rgb = truth.copy()
        rgb[2, 3] += 0.007
        result, report = refine_pose_rgbd(rgb, models, images, points, pixels, project)
        self.assertEqual(report["status"], "RGB_DEPTH_REFINED")
        self.assertGreater(report["valid_depth_points"], 20)
        self.assertLess(abs(result[2, 3] - truth[2, 3]), 0.007)
        self.assertLess(report["fused_reprojection_error_px"], report["rgb_reprojection_error_px"])
        self.assertGreater(report["planes"][0]["coverage_ratio"], 0.2)
        np.testing.assert_allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(result[:3, :3]), 1.0, places=10)
        self.assert_report_json(report)

    def test_multiple_faces_recover_pose(self):
        truth, models, images, points, pixels = scene(multiple=True)
        rgb = truth.copy()
        rgb[:3, :3] = Rotation.from_rotvec([0.006, -0.01, 0.001]).as_matrix() @ rgb[:3, :3]
        rgb[:3, 3] += [0.001, -0.001, 0.005]
        result, report = refine_pose_rgbd(rgb, models, images, points, pixels, project)
        self.assertEqual(report["status"], "RGB_DEPTH_REFINED")
        self.assertEqual(sum(item["status"] == "PLANE_ACCEPTED" for item in report["planes"]), 2)
        self.assertLess(np.linalg.norm(result[:3, 3] - truth[:3, 3]),
                        np.linalg.norm(rgb[:3, 3] - truth[:3, 3]))
        self.assertLess(report["fused_reprojection_error_px"], report["rgb_reprojection_error_px"])
        self.assert_report_json(report)

    def test_float32_projector_has_nonzero_finite_difference_derivatives(self):
        truth, models, images, points, pixels = scene()
        rgb = truth.copy()
        rgb[2, 3] += 0.006
        result, report = refine_pose_rgbd(
            rgb, models, images, points, pixels,
            lambda camera: project(np.asarray(camera, dtype=np.float32)).astype(np.float32))
        self.assertEqual(report["status"], "RGB_DEPTH_REFINED")
        self.assertLess(abs(result[2, 3] - truth[2, 3]), 0.002)
        self.assertLess(report["fused_reprojection_error_px"], report["rgb_reprojection_error_px"])

    def test_independent_depth_improves_noisy_corner_pnp_in_synthetic_trials(self):
        # A small 48-mm tag at 0.85 m, 0.35-px corner noise and independent
        # 1-mm axial depth noise. PnP + RGB-only LM already minimizes the
        # measured corner residual before depth refinement is attempted.
        # Fixed seeds include planar-PnP ambiguities and unchanged fallbacks.
        camera_matrix = np.array([[650.0, 0.0, 640.0], [0.0, 645.0, 360.0], [0.0, 0.0, 1.0]])
        corners = np.array([[-0.024, 0.024, 0.0], [0.024, 0.024, 0.0],
                            [0.024, -0.024, 0.0], [-0.024, -0.024, 0.0]])
        xx, yy = np.meshgrid(np.linspace(-0.016, 0.016, 17), np.linspace(-0.016, 0.016, 17))
        samples = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        truth = pose(translation=(0.01, -0.01, 0.85))
        camera = transform(truth, samples)
        pixels = project(camera)
        perfect_corners = project(transform(truth, corners))
        errors, accepted = [], 0
        for seed in range(20):
            random = np.random.default_rng(seed)
            measured_corners = perfect_corners + random.normal(0.0, 0.35, (4, 2))
            success, rvec, tvec = cv2.solvePnP(corners, measured_corners, camera_matrix, np.zeros(5),
                                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
            self.assertTrue(success)
            rvec, tvec = cv2.solvePnPRefineLM(corners, measured_corners, camera_matrix, np.zeros(5), rvec, tvec)
            rgb = np.eye(4)
            rgb[:3, :3] = cv2.Rodrigues(rvec)[0]
            rgb[:3, 3] = tvec.ravel()
            noisy_camera = camera * (1.0 + random.normal(0.0, 0.001, (len(camera), 1)) / camera[:, 2:3])
            fused, report = refine_pose_rgbd(rgb, {2: corners}, {2: measured_corners}, noisy_camera, pixels, project)
            accepted += report["status"] == "RGB_DEPTH_REFINED"
            errors.append([np.linalg.norm(rgb[:3, 3] - truth[:3, 3]),
                           np.linalg.norm(fused[:3, 3] - truth[:3, 3]),
                           np.degrees(Rotation.from_matrix(rgb[:3, :3] @ truth[:3, :3].T).magnitude()),
                           np.degrees(Rotation.from_matrix(fused[:3, :3] @ truth[:3, :3].T).magnitude())])
            self.assert_report_json(report)
        average = np.mean(errors, axis=0)
        self.assertGreaterEqual(accepted, 10)
        self.assertLess(average[1], average[0] * 0.65)
        self.assertLess(average[3], average[2])

    def test_background_is_removed_by_rgb_plane_gate(self):
        truth, models, images, points, pixels = scene()
        background = points * ((points[:, 2:3] + 0.2) / points[:, 2:3])
        contaminated = np.concatenate([points, background])
        all_pixels = np.concatenate([pixels, pixels])
        _, report = refine_pose_rgbd(truth, models, images, contaminated, all_pixels, project)
        self.assertEqual(report["status"], "RGB_DEPTH_REFINED")
        plane = report["planes"][0]
        self.assertLess(plane["gated_points"], plane["roi_points"])
        self.assertAlmostEqual(plane["inlier_ratio"], 1.0)
        self.assert_report_json(report)

    def test_ransac_excludes_near_plane_outliers(self):
        truth, models, images, points, pixels = scene()
        points[::5] *= (points[::5, 2:3] + 0.015) / points[::5, 2:3]
        _, report = refine_pose_rgbd(truth, models, images, points, pixels, project)
        self.assertEqual(report["status"], "RGB_DEPTH_REFINED")
        plane = report["planes"][0]
        self.assertGreater(plane["inlier_ratio"], 0.65)
        self.assertLess(plane["inlier_ratio"], 0.9)
        self.assertLess(plane["rms_m"], 0.001)

    def test_inconsistent_depth_returns_exact_rgb_pose(self):
        truth, models, images, points, pixels = scene()
        points *= (points[:, 2:3] + 0.1) / points[:, 2:3]
        result, report = refine_pose_rgbd(truth, models, images, points, pixels, project)
        np.testing.assert_array_equal(result, truth)
        self.assertEqual(report["status"], "DEPTH_REJECTED_RGB_DISAGREEMENT")
        self.assertEqual(report["valid_depth_points"], 0)
        self.assert_report_json(report)

    def test_per_marker_reprojection_growth_rejects_biased_depth(self):
        truth, models, images, points, pixels = scene(noise=0)
        points *= (points[:, 2:3] + 0.025) / points[:, 2:3]
        result, report = refine_pose_rgbd(
            truth, models, images, points, pixels, project, depth_weight=1.0,
            max_reprojection_increase_px=0.02)
        np.testing.assert_array_equal(result, truth)
        self.assertEqual(report["status"], "DEPTH_REJECTED_REPROJECTION")
        self.assertGreater(report["candidate_reprojection_error_px"], 0.02)
        self.assertEqual(report["fused_reprojection_error_px"], report["rgb_reprojection_error_px"])
        self.assert_report_json(report)

    def test_one_bad_marker_cannot_hide_behind_average_reprojection(self):
        truth, models, images, points, pixels = scene(multiple=True, noise=0)
        count = len(points) // 2
        points[:count] *= (points[:count, 2:3] + 0.004) / points[:count, 2:3]
        result, report = refine_pose_rgbd(
            truth, models, images, points, pixels, project, depth_weight=1.0,
            max_reprojection_increase_px=10.0, max_reprojection_error_px=0.5)
        self.assertEqual(report["status"], "DEPTH_REJECTED_REPROJECTION")
        # The aggregate would pass; the worst marker must still reject it.
        self.assertLess(report["candidate_reprojection_error_px"], 0.5)
        self.assertGreater(max(report["candidate_per_tag_reprojection_error_px"].values()), 0.5)
        np.testing.assert_array_equal(result, truth)

    def test_too_few_plane_inliers_are_rejected(self):
        truth, models, images, points, pixels = scene(noise=0)
        count = len(points) // 2
        points[:count] *= (points[:count, 2:3] + 0.015) / points[:count, 2:3]
        result, report = refine_pose_rgbd(truth, models, images, points, pixels, project)
        np.testing.assert_array_equal(result, truth)
        self.assertEqual(report["valid_depth_points"], 0)
        self.assertEqual(report["planes"][0]["status"], "INSUFFICIENT_PLANE_SUPPORT")

    def test_optimizer_budget_exhaustion_falls_back(self):
        truth, models, images, points, pixels = scene()
        rgb = truth.copy()
        rgb[2, 3] += 0.006
        result, report = refine_pose_rgbd(rgb, models, images, points, pixels, project, max_nfev=1)
        self.assertEqual(report["status"], "DEPTH_REJECTED_OPTIMIZATION")
        np.testing.assert_array_equal(result, rgb)
        self.assert_report_json(report)

    def test_projector_runtime_failures_do_not_escape_live_pipeline(self):
        truth, models, images, points, pixels = scene()
        for fail_after, expected in ((0, "RGB_INVALID_PROJECTION"), (1, "DEPTH_REJECTED_OPTIMIZATION")):
            calls = 0

            def unstable_project(camera):
                nonlocal calls
                calls += 1
                if calls > fail_after:
                    raise RuntimeError("synthetic camera projection failure")
                return project(camera)

            result, report = refine_pose_rgbd(truth, models, images, points, pixels, unstable_project)
            self.assertEqual(report["status"], expected)
            np.testing.assert_array_equal(result, truth)
            self.assert_report_json(report)

    def test_zero_weight_preserves_rgb_exactly(self):
        truth, models, images, points, pixels = scene()
        result, report = refine_pose_rgbd(truth, models, images, points, pixels, project, depth_weight=0)
        np.testing.assert_array_equal(result, truth)
        self.assertEqual(report["status"], "RGB_DEPTH_DISABLED")
        self.assertEqual(report["translation_delta_m"], 0.0)
        self.assertEqual(report["rotation_delta_deg"], 0.0)

    def test_empty_and_nonfinite_depth_fall_back_safely(self):
        truth, models, images, _, _ = scene()
        for points, pixels in ((np.empty((0, 3)), np.empty((0, 2))),
                               (np.full((25, 3), np.nan), np.zeros((25, 2))),
                               (np.ones((25, 3)), np.full((25, 2), np.inf))):
            with self.subTest(points=points.shape):
                result, report = refine_pose_rgbd(truth, models, images, points, pixels, project)
                np.testing.assert_array_equal(result, truth)
                self.assertEqual(report["status"], "INSUFFICIENT_DEPTH")
                self.assert_report_json(report)

    def test_nonfinite_samples_do_not_discard_valid_plane(self):
        truth, models, images, points, pixels = scene()
        points[:7] = np.nan
        pixels[7:10] = np.inf
        _, report = refine_pose_rgbd(truth, models, images, points, pixels, project)
        self.assertEqual(report["status"], "RGB_DEPTH_REFINED")
        self.assertEqual(report["finite_depth_points"], len(points) - 10)
        self.assert_report_json(report)

    def test_collinear_and_tiny_patch_samples_are_rejected(self):
        truth, models, images, _, _ = scene()
        for half_width in (0.0, 0.0001):
            x = np.linspace(-0.018, 0.018, 80)
            y = np.where(np.arange(80) % 2, half_width, -half_width)
            points = transform(truth, np.column_stack([x, y, np.zeros(80)]))
            result, report = refine_pose_rgbd(truth, models, images, points, project(points), project)
            np.testing.assert_array_equal(result, truth)
            self.assertEqual(report["status"], "INSUFFICIENT_DEPTH")
            self.assertEqual(report["valid_depth_points"], 0)
            self.assertEqual(report["planes"][0]["status"], "DEGENERATE_DEPTH_GEOMETRY")
            self.assert_report_json(report)

    def test_depth_sampling_is_bounded_and_deterministic(self):
        truth, models, images, points, pixels = scene()
        first, report1 = refine_pose_rgbd(truth, models, images, points, pixels, project, max_points_per_face=80)
        second, report2 = refine_pose_rgbd(truth, models, images, points, pixels, project, max_points_per_face=80)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(report1, report2)
        self.assertLessEqual(report1["valid_depth_points"], 80)
        self.assertEqual(report1["planes"][0]["sampled_points"], 80)

    def test_inputs_are_not_mutated(self):
        truth, models, images, points, pixels = scene()
        copies = [value.copy() for value in [truth, points, pixels, models[2], images[2]]]
        refine_pose_rgbd(truth, models, images, points, pixels, project)
        for actual, expected in zip([truth, points, pixels, models[2], images[2]], copies):
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_configuration_is_rejected_even_without_depth(self):
        truth, models, images, _, _ = scene()
        invalid = [{"depth_weight": value} for value in (-0.1, 1.1, np.nan, np.inf)] + [
            {"depth_sigma_m": 0}, {"pixel_sigma_px": np.nan}, {"min_depth_points": 2},
            {"max_points_per_face": 10}, {"max_points_per_face": 20.5},
            {"roi_inset_fraction": 0.5}, {"min_coverage_ratio": 0},
            {"max_reprojection_increase_px": -1}, {"max_reprojection_ratio": 0.9},
            {"max_translation_delta_m": np.inf}, {"ransac_iterations": 0},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                refine_pose_rgbd(truth, models, images, [], [], project, **options)

    def test_malformed_arrays_and_nonrigid_pose_are_rejected(self):
        truth, models, images, points, pixels = scene()
        with self.assertRaises(ValueError):
            refine_pose_rgbd(truth, models, images, points[:, :2], pixels, project)
        with self.assertRaises(ValueError):
            refine_pose_rgbd(truth, models, images, points[:-1], pixels, project)
        with self.assertRaises(ValueError):
            refine_pose_rgbd(truth, models, images, np.empty((0, 4)), np.empty((0, 2)), project)
        bad = truth.copy()
        bad[0, 0] = 3
        with self.assertRaises(ValueError):
            refine_pose_rgbd(bad, models, images, points, pixels, project)
        with self.assertRaises(ValueError):
            refine_pose_rgbd(truth, {2: np.zeros((4, 3))}, images, points, pixels, project)


if __name__ == "__main__":
    unittest.main()

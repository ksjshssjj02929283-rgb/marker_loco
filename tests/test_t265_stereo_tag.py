import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from marker6dof.camera import OpenCVCamera
from marker6dof.realsense import marker_object_points
from marker6dof.t265 import solve_stereo_tag_pose


def test_stereo_tag_pose_uses_both_eyes_to_recover_metric_pose():
    camera = OpenCVCamera(
        848, 800,
        np.array([[300.0, 0.0, 424.0], [0.0, 300.0, 400.0], [0.0, 0.0, 1.0]]),
        np.zeros(4), "equidistant",
    )
    tag = marker_object_points(0.0475)
    true_rotation = Rotation.from_euler("xyz", [16, -12, 8], degrees=True).as_matrix()
    true_translation = np.array([0.04, -0.02, 0.42])
    T_left_right = np.eye(4)
    T_left_right[0, 3] = 0.063
    left_points = (true_rotation @ tag.T).T + true_translation
    right_points = left_points - T_left_right[:3, 3]

    rvec, tvec, error_px = solve_stereo_tag_pose(
        camera, camera, camera.project_points(left_points),
        camera.project_points(right_points), tag, T_left_right,
    )

    assert np.linalg.norm(tvec - true_translation) < 0.001
    estimated_rotation = cv2.Rodrigues(rvec)[0]
    assert Rotation.from_matrix(estimated_rotation.T @ true_rotation).magnitude() < np.deg2rad(0.2)
    assert error_px < 0.01

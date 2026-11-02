"""Regression check for stereo extrinsic direction and joint pose recovery."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from marker6dof.camera import load_kalibr_camera


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/replay_stereo_joint_20261030.py"
SPEC = importlib.util.spec_from_file_location("replay_stereo_joint", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_joint_pose_recovers_known_stereo_projection():
    cameras = {"left": load_kalibr_camera(MODULE.CAMCHAIN, "cam0"),
               "right": load_kalibr_camera(MODULE.CAMCHAIN, "cam1")}
    calibration = yaml.safe_load(MODULE.EXTRINSICS.read_text())["T_1_0"]
    T_1_0 = np.eye(4)
    T_1_0[:3, :3] = Rotation.from_quat(calibration["quaternion_xyzw"]).as_matrix()
    T_1_0[:3, 3] = calibration["translation_m"]
    points = MODULE.object_corners(.05)
    truth = np.array([.08, -.07, .04, .025, .015, .55])
    T_0_tag = MODULE.transform(truth[:3], truth[3:])
    T_1_tag = T_1_0 @ T_0_tag
    rows = []
    for eye, T in (("left", T_0_tag), ("right", T_1_tag)):
        xyz = (T[:3, :3] @ points.T).T + T[:3, 3]
        pixels = cameras[eye].project_points(xyz)
        rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
        row = {f"r{axis}_rad": rvec[i] for i, axis in enumerate("xyz")}
        row.update({f"t{axis}_m": T[i, 3] for i, axis in enumerate("xyz")})
        for i, corner in enumerate(pixels):
            for j, axis in enumerate("xy"):
                row[f"corner_{i}_{axis}"] = corner[j]
        rows.append(row)
    estimate, _, left_error, right_error = MODULE.solve_pair(*rows, cameras, T_1_0, points)
    assert np.linalg.norm(estimate[3:] - truth[3:]) < 1e-7
    assert left_error < 1e-5
    assert right_error < 1e-5

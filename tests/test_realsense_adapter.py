from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from marker6dof.realsense import (
    DEFAULT_MARKER_SIZE_MM,
    detector_parameters,
    detect_markers,
    marker_object_points,
)
from marker6dof.tracking import load_rig, marker_object_points as tracking_marker_object_points
from marker6dof.t265 import ensure_t265_sdk, extrinsics_to_matrix, fuse_stereo_results


@pytest.mark.parametrize("pose_api", [
    None,
    "cv2.solvePnPGeneric(SOLVEPNP_IPPE_SQUARE) + joint multi-face solvePnP",
])
def test_d455_metadata_uses_lazy_sdk_without_hardware(tmp_path, monkeypatch, pose_api):
    from marker6dof.d455 import write_metadata

    camera_info = SimpleNamespace(
        name="name", serial_number="serial", firmware_version="firmware",
        product_line="product_line",
    )
    sdk = SimpleNamespace(__version__="fake-sdk-2.53", camera_info=camera_info)
    monkeypatch.setitem(sys.modules, "pyrealsense2", sdk)
    device_info = {"name": "Fake D455", "serial": "fake-serial", "firmware": "fake-firmware"}
    device = SimpleNamespace(
        supports=lambda key: key in device_info,
        get_info=lambda key: device_info[key],
    )
    args = SimpleNamespace(width=1280, height=720, fps=30, marker_size_mm=47.5)
    intrinsics = SimpleNamespace(
        width=1280, height=720, fx=600.0, fy=601.0, ppx=640.0, ppy=360.0,
        model="distortion.brown_conrady", coeffs=[0.1, 0.2, 0.0, 0.0, 0.0],
    )
    output = tmp_path / "metadata.json"

    write_metadata(output, args, device, intrinsics, 0.001, {3, 2}, pose_api=pose_api)

    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["software"]["librealsense"] == "fake-sdk-2.53"
    assert metadata["device"] == device_info
    assert metadata["stream"] == {"width": 1280, "height": 720, "fps": 30}
    assert metadata["aruco"]["accepted_ids"] == [2, 3]
    assert metadata["aruco"]["marker_black_square_size_m"] == 0.0475
    assert metadata["aruco"]["pose_api"] == (pose_api or "cv2.solvePnP(SOLVEPNP_IPPE_SQUARE)")
    assert metadata["color_intrinsics"] == {
        "width": 1280, "height": 720, "fx": 600.0, "fy": 601.0,
        "cx": 640.0, "cy": 360.0, "distortion_model": "distortion.brown_conrady",
        "coefficients": [0.1, 0.2, 0.0, 0.0, 0.0],
    }
    assert metadata["depth_scale_m_per_unit"] == 0.001


def test_marker_geometry_is_shared_with_tracking():
    marker_size_m = DEFAULT_MARKER_SIZE_MM / 1000
    expected = marker_object_points(marker_size_m)
    np.testing.assert_array_equal(expected, tracking_marker_object_points(marker_size_m))
    np.testing.assert_allclose(expected[0], [-0.02375, 0.02375, 0.0])
    assert expected.dtype == np.float32


def test_marker_geometry_rejects_non_positive_size():
    with pytest.raises(ValueError, match="positive"):
        marker_object_points(0.0)


def test_shared_detector_finds_synthetic_marker():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 2, 120, borderBits=1)
    image = np.full((180, 180), 255, dtype=np.uint8)
    image[30:150, 30:150] = marker

    corners, ids, rejected = detect_markers(image, dictionary, detector_parameters())

    assert ids is not None
    assert 2 in ids.ravel().tolist()
    assert len(corners) == 1
    assert isinstance(rejected, tuple | list)


def test_rig_size_must_match_runtime_marker_size():
    transform = np.eye(4).tolist()
    rig = {"reference_id": 2, "marker_size_m": 0.0475, "T_reference_tag": {"2": transform}}

    assert set(load_rig(rig, 0.0475)) == {2}
    with pytest.raises(ValueError, match="differs from rig calibration"):
        load_rig(rig, 0.048)


def test_rig_size_metadata_is_required():
    rig = {"reference_id": 2, "T_reference_tag": {"2": np.eye(4).tolist()}}

    with pytest.raises(ValueError, match="has no marker_size_m"):
        load_rig(rig, 0.0475)


def test_t265_extrinsics_are_target_from_source():
    class Extrinsics:
        rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        translation = [0.063, 0.0, 0.0]

    transform = extrinsics_to_matrix(Extrinsics())
    np.testing.assert_allclose(transform[:3, 3], [0.063, 0, 0])
    np.testing.assert_allclose(transform[:3, :3], np.eye(3))


def test_t265_sdk_guard_rejects_removed_support():
    class NewSdk:
        __version__ = "2.58.4"

    with pytest.raises(RuntimeError, match="no longer supports T265"):
        ensure_t265_sdk(NewSdk())


def test_t265_stereo_fusion_transforms_right_pose_to_left_frame():
    left_pose = np.eye(4)
    right_pose = np.eye(4)
    right_pose[0, 3] = 0.1
    right_to_left = np.eye(4)
    right_to_left[0, 3] = -0.1
    left = {"status": "SINGLE", "pose": left_pose, "visible_ids": [2],
            "used_ids": [2], "reprojection_error_px": 1.0}
    right = {"status": "SINGLE", "pose": right_pose, "visible_ids": [3],
             "used_ids": [3], "reprojection_error_px": 1.0}

    result = fuse_stereo_results(left, right, right_to_left)

    assert result["status"] == "STEREO"
    assert result["used_ids"] == [2, 3]
    np.testing.assert_allclose(result["pose"][:3, 3], [0.0, 0, 0])

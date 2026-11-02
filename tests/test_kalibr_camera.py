"""Camera-model and ROS2 timestamp integration regression tests."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from marker6dof.camera import OpenCVCamera, load_kalibr_camera
from marker6dof.realsense import marker_object_points
from marker6dof.tracking import RigTracker


def camera():
    return OpenCVCamera(1280, 960, np.array([[671., 0, 649.], [0, 672., 442.], [0, 0, 1]]),
                        np.array([-.038, .001, -.007, .002]), "equidistant")


def test_fisheye_round_trip_and_off_axis_rig_pose_without_sdk(monkeypatch):
    import marker6dof.realsense as rs
    monkeypatch.setattr(rs, "_realsense", lambda: pytest.fail("SDK must not be loaded"))
    cam = camera()
    rays = np.array([[0, 0, 1], [.7, .5, 1], [-.6, -.4, 1]])
    np.testing.assert_allclose(cam.normalized_points(cam.project_points(rays)), rays[:, :2], atol=1e-8)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [170, 15, 12], degrees=True).as_matrix()
    pose[:3, 3] = [.3, .1, .75]
    points = marker_object_points(.0475).astype(float)
    pixels = cam.project_points(points @ pose[:3, :3].T + pose[:3, 3])
    tracker = RigTracker({2: np.eye(4)}, .0475, min_edge_px=12)
    result = tracker.solve({2: pixels}, cam, 1.)
    assert result["status"] == "SINGLE"
    np.testing.assert_allclose(result["pose"], pose, atol=1e-7)
    # Treating these four coefficients as radtan would fail this geometric test.
    assert result["reprojection_error_px"] < 1e-7
    result = tracker.solve({}, cam, 2.)
    assert result["pose"] is None


def test_load_rejects_wrong_distortion_count(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "camera.yaml"
    p.write_text("cam0:\n  camera_model: pinhole\n  distortion_model: equidistant\n  distortion_coeffs: [0, 0, 0]\n  intrinsics: [671, 672, 649, 442]\n  resolution: [1280, 960]\n")
    with pytest.raises(ValueError, match="four"):
        load_kalibr_camera(p)


def test_duplicate_id_selection_is_order_independent_and_rejects_ambiguity():
    from marker6dof.commands.replay.replay_sensor_bag import select_detections
    real = np.array([[400, 300], [470, 300], [470, 370], [400, 370]], dtype=float)
    small = np.array([[300, 300], [310, 300], [310, 310], [300, 310]], dtype=float)
    for corners in ([real, small], [small, real]):
        selected, ids, dup = select_detections(corners, np.array([2, 2]), {2}, camera(), 12)
        np.testing.assert_array_equal(selected[2], real)
        assert ids == [2]
        assert dup["2"]["policy"] == "unique_viable_contour"
    selected, _, dup = select_detections([real, small * 2], np.array([2, 2]), {2}, camera(), 12)
    assert dup["2"]["policy"] == "dominant_area_contour"
    selected, _, dup = select_detections([real, real + 100], np.array([2, 2]), {2}, camera(), 12)
    assert selected == {}
    assert dup["2"]["policy"] == "ambiguous_duplicate_rejected"


def test_excluded_background_contour_preserves_target():
    from marker6dof.commands.replay.replay_sensor_bag import select_detections
    target = np.array([[880, 580], [910, 580], [910, 610], [880, 610]], dtype=float)
    background = np.array([[145, 850], [175, 850], [175, 880], [145, 880]], dtype=float)
    selected, ids, duplicates = select_detections(
        [background, target], np.array([2, 2]), {2}, camera(), 12,
        exclude_center_rect=(0, 700, 300, 1000),
    )
    np.testing.assert_array_equal(selected[2], target)
    assert ids == [2]
    assert duplicates == {}


def test_rosbag_decoding_preserves_header_and_storage_times(tmp_path):
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    from marker6dof.commands.replay.replay_sensor_bag import read_images
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    topic = "/camera/image/compressed"
    message_type = "sensor_msgs/msg/CompressedImage"
    header = types["std_msgs/msg/Header"](types["builtin_interfaces/msg/Time"](123, 456), "camera")
    image = np.full((10, 12, 3), 127, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    message = types[message_type](header, "png", encoded.ravel())
    bag = tmp_path / "bag"
    with Writer(bag, version=9) as writer:
        connection = writer.add_connection(topic, message_type, typestore=store)
        writer.write(connection, 124_000_000_789, store.serialize_cdr(message, message_type))
    frames = list(read_images(bag, topic))
    assert len(frames) == 1
    bag_ns, header_ns, decoded = frames[0]
    assert bag_ns == 124_000_000_789
    assert header_ns == 123_000_000_456
    np.testing.assert_array_equal(decoded, image)
    with pytest.raises(ValueError, match="no messages"):
        list(read_images(bag, "/missing"))

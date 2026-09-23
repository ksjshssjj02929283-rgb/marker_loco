from __future__ import annotations

import cv2
import numpy as np
import pytest

from marker6dof.realsense import detector_parameters, detect_markers, marker_object_points
from marker6dof.tracking import marker_object_points as tracking_marker_object_points


def test_marker_geometry_is_shared_with_tracking():
    expected = marker_object_points(0.048)
    np.testing.assert_array_equal(expected, tracking_marker_object_points(0.048))
    np.testing.assert_allclose(expected[0], [-0.024, 0.024, 0.0])
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

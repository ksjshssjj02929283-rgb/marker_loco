"""Regression checks for physical black-border corners on Kalibr grids."""

import cv2
import numpy as np
import pytest

from marker6dof.commands.calibration.calibrate_aprilgrid import _detector, _target_points


@pytest.mark.parametrize("size", [40, 60, 100])
def test_wide_border_detection_reaches_physical_square_edge(size):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((size + 80, size + 80), 230, np.uint8)
    image[40:40 + size, 40:40 + size] = cv2.aruco.generateImageMarker(
        dictionary, 0, size, borderBits=2,
    )
    image = cv2.GaussianBlur(image, (3, 3), .6)
    corners, ids, _ = _detector().detectMarkers(image)
    assert ids is not None and ids.ravel().tolist() == [0]
    lower, upper = 39.5, 39.5 + size
    expected = np.array([[lower, lower], [upper, lower], [upper, upper], [lower, upper]])
    # The old one-cell border/small window left corners 4–8 px inside a
    # synthetic black square, producing a wrong side-to-pitch ratio.
    np.testing.assert_allclose(corners[0].reshape(4, 2), expected, atol=.3)


def test_board_configuration_is_metric_black_edge_to_edge():
    board = _target_points(6, 6, .10, .3)
    assert len(board) == 36
    np.testing.assert_allclose(board[1] - board[0], [[.13, 0, 0]] * 4)
    np.testing.assert_allclose(board[6] - board[0], [[0, .13, 0]] * 4)
    np.testing.assert_allclose(np.linalg.norm(board[0] - np.roll(board[0], 1, axis=0), axis=1), .1)

from collections import deque
from types import SimpleNamespace

import numpy as np

from marker6dof.commands.tracking.live_dual_t265 import (
    _cube_center_pose, _display_trace, _nearest_pose_frame,
)


def _frame(timestamp, domain=1):
    return SimpleNamespace(
        get_timestamp=lambda: timestamp,
        get_frame_timestamp_domain=lambda: domain,
    )


def test_nearest_pose_uses_camera_timestamp_domain_and_closest_sample():
    camera = _frame(102.0)
    recent = deque([_frame(100.0), _frame(101.0, domain=2), _frame(105.0)])

    assert _nearest_pose_frame(recent, camera) is recent[0]
    assert _nearest_pose_frame([_frame(102.0, domain=2)], camera) is None


def test_display_trace_keeps_only_canvas_fields():
    trace = [{"smooth": [1, 2, 3], "segment": 2, "raw": [4, 5, 6], "q_raw": [0, 0, 0, 1]}]

    assert _display_trace(trace) == [{"smooth": [1, 2, 3], "segment": 2}]


def test_cube_center_offset_rotates_with_marker_and_keeps_reference_pose():
    reference = np.eye(4)
    reference[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    reference[:3, 3] = [1, 2, 3]

    center = _cube_center_pose(reference, np.array([0.01, 0.02, -0.03]))

    np.testing.assert_allclose(center[:3, 3], [0.98, 2.01, 2.97])
    np.testing.assert_allclose(reference[:3, 3], [1, 2, 3])
    assert _cube_center_pose(None, np.zeros(3)) is None

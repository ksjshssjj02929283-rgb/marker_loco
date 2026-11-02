import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import marker6dof.realsense_depth as adapter
from marker6dof.realsense_depth import (
    RealSenseDepthCloud,
    RealSensePoseRefiner,
    colorize_depth,
    depth_frame_timing,
    depth_points_in_color,
)


def test_colorize_depth_reports_metric_range_and_masks_invalid_samples():
    depth = np.array([[0, 1000], [2000, 5000]], dtype=np.uint16)
    preview, report = colorize_depth(depth, 0.001, min_depth_m=0.2, max_depth_m=3.0)
    assert preview.shape == (2, 2, 3)
    assert preview.dtype == np.uint8
    assert tuple(preview[0, 0]) == (0, 0, 0)
    assert report["depth_view_status"] == "READY"
    assert report["depth_min_m"] == 1.0
    assert report["depth_max_m"] == 5.0
    assert report["depth_median_m"] == 2.0
    assert report["depth_valid_ratio"] == .75
    assert report["depth_view_min_m"] == .2
    assert report["depth_view_max_m"] == 3.0
    json.dumps(report, allow_nan=False)


def test_colorize_depth_handles_empty_valid_frame():
    preview, report = colorize_depth(np.zeros((3, 4), dtype=np.uint16), .001)
    assert preview.shape == (3, 4, 3)
    assert not preview.any()
    assert report["depth_view_status"] == "NO_VALID_DEPTH"
    assert report["depth_valid_ratio"] == 0.0


def test_native_depth_vertices_use_rotation_and_axial_translation():
    vertices = np.array([[0.1, 0.0, 1.0], [-0.1, 0.1, 1.3]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("y", 8, degrees=True).as_matrix()
    transform[:3, 3] = [0.02, -0.01, 0.03]
    points, pixels = depth_points_in_color(
        vertices, np.array([[0.3, 0.4], [0.6, 0.5]]), transform, 640, 480,
    )
    np.testing.assert_allclose(points, vertices @ transform[:3, :3].T + transform[:3, 3])
    np.testing.assert_allclose(pixels, [[192, 192], [384, 240]])
    assert not np.allclose(points[:, 2], vertices[:, 2])


def test_cloud_filters_invalid_outside_roi_and_occluded_points():
    vertices = np.array([
        [0, 0, 1.2], [0, 0, 0.9], [0, 0, 0],
        [np.nan, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
    ])
    uv = np.array([[.5, .5], [.5, .5], [.5, .5], [.5, .5],
                   [1.1, .5], [.1, .1], [np.nan, .5]])
    points, pixels = depth_points_in_color(
        vertices, uv, np.eye(4), 100, 100,
        [np.array([[40, 40], [60, 40], [60, 60], [40, 60]])],
    )
    np.testing.assert_allclose(points, [[0, 0, .9]])
    np.testing.assert_allclose(pixels, [[50, 50]])


def test_cloud_empty_input_and_empty_regions():
    points, pixels = depth_points_in_color(np.empty((0, 3)), np.empty((0, 2)), np.eye(4), 10, 10)
    assert points.shape == (0, 3) and pixels.shape == (0, 2)
    points, _ = depth_points_in_color(np.array([[0, 0, 1]]), np.array([[.5, .5]]), np.eye(4), 10, 10, [])
    assert len(points) == 0


def _frame(timestamp, domain="hardware_clock"):
    return SimpleNamespace(get_timestamp=lambda: timestamp, get_frame_timestamp_domain=lambda: domain)


def test_depth_timing_rejects_missing_unsynchronized_and_mixed_clocks():
    color = _frame(100)
    assert depth_frame_timing(color, None) == ("NO_DEPTH_FRAME", None)
    assert depth_frame_timing(color, _frame(101)) == ("READY", 1.0)
    assert depth_frame_timing(color, _frame(130)) == ("DEPTH_TIME_MISMATCH", 30.0)
    assert depth_frame_timing(color, _frame(100, "system_time")) == ("DEPTH_TIMESTAMP_DOMAIN_MISMATCH", None)
    assert depth_frame_timing(color, _frame(float("nan"))) == ("DEPTH_INVALID_TIMESTAMP", None)
    with pytest.raises(ValueError):
        depth_frame_timing(color, _frame(100), float("nan"))


def test_sdk_cloud_maps_before_calculate_and_owns_transformed_buffers(monkeypatch):
    vertices = np.array([(0.1, 0.0, 1.0), (-0.1, 0.1, 1.3)],
                        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    uv = np.array([(.3, .4), (.6, .5)], dtype=[("u", "f4"), ("v", "f4")])
    original_vertices = vertices.view(np.float32).reshape(-1, 3).copy()
    rotation = Rotation.from_euler("xyz", [5, 8, -3], degrees=True).as_matrix()
    translation = np.array([.02, -.01, .03])
    extrinsics = SimpleNamespace(rotation=rotation.ravel(order="F").tolist(),
                                 translation=translation.tolist())
    color_profile = SimpleNamespace(get_intrinsics=lambda: SimpleNamespace(width=640, height=480))

    def get_extrinsics(target):
        assert target is color_profile
        return extrinsics

    depth_profile = SimpleNamespace(get_extrinsics_to=get_extrinsics)
    color = SimpleNamespace(profile=SimpleNamespace(as_video_stream_profile=lambda: color_profile))
    depth = SimpleNamespace(profile=SimpleNamespace(as_video_stream_profile=lambda: depth_profile))
    events = []

    def map_to(frame):
        assert frame is color
        events.append("map_to")

    def calculate(frame):
        assert frame is depth
        assert events == ["map_to"]
        events.append("calculate")
        return SimpleNamespace(get_vertices=lambda: vertices, get_texture_coordinates=lambda: uv)

    monkeypatch.setattr(adapter, "_realsense", lambda: SimpleNamespace(
        pointcloud=lambda: SimpleNamespace(map_to=map_to, calculate=calculate)))
    cloud = RealSenseDepthCloud()
    points, pixels = cloud.sample(depth, color)
    expected_points = original_vertices @ rotation.T + translation
    expected_pixels = uv.view(np.float32).reshape(-1, 2).astype(float) * [640, 480]
    np.testing.assert_allclose(points, expected_points)
    np.testing.assert_allclose(pixels, expected_pixels)
    assert events == ["map_to", "calculate"]
    assert not np.shares_memory(points, vertices)
    assert not np.shares_memory(pixels, uv)
    # Simulate SDK storage being reused immediately after sample() returns.
    vertices.view(np.float32)[:] = np.nan
    uv.view(np.float32)[:] = np.nan
    np.testing.assert_allclose(points, expected_points)
    np.testing.assert_allclose(pixels, expected_pixels)


class _UnexpectedSDKAccess:
    def __getattr__(self, name):
        pytest.fail(f"Unexpected SDK access: {name}")

    def __bool__(self):
        pytest.fail("Unexpected SDK frame truth check")


def _no_sdk():
    pytest.fail("SDK should not be loaded on an early-return path")


def test_zero_weight_does_not_touch_frames_cloud_or_sdk(monkeypatch):
    monkeypatch.setattr(adapter, "_realsense", _no_sdk)
    unavailable = _UnexpectedSDKAccess()
    refiner = RealSensePoseRefiner(unavailable, unavailable, unavailable, {}, {}, depth_weight=0)
    rgb = np.eye(4)
    result, report = refiner(rgb, [])
    assert result is rgb
    assert report["status"] == "DEPTH_DISABLED"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("weight", [0, .2])
@pytest.mark.parametrize("name,value", [
    ("max_translation_delta_m", 0),
    ("max_translation_delta_m", -1),
    ("max_translation_delta_m", np.nan),
    ("max_rotation_delta_deg", 0),
    ("max_rotation_delta_deg", 181),
    ("max_rotation_delta_deg", np.inf),
    ("max_reprojection_error_px", 0),
    ("max_reprojection_error_px", np.nan),
    ("max_reprojection_error_px", None),
    ("max_time_delta_ms", 0),
    ("max_time_delta_ms", -1),
    ("max_time_delta_ms", np.inf),
])
def test_refiner_validates_configuration_before_disabled_or_missing_depth(weight, name, value):
    with pytest.raises(ValueError, match=name):
        RealSensePoseRefiner(None, None, None, {}, {}, depth_weight=weight, **{name: value})


@pytest.mark.parametrize("weight", [np.nan, np.inf, -.1, 1.1, None])
def test_refiner_rejects_invalid_depth_weight_without_camera(weight):
    with pytest.raises(ValueError, match="depth_weight"):
        RealSensePoseRefiner(None, None, None, {}, {}, depth_weight=weight)


def test_refiner_accepts_finite_parameter_boundaries():
    refiner = RealSensePoseRefiner(None, None, None, {}, {},
                                  depth_weight=np.float32(1), max_rotation_delta_deg=180)
    assert refiner.options["depth_weight"] == 1.0
    assert refiner.options["max_rotation_delta_deg"] == 180.0


@pytest.mark.parametrize("depth,status", [
    (None, "NO_DEPTH_FRAME"),
    (_frame(130), "DEPTH_TIME_MISMATCH"),
    (_frame(100, "system_time"), "DEPTH_TIMESTAMP_DOMAIN_MISMATCH"),
    (_frame(np.nan), "DEPTH_INVALID_TIMESTAMP"),
])
def test_refiner_missing_or_incompatible_depth_never_builds_cloud(monkeypatch, depth, status):
    monkeypatch.setattr(adapter, "_realsense", _no_sdk)
    refiner = RealSensePoseRefiner(_UnexpectedSDKAccess(), _frame(100), depth, {}, {})
    rgb = np.eye(4)
    result, report = refiner(rgb, [])
    assert result is rgb
    assert report["status"] == status
    assert report["valid_depth_points"] == 0
    json.dumps(report, allow_nan=False)


def _refiner_scene(monkeypatch):
    """Small RGB-D scene routed through the real adapter and numeric core."""
    intrinsics = SimpleNamespace(width=1280, height=720, fx=650., fy=645., ppx=640., ppy=360.)

    def project_many(points):
        points = np.asarray(points, dtype=np.float64)
        return points[:, :2] / points[:, 2:3] * [650., 645.] + [640., 360.]

    def sdk_project(intr, point):
        assert intr is intrinsics
        point = np.asarray(point, dtype=np.float32)
        pixels = point[:2] / point[2] * np.array([intr.fx, intr.fy], dtype=np.float32)
        return (pixels + np.array([intr.ppx, intr.ppy], dtype=np.float32)).tolist()

    sdk = SimpleNamespace(rs2_project_point_to_pixel=sdk_project)
    monkeypatch.setattr(adapter, "_realsense", lambda: sdk)
    color, depth = _frame(100), _frame(102)
    profile = SimpleNamespace(get_intrinsics=lambda: intrinsics)
    color.profile = SimpleNamespace(as_video_stream_profile=lambda: profile)
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([.12, -.18, .02]).as_matrix()
    truth[:3, 3] = [.01, -.01, .65]
    model = np.array([[-.02375, .02375, 0], [.02375, .02375, 0],
                      [.02375, -.02375, 0], [-.02375, -.02375, 0]])
    corners = project_many(model @ truth[:3, :3].T + truth[:3, 3])
    xx, yy = np.meshgrid(np.linspace(-.0155, .0155, 19), np.linspace(-.0155, .0155, 19))
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    points = points @ truth[:3, :3].T + truth[:3, 3]
    pixels = project_many(points)
    calls = []

    def sample(depth_frame, color_frame, regions):
        assert depth_frame is depth and color_frame is color
        region_list = list(regions)
        assert len(region_list) == 1
        np.testing.assert_array_equal(region_list[0], corners)
        calls.append("sample")
        return points, pixels

    cloud = SimpleNamespace(sample=sample)
    # The unused marker lacks an image observation and must not be accessed.
    refiner = RealSensePoseRefiner(cloud, color, depth, {2: model, 7: model}, {2: corners})
    rgb = truth.copy()
    rgb[2, 3] += .006
    return SimpleNamespace(refiner=refiner, truth=truth, rgb=rgb, cloud=cloud, color=color,
                           depth=depth, profile=profile, sdk=sdk, calls=calls)


def test_refiner_float32_sdk_projection_improves_pose_and_preserves_diagnostics(monkeypatch):
    scene = _refiner_scene(monkeypatch)
    original = scene.rgb.copy()
    result, report = scene.refiner(scene.rgb, [2])
    assert report["status"] == "RGB_DEPTH_REFINED"
    assert report["timestamp_delta_ms"] == 2.0
    assert report["valid_depth_points"] >= 20
    assert abs(result[2, 3] - scene.truth[2, 3]) < .002
    assert report["fused_reprojection_error_px"] < report["rgb_reprojection_error_px"]
    assert len(report["planes"]) == 1 and report["planes"][0]["tag_id"] == 2
    assert scene.calls == ["sample"]
    np.testing.assert_array_equal(scene.rgb, original)
    json.dumps(report, allow_nan=False)


def _failure(exception):
    def raise_error(*args, **kwargs):
        raise exception
    return raise_error


@pytest.mark.parametrize("stage", ["domain", "timestamp", "cloud", "intrinsics", "sdk"])
def test_refiner_recovers_from_sdk_runtime_errors_in_every_adapter_stage(monkeypatch, stage):
    scene = _refiner_scene(monkeypatch)
    fail = _failure(RuntimeError(f"{stage} failed"))
    if stage == "domain":
        scene.depth.get_frame_timestamp_domain = fail
    elif stage == "timestamp":
        scene.depth.get_timestamp = fail
    elif stage == "cloud":
        scene.cloud.sample = fail
    elif stage == "intrinsics":
        scene.profile.get_intrinsics = fail
    else:
        monkeypatch.setattr(adapter, "_realsense", fail)
    result, report = scene.refiner(scene.rgb, [2])
    assert result is scene.rgb
    assert report["status"] == "DEPTH_SDK_ERROR"
    assert report["error"] == f"{stage} failed"
    assert report["valid_depth_points"] == 0
    assert report["timestamp_delta_ms"] == (None if stage in ("domain", "timestamp") else 2.0)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("after_calls,expected_status", [
    (0, "RGB_INVALID_PROJECTION"),
    (4, "DEPTH_REJECTED_OPTIMIZATION"),
])
def test_refiner_sdk_projection_runtime_errors_preserve_rgb(monkeypatch, after_calls, expected_status):
    scene = _refiner_scene(monkeypatch)
    real_project = scene.sdk.rs2_project_point_to_pixel
    calls = 0

    def unstable_project(intrinsics, point):
        nonlocal calls
        if calls >= after_calls:
            raise RuntimeError("projection failed")
        calls += 1
        return real_project(intrinsics, point)

    scene.sdk.rs2_project_point_to_pixel = unstable_project
    result, report = scene.refiner(scene.rgb, [2])
    np.testing.assert_array_equal(result, scene.rgb)
    assert report["status"] == expected_status
    assert report["timestamp_delta_ms"] == 2.0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("stage", ["cloud", "intrinsics", "projection", "image_corners", "object_corners"])
def test_refiner_does_not_hide_value_errors_or_malformed_inputs(monkeypatch, stage):
    scene = _refiner_scene(monkeypatch)
    fail = _failure(ValueError("malformed input"))
    if stage == "cloud":
        scene.cloud.sample = fail
    elif stage == "intrinsics":
        scene.profile.get_intrinsics = fail
    elif stage == "projection":
        scene.sdk.rs2_project_point_to_pixel = fail
    elif stage == "image_corners":
        scene.refiner.image_corners[2] = np.zeros((3, 2))
    else:
        scene.refiner.object_points[2] = np.zeros((4, 3))
    with pytest.raises(ValueError):
        scene.refiner(scene.rgb, [2])

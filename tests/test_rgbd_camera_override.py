from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from marker6dof.camera import OpenCVCamera
import marker6dof.realsense_depth as adapter
import marker6dof.rgbd as rgbd


def _camera():
    return OpenCVCamera(
        640, 480, np.array([[200., 0, 320.], [0, 210., 240.], [0, 0, 1.]]),
        np.array([.08, -.02, .001, -.002, .005]), "radtan",
    )


def _native_cloud(monkeypatch, vertices, transform, *, width=640, height=480):
    """Only native SDK geometry is available; SDK texture access must fail."""
    vertices = np.asarray(vertices, dtype=np.float32)
    extrinsics = SimpleNamespace(
        rotation=transform[:3, :3].ravel(order="F").tolist(),
        translation=transform[:3, 3].tolist(),
    )
    color_profile = SimpleNamespace(
        get_intrinsics=lambda: SimpleNamespace(width=width, height=height),
    )

    def get_extrinsics(target):
        assert target is color_profile
        return extrinsics

    depth_profile = SimpleNamespace(get_extrinsics_to=get_extrinsics)
    color = SimpleNamespace(profile=SimpleNamespace(as_video_stream_profile=lambda: color_profile))
    depth = SimpleNamespace(profile=SimpleNamespace(as_video_stream_profile=lambda: depth_profile))

    def unavailable(*args):
        pytest.fail("An RGB override must not use SDK texture coordinates or map_to")

    def calculate(frame):
        assert frame is depth
        return SimpleNamespace(get_vertices=lambda: vertices, get_texture_coordinates=unavailable)

    monkeypatch.setattr(adapter, "_realsense", lambda: SimpleNamespace(
        pointcloud=lambda: SimpleNamespace(map_to=unavailable, calculate=calculate),
    ))
    return adapter.RealSenseDepthCloud(), depth, color, vertices


def test_camera_override_reprojects_actual_color_geometry_and_keeps_nearest_surface(monkeypatch):
    camera = _camera()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [5, 8, -3], degrees=True).as_matrix()
    transform[:3, 3] = [.02, -.01, .03]
    color_points = np.array([
        [.1, .02, 1.2], [.075, .015, .9],  # Same color ray; nearer point wins.
        [-.1, -.02, 1.1], [2., 0, 1.], [1., 0, -.01],
    ])
    native = (color_points - transform[:3, 3]) @ transform[:3, :3]
    assert native[-1, 2] > 0  # Native-front points may still be behind RGB.
    native = np.vstack([native, [0, 0, 0], [np.nan, 0, 1], [0, 0, -1]])
    cloud, depth, color, storage = _native_cloud(monkeypatch, native, transform)
    projected = camera.project_points(color_points[:1])[0]
    region = projected + np.array([[-3, -3], [3, -3], [3, 3], [-3, 3]])
    projected_batches = []

    def project(points):
        assert np.isfinite(points).all() and np.all(points[:, 2] > 0)
        projected_batches.append(points.copy())
        return camera.project_points(points)

    override = SimpleNamespace(width=camera.width, height=camera.height, project_points=project)
    points, pixels = cloud.sample(depth, color, [region], camera=override)
    np.testing.assert_allclose(points, color_points[1:2], atol=1e-7)
    np.testing.assert_allclose(pixels, projected[None, :], atol=1e-5)
    assert sum(len(batch) for batch in projected_batches) == 4
    assert not np.shares_memory(points, storage)
    storage[:] = np.nan
    np.testing.assert_allclose(points, color_points[1:2], atol=1e-7)


def test_camera_override_empty_native_cloud_skips_projection(monkeypatch):
    cloud, depth, color, _ = _native_cloud(monkeypatch, [[np.nan, 0, 1], [0, 0, 0]], np.eye(4))

    def project(points):
        pytest.fail("Empty or invalid clouds must not be sent to OpenCV")

    camera = SimpleNamespace(width=640, height=480, project_points=project)
    points, pixels = cloud.sample(depth, color, camera=camera)
    assert points.shape == (0, 3) and pixels.shape == (0, 2)


def test_camera_override_checks_recorded_color_resolution(monkeypatch):
    cloud, depth, color, _ = _native_cloud(monkeypatch, [[0, 0, 1]], np.eye(4), width=1280)
    with pytest.raises(ValueError, match="resolution"):
        cloud.sample(depth, color, camera=_camera())


def test_refiner_uses_same_camera_for_cloud_sampling_and_optimization(monkeypatch):
    camera = _camera()
    color = SimpleNamespace(get_timestamp=lambda: 100., get_frame_timestamp_domain=lambda: "hardware")
    depth = SimpleNamespace(get_timestamp=lambda: 102., get_frame_timestamp_domain=lambda: "hardware")
    model = np.array([[-.02, .02, 0], [.02, .02, 0], [.02, -.02, 0], [-.02, -.02, 0]])
    pose = np.eye(4)
    pose[:3, 3] = [.01, -.01, .6]
    corners = camera.project_points(model + pose[:3, 3])
    points = np.array([[0, 0, .6], [.01, 0, .6]])
    pixels = camera.project_points(points)
    calls = []

    def sample(depth_frame, color_frame, regions, *, camera):
        assert depth_frame is depth and color_frame is color
        assert camera is expected_camera
        np.testing.assert_array_equal(list(regions), [corners])
        calls.append("sample")
        return points, pixels

    def refine(rgb_pose, objects, images, depth_points, depth_pixels, project, **options):
        assert project.__self__ is expected_camera
        np.testing.assert_array_equal(project(points), pixels)
        np.testing.assert_array_equal(depth_points, points)
        np.testing.assert_array_equal(depth_pixels, pixels)
        assert set(objects) == set(images) == {2}
        assert options["depth_weight"] == .2
        calls.append("refine")
        return rgb_pose.copy(), {"status": "RGB_DEPTH_REFINED", "valid_depth_points": len(points)}

    def no_sdk():
        pytest.fail("The RGB override must also replace the refiner's SDK projection")

    expected_camera = camera
    monkeypatch.setattr(adapter, "_realsense", no_sdk)
    monkeypatch.setattr(rgbd, "refine_pose_rgbd", refine)
    refiner = adapter.RealSensePoseRefiner(
        SimpleNamespace(sample=sample), color, depth, {2: model}, {2: corners}, camera=camera,
    )
    result, report = refiner(pose, [2])
    np.testing.assert_array_equal(result, pose)
    assert report["status"] == "RGB_DEPTH_REFINED"
    assert report["timestamp_delta_ms"] == 2.
    assert calls == ["sample", "refine"]

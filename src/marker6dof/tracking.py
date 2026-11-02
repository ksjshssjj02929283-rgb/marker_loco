"""Rigid multi-tag PnP and conservative, segmented relative trajectory state."""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .realsense import marker_object_points, normalized_image_points, project_realsense


def load_rig(data, marker_size_m):
    transforms = {int(k): np.asarray(v, dtype=float) for k, v in data['T_reference_tag'].items()}
    for tag_id, transform in transforms.items():
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f'Invalid rig transform for ID {tag_id}')
        rotation = transform[:3, :3]
        if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
            raise ValueError(f'Non-rigid transform for ID {tag_id}')
    if int(data['reference_id']) not in transforms:
        raise ValueError('Reference ID is missing from rig')
    # Scaling only marker corners would invalidate the inter-tag translations.
    calibrated_size = data.get('marker_size_m')
    if calibrated_size is None:
        raise ValueError('Rig calibration has no marker_size_m; rebuild or explicitly migrate it')
    if not math.isclose(float(calibrated_size), marker_size_m, rel_tol=1e-4):
        raise ValueError(
            f'Marker size {marker_size_m * 1000:g} mm differs from rig calibration '
            f'{float(calibrated_size) * 1000:g} mm; recalibrate or rescale the rig first'
        )
    return transforms


class RigTracker:
    """Fit all accepted corners directly, rather than average independently solved poses.

    IPPE's two planar solutions seed the fit; multi-face geometry scores them.
    Per-tag reprojection gates reject unsupported faces. Single-face estimates
    remain explicitly weaker and are never described as multi-tag agreement.
    Optional corner weighting/loss modes are exploratory; the default remains
    the existing unweighted least-squares fit.
    """

    def __init__(self, rig, marker_size_m, max_error_px=2.5, min_edge_px=24,
                 corner_weighting=None, corner_loss=None, corner_loss_scale_px=0.5):
        if corner_weighting not in (None, 'edge_size'):
            raise ValueError("corner_weighting must be None or 'edge_size'")
        if corner_loss not in (None, 'huber'):
            raise ValueError("corner_loss must be None or 'huber'")
        if corner_weighting is not None and corner_loss is not None:
            raise ValueError('choose either corner_weighting or corner_loss')
        if not math.isfinite(corner_loss_scale_px) or corner_loss_scale_px <= 0:
            raise ValueError('corner_loss_scale_px must be finite and positive')
        self.rig = rig
        self.local_points = marker_object_points(marker_size_m).astype(np.float64)
        self.points = {tag: (t[:3, :3] @ self.local_points.T).T + t[:3, 3] for tag, t in rig.items()}
        self.max_error = max_error_px
        self.min_edge = min_edge_px
        self.corner_weighting = corner_weighting
        self.corner_loss = corner_loss
        self.corner_loss_scale_px = corner_loss_scale_px
        self.previous = None
        self.previous_time = None

    def reset(self):
        self.previous = self.previous_time = None

    def solve(self, detections, intrinsics, timestamp, pose_refiner=None):
        """Estimate RGB pose, optionally refine, then gate the final pose in time.

        ``pose_refiner(pose, used_ids)`` returns a candidate and a JSON-safe
        diagnostic report. RGB corner support is independently rechecked here.
        """
        visible = sorted(detections)
        result = dict(status='LOST', reason='没有检测到已标定 Tag', visible_ids=visible,
                      used_ids=[], reprojection_error_px=None, tag_errors={}, rejected_ids=[], pose=None)
        valid, seeds = {}, []
        for tag, corners in detections.items():
            if tag not in self.rig:
                continue
            corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            edge = np.min(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1))
            inside = (corners[:, 0].min() > 3 and corners[:, 1].min() > 3 and
                      corners[:, 0].max() < intrinsics.width - 4 and corners[:, 1].max() < intrinsics.height - 4)
            if not np.isfinite(corners).all() or edge < self.min_edge or not inside:
                continue
            normalized = normalized_image_points(intrinsics, corners).astype(np.float64)
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                self.local_points, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            valid[tag] = (corners, normalized, float(edge))
            for rvec, tvec in zip(rvecs, tvecs):
                transform = np.eye(4)
                transform[:3, :3] = cv2.Rodrigues(rvec)[0]
                transform[:3, 3] = tvec.ravel()
                seeds.append(transform @ np.linalg.inv(self.rig[tag]))
        if not valid:
            result['status'] = 'WARN' if any(t in self.rig for t in visible) else 'LOST'
            if result['status'] == 'WARN':
                result['reason'] = 'Tag 太小、靠近画面边缘或位姿求解失败'
            self.reset()
            return result

        def errors(pose, tags):
            report = {}
            for tag in tags:
                camera_points = (pose[:3, :3] @ self.points[tag].T).T + pose[:3, 3]
                if not np.isfinite(camera_points).all() or np.min(camera_points[:, 2]) <= .01:
                    report[tag] = float('inf')
                else:
                    pixels = project_realsense(intrinsics, camera_points)
                    report[tag] = float(np.sqrt(np.mean(np.sum((pixels - valid[tag][0]) ** 2, axis=1))))
            return report

        def refine(seed, tags):
            points = np.concatenate([self.points[tag] for tag in tags])
            pixels = np.concatenate([valid[tag][1] for tag in tags])
            rvec = cv2.Rodrigues(seed[:3, :3])[0]
            tvec = seed[:3, 3].copy().reshape(3, 1)
            if self.corner_loss == 'huber':
                observed_pixels = np.concatenate([valid[tag][0] for tag in tags])

                def residual(parameters):
                    rotation = cv2.Rodrigues(parameters[:3])[0]
                    camera_points = (rotation @ points.T).T + parameters[3:]
                    # Keep the projection finite while the optimizer explores.
                    camera_points[:, 2] = np.maximum(camera_points[:, 2], .01)
                    return (project_realsense(intrinsics, camera_points) - observed_pixels).ravel()

                initial = np.concatenate([rvec.ravel(), tvec.ravel()])
                fitted = least_squares(
                    residual, initial, method='trf', loss='huber',
                    f_scale=self.corner_loss_scale_px, max_nfev=100,
                )
                pose = np.eye(4)
                pose[:3, :3] = cv2.Rodrigues(fitted.x[:3])[0]
                pose[:3, 3] = fitted.x[3:]
                return pose
            if self.corner_weighting == 'edge_size':
                # Larger observed tags usually provide more stable corners.
                # Keep this experimental weighting bounded so one large face
                # cannot overwhelm the other rigid-body constraints.
                edges = np.asarray([valid[tag][2] for tag in tags], dtype=float)
                relative = edges / np.median(edges)
                per_tag = np.clip(relative, .5, 2.0) ** 2
                weights = np.repeat(per_tag, 4)
                weights /= np.mean(weights)
                sqrt_weights = np.sqrt(weights)[:, None]

                def residual(parameters):
                    rotation = cv2.Rodrigues(parameters[:3])[0]
                    camera_points = (rotation @ points.T).T + parameters[3:]
                    z = np.maximum(camera_points[:, 2:3], 1e-6)
                    projected = camera_points[:, :2] / z
                    return ((projected - pixels) * sqrt_weights).ravel()

                initial = np.concatenate([rvec.ravel(), tvec.ravel()])
                fitted = least_squares(residual, initial, method='lm', max_nfev=80)
                pose = np.eye(4)
                pose[:3, :3] = cv2.Rodrigues(fitted.x[:3])[0]
                pose[:3, 3] = fitted.x[3:]
                return pose
            ok, rvec, tvec = cv2.solvePnP(points, pixels, np.eye(3), None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return seed
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(rvec)[0]
            pose[:3, 3] = tvec.ravel()
            return pose

        # Whole-tag consensus: a bad face must not drag the other faces away.
        if self.previous is not None and 0 < timestamp - self.previous_time < .2:
            seeds.append(self.previous)
        hypotheses = []
        for seed in seeds:
            initial_errors = errors(seed, valid)
            tags = [tag for tag, err in initial_errors.items() if err <= self.max_error * 2]
            if not tags:
                continue
            pose = refine(seed, tags)
            report = errors(pose, valid)
            used = [tag for tag, err in report.items() if err <= self.max_error]
            if not used:
                continue
            if used != tags:
                pose = refine(pose, used)
                report = errors(pose, valid)
                used = [tag for tag in used if report[tag] <= self.max_error]
            if used:
                rms = float(np.sqrt(np.mean([report[tag] ** 2 for tag in used])))
                hypotheses.append((pose, used, report, rms))
        if not hypotheses:
            result.update(status='WARN', reason='重投影误差过大，已拒绝该帧')
            self.reset()
            return result
        hypotheses.sort(key=lambda h: (-len(h[1]), h[3]))
        best = hypotheses[0]
        # Only use continuity to resolve similarly fitting hypotheses; never
        # select a materially worse fit just to make the trajectory smoother.
        if self.previous is not None and 0 < timestamp - self.previous_time < .2:
            similar = [h for h in hypotheses if len(h[1]) == len(best[1]) and h[3] <= best[3] + .15]
            best = min(similar, key=lambda h: np.linalg.norm(Rotation.from_matrix(self.previous[:3, :3].T @ h[0][:3, :3]).as_rotvec()))
        pose, used, report, rms = best
        if pose_refiner is not None:
            from .projection import validate_rigid_transform

            result['rgb_reprojection_error_px'] = rms
            candidate, depth_report = pose_refiner(pose.copy(), list(used))
            candidate = validate_rigid_transform(candidate, 'refined_pose')
            refined_errors = errors(candidate, valid)
            refined_rms = float(np.sqrt(np.mean([refined_errors[tag] ** 2 for tag in used])))
            # A refiner cannot weaken the original per-face RGB acceptance gate.
            if any(refined_errors[tag] > min(self.max_error, report[tag] + .25) for tag in used):
                depth_report['status'] = 'DEPTH_REJECTED_TRACKER_REPROJECTION'
                depth_report['fused_reprojection_error_px'] = rms
            else:
                pose, report, rms = candidate, refined_errors, refined_rms
            result['depth_report'] = depth_report
        result.update(used_ids=sorted(used), reprojection_error_px=rms,
                      tag_errors={str(tag): err if np.isfinite(err) else None for tag, err in report.items()},
                      rejected_ids=[tag for tag in visible if tag not in used])
        if self.previous is not None:
            dt = timestamp - self.previous_time
            distance = np.linalg.norm(pose[:3, 3] - self.previous[:3, 3])
            angle = np.linalg.norm(Rotation.from_matrix(self.previous[:3, :3].T @ pose[:3, :3]).as_rotvec())
            if 0 < dt < .2 and (distance > .02 + 3 * dt or angle > np.deg2rad(15 + 720 * dt)):
                result.update(status='WARN', reason='短时位姿突跳，轨迹在此断开')
                self.reset()
                return result
        self.previous, self.previous_time = pose.copy(), timestamp
        result.update(pose=pose, status='MULTI' if len(used) > 1 else 'SINGLE',
                      reason='多面角点联合定位' if len(used) > 1 else '单面定位，深度与姿态约束较弱')
        return result


class RelativeTrace:
    """Raw observations and separately smoothed display samples, with explicit gaps."""

    def __init__(self, limit=3000):
        self.limit = limit
        self.reset()

    def reset(self):
        self.initial = self.filtered = self.last_raw = self.last_time = None
        self.filtered_quaternion = None
        self.trace = deque(maxlen=self.limit)
        self.segment = 0
        self.broken = True

    def update(self, result, timestamp, initial_transform=None, display_pose=None):
        """Append a pose sample.

        ``initial_transform`` is an optional fixed transform applied before
        every pose.  It is used by world-frame composition to keep camera and
        TCP traces in one common origin instead of letting each trace choose a
        different first visible object pose.

        ``display_pose`` optionally supplies an already smoothed rigid pose in
        the same frame as the raw observation. Both poses use the raw trace's
        origin, and the display pose bypasses this trace's internal filter so
        related objects can share one smoothed rigid-body pose.
        """
        pose = result.pop('pose', None)
        result.update(position_m=None, smooth_position_m=None, quaternion_xyzw=None,
                      smooth_quaternion_xyzw=None, segment=self.segment, timestamp_s=timestamp)
        if pose is None:
            self.broken = True
            return result
        if display_pose is not None:
            from .projection import validate_rigid_transform

            display_pose = validate_rigid_transform(display_pose, 'display_pose')
        if initial_transform is not None:
            initial = np.asarray(initial_transform, dtype=np.float64)
            if initial.shape != (4, 4) or not np.isfinite(initial).all():
                raise ValueError('initial_transform must be a finite 4x4 matrix')
            self.initial = initial
        elif self.initial is None:
            self.initial = np.linalg.inv(pose)
        relative = self.initial @ pose
        raw = relative[:3, 3]
        raw_quaternion = Rotation.from_matrix(relative[:3, :3]).as_quat()
        dt = timestamp - self.last_time if self.last_time is not None else 0
        new_segment = self.broken or dt <= 0 or dt > .2
        if display_pose is not None:
            displayed = self.initial @ display_pose
            display_quaternion = Rotation.from_matrix(displayed[:3, :3]).as_quat()
            if new_segment:
                self.segment += 1
            elif np.dot(display_quaternion, self.filtered_quaternion) < 0:
                display_quaternion *= -1
            self.filtered = displayed[:3, 3].copy()
            self.filtered_quaternion = display_quaternion
        elif new_segment:
            self.segment += 1
            self.filtered = raw.copy()
            self.filtered_quaternion = raw_quaternion.copy()
        else:
            speed = np.linalg.norm(raw - self.last_raw) / dt
            # Adaptive first-order low-pass for DISPLAY only, in metres/seconds.
            cutoff = 2.0 + 8.0 * speed
            alpha = 1 - math.exp(-2 * math.pi * cutoff * dt)
            self.filtered += alpha * (raw - self.filtered)
            # q and -q are the same rotation. Keep one hemisphere before nlerp
            # so the displayed frame never flips because of quaternion sign.
            if np.dot(raw_quaternion, self.filtered_quaternion) < 0:
                raw_quaternion *= -1
            self.filtered_quaternion += alpha * (raw_quaternion - self.filtered_quaternion)
            self.filtered_quaternion /= np.linalg.norm(self.filtered_quaternion)
        self.broken = False
        self.last_time, self.last_raw = timestamp, raw.copy()
        result.update(position_m=raw.tolist(), smooth_position_m=self.filtered.tolist(),
                      quaternion_xyzw=raw_quaternion.tolist(),
                      smooth_quaternion_xyzw=self.filtered_quaternion.tolist(), segment=self.segment)
        self.trace.append(dict(t=timestamp, raw=raw.tolist(), smooth=self.filtered.tolist(),
                               q_raw=raw_quaternion.tolist(), q_smooth=self.filtered_quaternion.tolist(),
                               segment=self.segment, quality=result['status']))
        return result

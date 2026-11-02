"""Replay both T265 grippers into one static left-camera coordinate frame."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np

from marker6dof.commands.tracking.live_dual_t265 import (
    COLORS, _cube_center_pose, _draw_body_pose, _draw_detections,
    load_body, validate_bodies,
)
from marker6dof.projection import validate_rigid_transform
from marker6dof.realsense import detect_markers, detector_parameters, intrinsics_from_metadata, project_realsense
from marker6dof.smoothing import PoseSmoother
from marker6dof.t265 import fuse_stereo_results
from marker6dof.tracking import RelativeTrace, RigTracker


FIELDS = [
    "frame_index", "time_s", "body", "frame", "status", "visible_ids", "used_ids",
    "camera_sources", "reprojection_error_px", "segment", "x_m", "y_m", "z_m",
    "qx", "qy", "qz", "qw", "smooth_x_m", "smooth_y_m", "smooth_z_m",
    "smooth_qx", "smooth_qy", "smooth_qz", "smooth_qw",
]


def _position_fields(values, prefix):
    return dict(zip((f"{prefix}x_m", f"{prefix}y_m", f"{prefix}z_m"),
                    ("" if values is None else f"{float(value):.9f}" for value in
                     (values if values is not None else (None, None, None)))))


def _quaternion_fields(values, prefix):
    names = (f"{prefix}qx", f"{prefix}qy", f"{prefix}qz", f"{prefix}qw")
    return dict(zip(names, ("" for _ in names) if values is None else
                    (f"{float(value):.9f}" for value in values)))


def _csv_row(frame_index, time_s, name, data):
    row = {
        "frame_index": frame_index, "time_s": f"{time_s:.6f}", "body": name,
        "frame": "static_left_camera", "status": data["status"],
        "visible_ids": ",".join(map(str, data.get("visible_ids", []))),
        "used_ids": ",".join(map(str, data.get("used_ids", []))),
        "camera_sources": ",".join(data.get("camera_sources", [])),
        "reprojection_error_px": ("" if data.get("reprojection_error_px") is None else
                                  f"{data['reprojection_error_px']:.4f}"),
        "segment": data["segment"],
    }
    row.update(_position_fields(data["position_m"], ""))
    row.update(_position_fields(data["smooth_position_m"], "smooth_"))
    row.update(_quaternion_fields(data["quaternion_xyzw"], ""))
    row.update(_quaternion_fields(data["smooth_quaternion_xyzw"], "smooth_"))
    return row


def _draw_recent_trail(image, intrinsics, recent, transform, color):
    previous = None
    count = len(recent)
    for index, (segment, point) in enumerate(recent):
        camera_point = transform[:3, :3] @ point + transform[:3, 3]
        if camera_point[2] <= 0:
            previous = None
            continue
        pixel = tuple(np.rint(project_realsense(intrinsics, camera_point.reshape(1, 3))[0]).astype(int))
        if previous is not None and previous[0] == segment:
            weight = .3 + .7 * (index + 1) / max(count, 1)
            faded = tuple(int(channel * weight) for channel in color)
            cv2.line(image, previous[1], pixel, faded, 2, cv2.LINE_AA)
        previous = segment, pixel


def _draw_frame_hud(image, eye, time_s, results):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 78), (11, 18, 24), -1)
    cv2.addWeighted(overlay, .82, image, .18, 0, image)
    cv2.putText(image, f"T265 {eye.upper()}  |  {time_s:05.1f}s", (18, 27),
                cv2.FONT_HERSHEY_SIMPLEX, .67, (236, 243, 247), 2, cv2.LINE_AA)
    for name, x in (("left", 18), ("right", 430)):
        result = results[name]
        ids = ",".join(map(str, result.get("used_ids", []))) or "-"
        label = f"{name[0].upper()} {result['status']}  ID {ids}"
        cv2.putText(image, label, (x, 61), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, COLORS[name], 2, cv2.LINE_AA)


def _style_3d(ax):
    pane = (0.08, 0.13, 0.17, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane)
        axis.pane.set_edgecolor("#2a3e4b")
        axis._axinfo["grid"]["color"] = (0.5, 0.63, 0.7, 0.2)


def _plot_trajectories(path, trajectories, task_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 6), constrained_layout=True, facecolor="#0b1117")
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122)]
    for ax in axes:
        ax.set_facecolor("#14202a")
    _style_3d(axes[0])
    for name, color in (("left", "#49bf7c"), ("right", "#cb68cc")):
        by_segment = {}
        for segment, _, point in trajectories[name]:
            by_segment.setdefault(segment, []).append(np.asarray(point) * 1000)
        for index, points in enumerate(by_segment.values()):
            xyz = np.asarray(points)
            label = name if index == 0 else None
            axes[0].plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=2, label=label)
            axes[1].plot(xyz[:, 0], xyz[:, 2], color=color, lw=2, label=label)
        if trajectories[name]:
            start = np.asarray(trajectories[name][0][2]) * 1000
            end = np.asarray(trajectories[name][-1][2]) * 1000
            axes[0].scatter(*start, color=color, s=45, marker="o")
            axes[0].scatter(*end, color=color, s=65, marker="D")
            axes[1].scatter(start[0], start[2], color=color, s=45, marker="o")
            axes[1].scatter(end[0], end[2], color=color, s=65, marker="D")
        if not by_segment:
            axes[1].plot([], [], color=color, label=f"{name} (no pose)")
    axes[0].set(xlabel="X right / mm", ylabel="Y down / mm", zlabel="Z forward / mm",
                title="3D cube-center paths")
    axes[1].set(xlabel="X right / mm", ylabel="Z forward / mm", title="Top view · X / Z")
    axes[1].grid(alpha=.18)
    axes[1].axis("equal")
    axes[1].legend(loc="best")
    fig.suptitle(f"{task_name}  ·  Left / Right in one camera frame", fontsize=17, weight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_individual(path, trajectory, name, task_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    color = "#49bf7c" if name == "left" else "#cb68cc"
    fig = plt.figure(figsize=(14, 7), constrained_layout=True, facecolor="#0b1117")
    ax3 = fig.add_subplot(121, projection="3d")
    ax_time = fig.add_subplot(222)
    ax_top = fig.add_subplot(224)
    for ax in (ax3, ax_time, ax_top):
        ax.set_facecolor("#14202a")
    _style_3d(ax3)
    by_segment = {}
    for segment, time_s, point in trajectory:
        by_segment.setdefault(segment, []).append((time_s, np.asarray(point) * 1000))
    for segment_index, samples in enumerate(by_segment.values()):
        times = np.asarray([sample[0] for sample in samples])
        xyz = np.asarray([sample[1] for sample in samples])
        ax3.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=2)
        ax_top.plot(xyz[:, 0], xyz[:, 2], color=color, lw=2)
        for axis, channel in enumerate(("#ff8c8c", "#8de5ac", "#8bbdff")):
            ax_time.plot(times, xyz[:, axis], color=channel, lw=1.6,
                         label="XYZ"[axis] if segment_index == 0 else None)
    if trajectory:
        start = np.asarray(trajectory[0][2]) * 1000
        end = np.asarray(trajectory[-1][2]) * 1000
        ax3.scatter(*start, color=color, s=50, marker="o")
        ax3.scatter(*end, color=color, s=75, marker="D")
        ax_top.scatter(start[0], start[2], color=color, s=50, marker="o", label="start")
        ax_top.scatter(end[0], end[2], color=color, s=75, marker="D", label="end")
    else:
        ax3.text2D(.5, .5, "No valid pose", transform=ax3.transAxes, ha="center")
    ax3.set(xlabel="X / mm", ylabel="Y / mm", zlabel="Z / mm", title="3D trajectory")
    ax_time.set(xlabel="Time / s", ylabel="Position / mm", title="Position over time")
    ax_top.set(xlabel="X right / mm", ylabel="Z forward / mm", title="Top view · X / Z")
    ax_time.grid(alpha=.18)
    ax_top.grid(alpha=.18)
    ax_top.axis("equal")
    if trajectory:
        ax_time.legend(loc="best")
        ax_top.legend(loc="best")
    fig.suptitle(f"{task_name}  ·  {name.upper()} cube center  ·  {len(trajectory)} valid frames",
                 color=color, fontsize=17, weight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_gallery(path, task_name, summary):
    title = escape(task_name)
    camera_label = "T265 左鱼眼单目" if summary.get("camera_eyes") == ["left"] else "T265 同步双鱼眼"
    video_label = "左鱼眼视频与投影轨迹" if summary.get("camera_eyes") == ["left"] else "双鱼眼视频与投影轨迹"
    stats = "".join(
        f'<div class="stat"><strong>{summary[key]}</strong><span>{label}</span></div>'
        for key, label in (("frames", "共同视频帧"), ("left_valid_frames", "左夹爪有效帧"),
                           ("right_valid_frames", "右夹爪有效帧"),
                           ("both_valid_frames", "双夹爪同帧有效"))
    )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · 双夹爪轨迹</title>
<style>body{{margin:0;background:#0b1117;color:#edf5f8;font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:30px}}h1{{font-size:28px;margin:0 0 8px}}
p{{color:#9eb3c1;margin:0 0 24px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}}
.stat{{background:#14202a;border:1px solid #29404e;border-radius:10px;padding:15px}}
.stat strong{{display:block;font-size:24px}}.stat span{{color:#9eb3c1;font-size:12px}}
.card{{background:#14202a;border:1px solid #29404e;border-radius:12px;padding:14px;margin-bottom:18px}}
h2{{font-size:18px;margin:0 0 12px}}video,img{{display:block;width:100%;border-radius:6px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.left{{color:#72e0a2}}.right{{color:#e58ce5}}
@media(max-width:850px){{.stats{{grid-template-columns:repeat(2,1fr)}}.pair{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>{title} · 双夹爪 Marker 中心轨迹</h1>
<p>{camera_label} · 左相机静止坐标系 · 绿色左夹爪 / 紫色右夹爪 · 圆点起点 / 菱形终点</p>
<div class="stats">{stats}</div>
<section class="card"><h2>{video_label}</h2><video controls preload="metadata" src="annotated_dual.mp4"></video></section>
<div class="pair"><section class="card"><h2 class="left">左夹爪独立轨迹</h2><img src="left_trajectory.png"></section>
<section class="card"><h2 class="right">右夹爪独立轨迹</h2><img src="right_trajectory.png"></section></div>
<section class="card"><h2>同一坐标系中的双轨迹</h2><img src="dual_trajectories.png"></section>
<p>中心位置由多面 Marker 几何估计，并非夹爪 TCP。失锁区间断线；CSV 中对应位姿留空。</p>
</main></body></html>'''
    path.write_text(page, encoding="utf-8")


def _recorded_frame_times(session, fps):
    path = session / "frames.csv"
    if not path.exists():
        return {}, fps, "frame_index / recorded_fps"
    with path.open(newline="", encoding="utf-8") as stream:
        stamps_ms = {int(row["frame_index"]): float(row["camera_timestamp_ms"])
                     for row in csv.DictReader(stream)}
    if not stamps_ms:
        return {}, fps, "frame_index / recorded_fps"
    first = stamps_ms[min(stamps_ms)]
    times = {index: (stamp - first) / 1000.0 for index, stamp in stamps_ms.items()}
    last_index = max(times)
    duration_s = times[last_index]
    video_fps = last_index / duration_s if last_index > 0 and duration_s > 0 else fps
    return times, video_fps, "recorded_left_fisheye_timestamp"


def replay(session: Path, output_dir: Path, bodies: dict, *, max_frames: int = 0,
           max_error_px: float = 3.5, min_edge_px: float = 24.0,
           eyes: tuple[str, ...] = ("left", "right")) -> dict:
    if eyes not in (("left",), ("left", "right")):
        raise ValueError("eyes must be left-only or stereo")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    source = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    fps = float(source["stream"]["fps"])
    if fps <= 0:
        raise ValueError("recorded FPS must be positive")
    frame_times, video_fps, time_source = _recorded_frame_times(session, fps)
    intrinsics = {eye: intrinsics_from_metadata(source["fisheye_intrinsics"][eye])
                  for eye in ("left", "right")}
    T_left_right = validate_rigid_transform(source["T_left_right"], "T_left_right")
    T_right_left = np.linalg.inv(T_left_right)
    captures = {eye: cv2.VideoCapture(str(session / f"fisheye_{eye}.mp4"))
                for eye in eyes}
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise RuntimeError("both recorded fisheye videos must be readable")
    source_frame_counts = {eye: int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                           for eye, capture in captures.items()}

    output_dir.mkdir(parents=True)
    width = int(source["stream"]["width"])
    height = int(source["stream"]["height"])
    video = cv2.VideoWriter(str(output_dir / "annotated_dual.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width * len(eyes), height))
    if not video.isOpened():
        for capture in captures.values():
            capture.release()
        raise RuntimeError("could not create annotated_dual.mp4")
    trackers = {name: {eye: RigTracker(body["rig"], body["marker_size_m"],
                                      max_error_px, min_edge_px)
                       for eye in eyes}
                for name, body in bodies.items()}
    smoothers = {name: PoseSmoother() for name in bodies}
    traces = {name: RelativeTrace(limit=200) for name in bodies}
    recent = {name: deque(maxlen=180) for name in bodies}
    trajectories = {name: [] for name in bodies}
    counts = Counter()
    parameters = detector_parameters()
    identity = np.eye(4)
    frame_index = 0
    ended_with_eye_mismatch = False
    try:
        with (output_dir / "dual_center_poses.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            while max_frames <= 0 or frame_index < max_frames:
                frames = {eye: captures[eye].read() for eye in captures}
                if not any(ok for ok, _ in frames.values()):
                    break
                if not all(ok for ok, _ in frames.values()):
                    ended_with_eye_mismatch = True
                    break
                images = {eye: frame for eye, (_, frame) in frames.items()}
                grays = {eye: cv2.cvtColor(images[eye], cv2.COLOR_BGR2GRAY) for eye in images}
                time_s = frame_times.get(frame_index, frame_index / fps)
                results = {}
                for name, body in bodies.items():
                    eye_results = {}
                    for eye in eyes:
                        corners, ids, _ = detect_markers(grays[eye], body["dictionary"], parameters)
                        if ids is not None:
                            selected = [index for index, tag_id in enumerate(ids.reshape(-1))
                                        if int(tag_id) in body["rig"]]
                            corners = [corners[index] for index in selected]
                            ids = ids[selected] if selected else None
                        detections = _draw_detections(images[eye], corners, ids, name)
                        eye_results[eye] = trackers[name][eye].solve(
                            detections, intrinsics[eye], time_s)
                    result = (fuse_stereo_results(eye_results["left"], eye_results["right"], T_left_right)
                              if len(eyes) == 2 else dict(eye_results["left"], camera_sources=["left"]))
                    center_pose = _cube_center_pose(result.get("pose"), body["cube_center_reference_m"])
                    display_pose = smoothers[name].update(center_pose, time_s)
                    data = traces[name].update(dict(result, pose=center_pose), time_s,
                                               initial_transform=identity, display_pose=display_pose)
                    data["camera_sources"] = result.get("camera_sources", [])
                    writer.writerow(_csv_row(frame_index, time_s, name, data))
                    if center_pose is not None:
                        counts[name] += 1
                        if result.get("status") == "STEREO":
                            counts[f"{name}_stereo"] += 1
                        position = np.asarray(data["smooth_position_m"])
                        entry = (data["segment"], position)
                        trajectories[name].append((data["segment"], time_s, position))
                        recent[name].append(entry)
                    for eye in eyes:
                        eye_pose = center_pose if eye == "left" or center_pose is None else T_right_left @ center_pose
                        _draw_body_pose(images[eye], intrinsics[eye], eye_pose, name,
                                        body["marker_size_m"])
                    results[name] = data
                if all(results[name]["position_m"] is not None for name in bodies):
                    counts["both"] += 1
                for name in bodies:
                    for eye in eyes:
                        transform = identity if eye == "left" else T_right_left
                        _draw_recent_trail(images[eye], intrinsics[eye], recent[name],
                                           transform, COLORS[name])
                for eye in eyes:
                    _draw_frame_hud(images[eye], eye, time_s, results)
                video.write(np.hstack([images[eye] for eye in eyes]))
                frame_index += 1
    finally:
        video.release()
        for capture in captures.values():
            capture.release()

    task_name = session.parent.name.replace("_", " ")
    _plot_trajectories(output_dir / "dual_trajectories.png", trajectories, task_name)
    for name in bodies:
        _plot_individual(output_dir / f"{name}_trajectory.png", trajectories[name], name, task_name)
    summary = {
        "session": str(session.resolve()), "frame": "static_left_camera",
        "time_source": time_source, "source_fps": fps,
        "annotated_video_fps": video_fps, "frames": frame_index,
        "source_frame_counts": source_frame_counts,
        "ended_with_eye_mismatch": ended_with_eye_mismatch,
        "source_summary_present": (session / "summary.json").exists(),
        "left_valid_frames": counts["left"], "right_valid_frames": counts["right"],
        "both_valid_frames": counts["both"],
        "left_stereo_frames": counts["left_stereo"],
        "right_stereo_frames": counts["right_stereo"],
        "left_valid_fraction": counts["left"] / frame_index if frame_index else 0,
        "right_valid_fraction": counts["right"] / frame_index if frame_index else 0,
        "both_valid_fraction": counts["both"] / frame_index if frame_index else 0,
        "left_rig": bodies["left"]["dictionary_name"],
        "right_rig": bodies["right"]["dictionary_name"],
        "camera_eyes": list(eyes),
        "note": "Positions are cube centers in the stationary left fisheye frame; blank CSV poses mean tracking lost.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    _write_gallery(output_dir / "index.html", task_name, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--left-rig", type=Path, required=True)
    parser.add_argument("--right-rig", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0, help="0 replays the entire session")
    args = parser.parse_args()
    bodies = {"left": load_body(args.left_rig, "left"),
              "right": load_body(args.right_rig, "right")}
    validate_bodies(bodies["left"], bodies["right"])
    summary = replay(args.session, args.output_dir, bodies, max_frames=args.max_frames)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export synchronized T265 stereo and two animated cube-center trajectories.

Run with ``.venv-t265/bin/python scripts/export_t265_task_videos.py``.
The input is the replay output, so this script never needs a connected camera.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("01_open_lid_153305", "OPEN LID"),
    ("02_fold_item_153420", "FOLD ITEM"),
    ("03_tidy_table_153506", "TIDY TABLE"),
)
OUTPUT_SIZE = (1920, 1280)
PANEL_SIZE = (920, 320)
COLORS = {"left": (120, 225, 140), "right": (225, 115, 225)}
EGO_COLOR = (132, 214, 246)
BG = (17, 12, 9)
WHITE = (243, 242, 238)
MUTED = (174, 157, 139)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(image: np.ndarray, label: str, xy: tuple[int, int], size: float,
        color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(image, label, xy, FONT, size, color, thickness, cv2.LINE_AA)


def load_rows(path: Path, frames: int) -> dict[str, list[dict]]:
    rows = {name: [None] * frames for name in COLORS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            index = int(row["frame_index"])
            name = row["body"]
            if name not in rows or index < 0 or index >= frames or rows[name][index] is not None:
                raise ValueError(f"invalid or duplicate CSV row: {name} frame {index}")
            rows[name][index] = row
    for name, samples in rows.items():
        if any(row is None for row in samples):
            raise ValueError(f"missing {name} CSV rows")
    return rows


def position_mm(row: dict) -> np.ndarray | None:
    values = [row[f"smooth_{axis}_m"] for axis in "xyz"]
    if not all(values):
        return None
    point = np.asarray([float(value) * 1000 for value in values], dtype=np.float64)
    if not np.isfinite(point).all():
        raise ValueError("nonfinite trajectory position")
    return point


def project_xyz(point: np.ndarray) -> np.ndarray:
    """Fixed axonometric view of camera X/right, Y/down and Z/forward."""
    x, y, z = point
    yaw, pitch = -0.64, 0.55
    horizontal = np.cos(yaw) * x - np.sin(yaw) * z
    depth = np.sin(yaw) * x + np.cos(yaw) * z
    vertical = np.cos(pitch) * y - np.sin(pitch) * depth
    return np.array([horizontal, vertical])


class TrajectoryPanel:
    def __init__(self, name: str, rows: list[dict]):
        self.name = name
        self.rows = rows
        self.color = COLORS.get(name, EGO_COLOR)
        self.positions = [position_mm(row) for row in rows]
        self.projected = [project_xyz(p) if p is not None else None for p in self.positions]
        self.segments = [row["segment"] for row in rows]
        self.last_index = -1
        self.base = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), (31, 25, 20), np.uint8)
        self._prepare()
        self.revealed = self.base.copy()

    def _prepare(self) -> None:
        width, height = PANEL_SIZE
        cv2.rectangle(self.base, (0, 0), (width - 1, height - 1), (66, 55, 43), 1)
        cv2.rectangle(self.base, (1, 1), (7, height - 2), self.color, -1)
        points = np.asarray([p for p in self.projected if p is not None])
        if len(points):
            low, high = points.min(axis=0), points.max(axis=0)
        else:
            low, high = np.array([-50., -50.]), np.array([50., 50.])
        center = (low + high) / 2
        span = np.maximum(high - low, 40.0)
        plot_left, plot_top, plot_width, plot_height = 35, 73, width - 70, height - 130
        self.scale = min(plot_width / span[0], plot_height / span[1]) * .89
        self.offset = np.array([plot_left + plot_width / 2, plot_top + plot_height / 2])
        self.center = center
        for index in range(1, 5):
            x = plot_left + index * plot_width // 5
            y = plot_top + index * plot_height // 5
            cv2.line(self.base, (x, plot_top), (x, plot_top + plot_height), (48, 40, 32), 1)
            cv2.line(self.base, (plot_left, y), (plot_left + plot_width, y), (48, 40, 32), 1)
        for index in range(1, len(self.projected)):
            if not self._connected(index):
                continue
            cv2.line(self.base, self.pixel(index - 1), self.pixel(index),
                     tuple(int(channel * .27) for channel in self.color), 2, cv2.LINE_AA)
        first = next((index for index, p in enumerate(self.positions) if p is not None), None)
        if first is not None:
            cv2.circle(self.base, self.pixel(first), 5, self.color, 1, cv2.LINE_AA)
            put(self.base, "START", (self.pixel(first)[0] + 9, self.pixel(first)[1] - 6), .37, MUTED)

    def _connected(self, index: int) -> bool:
        return (self.projected[index] is not None and
                self.projected[index - 1] is not None and
                self.segments[index] == self.segments[index - 1])

    def pixel(self, index: int) -> tuple[int, int]:
        location = self.offset + (self.projected[index] - self.center) * self.scale
        return int(round(location[0])), int(round(location[1]))

    def render(self, index: int) -> np.ndarray:
        if index != self.last_index + 1:
            raise ValueError("frames must be rendered in order")
        if index > 0 and self._connected(index):
            cv2.line(self.revealed, self.pixel(index - 1), self.pixel(index),
                     self.color, 3, cv2.LINE_AA)
        self.last_index = index
        image = self.revealed.copy()
        row, point = self.rows[index], self.positions[index]
        title = ("T265 EGO  /  VIO CAMERA" if self.name == "ego" else
                 f"{self.name.upper()} GRIPPER  /  CUBE CENTER")
        put(image, title, (26, 39), .75, self.color, 2)
        status = row["status"]
        status_color = self.color if point is not None else (105, 145, 240)
        put(image, status, (width_for_status(status), 39), .56, status_color, 2)
        if point is not None:
            pixel = self.pixel(index)
            cv2.circle(image, pixel, 11, (23, 21, 20), -1, cv2.LINE_AA)
            cv2.circle(image, pixel, 8, self.color, -1, cv2.LINE_AA)
            cv2.circle(image, pixel, 14, self.color, 1, cv2.LINE_AA)
            xyz = f"X {point[0]:+06.0f}   Y {point[1]:+06.0f}   Z {point[2]:+06.0f} mm"
        else:
            xyz = "POSE LOST  /  TRAJECTORY GAP"
        put(image, xyz, (32, PANEL_SIZE[1] - 24), .59, WHITE)
        return image


class CombinedTrajectoryPanel:
    """Draw all three paths at one fixed scale in their shared VIO frame."""

    def __init__(self, rows: dict[str, list[dict]]):
        self.rows = rows
        self.size = (1860, 320)
        self.colors = dict(COLORS, ego=EGO_COLOR)
        self.positions = {name: [position_mm(row) for row in samples]
                          for name, samples in rows.items()}
        self.projected = {name: [project_xyz(p) if p is not None else None
                                 for p in positions]
                          for name, positions in self.positions.items()}
        points = np.asarray([point for samples in self.projected.values()
                             for point in samples if point is not None])
        low, high = points.min(axis=0), points.max(axis=0)
        self.center = (low + high) / 2
        span = np.maximum(high - low, 40.0)
        self.scale = min(1750 / span[0], 190 / span[1]) * .9
        self.offset = np.array([930, 166])
        self.base = np.full((320, 1860, 3), (31, 25, 20), dtype=np.uint8)
        cv2.rectangle(self.base, (0, 0), (1859, 319), (66, 55, 43), 1)
        for j in range(1, 7):
            x = 40 + j * 1780 // 7
            cv2.line(self.base, (x, 75), (x, 260), (48, 40, 32), 1)
        for j in range(1, 4):
            y = 75 + j * 185 // 4
            cv2.line(self.base, (40, y), (1820, y), (48, 40, 32), 1)
        for name in ("ego", "left", "right"):
            for index in range(1, len(rows[name])):
                if self.connected(name, index):
                    cv2.line(self.base, self.pixel(name, index - 1), self.pixel(name, index),
                             tuple(int(c * .23) for c in self.colors[name]), 2, cv2.LINE_AA)
        self.revealed = self.base.copy()
        self.last_index = -1

    def connected(self, name: str, index: int) -> bool:
        points = self.projected[name]
        return (points[index] is not None and points[index - 1] is not None and
                self.rows[name][index]["segment"] == self.rows[name][index - 1]["segment"])

    def pixel(self, name: str, index: int) -> tuple[int, int]:
        location = self.offset + (self.projected[name][index] - self.center) * self.scale
        return int(round(location[0])), int(round(location[1]))

    def render(self, index: int) -> np.ndarray:
        if index != self.last_index + 1:
            raise ValueError("combined paths must be rendered in frame order")
        for name in ("ego", "left", "right"):
            if index > 0 and self.connected(name, index):
                cv2.line(self.revealed, self.pixel(name, index - 1), self.pixel(name, index),
                         self.colors[name], 3, cv2.LINE_AA)
        self.last_index = index
        image = self.revealed.copy()
        put(image, "SHARED VIO FRAME  /  ALL THREE TRAJECTORIES", (32, 41), .75, WHITE, 2)
        for name, x in (("left", 1120), ("right", 1355), ("ego", 1610)):
            cv2.circle(image, (x, 34), 7, self.colors[name], -1, cv2.LINE_AA)
            put(image, name.upper(), (x + 14, 41), .56, self.colors[name], 2)
        for name in ("ego", "left", "right"):
            if self.positions[name][index] is not None:
                location = self.pixel(name, index)
                cv2.circle(image, location, 11, (23, 21, 20), -1, cv2.LINE_AA)
                cv2.circle(image, location, 7, self.colors[name], -1, cv2.LINE_AA)
        ego = self.positions["ego"][index]
        label = ("EGO POSE LOST" if ego is None else
                 f"EGO X {ego[0]:+06.0f}   Y {ego[1]:+06.0f}   Z {ego[2]:+06.0f} mm")
        put(image, label, (35, 295), .6, EGO_COLOR)
        put(image, "Left / right are cube centers; all paths use one scale", (1110, 295), .46, MUTED)
        return image


def width_for_status(status: str) -> int:
    return PANEL_SIZE[0] - max(160, len(status) * 16 + 35)


def load_ego_rows(path: Path, frames: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        samples = list(csv.DictReader(stream))
    if len(samples) != frames or any(int(row["frame_index"]) != index
                                      for index, row in enumerate(samples)):
        raise ValueError("ego pose CSV does not match video frames")
    segment = 0
    previous_valid = False
    for row in samples:
        valid = bool(row["x_m"])
        if valid and not previous_valid:
            segment += 1
        previous_valid = valid
        row["segment"] = str(segment)
        row["status"] = "VIO" if valid else row["status"]
        for axis in "xyz":
            row[f"smooth_{axis}_m"] = row[f"{axis}_m"]
    return samples


def update_galleries(result_dir: Path) -> None:
    gallery = result_dir / "index.html"
    if not gallery.exists():
        return
    page = gallery.read_text(encoding="utf-8")
    if "task_visualization.mp4" not in page:
        target = ('<section class="card"><h2>左鱼眼视频与投影轨迹</h2>'
                  if "01_monocular_static" in str(result_dir) else
                  '<section class="card"><h2>双鱼眼视频与投影轨迹</h2>')
        replacement = ('<section class="card"><h2>完整可视化视频 · 双鱼眼与左右动态轨迹</h2>'
                       '<video controls preload="metadata" src="task_visualization.mp4"></video>'
                       '<p><a href="task_visualization.mp4" download>下载 MP4</a></p></section>\n'
                       + target)
        if target not in page:
            raise ValueError(f"unexpected gallery layout: {gallery}")
        gallery.write_text(page.replace(target, replacement, 1), encoding="utf-8")
    overview = result_dir.parent / "index.html"
    if overview.exists():
        page = overview.read_text(encoding="utf-8")
        name = result_dir.name
        video_href = f'{name}/task_visualization.mp4'
        if video_href not in page:
            target = f'<a class="button" href="{name}/index.html">'
            replacement = (f'<a class="button" href="{video_href}" download>下载完整视频 MP4</a> '
                           + target)
            if target not in page:
                raise ValueError(f"unexpected overview layout: {overview}")
            overview.write_text(page.replace(target, replacement, 1), encoding="utf-8")


def export_one(result_dir: Path, title: str, *, max_frames: int = 0) -> dict:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    expected = int(summary["frames"])
    fps = float(summary["annotated_video_fps"])
    rows = load_rows(result_dir / "dual_center_poses.csv", expected)
    panels = {name: TrajectoryPanel(name, rows[name]) for name in COLORS}
    vio = summary.get("coordinate_mode") == "vio"
    if vio:
        panels["ego"] = TrajectoryPanel("ego", load_ego_rows(result_dir / "ego_pose.csv", expected))
        combined = CombinedTrajectoryPanel(dict(rows, ego=panels["ego"].rows))
    capture = cv2.VideoCapture(str(result_dir / "annotated_dual.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"cannot read annotated video in {result_dir}")
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_count != expected:
        capture.release()
        raise ValueError(f"video has {source_count} frames; CSV has {expected}")
    frames = min(expected, max_frames) if max_frames else expected
    mono = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 848
    output_size = (1920, 920) if mono else (1920, 1600) if vio else OUTPUT_SIZE
    mode = ("MONOCULAR / STATIC" if mono else
            "STEREO / VIO WORLD" if vio else
            "STEREO / STATIC")
    output = result_dir / ("task_visualization_preview.mp4" if max_frames else "task_visualization.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{output_size[0]}x{output_size[1]}", "-r", f"{fps:.9f}",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in range(frames):
            ok, source = capture.read()
            if not ok or source.shape[:2] != (800, 848 if mono else 1696):
                raise RuntimeError(f"cannot decode source frame {index} in {result_dir}")
            canvas = np.full((output_size[1], output_size[0], 3), BG, dtype=np.uint8)
            put(canvas, "T265  /  DUAL GRIPPER", (38, 47), .75, WHITE, 2)
            put(canvas, f"{title}  /  {mode}", (520, 47), .65, (218, 190, 105), 2)
            time_s = float(rows["left"][index]["time_s"])
            put(canvas, f"{time_s:06.1f} s  /  {index + 1:04d} - {expected:04d}",
                (1500, 47), .67, MUTED, 2)
            cv2.rectangle(canvas, (0, 64), (output_size[0] - 1, 68), (52, 45, 37), -1)
            cv2.rectangle(canvas, (0, 64), (int(output_size[0] * (index + 1) / expected), 68),
                          (199, 192, 116), -1)
            if mono:
                canvas[80:910, 60:940] = cv2.resize(source, (880, 830), interpolation=cv2.INTER_AREA)
                cv2.rectangle(canvas, (59, 79), (940, 910), (79, 65, 50), 2)
                canvas[80:400, 970:1890] = panels["left"].render(index)
                canvas[430:750, 970:1890] = panels["right"].render(index)
                put(canvas, "LEFT FISHEYE ONLY", (1005, 811), .70, MUTED, 2)
                put(canvas, "STATIC CAMERA FRAME  /  CUBE CENTER", (1005, 864), .58, MUTED)
            else:
                canvas[80:910, 80:1840] = cv2.resize(source, (1760, 830), interpolation=cv2.INTER_AREA)
                cv2.rectangle(canvas, (79, 79), (1840, 910), (79, 65, 50), 2)
                canvas[930:1250, 30:950] = panels["left"].render(index)
                canvas[930:1250, 970:1890] = panels["right"].render(index)
                if vio:
                    canvas[1260:1580, 30:1890] = combined.render(index)
            encoder.stdin.write(canvas.tobytes())
            if index % 150 == 0:
                print(f"{title}: {index + 1}/{frames}", flush=True)
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
    except Exception:
        encoder.kill()
        encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
    report = {"video": str(output), "frames": frames, "fps": fps,
              "size": list(output_size), "duration_s": frames / fps,
              "bytes": output.stat().st_size}
    if not max_frames:
        update_galleries(result_dir)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[name for name, _ in TASKS],
                        help="export one task; default exports all three")
    parser.add_argument("--result-root", type=Path, default=ROOT / "outputs/t265_tasks",
                        help="directory containing the three task replay results")
    parser.add_argument("--max-frames", type=int, default=0, help="short preview for verification")
    args = parser.parse_args()
    for name, title in TASKS:
        if args.task is None or args.task == name:
            export_one(args.result_root / name, title,
                       max_frames=args.max_frames)


if __name__ == "__main__":
    main()

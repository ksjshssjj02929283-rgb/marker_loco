#!/usr/bin/env python3
"""Animate held-out laser and camera trajectories from evaluation CSV files.

This is a visualization of numerical trajectories, not camera footage. The
camera position at every shown timestamp uses an SE(3) fitted on the opposite
chronological half of that recording.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1280, 720
BG, NAVY, ORANGE, TEAL = "#F5F8FB", "#26394A", "#D87531", "#237A6B"
COLORS = {"first_to_second": ORANGE, "second_to_first": TEAL}
MAIN = (50, 155, 790, 600)
PROGRESS = (835, 155, 1230, 365)
ERROR = (835, 425, 1230, 600)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "Bold" if bold else "Regular"
    paths = (Path(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{suffix}.ttc"),
             Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"))
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


FONTS = {size: font(size) for size in (15, 18, 20, 24, 32)}
BOLD = {size: font(size, bold=True) for size in (18, 20, 24, 32)}


def load(case: Path) -> dict:
    path = case / "cross_validated_trajectories.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["gt_index"]))
    if not rows:
        raise ValueError(f"no trajectory rows: {path}")
    expected = list(range(len(rows)))
    if [int(row["gt_index"]) for row in rows] != expected:
        raise ValueError(f"GT indexes must be consecutive: {path}")
    summary = json.loads((case / "summary.json").read_text(encoding="utf-8"))
    if summary.get("cross_validation") is None:
        raise ValueError(f"cross validation unavailable: {case}")
    truth = np.array([[float(row[f"gt_{axis}_m"]) for axis in "xyz"] for row in rows]) * 1000
    aligned = np.array([[float(row[f"aligned_{axis}_m"]) if row["paired"] == "True" else np.nan
                         for axis in "xyz"] for row in rows]) * 1000
    times = np.array([float(row["gt_time_s"]) for row in rows])
    errors = np.array([float(row["residual_mm"]) if row["paired"] == "True" else np.nan
                       for row in rows])
    folds = [row["fold"] for row in rows]
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"GT times must increase: {case}")
    return dict(rows=rows, truth=truth, aligned=aligned, times=times, errors=errors,
                folds=folds, summary=summary, source=path)


def mapper(values: np.ndarray, rect: tuple[int, int, int, int], *, equal: bool = False):
    values = np.asarray(values)
    lo, hi = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
    span = np.maximum(hi - lo, 1.0)
    left, top, right, bottom = rect
    width, height = right - left - 40, bottom - top - 40
    if equal:
        scale = min(width / span[0], height / span[1]) * 0.94
        cx, cy = (lo + hi) / 2
        return lambda x, y: (int((left + right) / 2 + (x - cx) * scale),
                             int((top + bottom) / 2 - (y - cy) * scale))
    xlo, xhi = lo[0] - span[0] * .04, hi[0] + span[0] * .04
    ylo, yhi = lo[1] - span[1] * .08, hi[1] + span[1] * .08
    return lambda x, y: (int(left + 20 + (x - xlo) / (xhi - xlo) * width),
                         int(bottom - 20 - (y - ylo) / (yhi - ylo) * height))


def circle(draw: ImageDraw.ImageDraw, point: tuple[int, int], radius: int, fill: str,
           outline: str = "white") -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill,
                 outline=outline, width=2)


def panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle((left, top, right, bottom), radius=12,
                           fill="white", outline="#DCE5EA", width=2)
    draw.text((left + 18, top + 10), title, font=BOLD[18], fill=NAVY)


def continuous_segments(points: list[tuple[int, int] | None], folds: list[str]) -> list[tuple[int, int, str]]:
    segments = []
    start = None
    for index, point in enumerate(points):
        if point is None:
            if start is not None:
                segments.append((start, index - 1, COLORS[folds[start]]))
                start = None
        elif start is None or folds[index] != folds[start]:
            if start is not None:
                segments.append((start, index - 1, COLORS[folds[start]]))
            start = index
    if start is not None:
        segments.append((start, len(points) - 1, COLORS[folds[start]]))
    return segments


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int] | None],
                  start: int, end: int, color: str, width: int) -> None:
    if end > start:
        draw.line(points[start:end + 1], fill=color, width=width, joint="curve")


def render(case: Path, *, seconds: float = 16.0, fps: int = 15) -> Path:
    data = load(case)
    truth, aligned, times, errors, folds = (data[key] for key in
                                            ("truth", "aligned", "times", "errors", "folds"))
    origin = truth[0]
    truth = truth - origin
    aligned = aligned - origin
    direction = np.linalg.svd(truth - truth.mean(axis=0), full_matrices=False)[2][0]
    if np.dot(truth[-1] - truth[0], direction) < 0:
        direction *= -1
    progress_gt = truth @ direction
    progress_cam = aligned @ direction
    elapsed = times - times[0]
    xy_all = np.vstack((truth[:, :2], aligned[np.isfinite(aligned).all(axis=1), :2]))
    xy_map = mapper(xy_all, MAIN, equal=True)
    progress_all = np.column_stack((np.r_[elapsed, elapsed[np.isfinite(progress_cam)]],
                                    np.r_[progress_gt, progress_cam[np.isfinite(progress_cam)]]))
    progress_map = mapper(progress_all, PROGRESS)
    max_error = max(10.0, float(np.nanmax(errors)) * 1.05)
    error_map = mapper(np.array([[0, 0], [elapsed[-1], max_error]]), ERROR)
    xy_gt = [xy_map(*point[:2]) for point in truth]
    xy_cam = [xy_map(*point[:2]) if np.isfinite(point).all() else None for point in aligned]
    curve_gt = [progress_map(t, p) for t, p in zip(elapsed, progress_gt)]
    curve_cam = [progress_map(t, p) if np.isfinite(p) else None
                 for t, p in zip(elapsed, progress_cam)]
    curve_error = [error_map(t, e) if np.isfinite(e) else None
                   for t, e in zip(elapsed, errors)]
    xy_segments = continuous_segments(xy_cam, folds)
    curve_segments = continuous_segments(curve_cam, folds)
    error_segments = continuous_segments(curve_error, folds)
    split = next((index for index in range(1, len(folds)) if folds[index] != folds[0]), None)
    label = f"{case.parent.name} / {case.name} / {case.parent.parent.name}"
    cv = data["summary"]["cross_validation"]
    rpe = json.loads((case / "relative_translation_rpe_0.1s.json").read_text(encoding="utf-8"))
    if rpe["source_csv_sha256"] != hashlib.sha256(data["source"].read_bytes()).hexdigest():
        raise ValueError(f"relative translation metric is stale: {case}")

    background = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(background)
    draw.text((50, 20), "动态轨迹跨段 SE(3) 对齐", font=BOLD[32], fill=NAVY)
    draw.text((50, 68), label, font=FONTS[20], fill="#526575")
    rpe_color = TEAL if rpe["mae_mm"] <= 2.0 else NAVY
    draw.text((50, 97),
              f"0.1 s 相对平移 RPE MAE {rpe['mae_mm']:.2f} mm  ·  {rpe['pairs']} 个有效区间",
              font=BOLD[20], fill=rpe_color)
    draw.text((50, 126),
              f"跨段 3D 位置 RMSE {cv['heldout_position_rmse_mm']:.2f} mm  ·  P95 {cv['heldout_position_p95_mm']:.2f} mm  ·  覆盖 {cv['heldout_coverage']:.1%}",
              font=FONTS[18], fill=NAVY)
    if rpe["mae_mm"] <= 2.0:
        draw.rounded_rectangle((970, 62, 1230, 102), radius=12,
                               fill="#DDF2E9", outline="#9DCDBD", width=2)
        draw.text((989, 68), "约 2 mm 相对平移", font=BOLD[18], fill=TEAL)
    panel(draw, MAIN, "XY 轨迹：同一时刻的激光点与相机点")
    panel(draw, PROGRESS, "沿主运动轴的位移")
    panel(draw, ERROR, "跨段 3D 位置误差")
    for rect in (PROGRESS, ERROR):
        l, t, r, b = rect
        for fraction in (.25, .5, .75):
            x = int(l + 20 + fraction * (r - l - 40))
            draw.line((x, t + 44, x, b - 20), fill="#ECF0F2", width=1)
    draw_polyline(draw, xy_gt, 0, len(xy_gt) - 1, "#A5B2BD", 2)
    draw_polyline(draw, curve_gt, 0, len(curve_gt) - 1, "#A5B2BD", 2)
    for segments, points in ((xy_segments, xy_cam), (curve_segments, curve_cam),
                             (error_segments, curve_error)):
        for start, end, color in segments:
            draw_polyline(draw, points, start, end, "#E8CFB9" if color == ORANGE else "#B7D6D0", 2)
    if split is not None:
        x, _ = progress_map(elapsed[split], progress_gt[split])
        draw.line((x, PROGRESS[1] + 42, x, PROGRESS[3] - 19),
                  fill="#9FAAB2", width=2)
        x, _ = error_map(elapsed[split], 0)
        draw.line((x, ERROR[1] + 42, x, ERROR[3] - 19), fill="#9FAAB2", width=2)
    for x, color, text_label in ((60, NAVY, "激光真值"), (245, TEAL, "前半段相机"),
                                 (480, ORANGE, "后半段相机")):
        circle(draw, (x, 640), 7, color, BG)
        draw.text((x + 18, 625), text_label, font=FONTS[18], fill=NAVY)
    draw.line((725, 640, 757, 640), fill="#C94C40", width=4)
    draw.text((767, 625), "当前三维位置差", font=FONTS[18], fill=NAVY)
    draw.text((50, 674),
              "动画：跨段 SE(3) 对齐轨迹；0.1 s RPE：同一验证半段内，比较三维位移的方向与长度。",
              font=FONTS[15], fill="#5D6F7C")

    frames = max(2, int(round(seconds * fps)))
    active_seconds = max(0.5, seconds - 1.4)
    output = case / "trajectory_cross_validation.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in range(frames):
            wall_time = frame / fps
            fraction = np.clip((wall_time - .5) / active_seconds, 0, 1)
            index = int(np.searchsorted(times, times[0] + fraction * (times[-1] - times[0]), side="right") - 1)
            index = max(0, min(index, len(times) - 1))
            image = background.copy()
            draw = ImageDraw.Draw(image)
            draw_polyline(draw, xy_gt, 0, index, NAVY, 4)
            draw_polyline(draw, curve_gt, 0, index, NAVY, 3)
            for segments, points in ((xy_segments, xy_cam), (curve_segments, curve_cam),
                                     (error_segments, curve_error)):
                for start, end, color in segments:
                    draw_polyline(draw, points, start, min(end, index), color, 3)
            x, _ = progress_map(elapsed[index], progress_gt[index])
            draw.line((x, PROGRESS[1] + 42, x, PROGRESS[3] - 19), fill="#647787", width=2)
            x, _ = error_map(elapsed[index], 0)
            draw.line((x, ERROR[1] + 42, x, ERROR[3] - 19), fill="#647787", width=2)
            circle(draw, xy_gt[index], 9, NAVY)
            circle(draw, curve_gt[index], 6, NAVY)
            status = "相机无有效配对"
            if xy_cam[index] is not None:
                color = COLORS[folds[index]]
                draw.line((*xy_gt[index], *xy_cam[index]), fill="#C94C40", width=4)
                circle(draw, xy_cam[index], 9, color)
                circle(draw, curve_cam[index], 6, color)
                circle(draw, curve_error[index], 6, color)
                status = f"当前位置差 {errors[index]:.1f} mm"
            phase = "前半段" if folds[index] == "second_to_first" else "后半段"
            draw.text((1000, 610), f"{phase}  ·  {elapsed[index]:.1f} / {elapsed[-1]:.1f} s",
                      font=BOLD[18], fill=NAVY)
            draw.text((1000, 637), status, font=BOLD[20],
                      fill="#C94C40" if xy_cam[index] is not None else "#5D6F7C")
            draw.rounded_rectangle((50, 701, 1230, 709), radius=4, fill="#DBE3E8")
            draw.rounded_rectangle((50, 701, int(50 + 1180 * fraction), 709), radius=4, fill=TEAL)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {case}: {stderr}")
    except Exception:
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise
    source_hash = hashlib.sha256(data["source"].read_bytes()).hexdigest()
    summary_hash = hashlib.sha256((case / "summary.json").read_bytes()).hexdigest()
    (case / "trajectory_video_manifest.json").write_text(json.dumps({
        "source_csv_sha256": source_hash, "source_summary": "summary.json",
        "source_summary_sha256": summary_hash,
        "video": output.name, "fps": fps, "frames": frames,
        "duration_s": frames / fps, "source_duration_s": float(elapsed[-1]),
        "relative_translation_interval_s": rpe["interval_s"],
        "relative_translation_mae_mm": rpe["mae_mm"],
        "relative_translation_pairs": rpe["pairs"],
        "relative_translation_metric_sha256": hashlib.sha256(
            (case / "relative_translation_rpe_0.1s.json").read_bytes()).hexdigest(),
        "heldout_position_rmse_mm": cv["heldout_position_rmse_mm"],
        "method": "time-proportional animation of held-out paired trajectories; no invented positions or camera footage"
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, action="append", help="evaluation case directory; repeat for more clips")
    parser.add_argument("--all", type=Path, help="render every valid case under this evaluation root")
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.seconds <= 2 or not 5 <= args.fps <= 60:
        parser.error("seconds must exceed 2 and fps must be between 5 and 60")
    cases = list(args.case_dir or [])
    if args.all:
        cases.extend(p.parent for p in sorted(args.all.glob("*/marker*/*/cross_validated_trajectories.csv")))
    if not cases:
        parser.error("provide --case-dir or --all")
    for case in dict.fromkeys(cases):
        output = render(case, seconds=args.seconds, fps=args.fps)
        print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

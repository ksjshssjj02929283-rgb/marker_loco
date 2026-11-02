#!/usr/bin/env python3
"""Render traceable RGB contact sheets from the 2026-09-30 D455 recordings."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("captures/d455_aruco")
OUT = Path("outputs/report_visuals_20260930")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SCENES = (
    ("static_d300", "约 0.3 m"),
    ("static_d500", "约 0.5 m"),
    ("static_d800", "约 0.8 m"),
    ("static_d1000", "约 1.0 m"),
    ("zdisp_0mm", "标称 0 mm"),
    ("zdisp_50mm", "标称 50 mm"),
    ("zdisp_100mm", "标称 100 mm"),
    ("zdisp_150mm", "标称 150 mm"),
    ("zdisp_200mm", "标称 200 mm"),
)


def panel(scene: str, label: str) -> Image.Image:
    session = sorted((ROOT / scene).glob("2*"))[-1]
    capture = cv2.VideoCapture(str(session / "color_raw.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {session / 'color_raw.mp4'}")
    frame_index = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {session}")

    rows = [row for row in csv.DictReader((session / "poses.csv").open(encoding="utf-8"))
            if int(row["frame_index"]) == frame_index]
    # Leave the scale/tape and marker visible; exclude the person's face at right.
    frame = frame[95:645, 80:900].copy()
    colors = {2: (32, 194, 68), 3: (255, 172, 36), 4: (38, 163, 243), 5: (214, 99, 200)}
    for row in rows:
        tag_id = int(row["tag_id"])
        corners = [
            (round(float(row[f"corner_{i}_x"])) - 80,
             round(float(row[f"corner_{i}_y"])) - 95)
            for i in range(4)
        ]
        points = cv2.convexHull(np.array(corners, dtype="int32"))
        cv2.polylines(frame, [points], True, colors.get(tag_id, (255, 255, 255)), 3, cv2.LINE_AA)
        x, y = corners[0]
        label_x = x - 25 if tag_id == 3 else x + 10
        cv2.putText(frame, f"ID {tag_id}", (label_x, max(y - 9, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, colors.get(tag_id, (255, 255, 255)), 2, cv2.LINE_AA)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((820, 550))
    board = Image.new("RGB", (844, 640), "#f7f9fb")
    board.paste(image, (12, 54))
    draw = ImageDraw.Draw(board)
    draw.text((16, 9), label, font=ImageFont.truetype(FONT, 26), fill="#16232b")
    draw.text((16, 609), f"{session.name}  |  原始 RGB 中间帧 {frame_index}  |  检测 ID: "
              + (", ".join(row["tag_id"] for row in rows) if rows else "无"),
              font=ImageFont.truetype(FONT, 15), fill="#354953")
    return board


def sheet(items: tuple[tuple[str, str], ...], filename: str, title: str, columns: int) -> None:
    rows = (len(items) + columns - 1) // columns
    width = columns * 844 + (columns + 1) * 12
    height = rows * 640 + (rows + 1) * 12 + 66
    canvas = Image.new("RGB", (width, height), "#e7edef")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), title, font=ImageFont.truetype(FONT, 32), fill="#14242b")
    for index, (scene, label) in enumerate(items):
        x = 12 + (index % columns) * (844 + 12)
        y = 78 + (index // columns) * (640 + 12)
        canvas.paste(panel(scene, label), (x, y))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / filename, optimize=True)
    print(OUT / filename)


def main() -> None:
    sheet(SCENES[:4], "rgb_static_four_distances.png", "D455 静态四距离 | 原始 RGB 与检测角点", 2)
    sheet(SCENES[4:], "rgb_nominal_displacement_five_positions.png",
          "D455 标称位移五位置 | 原始 RGB 与检测角点", 3)


if __name__ == "__main__":
    main()

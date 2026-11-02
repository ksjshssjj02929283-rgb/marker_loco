#!/usr/bin/env python3
"""Build one local review page for completed dual-gripper T265 task replays."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


TASK_LABELS = {
    "01_open_lid": "打开盖子",
    "02_fold_item": "折叠物品",
    "03_tidy_table": "整理桌面",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/t265_tasks"))
    args = parser.parse_args()
    cards = []
    for directory in sorted(args.root.iterdir()):
        summary_path = directory / "summary.json"
        if not directory.is_dir() or not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        task = next((label for prefix, label in TASK_LABELS.items()
                     if directory.name.startswith(prefix)), directory.name)
        coverage = 100 * data["both_valid_fraction"]
        note = ("第二次录制；左右视频末尾相差 1 帧，使用共同帧。"
                if data.get("ended_with_eye_mismatch") else "左右鱼眼完整同步回放。")
        relative = escape(directory.name)
        cards.append(f'''
<article class="card"><a href="{relative}/index.html"><img src="{relative}/dual_trajectories.png" alt="{escape(task)}双夹爪轨迹"></a>
<div class="body"><p class="eyebrow">T265 / DUAL MARKER / TASK</p><h2>{escape(task)}</h2>
<p class="meta">共同视频 {data['frames']} 帧 · 双夹爪同帧有效 {data['both_valid_frames']} 帧
<strong>{coverage:.1f}%</strong></p><p class="note">{note}</p>
<a class="button" href="{relative}/index.html">查看双鱼眼视频与独立轨迹 →</a></div></article>''')
    if not cards:
        parser.error(f"no completed replay summaries in {args.root}")
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>T265 双夹爪三任务结果</title>
<style>body{{margin:0;background:#0b1117;color:#edf5f8;font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1480px;margin:auto;padding:40px 28px}}h1{{font-size:32px;margin:0 0 8px}}
.intro{{color:#9eb3c1;max-width:900px;margin:0 0 30px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}
.card{{background:#14202a;border:1px solid #29404e;border-radius:12px;overflow:hidden}}
.card img{{width:100%;aspect-ratio:1.7;object-fit:cover;display:block}}
.body{{padding:20px}}.eyebrow{{font:11px ui-monospace,monospace;letter-spacing:.14em;color:#72e0a2;margin:0}}
h2{{font-size:22px;margin:8px 0 14px}}.meta{{color:#d9e6ed}}strong{{color:#72e0a2;display:block;font-size:26px}}
.note{{color:#9eb3c1;min-height:45px}}.button{{display:inline-block;padding:10px 14px;background:#20493a;color:#e8fff3;
text-decoration:none;border-radius:7px;font-weight:600}}.button:hover{{background:#2b624d}}
footer{{color:#9eb3c1;margin-top:24px;font-size:13px}}
@media(max-width:980px){{.grid{{grid-template-columns:1fr}}.card img{{aspect-ratio:2.4}}}}</style></head>
<body><main><h1>双夹爪三任务轨迹结果</h1>
<p class="intro">T265 左右鱼眼同步录制。绿色为左夹爪、紫色为右夹爪；轨迹点位于各自 Marker 立方体的估计中心，使用静止左鱼眼坐标系。</p>
<div class="grid">{''.join(cards)}</div>
<footer>点击任务进入双目视频、左右独立轨迹和同坐标系对比图。相机移动时，该坐标系不能视为世界坐标；中心点仍需实物尺寸复核。</footer>
</main></body></html>'''
    output = args.root / "index.html"
    output.write_text(page, encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

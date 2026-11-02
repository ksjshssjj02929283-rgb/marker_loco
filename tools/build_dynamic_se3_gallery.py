#!/usr/bin/env python3
"""Build a browsable gallery for two-segment laser trajectory validation."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


SEQUENCES = [
    (marker, name)
    for marker, tracks in (("marker50", 4), ("marker60", 4), ("marker70", 1))
    for name in ("continuous_static_20_100", *(f"dynamic_track_{index:02d}" for index in range(1, tracks + 1)))
]


def collect(root: Path) -> list[dict]:
    rows = []
    for marker, sequence in SEQUENCES:
        for camera in ("left", "right"):
            folder = root / camera / marker / sequence
            path = folder / "summary.json"
            if not path.exists():
                rows.append(dict(marker=marker, sequence=sequence, camera=camera, status="missing"))
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            cv = summary.get("cross_validation") or {}
            rows.append(dict(
                marker=marker, sequence=sequence, camera=camera,
                status=summary.get("status"),
                same_fit_rmse_mm=summary.get("trajectory_rmse_mm"),
                heldout_rmse_mm=cv.get("heldout_position_rmse_mm"),
                heldout_p95_mm=cv.get("heldout_position_p95_mm"),
                heldout_coverage=cv.get("heldout_coverage"),
                heldout_groups=cv.get("heldout_groups"),
                quality_flags=";".join(summary.get("quality_flags", [])),
                plot=str((folder / "trajectory_cross_validation.png").relative_to(root)),
                video=(str((folder / "trajectory_cross_validation.mp4").relative_to(root))
                       if (folder / "trajectory_cross_validation.mp4").exists() else ""),
                svg=str((folder / "trajectory_cross_validation.svg").relative_to(root)),
                pairs=str((folder / "cross_validated_trajectories.csv").relative_to(root)),
                summary=str(path.relative_to(root))))
    return rows


def fmt(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.1f}{suffix}"


def make_gallery(root: Path, rows: list[dict]) -> None:
    available = [row for row in rows if row.get("heldout_rmse_mm") is not None]
    table = []
    cards = []
    for row in rows:
        key = f"{row['marker']} / {row['sequence']} / {row['camera']}"
        anchor = f"{row['marker']}-{row['sequence']}-{row['camera']}"
        warning = "<span class='warn'>几何近直线</span>" if "near_collinear" in row.get("quality_flags", "") else ""
        metric = (f"{fmt(row.get('heldout_rmse_mm'))} mm" if row.get("heldout_rmse_mm") is not None else "未评估")
        table.append(f"<tr><td><a href='#{anchor}'>{html.escape(key)}</a></td>"
                     f"<td>{metric}</td><td>{fmt(row.get('heldout_p95_mm'), ' mm')}</td>"
                     f"<td>{fmt(100 * row['heldout_coverage'], '%') if row.get('heldout_coverage') is not None else '—'}</td>"
                     f"<td>{fmt(row.get('same_fit_rmse_mm'), ' mm')}</td><td>{warning}</td></tr>")
        if row.get("heldout_rmse_mm") is None:
            continue
        src = html.escape(row["plot"], quote=True)
        video = html.escape(row["video"], quote=True)
        visual = (f"<video controls playsinline preload='none' poster='{src}'>"
                  f"<source src='{video}' type='video/mp4'>浏览器不支持此视频。</video>"
                  if video else f"<a href='{src}' target='_blank'><img loading='lazy' src='{src}' alt='{html.escape(key, quote=True)} 的轨迹投影、主运动轴时间曲线及误差曲线'></a>")
        video_link = f"<a href='{video}'>下载 MP4</a>" if video else ""
        cards.append(f"<article id='{anchor}' class='card'>"
                     f"<div class='cardhead'><h3>{html.escape(key)}</h3><span>{warning}</span></div>"
                     f"<p><strong>跨段 RMSE {metric}</strong> · P95 {fmt(row['heldout_p95_mm'])} mm"
                     f" · 有效覆盖 {fmt(100 * row['heldout_coverage'])}%"
                     f" · 同段拟合 {fmt(row['same_fit_rmse_mm'])} mm</p>"
                     f"{visual}<div class='links'>{video_link}"
                     f"<a href='{src}'>放大 PNG</a>"
                     f"<a href='{html.escape(row['svg'], quote=True)}'>SVG</a>"
                     f"<a href='{html.escape(row['pairs'], quote=True)}'>逐点 CSV</a>"
                     f"<a href='{html.escape(row['summary'], quote=True)}'>拟合参数 JSON</a></div></article>")
    count = len(available)
    video_count = sum(bool(row.get("video")) for row in available)
    near_collinear = sum("near_collinear" in row.get("quality_flags", "") for row in available)
    featured = []
    for camera, marker, sequence, title, explanation in (
            ("left", "marker60", "continuous_static_20_100", "跨段误差明显", None),
            ("right", "marker60", "dynamic_track_02", "轨迹较一致", "深蓝真值与橙绿虚线在多个投影中贴近。")):
        row = next(item for item in rows if item["camera"] == camera and item["marker"] == marker
                   and item["sequence"] == sequence)
        if row.get("heldout_rmse_mm") is not None:
            src = html.escape(row["plot"], quote=True)
            video = html.escape(row["video"], quote=True)
            visual = (f"<video controls playsinline preload='metadata' poster='{src}'>"
                      f"<source src='{video}' type='video/mp4'>浏览器不支持此视频。</video>"
                      if video else f"<a href='{src}' target='_blank'><img src='{src}' alt='{title}的轨迹叠加图'></a>")
            if explanation is None:
                explanation = (f"同段拟合 {fmt(row['same_fit_rmse_mm'])} mm；"
                               f"跨段验证 {fmt(row['heldout_rmse_mm'])} mm。")
            featured.append(f"<article class='card'><h3>{title} · {marker} / {sequence} / {camera}</h3>"
                            f"<p>{explanation} 图中可查看不同时间段的轨迹偏差。</p>"
                            f"{visual}</article>")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>动态轨迹跨段 SE(3) 验证</title><style>
:root {{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#26394a;background:#f5f8fa}}
body {{margin:0}} main {{max-width:1500px;margin:auto;padding:30px 28px 70px}}
h1 {{font-size:31px;margin:0 0 8px}} h2 {{margin:36px 0 16px}} p {{line-height:1.65}}
.lead {{color:#526575;max-width:1000px}} .stats {{display:flex;gap:12px;flex-wrap:wrap;margin:23px 0}}
.stat {{background:white;border:1px solid #dce4e8;border-radius:12px;padding:14px 20px;min-width:170px}}
.stat strong {{display:block;font-size:25px;color:#1f526d}} .stat span {{color:#5e707d;font-size:13px}}
.note {{background:#fff9e8;border-left:4px solid #c98726;padding:12px 18px;border-radius:6px}}
.tablewrap {{overflow:auto;background:white;border-radius:10px;border:1px solid #dce4e8}}
table {{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td {{padding:10px 13px;border-bottom:1px solid #e5eaed;text-align:left;white-space:nowrap}}
th {{position:sticky;top:0;background:#ebf0f3}} tr:hover td {{background:#f7fafb}}
a {{color:#146088;text-decoration:none}} a:hover {{text-decoration:underline}}
.warn {{background:#fff0d5;color:#865414;border-radius:99px;padding:4px 9px;font-size:12px}}
.cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,670px),1fr));gap:20px}}
.card {{background:white;border:1px solid #dce4e8;border-radius:12px;padding:16px;box-shadow:0 4px 18px #1d394a0c;scroll-margin-top:20px}}
.cardhead {{display:flex;justify-content:space-between;align-items:center;gap:12px}} h3 {{font-size:18px;margin:0}}
.card p {{margin:8px 0 12px;font-size:14px}} .card img,.card video {{width:100%;display:block;border:1px solid #e3e8eb;border-radius:6px}}
.links {{display:flex;gap:16px;flex-wrap:wrap;padding-top:12px;font-size:13px}}
</style></head><body><main>
<h1>动态轨迹跨段 SE(3) 验证</h1>
<p class="lead">点击视频播放同步轨迹动画：深蓝点是激光真值，橙/绿点是相机轨迹，红线表示当下的三维位置差。两段连续时间各自作为一次验证段：从另一段估计时间偏移与固定尺度刚体变换，再转换本段并计算逐点位置误差。静态图保留 XY、XZ、YZ 投影与误差曲线。</p>
<div class="stats"><div class="stat"><strong>{count}/{len(rows)}</strong><span>成功完成跨段验证</span></div>
<div class="stat"><strong>{video_count}</strong><span>可播放的轨迹动画</span></div>
<div class="stat"><strong>{near_collinear}</strong><span>近直线几何警示</span></div>
<div class="stat"><strong>1.0</strong><span>固定尺度，无缩放拟合</span></div></div>
<p class="note">RMSE 是跨段对齐后的激光靶点与 Tag 中心位置差，包含未知靶点安装偏移和时间估计误差；没有姿态真值，不能据此宣称 6DoF 绝对精度。不同轨迹运动幅度不同，不合并成一个总分。近直线轨迹的绕运动轴旋转约束很弱。</p>
<h2>先看两段轨迹</h2><div class="cards">{''.join(featured)}</div>
<h2>逐段结果</h2><div class="tablewrap"><table><thead><tr><th>轨迹 / 相机</th><th>跨段 RMSE</th><th>跨段 P95</th><th>覆盖</th><th>同段拟合 RMSE</th><th>提示</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<h2>轨迹叠加与逐点误差</h2><div class="cards">{''.join(cards)}</div>
</main></body></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")


def make_report(root: Path, rows: list[dict]) -> None:
    lines = ["# 动态轨迹两段交叉验证", "", "在一段连续轨迹的前半段估时间偏移并拟合固定尺度 SE(3)，在后半段评估；再交换。每个激光时间组只作为一次验证点。图中深蓝色为激光真值，橙/绿虚线为不同验证段的相机轨迹。完整图册见 [index.html](index.html)。", "",
             "| 序列 | 左目跨段 RMSE / P95 / 覆盖 | 右目跨段 RMSE / P95 / 覆盖 |", "| --- | ---: | ---: |"]
    for marker, sequence in SEQUENCES:
        pair = [next(row for row in rows if row["marker"] == marker and row["sequence"] == sequence and row["camera"] == camera)
                for camera in ("left", "right")]
        cells = []
        for row in pair:
            if row.get("heldout_rmse_mm") is None:
                cells.append("—")
            else:
                cells.append(f"{row['heldout_rmse_mm']:.2f} / {row['heldout_p95_mm']:.2f} mm / {row['heldout_coverage']:.1%}")
        lines.append(f"| {marker} / {sequence} | {cells[0]} | {cells[1]} |")
    lines += ["", "## 如何读图与视频", "", "视频同步播放两段轨迹的激光点与相机点，用红线显示当前三维位置差，并展示运动进度和逐点残差。视频由数值轨迹生成，不是原始相机画面。静态图展示 XY、XZ、YZ 投影与沿主运动轴位移的时间曲线；XZ、YZ 的 Z 轴作显示放大。底部显示各验证点的三维位置误差，竖线标识前后半段分界。每段缺失的相机配对点不连线、不参与误差计算。", "",
              "## 解释边界", "", "跨段指标避免用同一批点同时拟合和评分，但仍由同一录制内另一段提供时间偏移与 SE(3)，不是独立外参标定或绝对精度。激光只有靶点位置、没有姿态真值；相机给的是 Tag 中心，二者固定空间偏移未独立测量。近直线轨迹有旋转几何退化；详情见每段 summary.json 的奇异值比例与质量标记。尺度始终固定为 1，没有 Sim(3) 缩放。", "",
              "## 复现", "", "输入是本机已录制的左右目 replay 与 `/home/dex/20261030/GT` 激光原始数据。具体阈值与输入映射见左右目 `evaluation_protocol.json`。", "",
              "```bash",
              ".venv/bin/python tools/eval_laser_20261030.py --replay-root outputs/accuracy_20261030_kalibr/replay --gt-root /home/dex/20261030/GT --output-dir outputs/dynamic_se3_cross_validation_20261102/left --manifest outputs/accuracy_20261030_kalibr/manifest_filtered_left.json",
              ".venv/bin/python tools/eval_laser_20261030.py --replay-root outputs/accuracy_20261030_kalibr/replay_right --gt-root /home/dex/20261030/GT --output-dir outputs/dynamic_se3_cross_validation_20261102/right --manifest outputs/accuracy_20261030_kalibr/manifest_filtered_right.json",
              ".venv/bin/python tools/render_dynamic_se3_video.py --all outputs/dynamic_se3_cross_validation_20261102",
              ".venv/bin/python tools/build_dynamic_se3_gallery.py --root outputs/dynamic_se3_cross_validation_20261102",
              "```", ""]
    for camera, marker, sequence in (("left", "marker50", "continuous_static_20_100"),
                                     ("left", "marker60", "continuous_static_20_100"),
                                     ("right", "marker60", "dynamic_track_02")):
        image = f"{camera}/{marker}/{sequence}/trajectory_cross_validation.png"
        video = f"{camera}/{marker}/{sequence}/trajectory_cross_validation.mp4"
        lines += [f"### {marker} / {sequence} / {camera}", "", f"![轨迹叠加与跨段误差]({image})", ""]
        if (root / video).exists():
            lines += [f"[播放轨迹动画]({video})", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = collect(root)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_gallery(root, rows)
    make_report(root, rows)
    print(f"{sum(row.get('heldout_rmse_mm') is not None for row in rows)}/{len(rows)} trajectories: {root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render a local Chinese sensor-range review from replay/evaluation artifacts."""

import argparse
import html
import json
from pathlib import Path


def render(root):
    root = Path(root)
    experiments = [(name, json.loads((root / name / "evaluation/static_depths.json").read_text()))
                   for name in ("37_5mm", "47_5mm")]
    comparison = []
    sections = []
    for name, result in experiments:
        replay = result["replay"]
        points = result["static_points"]
        fmt = lambda v: "—" if v is None else f"{v:.2f} m"
        comparison.append(f"<tr><td>{replay['marker_size_mm']:g} mm</td>"
                          f"<td>{fmt(result['high_coverage_max_static_z_m'])}</td>"
                          f"<td>{fmt(result['strict_24px_high_coverage_max_static_z_m'])}</td>"
                          f"<td>{fmt(result['observed_max_static_z_m'])}</td></tr>")
        rows = []
        for p in points:
            selection = "自动静止" if p["selection"] == "automatic" else "画面复核停留"
            segment = "前段" if p["sequence"] == "primary" else "视角变化后"
            delta = p.get("previous_delta_z_cm")
            delta_text = "—" if delta is None else f"{delta:+.1f}"
            quality = "低覆盖，仅参考" if p["coverage"] < .5 else ("远处，门槛已放宽" if p["strict_24px_fraction"] < .8 else "通过 24 px 门槛")
            row_class = ' class="sparse"' if p["coverage"] < .5 else ""
            rows.append(f"<tr{row_class}><td>{p['index']}</td><td>{segment}<br>{selection}</td>"
                        f"<td><button onclick=\"seek('{name}',{p['start_s']})\">{p['start_s']:.1f}–{p['end_s']:.1f} s ▶</button></td>"
                        f"<td><b>{p['z_median_m']:.3f}</b></td><td>{p['range_median_m']:.3f}</td>"
                        f"<td>{p['z_std_mm']:.1f}</td><td>{p['coverage']:.0%}<br>{p['accepted_frames']}/{p['total_frames']} 帧</td>"
                        f"<td>{p['edge_median_px']:.1f}</td><td>{delta_text}</td><td>{quality}</td></tr>")
        first = points[:3]
        steps = "、".join(f"{p['previous_delta_z_cm']:.2f} cm" for p in first[1:])
        sections.append(f"""<section id="{name}"><h2>{replay['marker_size_mm']:g} mm marker</h2>
<p>完整重放 {replay['frames']} 帧，时长 {replay['duration_s']:.1f} s。前三个静止点的相邻深度变化：{steps}。
录制约 {result['parameters']['sequence_end_s']:g} s 出现遮挡与视角变化，表内分段处理；跨分段不计算位移。</p>
<p><a href="{name}/poses.csv">逐帧位姿 CSV</a> · <a href="{name}/frames.jsonl">逐帧检测与重复 ID 诊断</a> ·
<a href="{name}/evaluation/static_depths.csv">停留点 CSV</a> · <a href="{name}/evaluation/static_depths.json">完整评估 JSON</a></p>
<div class="scroll"><table><thead><tr><th>点</th><th>选取方式</th><th>时间 / 回放</th><th>深度 Z (m)</th><th>直线距离 (m)</th><th>Z 波动 σ (mm)</th><th>定位覆盖率</th><th>最短边 (px)</th><th>相邻 ΔZ (cm)</th><th>解释</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<img src="{name}/evaluation/depth_visibility.png" alt="深度、识别覆盖率和角点像素尺寸曲线">
<video id="video_{name}" controls preload="metadata" src="{name}/annotated.mp4"></video>
<details><summary>运行参数与来源</summary><pre>{html.escape(json.dumps(replay, ensure_ascii=False, indent=2))}</pre></details></section>""")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>1014 相机适配与 Marker 距离初评</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#182a39;background:#f4f6f8;margin:0;padding:28px}}
main{{max-width:1260px;margin:auto}}h1{{font-size:30px}}section,.intro{{background:white;padding:24px;border-radius:12px;margin:20px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}}
th{{background:#eaf1f6}}.scroll{{overflow:auto}}.sparse{{background:#fff3df}}img,video{{width:100%;margin-top:20px}}video{{max-height:650px;background:#111}}
a{{color:#176dad}}button{{cursor:pointer;border:0;border-radius:5px;background:#eaf1f6;padding:6px;color:#145a87}}pre{{font-size:12px;white-space:pre-wrap}}
.note{{border-left:4px solid #dda341;padding-left:16px}}@media(max-width:700px){{body{{padding:12px}}section,.intro{{padding:16px}}}}
</style></head><body><main><h1>1014 相机适配与 Marker 距离初评</h1>
<p>2026-10-14 测试数据 · 37.5 mm / 47.5 mm · cam0 前相机 · 单目视觉深度</p>
<div class="intro"><h2>已经跑通</h2><p>ROS2 CompressedImage → 原始 1280×960 图像 → ArUco 亚像素角点 → Kalibr equidistant 去畸变 →
现有 RigTracker 的 IPPE / PnP 和原图重投影门控。两组共重放 4650 帧，无需相机 SDK。cam1 后相机未录到图像；本次没有双目、IMU 或深度传感器融合。</p>
<p>下面“深度”是相机光学坐标系的 Z，即相机到 ID 2 中心的轴向距离；直线距离为 √(X²+Y²+Z²)。marker 尺寸按用户给定的黑框边长使用。
没有套用其他相机或不同尺寸 marker 的多面标定。位姿为未经显示平滑的单面观测。</p>
<table><thead><tr><th>Marker</th><th>有效帧 ≥80% 的最远停留点</th><th>且 ≥80% 观测满足 24 px</th><th>最远稀疏观测停留点</th></tr></thead><tbody>{''.join(comparison)}</tbody></table>
<p class="note">近距离约 0.33、0.53、0.74 m 的台阶清楚，前三点变化接近 20 cm。远处定位逐渐间歇。
表中的最远点仅代表这两段录制里成功估计过的距离，不能当成相机的精确最大量程，也不适合据此认定两种尺寸的极限优劣。</p>
<p>12 px 为本次距离探索门槛；原有 24 px 门槛的通过比例也保留。稀疏远点的深度只统计成功帧，丢失帧没有深度。
Z 波动含手持微动与定位噪声，不是绝对误差。未提供绝对距离真值，未拟合尺度，也未把每一段强行匹配为 20 cm。</p>
<p>背景误检会产生重复 ID：先筛去过小和贴边候选；唯一合格候选或面积至少为次候选 4 倍的候选可用；大小接近的重复候选拒绝。
此规则假设画面中每个 ID 只有一个真实 marker，所有重复候选角点与选择依据保存在 JSONL。</p></div>
{''.join(sections)}
<section><h2>复现与验收</h2><p>复现命令见 <a href="reproduce.sh">reproduce.sh</a>。输入原始录制与标定文件保留，生成结果位于 outputs/sensor_1014/。
自动候选以 0.8 s 窗口、20 mm/s 和至少 1.5 s 停留筛选；远点的复核窗口在脚本中明确列出。点击时间可跳转到对应视频片段。
视频使用名义 30 Hz，统计使用原始图像 header 时间戳；逐帧 CSV 同时保留 bag 存储时间。</p>
<p>模型依据：<a href="https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html">OpenCV fisheye 投影/去畸变</a>；
<a href="https://ternaris.gitlab.io/rosbags/index.html">Rosbags 离线读取</a>。代码验证记录见 <a href="validation.txt">validation.txt</a>。</p></section>
</main><script>function seek(name,t){{const v=document.getElementById('video_'+name);v.currentTime=t;v.scrollIntoView({{behavior:'smooth',block:'center'}});}}</script></body></html>"""
    (root / "report.html").write_text(document, encoding="utf-8")
    return root / "report.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

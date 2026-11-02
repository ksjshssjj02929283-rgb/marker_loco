#!/usr/bin/env python3
"""Build a left/right camera comparison from the October 30 laser evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_results(path: Path) -> dict[str, dict]:
    items = json.loads((path / "summary.json").read_text(encoding="utf-8"))["results"]
    return {(item["name"] if "name" in item else f"{item['marker']}/static"): item
            for item in items}


def make_rows(left: dict[str, dict], right: dict[str, dict]) -> list[dict]:
    rows = []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name, {}), right.get(name, {})
        if "pair_rmse_mm" in a or "pair_rmse_mm" in b:
            metric = "static_pair_distance"
            a_rmse, b_rmse = a.get("pair_rmse_mm"), b.get("pair_rmse_mm")
            a_coverage = f"{a.get('valid_points', 0)}/{a.get('available_points', 0)}"
            b_coverage = f"{b.get('valid_points', 0)}/{b.get('available_points', 0)}"
            a_offset = b_offset = None
            a_1s = b_1s = None
            a_5s = b_5s = None
            a_traj, b_traj = None, None
        else:
            metric = "dynamic"
            a_rmse = a.get("relative_all_pairs", {}).get("rmse_mm")
            b_rmse = b.get("relative_all_pairs", {}).get("rmse_mm")
            a_coverage = f"{a.get('paired_groups', 0)}/{a.get('gt_groups_eligible', 0)}"
            b_coverage = f"{b.get('paired_groups', 0)}/{b.get('gt_groups_eligible', 0)}"
            a_offset, b_offset = a.get("time_offset_s"), b.get("time_offset_s")
            a_traj, b_traj = a.get("trajectory_rmse_mm"), b.get("trajectory_rmse_mm")
            a_intervals = {entry["interval_s"]: entry.get("rmse_mm") for entry in a.get("intervals", [])}
            b_intervals = {entry["interval_s"]: entry.get("rmse_mm") for entry in b.get("intervals", [])}
            a_1s, b_1s = a_intervals.get(1.0), b_intervals.get(1.0)
            a_5s, b_5s = a_intervals.get(5.0), b_intervals.get(5.0)
        rows.append({
            "dataset": name,
            "metric": metric,
            "left_valid": a_coverage,
            "right_valid": b_coverage,
            "left_time_offset_s": a_offset,
            "right_time_offset_s": b_offset,
            "left_all_pair_rmse_mm": a_rmse,
            "right_all_pair_rmse_mm": b_rmse,
            "left_1s_rmse_mm": a_1s,
            "right_1s_rmse_mm": b_1s,
            "left_5s_rmse_mm": a_5s,
            "right_5s_rmse_mm": b_5s,
            "left_aligned_trajectory_rmse_mm": a_traj,
            "right_aligned_trajectory_rmse_mm": b_traj,
            "left_quality_flags": ";".join(a.get("quality_flags", [])),
            "right_quality_flags": ";".join(b.get("quality_flags", [])),
        })
    return rows


def write_comparison(root: Path, output: Path) -> None:
    left = load_results(root / "evaluation_left")
    right = load_results(root / "evaluation_right")
    rows = make_rows(left, right)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "left_right_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dynamic = [row for row in rows if row["metric"] == "dynamic"]
    static = [row for row in rows if row["metric"] == "static_pair_distance"]
    lines = [
        "# 2026-10-30 激光真值精度评估：左右相机交叉核对",
        "",
        "两路 UVC 图像分别用本地候选 camchain 回放。ID/黑码边长为 marker50: ID 2 / 37.5 mm，marker60: ID 2 / 50 mm，marker70: ID 6 / 60 mm。图像是单目已知尺寸 PnP，最短边门限 12 px、重投影误差门限 2.5 px；表中覆盖率是实际成功位姿与 GT 时间组配对数。",
        "",
        "激光给出的是靶点位置，相机给出的是 Tag 中心位置；两者固定外参没有独立测量。因此静态和动态的主结果使用两时刻位移长度误差，不做空间拟合。动态的对齐轨迹 RMSE 使用同一段数据拟合一个时移和固定尺度 SE(3)，是事后对齐残差，不能解释成独立绝对定位精度。GT 没有姿态，不能评估 6DoF 姿态精度。",
        "",
        "动态时移遵循 `laser_time = camera_header_time + offset`。两路、各数据段估出的 offset 接近，说明相机与激光时钟之间的约 10.3 秒差值在本批采集里较稳定；匹配仍只在有效位姿帧之间插值，且最大间隔 0.1 秒。",
        "",
        "## 静态点位间距",
        "",
        "静态每个距离只有一条激光真值，故按 9 个点形成的 36 组距离比较；数值为 36 组误差的 RMSE。",
        "",
        "| Marker | 左路 RMSE (mm) | 右路 RMSE (mm) | 左路相邻 10 cm 误差 (mm) | 右路相邻 10 cm 误差 (mm) |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for marker in ("marker50/static", "marker60/static"):
        a, b = left.get(marker, {}), right.get(marker, {})
        if not a and not b:
            continue
        def adjacent(summary_dir: Path, label: str) -> str:
            path = summary_dir / label / "static_pairs.csv"
            if not path.is_file():
                return "—"
            with path.open(newline="", encoding="utf-8") as stream:
                values = [float(row["error_mm"]) for row in csv.DictReader(stream)
                          if int(row["to_cm"]) - int(row["from_cm"]) == 10]
            return ", ".join(f"{value:+.1f}" for value in values)
        label = marker.split("/")[0]
        lines.append(f"| {label} | {a.get('pair_rmse_mm', float('nan')):.2f} | {b.get('pair_rmse_mm', float('nan')):.2f} | {adjacent(root / 'evaluation_left', label)} | {adjacent(root / 'evaluation_right', label)} |")

    lines += [
        "",
        "marker50 左路 90→100 cm 段误差约 +40.7 mm，而右路约 −6.1 mm；这个单点差异不应被总体 RMSE 掩盖，建议作为远距离稳定性问题复查。其余 marker50 相邻段多数约 −3 到 −12 mm。marker60 两路相邻误差大多在 −15 到 +2 mm。",
        "",
        "## 动态轨迹",
        "",
        "| 数据 | 配对组 左/右 | 时移 左/右 (s) | 全部点对位移 RMSE 左/右 (mm) | 1 秒位移 RMSE 左/右 (mm) | 5 秒位移 RMSE 左/右 (mm) | 拟合轨迹 RMSE 左/右 (mm) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dynamic:
        name = row["dataset"]
        def fmt(v): return "—" if v is None else f"{float(v):.2f}"
        lines.append("| {} | {} / {} | {} / {} | {} / {} | {} / {} | {} / {} | {} / {} |".format(
            name, row["left_valid"], row["right_valid"],
            fmt(row["left_time_offset_s"]), fmt(row["right_time_offset_s"]),
            fmt(row["left_all_pair_rmse_mm"]), fmt(row["right_all_pair_rmse_mm"]),
            fmt(row["left_1s_rmse_mm"]), fmt(row["right_1s_rmse_mm"]),
            fmt(row["left_5s_rmse_mm"]), fmt(row["right_5s_rmse_mm"]),
            fmt(row["left_aligned_trajectory_rmse_mm"]), fmt(row["right_aligned_trajectory_rmse_mm"])))
    lines += [
        "",
        "marker60 连续静止序列存在同 ID 的多个候选轮廓，回放器按歧义策略拒绝了一部分帧：左路有效 266/984、右路 377/984 个可用 GT 组，结果覆盖较低，不能与完整轨迹的指标等权解读。marker70 连续序列覆盖也只有约 69%～75%。每段的详细重投影分布、严格 24 px 覆盖率、GT 组内离散度及质量标记在左右 evaluation 子目录的 `summary.json` 中。",
        "",
        "## 复核文件",
        "",
        "- 左路详细报告：`evaluation_left/report.md`",
        "- 右路详细报告：`evaluation_right/report.md`",
        "- 逐数据集左右对照：`left_right_comparison.csv`",
        "- 1 秒位移误差图：`dynamic_1s_error.png`",
        "",
        "回放与评估使用的相机内参是候选配置，尚未用本次 UVC 相机的独立标定数据确认。因此本结果适合作为本批实验的相对精度评估和复查依据，不应作为最终经标定的绝对精度声明。",
        "",
    ]
    (output / "dual_camera_report.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    labels = [row["dataset"].replace("/", "\n") for row in dynamic]
    x = list(range(len(dynamic)))
    width = 0.38
    left_values = [float(row["left_1s_rmse_mm"] or 0) for row in dynamic]
    right_values = [float(row["right_1s_rmse_mm"] or 0) for row in dynamic]
    ax.bar([value - width / 2 for value in x], left_values, width, label="左相机")
    ax.bar([value + width / 2 for value in x], right_values, width, label="右相机")
    ax.set_ylabel("1 秒位移长度误差 RMSE (mm)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output / "dynamic_1s_error.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/accuracy_20261030"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    write_comparison(args.root, args.output_dir or args.root)
    print(f"comparison report: {(args.output_dir or args.root) / 'dual_camera_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create reader-first plots for the two-layer Feishu accuracy report."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = Path("outputs/accuracy_20261030_kalibr")
DETAIL = ROOT / "details"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})


def rows(name):
    with (DETAIL / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


jitter = rows("static_pose_jitter.csv")
pairs = rows("static_pair_errors_all.csv")
intervals = rows("dynamic_interval_metrics.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


# Condition matrix: farthest tested nominal distance for each static criterion.
fig, ax = plt.subplots(figsize=(10.5, 4.8))
labels = ["marker50 左", "marker50 右", "marker60 左", "marker60 右"]
values = np.array([[70, 60, 40, 40], [80, 70, 50, 40], [90, 80, 50, 40], [80, 80, 60, 50]])
im = ax.imshow(values, cmap="YlGn", vmin=20, vmax=100, aspect="auto")
ax.set_xticks(range(4), ["静止 RMS\n≤3 mm", "静止 P95\n≤3 mm", "点对 RMSE\n≤3 mm", "窗口内每点\n绝对误差≤3 mm"])
ax.set_yticks(range(4), labels)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"20–{values[i, j]} cm", ha="center", va="center", fontsize=11,
                color="black" if values[i, j] < 75 else "white", fontweight="bold")
ax.set_title("3 mm 静态条件速览：从 20 cm 起的最远已测达标窗口")
ax.set_xlabel("判据；范围只代表本批每 10 cm 离散点")
fig.colorbar(im, ax=ax, label="最远端点（cm）", shrink=.8)
save(fig, "two_layer_static_condition_matrix")


# Per-segment error and endpoint jitter.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
segments = [f"{x}–{x+10}" for x in range(20, 100, 10)]
xs = np.arange(len(segments))
colors = {"marker50 left": "#2563eb", "marker50 right": "#f97316",
          "marker60 left": "#16a34a", "marker60 right": "#9333ea"}
for marker in ("marker50", "marker60"):
    for side in ("left", "right"):
        key = marker + " " + side
        vals = [float(next(r["error_mm"] for r in pairs if r["marker"] == marker and r["camera"] == side
                           and int(r["from_cm"]) == x and int(r["to_cm"]) == x + 10)) for x in range(20, 100, 10)]
        ax1.plot(xs, vals, marker="o", label=key.replace("left", "左").replace("right", "右"), color=colors[key])
        end = [float(next(r["jitter_rms_mm"] for r in jitter if r["marker"] == marker and r["camera"] == side
                          and int(r["distance_cm"]) == x + 10)) for x in range(20, 100, 10)]
        ax2.plot(xs, end, marker="o", label=key.replace("left", "左").replace("right", "右"), color=colors[key])
ax1.axhspan(-3, 3, color="#16a34a", alpha=.08, label="±3 mm 区间")
ax1.axhline(3, color="#111827", linestyle=":", label="3 mm 参考线")
ax1.axhline(-3, color="#111827", linestyle=":")
ax2.axhline(3, color="#111827", linestyle=":", label="3 mm 参考线")
ax1.set_title("逐段静态点间距误差")
ax1.set_ylabel("相机距离 − 激光距离（mm）")
ax2.set_title("每段终点静止抖动 RMS")
ax2.set_ylabel("径向抖动 RMS（mm）")
for ax in (ax1, ax2):
    ax.set_xticks(xs, segments, rotation=35)
    ax.grid(axis="y", alpha=.25)
ax1.legend(fontsize=8, ncol=2)
save(fig, "two_layer_static_segment_detail")


# Compact dynamic overview: pass count / available sequence-camera pairs.
intervals_s = (.5, 1., 2., 5.)
markers = ("marker50", "marker60", "marker70")
pass_pct, labels = [], []
for marker in markers:
    p_row, label_row = [], []
    for dt in intervals_s:
        rr = [r for r in intervals if r["marker"] == marker and float(r["interval_s"]) == dt]
        good = sum(float(r["error_rmse_mm"]) <= 3 for r in rr)
        p_row.append(100 * good / len(rr) if rr else 0)
        label_row.append(f"{good}/{len(rr)}")
    pass_pct.append(p_row)
    labels.append(label_row)
fig, ax = plt.subplots(figsize=(9.5, 4.0))
im = ax.imshow(np.asarray(pass_pct), cmap="YlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(4), ["0.5 s", "1 s", "2 s", "5 s"])
ax.set_yticks(range(3), [m.replace("marker", "marker ") for m in markers])
for i in range(3):
    for j in range(4):
        ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=13,
                color="white" if pass_pct[i][j] >= 60 else "black", fontweight="bold")
ax.set_title("动态位移 RMSE≤3 mm：达标序列数 / 可评估序列数")
ax.set_xlabel("位移时间间隔（DT、DJ）")
fig.colorbar(im, ax=ax, label="达标比例（%）", shrink=.85)
save(fig, "two_layer_dynamic_summary")

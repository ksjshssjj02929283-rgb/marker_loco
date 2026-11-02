#!/usr/bin/env python3
"""Render accuracy charts from recorded D455 sessions (no camera needed).

Two figures, saved under outputs/accuracy_test/:

  sigma_vs_distance.png   — static repeatability (std of camera-frame
                            translation) of the reference face vs distance.
  displacement_error.png  — known Z displacement: cumulative measured-vs-true,
                            plus per-step error for the 50 mm increments.

Reads poses.csv directly; the reference face is tag_id 2 (fallback: any face).
Usage:
    .venv/bin/python scripts/plot_accuracy.py
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/accuracy_test")


def latest_session(scene: str) -> Path:
    dirs = sorted(glob.glob(f"captures/d455_aruco/{scene}/2*"))
    if not dirs:
        raise SystemExit(f"no session found for {scene}")
    return Path(dirs[-1])


def face_series(session: Path, tag_id: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return (tvec [N,3] in mm, reproj_err [N]) for the given face."""
    rows = list(csv.DictReader(open(session / "poses.csv", encoding="utf-8")))
    rows = [r for r in rows if int(r["tag_id"]) == tag_id] or rows
    t = np.array([[float(r[a]) for a in ("tx_m", "ty_m", "tz_m")] for r in rows]) * 1000
    e = np.array([float(r["reprojection_error_px"]) for r in rows])
    return t, e


def static_chart():
    scenes = ["static_d300", "static_d500", "static_d800", "static_d1000"]
    dist, sigma = [], []
    for sc in scenes:
        sess = latest_session(sc)
        t, _ = face_series(sess)
        dist.append(t[:, 2].mean())
        sigma.append(t.std(axis=0))
    dist = np.array(dist)
    sigma = np.array(sigma)  # [N, 3] tx/ty/tz

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=130)
    for i, label in enumerate(("tx", "ty", "tz")):
        ax.plot(dist, sigma[:, i], "o-", label=label, linewidth=1.6, markersize=5)
    ax.axhline(5.0, color="tab:red", ls="--", lw=1, alpha=0.7)
    ax.text(dist[-1], 5.0 + 0.15, "acceptance: σ ≤ 5 mm @ 1.0 m", color="tab:red",
            fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("distance (mm)")
    ax.set_ylabel("std of translation (mm)")
    ax.set_title("Static repeatability vs distance (reference face 2)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sigma_vs_distance.png")
    plt.close(fig)


def displacement_chart():
    scenes = ["zdisp_0mm", "zdisp_50mm", "zdisp_100mm", "zdisp_150mm", "zdisp_200mm"]
    P = []
    for sc in scenes:
        t, _ = face_series(latest_session(sc))
        P.append(t.mean(axis=0))
    P = np.array(P)  # [5, 3] mm

    # Fit a 3D line (tape direction) to remove the tape-vs-axis tilt.
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c)
    d = Vt[0]
    if d[2] < 0:
        d = -d
    s = (P - c) @ d
    s -= s[0]                       # cumulative along-track (mm), origin at P0

    true_cum = np.arange(5) * 50.0
    step_err = np.diff(s) - 50.0    # per-step error vs 50 mm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=130)

    ax1.plot(true_cum, true_cum, "k--", lw=1, label="true")
    ax1.plot(true_cum, s, "o-", lw=1.6, label="measured", markersize=5)
    ax1.set_xlabel("true displacement (mm)")
    ax1.set_ylabel("measured along-track (mm)")
    ax1.set_title("Known Z displacement: 200 mm total")
    ax1.legend()
    ax1.grid(alpha=0.3)

    colors = ["tab:green" if abs(e) <= 1 else "tab:orange" if abs(e) <= 3 else "tab:red"
              for e in step_err]
    ax2.bar(np.arange(4) + 1, step_err, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="k", lw=0.8)
    for i, e in enumerate(step_err):
        ax2.text(i + 1, e + (0.06 if e >= 0 else -0.06), f"{e:+.2f}",
                 ha="center", va="bottom" if e >= 0 else "top", fontsize=8)
    ax2.set_xticks(np.arange(4) + 1)
    ax2.set_xticklabels(["0→50", "50→100", "100→150", "150→200"])
    ax2.set_ylabel("step error (mm)")
    ax2.set_title("Per-50 mm-step error")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Total error = {s[-1] - 200:+.2f} mm  ({100 * (s[-1] - 200) / 200:+.2f}%)",
                 fontsize=9, color="tab:blue")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "displacement_error.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    static_chart()
    displacement_chart()
    print(f"wrote {OUT / 'sigma_vs_distance.png'}")
    print(f"wrote {OUT / 'displacement_error.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

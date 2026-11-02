#!/usr/bin/env python3
"""Static-noise baseline for recorded D455/T265 sessions.

For each static session's poses.csv, group single-face poses by tag_id and report
the per-axis standard deviation of the camera-frame translation (repeatability /
precision). This is a noise baseline, NOT absolute accuracy — it cannot reveal a
systematic bias in intrinsics or marker size.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


AXES = ("tx_m", "ty_m", "tz_m")


def analyze_session(session: Path) -> dict:
    poses = session / "poses.csv"
    if not poses.exists():
        return {"session": str(session), "error": "no poses.csv"}

    rows = list(csv.DictReader(poses.open(newline="", encoding="utf-8")))
    by_tag: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_tag[int(row["tag_id"])].append(row)

    tags = {}
    for tag_id, tag_rows in sorted(by_tag.items()):
        pos = np.array([[float(r[a]) for a in AXES] for r in tag_rows])  # meters
        err = np.array([float(r["reprojection_error_px"]) for r in tag_rows])
        # Trim the first ~0.5 s of a session if it starts moving; keep all frames otherwise.
        tags[str(tag_id)] = {
            "frames": len(tag_rows),
            "mean_mm": {a: float(np.mean(pos[:, i]) * 1000) for i, a in enumerate(AXES)},
            "std_mm": {a: float(np.std(pos[:, i]) * 1000) for i, a in enumerate(AXES)},
            "std_norm_mm": float(np.std(np.linalg.norm(pos, axis=1)) * 1000),
            "median_reproj_px": float(np.median(err)),
            "p95_reproj_px": float(np.percentile(err, 95)),
        }

    # Pooled per-axis std across tags (weighted by frames).
    pooled = {}
    for i, a in enumerate(AXES):
        vals = np.concatenate(
            [np.array([float(r[a]) for r in by_tag[t]]) for t in by_tag]
        )
        pooled[a] = float(np.std(vals) * 1000)

    return {"session": str(session), "pooled_std_mm": pooled, "tags": tags}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", type=Path, help="Session dirs (or globs)")
    parser.add_argument("--output", type=Path, default=Path("outputs/accuracy_test/static_noise.json"))
    args = parser.parse_args()

    sessions = args.sessions
    if not sessions:
        sessions = sorted(Path("captures/d455_aruco/static").glob("*/"))

    results = [analyze_session(s) for s in sessions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Compact table: per-tag std is the true noise metric. Pooling raw positions
    # across tags mixes in the fixed inter-face offset, so do not report it.
    print(f"{'session':24} {'tag':4} {'frames':7} {'std tx/ty/tz (mm)':20} {'reproj p50/p95 (px)'}")
    for r in results:
        if "error" in r:
            print(f"{r['session']:24}  ERROR {r['error']}")
            continue
        for tag_id, t in r["tags"].items():
            s = t["std_mm"]
            print(
                f"{Path(r['session']).name:24} "
                f"{tag_id:4} {t['frames']:7} "
                f"{s['tx_m']:6.2f}/{s['ty_m']:6.2f}/{s['tz_m']:6.2f}     "
                f"{t['median_reproj_px']:.2f}/{t['p95_reproj_px']:.2f}"
            )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

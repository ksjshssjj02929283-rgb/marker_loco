#!/usr/bin/env python3
"""Evaluate stereo joint dynamic trajectories using the prior laser protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_laser_20261030 import MARKERS, evaluate_dynamic
from evaluate_relative_translation_rpe import score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=Path("/home/dex/20261030/GT"))
    args = parser.parse_args()
    results = []
    for marker, spec in MARKERS.items():
        names = [p.name for p in sorted((args.replay_root / marker).glob("*")) if p.is_dir()]
        for name in names:
            gt_file = (spec["continuous"] if name == "continuous_static_20_100"
                       else f"track{int(name.rsplit('_', 1)[1])}.txt" if marker != "marker70"
                       else "track.txt")
            gt_path = args.gt_root / spec["gt_dir"] / gt_file
            out = args.output_root / marker / name
            result = evaluate_dynamic(marker, name, args.replay_root, gt_path, out, {},
                                      max_gap_s=0.1, min_coverage=0.65,
                                      min_motion_fraction=0.75, scan_step_s=0.05,
                                      scan_margin_s=20.0, max_group_spread_mm=60.0)
            if result["status"] == "ok" and (out / "cross_validated_trajectories.csv").exists():
                rpe = score(out, interval_s=0.1)
                (out / "relative_translation_rpe_0.1s.json").write_text(
                    json.dumps(rpe, ensure_ascii=False, indent=2) + "\n")
                result["rpe_0p1s"] = rpe
            print(json.dumps({"case": result["name"], "status": result["status"],
                              "rpe_mae_mm": result.get("rpe_0p1s", {}).get("mae_mm")}), flush=True)
            results.append(result)
    (args.output_root / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

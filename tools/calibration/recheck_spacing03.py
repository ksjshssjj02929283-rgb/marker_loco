"""Reproduce the corrected spacing=0.3 calibration's historical evaluations."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


PROJECT = Path(__file__).resolve().parents[2]
CAMCHAIN = PROJECT / "outputs/d455_aprilgrid_1020_spacing03_refined/camchain.yaml"
MARKER = PROJECT / "outputs/d455_aprilgrid_1020_spacing03_refined_ab"
LASER = PROJECT / "outputs/laser_spacing03_refined_20261020"


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["marker", "laser", "laser-analyze"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--method", choices=["opencv", "kalibr"], default="opencv")
    args = parser.parse_args()
    global CAMCHAIN, MARKER, LASER
    if args.method == "kalibr":
        CAMCHAIN = PROJECT / "kalibr_d455/camchain-spacing03.yaml"
        MARKER = PROJECT / "outputs/d455_aprilgrid_1020_spacing03_kalibr_ab"
        LASER = PROJECT / "outputs/laser_spacing03_kalibr_20261020"
    if args.mode == "marker":
        previous = PROJECT / "outputs/marker_intrinsics_recheck_20261020"
        manifest = json.loads((previous / "run_manifest.json").read_text())
        manifest.pop("requested_intrinsics_rounded", None)
        manifest["camchain_source"] = str(CAMCHAIN)
        manifest["camchain_sha256"] = hashlib.sha256(CAMCHAIN.read_bytes()).hexdigest()
        manifest["board"] = dict(tagCols=6, tagRows=6, tagSize=.10, tagSpacing=.3)
        shutil.copy2(CAMCHAIN, MARKER / "camchain.yaml")
        (MARKER / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        analysis = module(previous / "analyze_run.py")
        result, trajectory = analysis.analyze(MARKER, PROJECT / "outputs/d455_aprilgrid_1019_ab")
        analysis.figures(MARKER, result, trajectory)
        # The historical report generator contains prose for spacing=.5;
        # deliberately reuse only its paired-observation analysis and plots.
        print(json.dumps({"consistent": all(c["passed"] for c in result["consistency"].values()),
                          "metrics": {m: {k: v["metrics_excluding_anchor"] for k, v in d.items()
                                          if k != "static"} for m, d in result["models"].items()}}, indent=2))
    elif args.mode == "laser":
        runner = module(PROJECT / "outputs/laser_intrinsics_recheck_20261020/run_laser.py")
        LASER.mkdir(parents=True, exist_ok=True)
        runner.OUT, runner.CAMCHAIN = LASER, CAMCHAIN
        sys.argv = [sys.argv[0], "--workers", str(args.workers)]
        runner.main()
    else:
        analysis = module(PROJECT / "outputs/laser_intrinsics_recheck_20261020/analyze_laser.py")
        # Preserve all numerical verification; omit the legacy .5 prose.
        analysis.build_report = lambda *unused: None
        sys.argv = [sys.argv[0], "--output-root", str(LASER)]
        analysis.main()


if __name__ == "__main__":
    main()

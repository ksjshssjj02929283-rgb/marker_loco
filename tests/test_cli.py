"""Regression checks for public CLI behavior after moving scripts into the package."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_python(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("adapter", ["record_d455", "record_t265", "live_dashboard"])
def test_hardware_cli_help_without_realsense(adapter: str, tmp_path: Path):
    program = (
        "import sys; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "sys.modules['pyrealsense2'] = None; "
        "from marker6dof import cli; "
        f"sys.argv = [{adapter!r}, '--help']; "
        f"sys.exit(getattr(cli, {adapter!r})())"
    )
    result = run_python(["-I", "-c", program], tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--help" in result.stdout


@pytest.mark.parametrize("script", ["run_video.py", "generate_demo.py"])
def test_legacy_script_runs_from_another_directory(script: str, tmp_path: Path):
    result = run_python([str(ROOT / "scripts" / script), "--help"], tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--config" in result.stdout


def test_original_cli_preserves_returned_exit_status(tmp_path: Path):
    # A console-script wrapper must preserve the command's integer failure status.
    program = (
        "import sys; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "sys.modules['pyrealsense2'] = None; "
        "from marker6dof import cli, d455; "
        "d455.main = lambda: 7; "
        "sys.exit(cli.record_d455())"
    )
    result = run_python(["-I", "-c", program], tmp_path)

    assert result.returncode == 7, result.stdout + result.stderr


def test_legacy_evaluator_reports_failure_with_nonzero_exit(tmp_path: Path):
    truth_path = tmp_path / "truth.csv"
    estimate_path = tmp_path / "estimates.csv"
    pose = {"frame_index": 0, "tx_m": 0, "ty_m": 0, "tz_m": 1,
            "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    for filename, row in (
        (truth_path, pose),
        (estimate_path, {**pose, "status": "MEASURED", "reason": "ok"}),
    ):
        with filename.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    output = tmp_path / "evaluation.json"

    result = run_python(
        [str(ROOT / "scripts" / "evaluate_demo.py"),
         "--ground-truth", str(truth_path), "--estimates", str(estimate_path),
         "--output", str(output), "--max-position-mm", "-1"],
        tmp_path,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False

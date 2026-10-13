"""Small installed-command adapters for the repository scripts."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _run(script: str) -> None:
    root = Path(__file__).resolve().parents[2]
    sys.argv[0] = str(root / "scripts" / script)
    runpy.run_path(sys.argv[0], run_name="__main__")


def run_video() -> None:
    _run("run_video.py")


def build_relative() -> None:
    _run("build_relative_trajectory.py")


def track_d455() -> None:
    _run("track_d455.py")

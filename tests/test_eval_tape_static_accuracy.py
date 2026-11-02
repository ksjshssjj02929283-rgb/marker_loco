from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_tape_static_accuracy.py"


def write_session(root: Path, name: str, z_m: float) -> Path:
    session = root / name
    session.mkdir()
    with (session / "poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag_id", "tx_m", "ty_m", "tz_m", "reprojection_error_px"])
        writer.writeheader()
        for offset in (-0.0005, 0.0, 0.0005):
            writer.writerow({"tag_id": 2, "tx_m": 0.0, "ty_m": 0.0, "tz_m": z_m + offset, "reprojection_error_px": 0.2})
    return session


def test_tape_static_accuracy_reports_raw_and_baseline_corrected_errors(tmp_path: Path):
    near = write_session(tmp_path, "near", 0.205)  # +5 mm mounting/reference offset
    far = write_session(tmp_path, "far", 0.815)  # +15 mm raw, +10 mm after zeroing at 20 cm
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--point", f"20=20={near}",
            "--point", f"80=80={far}",
            "--output", str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["baseline_raw_error_mm"] == 5.0
    assert result["points"][1]["raw_error_mm"] == 15.0
    assert result["points"][1]["baseline_corrected_error_mm"] == 10.0
    assert "zeroed err" in completed.stdout

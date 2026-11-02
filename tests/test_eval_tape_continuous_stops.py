from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_tape_continuous_stops.py"


def test_extracts_static_platforms_in_recording_order(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    with (session / "poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "host_timestamp_s", "tag_id", "tx_m", "ty_m", "tz_m", "reprojection_error_px"])
        writer.writeheader()
        frame = 0
        for z in (0.255, 0.355, 0.465):
            for _ in range(120):  # 4 seconds at 30 Hz
                writer.writerow({"frame_index": frame, "host_timestamp_s": frame / 30, "tag_id": 3, "tx_m": 0, "ty_m": 0, "tz_m": z, "reprojection_error_px": 0.2})
                frame += 1
            for fraction in range(10):
                writer.writerow({"frame_index": frame, "host_timestamp_s": frame / 30, "tag_id": 3, "tx_m": 0, "ty_m": 0, "tz_m": z + fraction * 0.01, "reprojection_error_px": 0.2})
                frame += 1
    output = tmp_path / "result.json"
    subprocess.run([sys.executable, str(SCRIPT), "--session", str(session), "--distances-cm", "30,40,50", "--output", str(output)], check=True)
    result = json.loads(output.read_text())
    assert [point["tape_distance_mm"] for point in result["points"]] == [300, 400, 500]
    assert len(result["points"]) == 3

"""Replay the three task recordings with only the T265 left fisheye."""

from pathlib import Path

from marker6dof.commands.replay.replay_dual_t265 import replay
from marker6dof.commands.tracking.live_dual_t265 import load_body


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("01_open_lid_153305", "01_open_lid", "20261102_153305", 743),
    ("02_fold_item_153420", "02_fold_item", "20261102_153420", 708),
    ("03_tidy_table_153506", "03_tidy_table", "20261102_153506", 876),
)


def main() -> None:
    bodies = {
        "left": load_body(ROOT / "configs/local/t265_left_5x5_54p6_rig.json", "left"),
        "right": load_body(ROOT / "configs/local/t265_right_6x6_56_rig.json", "right"),
    }
    for name, task, stamp, frames in TASKS:
        output = ROOT / "outputs/t265_stages/01_monocular_static" / name
        if output.exists():
            print(f"already exists: {output}")
            continue
        session = ROOT / "captures/t265_tasks" / task / stamp
        summary = replay(session, output, bodies, max_frames=frames, eyes=("left",))
        print(f"{name}: {summary['frames']} frames, "
              f"left={summary['left_valid_frames']}, right={summary['right_valid_frames']}")


if __name__ == "__main__":
    main()

"""Reject local data and oversized blobs from the Git index."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_BLOB_BYTES = 10 * 1024 * 1024
LOCAL_ROOTS = {
    "captures", "outputs", "local", "graphify-out",
    "build", "dist", ".analysis_upstreams", "ego_umi_slam",
    ".aws", ".codex",
}
RECORDING_SUFFIXES = {
    ".bag", ".db3", ".mcap", ".mp4", ".avi", ".mkv",
    ".larkcache.linco", ".log",
}


def git(*args: str, input_data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        input=input_data, capture_output=True, check=True,
    ).stdout


def main() -> int:
    entries = []
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_id, stage = metadata.split()
        if stage != b"0":
            print(f"Unmerged index entry: {raw_path!r}", file=sys.stderr)
            return 1
        entries.append((mode, object_id, raw_path.decode("utf-8", "surrogateescape")))

    problems = []
    for mode, _, path in entries:
        parts = Path(path).parts
        suffix = path.lower()
        if mode == b"160000":
            problems.append(f"nested repository/submodule: {path}")
        if (parts[0] in LOCAL_ROOTS or parts[0] == ".venv"
                or parts[0].startswith(".venv-")
                or parts[:2] == ("configs", "local")
                or path == ".claude/settings.local.json"):
            problems.append(f"local-only path: {path}")
        if suffix.endswith(tuple(RECORDING_SUFFIXES)):
            problems.append(f"recording or runtime file: {path}")

    if entries:
        query = b"".join(object_id + b"\n" for _, object_id, _ in entries)
        info = git("cat-file", "--batch-check=%(objecttype) %(objectsize)", input_data=query)
        for (_, _, path), line in zip(entries, info.splitlines(), strict=True):
            object_type, size = line.split()
            if object_type == b"blob" and int(size) > MAX_BLOB_BYTES:
                problems.append(f"blob exceeds 10 MiB ({int(size)} bytes): {path}")

    if problems:
        print("Git index hygiene check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"Git index hygiene check passed ({len(entries)} paths, 10 MiB limit).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Git index hygiene check could not run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

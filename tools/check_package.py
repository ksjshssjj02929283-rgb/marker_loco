#!/usr/bin/env python3
"""Build and exercise a wheel outside the checkout without downloading packages.

The temporary environment borrows dependency directories from this interpreter.
Its marker6dof package and console scripts must come from the wheel installation.
Run after installing the project's development dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import venv


ROOT = Path(__file__).resolve().parents[1]
DEMO_FRAMES = 12

INSTALL_AUDIT = r'''
import json
from importlib import metadata, resources
from pathlib import Path
import sys

import marker6dof

environment = Path(sys.prefix).resolve()
module = Path(marker6dof.__file__).resolve()
distribution = metadata.distribution("marker-6dof-tracking")
distribution_root = Path(distribution.locate_file("")).resolve()
if not module.is_relative_to(environment):
    raise RuntimeError(f"Imported marker6dof outside the wheel environment: {module}")
if not distribution_root.is_relative_to(environment):
    raise RuntimeError(f"Found inherited package metadata: {distribution_root}")
for filename in ("live_dashboard.html", "session_review.html"):
    content = resources.files("marker6dof.web").joinpath(filename).read_text(encoding="utf-8")
    if "<html" not in content.lower():
        raise RuntimeError(f"Missing or invalid installed HTML resource: {filename}")
entry_points = sorted(
    ({"name": entry.name, "value": entry.value}
     for entry in distribution.entry_points if entry.group == "console_scripts"),
    key=lambda entry: entry["name"],
)
if not entry_points:
    raise RuntimeError("The wheel installed no console scripts")
print(json.dumps({"module": str(module), "entry_points": entry_points}))
'''

# Exercise generated console scripts with the optional camera SDK unavailable.
HELP_WITHOUT_SDK = r'''
import runpy
import sys

script = sys.argv[1]
sys.modules["pyrealsense2"] = None
sys.argv = [script, "--help"]
runpy.run_path(script, run_name="__main__")
'''


def run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    if (result.returncode == 0) != expect_success:
        expected = "zero" if expect_success else "nonzero"
        raise RuntimeError(
            f"Expected {expected} exit status, got {result.returncode}: {command!r}\n"
            f"{result.stdout}"
        )
    return result


def main() -> int:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")

    with tempfile.TemporaryDirectory(prefix="marker6dof-package-") as temporary:
        working = Path(temporary).resolve()
        if working.is_relative_to(ROOT):
            raise RuntimeError("Package checks must run outside the checkout; change TMPDIR")

        wheels = working / "wheels"
        wheels.mkdir()
        print("Building wheel without network access...", flush=True)
        run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--no-index", "--wheel-dir", str(wheels), str(ROOT)],
            cwd=working,
            environment=environment,
        )
        built = list(wheels.glob("marker_6dof_tracking-*.whl"))
        if len(built) != 1:
            raise RuntimeError(f"Expected one project wheel, found {built}")

        isolated = working / "environment"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(isolated)
        executable_dir = isolated / ("Scripts" if os.name == "nt" else "bin")
        python = executable_dir / ("python.exe" if os.name == "nt" else "python")
        site_packages = Path(run(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            cwd=working,
            environment=environment,
        ).stdout.strip())
        dependency_paths = sorted({
            sysconfig.get_path(key) for key in ("purelib", "platlib")
            if sysconfig.get_path(key) and Path(sysconfig.get_path(key)).is_dir()
        })
        # Plain paths borrow libraries without executing the parent environment's
        # editable-install .pth files. Never modify or uninstall parent packages.
        (site_packages / "borrowed_dependencies.pth").write_text(
            "\n".join(dependency_paths) + "\n", encoding="utf-8"
        )
        run(
            [str(python), "-I", "-m", "pip", "install", "--no-deps", "--no-index",
             "--ignore-installed", str(built[0])],
            cwd=working,
            environment=environment,
        )
        audit = json.loads(run(
            [str(python), "-I", "-c", INSTALL_AUDIT],
            cwd=working,
            environment=environment,
        ).stdout)
        print(f"Wheel import verified in temporary environment: {audit['module']}", flush=True)
        print("Only dependency directories are borrowed; installed HTML resources verified.", flush=True)

        checked = 0
        for entry in audit["entry_points"]:
            if ".commands.reporting." in entry["value"]:
                continue  # Reporting depends on its separately installed extra.
            script = executable_dir / (entry["name"] + (".exe" if os.name == "nt" else ""))
            if not script.is_file():
                raise RuntimeError(f"Missing installed console script: {script}")
            if os.name == "nt":
                # Windows console scripts are executable launchers, not Python files.
                help_command = [str(python), "-I", "-c",
                    "import sys; from importlib.metadata import distribution; "
                    "sys.modules['pyrealsense2'] = None; "
                    "entry = next(e for e in distribution('marker-6dof-tracking').entry_points "
                    "if e.group == 'console_scripts' and e.name == sys.argv[1]); "
                    "sys.argv = [entry.name, '--help']; sys.exit(entry.load()())", entry["name"]]
            else:
                help_command = [str(python), "-I", "-c", HELP_WITHOUT_SDK, str(script)]
            help_result = run(help_command, cwd=working, environment=environment)
            if "usage:" not in help_result.stdout.lower():
                raise RuntimeError(f"Console script did not print help: {entry['name']}")
            checked += 1
        print(f"Verified --help for {checked} console scripts with the camera SDK disabled.", flush=True)

        def installed(name: str, *arguments: str) -> list[str]:
            script = executable_dir / (name + (".exe" if os.name == "nt" else ""))
            if not script.is_file():
                raise RuntimeError(f"Missing required demo command: {name}")
            return [str(script), *arguments] if os.name == "nt" else [str(python), "-I", str(script), *arguments]

        generated = working / "generated"
        estimates = working / "estimates"
        evaluation = working / "evaluation.json"
        config = str(ROOT / "configs" / "demo_calibration.json")
        run(
            installed("marker-generate-demo", "--config", config, "--frames", str(DEMO_FRAMES),
                      "--output-dir", str(generated)),
            cwd=working,
            environment=environment,
        )
        run(
            installed("marker-run-video", "--config", config, "--video", str(generated / "demo_input.avi"),
                      "--output-dir", str(estimates), "--no-annotated-video"),
            cwd=working,
            environment=environment,
        )
        evaluation_arguments = [
            "--ground-truth", str(generated / "ground_truth.csv"),
            "--estimates", str(estimates / "tcp_poses.csv"),
        ]
        run(
            installed("marker-evaluate-demo", *evaluation_arguments, "--output", str(evaluation)),
            cwd=working,
            environment=environment,
        )
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        if report.get("passed") is not True or report.get("frames") != DEMO_FRAMES:
            raise RuntimeError(f"Installed demo pipeline failed its accuracy check: {report}")

        rejected = working / "rejected.json"
        run(
            installed("marker-evaluate-demo", *evaluation_arguments, "--output", str(rejected),
                      "--max-position-mm", "-1"),
            cwd=working,
            environment=environment,
            expect_success=False,
        )
        if json.loads(rejected.read_text(encoding="utf-8")).get("passed") is not False:
            raise RuntimeError("The deliberately failing evaluation did not record passed=false")
        print(f"Installed demo pipeline passed ({DEMO_FRAMES} frames); failed evaluation returns nonzero.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Package check failed: {error}", file=sys.stderr)
        raise SystemExit(1)

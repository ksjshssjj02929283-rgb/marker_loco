# Developer Tools

- `check_package.py` builds and checks an installable wheel in a temporary environment. Run it with `make package-check`.
- `calibration/` contains AprilGrid geometry, refinement, and historical calibration comparison utilities. They require recordings under `captures/calibration/d455/` and generated files under `outputs/`.
Laser based pose and RGB-D gate experiments that depend on unpublished local data are kept under the ignored `local/evaluation/` directory.

These utilities support development and reproducible project experiments. User facing commands are implemented under `src/marker6dof/commands/` and registered in `pyproject.toml`.

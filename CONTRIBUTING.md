# Contributing

1. Create a virtual environment and install `python -m pip install -e ".[dev,d455]"` (or `make venv` then `make install`). Use `.[dev]` on machines without a D455.
2. Run `make test` and `make demo` before opening a pull request. `make test` disables pytest plugin autoload so the ROS global plugin does not interfere; `make demo` runs the synthetic closed loop and fails on non-MEASURED frames or when the position (15 mm) / rotation (8°) thresholds are exceeded.
3. Keep real camera recordings, serial numbers, local calibration and generated videos out of git (`captures/`, `outputs/`, `*.mp4`, `*.avi`, `*.db3` are ignored).
4. Record real samples with `scripts/record_d455.py` and verify them offline with `scripts/replay_d455.py` before relying on live hardware.
5. Document coordinate conventions and units for every new pose field.
6. Add a focused regression test for changes to geometry, gating or output schema.

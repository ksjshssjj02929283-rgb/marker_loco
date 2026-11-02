# D455 Kalibr Calibration Artifacts

This directory keeps the small, versioned AprilGrid target definitions, camchain YAML files, and textual calibration results used by the D455 calibration studies. See [calibration methods](../docs/d455_calibration_methods_20261020.md) for the measured board geometry, model comparison, and validation limits.

Original `.bag` recordings are stored locally in `captures/calibration/kalibr_d455/`; conversion logs remain local and are ignored by Git. The PDF report is a retained historical result. The installable tracking package does not load files from this directory automatically. Pass the selected camchain explicitly to evaluation or replay commands.

# D455 + ArUco quick validation

This smoke test uses the vendor/open-source path directly:

- RealSense SDK 2.0 / `pyrealsense2` for D455 RGB-D capture, alignment,
  intrinsics, device timestamps, depth scale, and `.db3` recording.
- OpenCV ArUco for `DICT_4X4_50` detection and subpixel corners, plus OpenCV
  `solvePnP(SOLVEPNP_IPPE_SQUARE)` for planar-tag pose estimation. Pixel/ray
  conversion uses the RealSense SDK so the D455's reported distortion model is
  handled by the vendor implementation.

## Run

From the repository root:

```bash
.venv-hifi/bin/python experiments/d455_aruco/record_and_track.py \
  --marker-size-mm 50 \
  --ids 2,3,4,5 \
  --duration 30
```

Use `--marker-size-mm 40` if you printed the version whose complete paper
cutout is 50 mm and whose black square is 40 mm. Always measure the black outer
edge with a ruler. Press `Q` or `Esc` to stop early.

For a terminal without a display:

```bash
.venv-hifi/bin/python experiments/d455_aruco/record_and_track.py \
  --marker-size-mm 50 --ids 2,3,4,5 --duration 30 --no-preview
```

Raw RGB-D is roughly 3 GB per 30 seconds at the default stream settings. Add
`--no-raw` for a lightweight run that keeps only annotated MP4 and CSV/JSON.

## Output

Each run creates `captures/d455_aruco/YYYYMMDD_HHMMSS/`:

- `raw_realsense.db3`: original color/depth streams and device metadata written
  by the current RealSense SDK recorder (older SDK releases used `.bag`).
- `annotated.mp4`: detections and coordinate axes for visual inspection.
- `color_raw.mp4`: clean color stream without overlays, kept for offline replay.
- `poses.csv`: per-detection PnP pose, quaternion, reprojection error, aligned
  depth median, and image corners.
- `frames.csv`: every frame's device/host timestamps and visible target IDs.
- `metadata.json`: D455 serial/firmware, color intrinsics, distortion, depth
  scale, marker settings, and coordinate convention.
- `summary.json`: frame count and target visibility coverage.

`tvec` is tag position in the D455 color-camera frame: +X right, +Y down, +Z
forward, in metres. It is not yet the gripper TCP pose. A single tag is enough
for this smoke test; converting it to the gripper/TCP frame requires a fixed
measured `T_tag_gripper` or `T_tag_TCP` transform.

## Offline replay

Replay a recorded session without the camera (uses only `color_raw.mp4` and
`metadata.json`, no `pyrealsense2`):

```bash
.venv/bin/python scripts/replay_d455.py \
  --session captures/d455_aruco/YYYYMMDD_HHMMSS \
  --output-dir outputs/replay \
  --min-coverage 0.5
```

## First recording motion

1. Hold the gripper still for 5 seconds at roughly 0.5 m.
2. Translate left/right and up/down slowly, keeping the tag facing the camera.
3. Move closer/farther within roughly 0.4-1.0 m.
4. Rotate slowly about all three axes, avoiding severe grazing angles.
5. Hold still again for 5 seconds.

For the first pass, inspect detection coverage and continuity rather than
claiming absolute accuracy. PnP accuracy still depends on the printed black-edge
measurement and the physical tag being flat.

After a recording, estimate a conservative relative trajectory with:

```bash
.venv-hifi/bin/python experiments/d455_aruco/relative_trajectory.py \
  --poses captures/d455_aruco/YYYYMMDD_HHMMSS/poses.csv \
  --output captures/d455_aruco/YYYYMMDD_HHMMSS/relative_trajectory.csv \
  --reference-id 2 --min-overlap 30
```

The tool only stitches tags that share at least 30 frames and whose fixed
relative transform is supported by robust inliers. It outputs a trajectory in
the initial ID-2 frame and a JSON report with accepted overlap edges. It will
not claim a reliable full four-tag trajectory when an edge, such as ID 4 to ID
5, has only one overlap frame.

To turn the relative trajectory into a metric TCP trajectory, supply a measured
fixed `T_reference_TCP` transform (see `TCP_CALIBRATION.md` for how to measure
it and `tcp_transform.template.json` for the format):

```bash
.venv-hifi/bin/python experiments/d455_aruco/relative_trajectory.py \
  --poses captures/d455_aruco/YYYYMMDD_HHMMSS/poses.csv \
  --output captures/d455_aruco/YYYYMMDD_HHMMSS/relative_trajectory.csv \
  --reference-id 2 --min-overlap 60 \
  --rig-output captures/d455_aruco/YYYYMMDD_HHMMSS/rig_calibration_id2.json \
  --tcp-transform experiments/d455_aruco/tcp_transform.json \
  --tcp-output captures/d455_aruco/YYYYMMDD_HHMMSS/tcp_poses.csv
```

The resulting `tcp_poses.csv` is the metric TCP pose in the D455 color-camera
frame (+X right, +Y down, +Z forward, metres, quaternion xyzw). It is still a
camera-frame pose, not a world/SLAM pose.

## Live browser trajectory

Export a fixed multi-tag rig relation from the high-coverage recording:

```bash
.venv-hifi/bin/python experiments/d455_aruco/relative_trajectory.py \
  --poses captures/d455_aruco/20260921_151219/poses.csv \
  --output captures/d455_aruco/20260921_151219/relative_trajectory.csv \
  --reference-id 2 --min-overlap 60 \
  --rig-output captures/d455_aruco/20260921_151219/rig_calibration_id2.json
```

Then start the dashboard and open `http://127.0.0.1:8765`:

```bash
.venv-hifi/bin/python experiments/d455_aruco/live_dashboard.py \
  --serial 353522302024 \
  --marker-size-mm 48 \
  --rig captures/d455_aruco/20260921_151219/rig_calibration_id2.json
```

The live pose uses a joint corner fit across all accepted calibrated tags. The
shown trajectory starts at the first accepted live pose and is therefore a
relative ID-2 rigid-body trajectory, not a TCP/world trajectory. The browser
shows raw observations as a thin reference and an adaptive low-pass result as
the main trajectory. It never connects separate segments across a rejected or
lost interval, so reacquisition cannot create a false straight-line jump.

The 3D 6DoF view combines the smoothed XYZ path with sampled orientation
frames. Red, green, and blue axes are the tracked rigid body's local X, Y, and
Z directions; the larger frame is the current orientation. Drag the 3D canvas
to orbit, use the mouse wheel to zoom, and use **重置视角** to restore the
default camera. The numeric RPY display is derived from the smoothed relative
quaternion and is expressed in degrees.

`MULTI` means at least two tag faces support the joint pose, `SINGLE` is a
weaker single-face estimate, `WARN` is rejected by geometric/temporal gates,
and `LOST` means that no calibrated face produced a pose. Use **重置原点** to
clear the trace and choose a new relative origin. Stop the server with
`Ctrl-C`.

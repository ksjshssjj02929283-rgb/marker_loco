"""Magnify one already processed A/B interval without resetting either filter."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from marker6dof.projection import draw_realsense_frame, draw_realsense_polyline
from marker6dof.realsense import intrinsics_from_metadata, project_realsense
from marker6dof.commands.reporting.video_encoding import encode_comparison_videos


FIELDS = ('raw_reference', 'raw_tcp', 'old_camera_tcp', 'new_camera_tcp')


def load_rows(directory):
    with (directory / 'comparison.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['frame_index'] = int(row['frame_index'])
        row['elapsed_s'] = float(row['elapsed_s'])
        row['segment'] = int(row['segment'])
        for field in FIELDS:
            if not row[field + '_x_m']:
                row[field] = None
                continue
            transform = np.eye(4)
            transform[:3, 3] = [float(row[field + '_' + axis + '_m']) for axis in 'xyz']
            transform[:3, :3] = Rotation.from_quat([
                float(row[field + '_q' + axis]) for axis in 'xyzw'
            ]).as_matrix()
            row[field] = transform
    return rows


def stats(rows, field):
    points = np.array([row[field][:3, 3] for row in rows if row['raw_tcp'] is not None])
    radius = np.linalg.norm(points - points.mean(axis=0), axis=1) * 1000
    return {'rms_mm': float(np.sqrt(np.mean(radius ** 2))),
            'radial_p95_mm': float(np.percentile(radius, 95)),
            'peak_to_peak_xyz_mm': (np.ptp(points, axis=0) * 1000).tolist()}


def render_detail_chart(directory, selected, detail):
    from marker6dof.commands.reporting.render_smoothing_comparison import _pyplot, segmented_series

    plt = _pyplot()
    series = (
        ('raw_reference', 'Reference origin', '#8b5cf6'),
        ('raw_tcp', 'Raw TCP', '#6b7280'),
        ('old_camera_tcp', 'Before display', '#d97706'),
        ('new_camera_tcp', 'After display', '#0284c7'),
    )
    origin = np.asarray(detail['common_coordinate_origin_m']) * 1000
    fig = plt.figure(figsize=(11, 10), dpi=150)
    grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.3])
    axes = [fig.add_subplot(grid[index]) for index in range(3)]
    for field, label, color in series:
        if field == 'raw_reference':
            continue
        times, points = segmented_series(selected, field)
        for axis_index, axis in enumerate(axes):
            axis.plot(times, points[:, axis_index] - origin[axis_index], color=color,
                      label=label, linewidth=1.2, alpha=.8)
    for axis_index, axis in enumerate(axes):
        axis.set_ylabel('XYZ'[axis_index] + ' / mm')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].set_title('Same time window and common raw-TCP mean origin; no filter restart')
    axes[-1].set_xlabel('Original device elapsed time (s)')
    bars_axis = fig.add_subplot(grid[3])
    values = [detail['window_statistics'][field]['rms_mm'] for field, _, _ in series]
    bars = bars_axis.bar([label for _, label, _ in series], values,
                         color=[color for _, _, color in series], width=.65)
    bars_axis.bar_label(bars, fmt='%.3f', padding=3)
    bars_axis.set_ylabel('3D RMS around each mean / mm')
    bars_axis.grid(axis='y', alpha=.2)
    bars_axis.margins(y=.2)
    fig.suptitle(f'Detail {detail["start_s"]:g}-{detail["end_s"]:g} s; '
                 f'{detail["accepted_frames"]} shared accepted frames')
    fig.text(.5, .012, 'Short exploratory interval includes settling; not independently confirmed stationary or absolute accuracy.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, .96])
    fig.savefig(directory / 'detail_timeseries.png')
    plt.close(fig)


def render_detail(directory, start, end, slow_motion=4.0):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    rows = load_rows(directory)
    if not (0 <= start < end <= rows[-1]['elapsed_s']) or slow_motion <= 0:
        raise ValueError('Interval must fit the actual recording; slow-motion factor must be positive')
    selected = [row for row in rows if start <= row['elapsed_s'] <= end]
    valid = [row for row in selected if row['raw_tcp'] is not None]
    if len(valid) < 3:
        raise ValueError('At least three accepted poses are required')
    session = Path(summary['session'])
    metadata = summary['source_metadata']
    intrinsics = intrinsics_from_metadata(metadata['color_intrinsics'])
    marker_size = float(metadata['aruco']['marker_black_square_size_m'])
    fps = float(metadata['stream']['fps'])
    positions = np.array([row[field][:3, 3] for row in valid
                          for field in ('raw_tcp', 'old_camera_tcp', 'new_camera_tcp')])
    reference_origin = np.mean([row['raw_tcp'][:3, 3] for row in valid], axis=0)
    plot_points = (positions - reference_origin) * 1000
    center = (plot_points.min(axis=0) + plot_points.max(axis=0)) / 2
    span_mm = max(float(np.ptp(plot_points, axis=0).max()) * 1.25, 4.0)
    grid_mm = 10.0 ** np.floor(np.log10(span_mm / 5))
    if span_mm / grid_mm > 10:
        grid_mm *= 5
    camera_points = np.array([row[field][:3, 3] for row in valid
                             for field in ('raw_reference', 'raw_tcp', 'new_camera_tcp')])
    pixels = project_realsense(intrinsics, camera_points)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    width, height = int(metadata['stream']['width']), int(metadata['stream']['height'])
    if not len(pixels):
        raise ValueError('No finite projections for the shared RGB crop')
    low = np.floor(pixels.min(axis=0) - 80).astype(int)
    high = np.ceil(pixels.max(axis=0) + 80).astype(int)
    low = np.maximum(low, [0, 0]); high = np.minimum(high, [width, height])
    if np.any(high - low < 20):
        low, high = np.array([0, 0]), np.array([width, height])
    x0, y0 = low; x1, y1 = high
    output = directory / 'detail_closeup.mp4'
    capture = cv2.VideoCapture(str(session / 'color_raw.mp4'))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_motion, (1600, 900))
    if not capture.isOpened() or not writer.isOpened():
        capture.release(); writer.release()
        raise RuntimeError('Cannot open detail video input/output')
    history_rows = []
    last_time = last_segment = None
    written = 0
    try:
        for index in range(selected[-1]['frame_index'] + 1):
            ok, rgb = capture.read()
            if not ok:
                raise ValueError(f'Source video missing frame {index}')
            row = rows[index]
            if not start <= row['elapsed_s'] <= end:
                continue
            gap = (last_time is not None and (row['elapsed_s'] - last_time > .2
                                            or row['segment'] != last_segment))
            if gap or row['raw_tcp'] is None:
                history_rows.append(None)
            history_rows.append(row if row['raw_tcp'] is not None else None)
            last_time, last_segment = row['elapsed_s'], row['segment']
            canvas = np.full((900, 1600, 3), (26, 29, 34), np.uint8)
            for branch, (axis_field, plot_field, title, color) in enumerate((
                ('raw_tcp', 'old_camera_tcp', 'BEFORE: raw RGB axes / legacy trajectory', (0, 165, 255)),
                ('new_camera_tcp', 'new_camera_tcp', 'AFTER: same smoothed pose for axes / trajectory', (255, 180, 50)),
            )):
                offset = branch * 800
                cv2.putText(canvas, title, (offset + 16, 30), cv2.FONT_HERSHEY_SIMPLEX, .60, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(canvas, f'device t={row["elapsed_s"]:.3f}s  frame={index}  {row["status"]}  {slow_motion:g}x slower playback',
                            (offset + 16, 57), cv2.FONT_HERSHEY_SIMPLEX, .53, (190, 200, 210), 1, cv2.LINE_AA)
                annotated = rgb.copy()
                segment = []
                for sample in history_rows + [None]:
                    if sample is None:
                        if len(segment) > 1:
                            draw_realsense_polyline(annotated, intrinsics, segment, color, 2)
                        segment = []
                    else:
                        segment.append(sample[axis_field][:3, 3])
                if row[axis_field] is not None:
                    draw_realsense_frame(annotated, intrinsics, row[axis_field], marker_size)
                crop = annotated[y0:y1, x0:x1]
                ratio = min(768 / crop.shape[1], 420 / crop.shape[0])
                preview = cv2.resize(crop, (round(crop.shape[1] * ratio), round(crop.shape[0] * ratio)))
                px, py = offset + 16 + (768 - preview.shape[1]) // 2, 75 + (420 - preview.shape[0]) // 2
                canvas[py:py + preview.shape[0], px:px + preview.shape[1]] = preview
                for plane, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                    left, top, size = offset + 18 + plane * 260, 550, 230
                    cv2.putText(canvas, 'XYZ'[a] + 'XYZ'[b] + ' / mm', (left + 65, top - 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1, cv2.LINE_AA)

                    def mapped(position):
                        value = (position - reference_origin) * 1000
                        delta = (value[[a, b]] - center[[a, b]]) * size / span_mm
                        return tuple(np.rint([left + size / 2, top + size / 2] + delta * [1, -1]).astype(int))

                    for axis in (a, b):
                        minimum = center[axis] - span_mm / 2
                        for tick in np.arange(np.ceil(minimum / grid_mm) * grid_mm, minimum + span_mm, grid_mm):
                            pixel = int(round((tick - minimum) * size / span_mm))
                            if axis == a:
                                cv2.line(canvas, (left + pixel, top), (left + pixel, top + size), (55, 58, 65), 1)
                            else:
                                cv2.line(canvas, (left, top + size - pixel), (left + size, top + size - pixel), (55, 58, 65), 1)
                    for field, curve_color, thickness in (('raw_tcp', (110, 115, 120), 1), (plot_field, color, 2)):
                        previous = None
                        for sample in history_rows:
                            if sample is None:
                                previous = None
                                continue
                            current = mapped(sample[field][:3, 3])
                            if previous is not None:
                                cv2.line(canvas, previous, current, curve_color, thickness, cv2.LINE_AA)
                            previous = current
                        if previous is not None:
                            cv2.circle(canvas, previous, 4, curve_color, -1, cv2.LINE_AA)
                cv2.putText(canvas, f'SHARED SCALE: span {span_mm:.2f} mm; grid {grid_mm:g} mm. Gray = identical raw TCP.',
                            (offset + 16, 818), cv2.FONT_HERSHEY_SIMPLEX, .51, (210, 215, 220), 1, cv2.LINE_AA)
                cv2.putText(canvas, 'Common origin/crop/scale; no filter restart. Short window includes settling.',
                            (offset + 16, 848), cv2.FONT_HERSHEY_SIMPLEX, .51, (170, 195, 230), 1, cv2.LINE_AA)
            writer.write(canvas); written += 1
    finally:
        capture.release(); writer.release()
    encode_comparison_videos(directory, ('detail_closeup.mp4',))
    result = {
        'start_s': start, 'end_s': end, 'frames': len(selected), 'accepted_frames': len(valid),
        'written_frames': written, 'slow_motion_factor': slow_motion, 'playback_fps': fps / slow_motion,
        'shared_rgb_roi_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'shared_plot_span_mm': span_mm, 'shared_grid_mm': float(grid_mm),
        'common_coordinate_origin_m': reference_origin.tolist(),
        'window_statistics': {field: stats(valid, field) for field in FIELDS},
        'selection_note': 'Exploratory final low-image-motion interval selected from corner/reference observations, not by maximizing smoothing improvement. About one second; not independently confirmed stationary and includes filter settling.',
        'rendering_note': 'Only crop and display scale/playback speed change; original measured values and complete-recording filter states are retained. Both branches use identical RGB crop and mm/pixel scale.',
    }
    render_detail_chart(directory, selected, result)
    (directory / 'detail_window.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--slow-motion', type=float, default=4.0)
    args = parser.parse_args(argv)
    render_detail(args.comparison_dir, args.start, args.end, args.slow_motion)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

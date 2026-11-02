from __future__ import annotations

import builtins
import csv
import json

import numpy as np
import pytest

from marker6dof.commands.reporting import render_smoothing_comparison as report


def write_comparison(directory, windows=None):
    summary = {
        'session': 'captures/<script>alert(1)</script>',
        'static_windows': windows if windows is not None else [{
            'start_s': 0, 'end_s': .1, 'frames': 4, 'accepted_frames': 3,
            'raw': {'rms_3d_mm': .7}, 'old': {'rms_3d_mm': .3},
            'new': {'rms_3d_mm': .15},
        }],
    }
    (directory / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    fields = ['elapsed_s', 'segment'] + [
        f'{prefix}_{axis}_m' for prefix, _, _ in report.SERIES for axis in 'xyz'
    ]
    rows = []
    for frame, segment in enumerate([1, 1, 1, 2]):
        row = {'elapsed_s': frame / 30, 'segment': segment}
        for prefix, _, _ in report.SERIES:
            row.update({f'{prefix}_{axis}_m': '' if frame == 2 else frame * .001 for axis in 'xyz'})
        rows.append(row)
    with (directory / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_render_portable_report_with_same_source_videos_and_truth_limits(tmp_path):
    pytest.importorskip('matplotlib')
    write_comparison(tmp_path)
    for filename in ('before.mp4', 'after.mp4', 'side_by_side.mp4'):
        (tmp_path / filename).write_bytes(b'video fixture')
    outputs = report.render_comparison(tmp_path)
    for name in ('trajectory_xyz.png', 'window_rms.png'):
        assert (tmp_path / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    page = (tmp_path / 'report.html').read_text(encoding='utf-8')
    for filename in ('before.mp4', 'after.mp4', 'side_by_side.mp4'):
        assert f'src="{filename}"' in page
    assert '<script>alert' not in page
    assert '&lt;script&gt;' in page
    assert '同一 RigTracker' in page
    assert 'RGB 叠加使用原始' in page
    assert '指定窗口不等于已确认静止' in page
    assert '静止重复性也不等于绝对 TCP 精度' in page
    assert '显示滞后' in page
    assert '0.700' in page and '0.300' in page and '0.150' in page
    assert outputs['report'] == str(tmp_path / 'report.html')


@pytest.mark.parametrize('windows', [[], [{
    'start_s': 1, 'end_s': 2, 'accepted_frames': 0, 'raw': None, 'old': None, 'new': None,
}], [{
    'start_s': 1, 'end_s': 2, 'accepted_frames': 20,
    'raw': {'rms_3d_mm': .4}, 'old': {'rms_3d_mm': float('nan')}, 'new': None,
}]])
def test_missing_window_metrics_do_not_become_zero_bars(tmp_path, monkeypatch, windows):
    pytest.importorskip('matplotlib')
    from matplotlib.axes import Axes

    write_comparison(tmp_path, windows)

    def unexpected_bar(*args, **kwargs):
        raise AssertionError('Invalid or missing window metrics must not be plotted as bars')

    monkeypatch.setattr(Axes, 'bar', unexpected_bar)
    report.render_comparison(tmp_path)
    page = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert '视频尚未生成' in page
    assert '缺失数据不会按 0 绘制' in page or '暂无有效数据' in page


def test_plot_series_preserves_samples_and_breaks_at_loss_segment_or_long_gap():
    rows = []
    for timestamp, segment, point in [(0, 1, .1), (.033, 1, None), (.066, 2, .2), (.4, 2, .3)]:
        row = {'elapsed_s': str(timestamp), 'segment': str(segment)}
        row.update({f'raw_tcp_{axis}_m': '' if point is None else str(point) for axis in 'xyz'})
        rows.append(row)
    times, points = report.segmented_series(rows, 'raw_tcp')
    np.testing.assert_allclose(points[np.isfinite(points).all(axis=1)],
                               [[100] * 3, [200] * 3, [300] * 3])
    assert len(times) == 6
    assert np.isnan(points[1]).all()  # Lost sample.
    assert np.isnan(points[2]).all()  # New segment.
    assert np.isnan(points[4]).all()  # Gap beyond 0.2 seconds.


def test_bad_schema_is_rejected_before_writing_or_loading_plot_dependency(tmp_path, monkeypatch):
    (tmp_path / 'summary.json').write_text('{"static_windows": []}', encoding='utf-8')
    (tmp_path / 'comparison.csv').write_text('elapsed_s,segment\n0,1\n', encoding='utf-8')

    def unexpected_import():
        raise AssertionError('Input validation should precede optional reporting imports')

    monkeypatch.setattr(report, '_pyplot', unexpected_import)
    with pytest.raises(ValueError, match='missing columns'):
        report.render_comparison(tmp_path)
    assert not (tmp_path / 'report.html').exists()
    assert not (tmp_path / 'trajectory_xyz.png').exists()


def test_missing_matplotlib_has_clear_optional_dependency_error(monkeypatch):
    original_import = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name.startswith('matplotlib'):
            raise ModuleNotFoundError('matplotlib is unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_matplotlib)
    with pytest.raises(RuntimeError, match='reporting dependencies'):
        report._pyplot()


def test_motion_variation_report_is_distinct_from_stationary_noise_and_accuracy(tmp_path):
    pytest.importorskip('matplotlib')
    write_comparison(tmp_path)
    source = tmp_path / 'summary.json'
    summary = json.loads(source.read_text(encoding='utf-8'))
    summary['motion_variation'] = {
        'eligible_frame_triples': 42, 'units': 'mm/s^2',
        'raw': {'rms': 1200.1, 'median': 1000.2, 'p95': 2100.3},
        'old': {'rms': 800.1, 'median': 600.2, 'p95': 1700.3},
        'new': {'rms': 500.1, 'median': 300.2, 'p95': 900.3},
    }
    source.write_text(json.dumps(summary), encoding='utf-8')
    outputs = report.render_comparison(tmp_path)
    assert (tmp_path / 'motion_variation.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert outputs['motion_variation_chart'] == str(tmp_path / 'motion_variation.png')
    page = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert '有效三帧组数：42' in page
    assert 'mm/s²' in page
    assert '1200.100' in page and '800.100' in page and '500.100' in page
    assert '数值较低表示显示曲线的短时变化减少' in page
    assert '不能当成静止噪声、绝对定位精度' in page
    assert '排除失锁、断段和长间隔' in page
    assert 'src="motion_variation.png"' in page


@pytest.mark.parametrize('motion', [
    {'eligible_frame_triples': 0, 'units': 'mm/s^2'},
    {'eligible_frame_triples': 3, 'units': 'm/s^2',
     'raw': {'rms': .1}, 'old': {'rms': .05}, 'new': {'rms': .02}},
    {'eligible_frame_triples': 3, 'units': 'mm/s^2',
     'raw': {'rms': 100}, 'old': {'rms': float('nan')}, 'new': {'rms': 20}},
])
def test_missing_invalid_or_wrong_unit_motion_metrics_are_not_zero_bars(tmp_path, monkeypatch, motion):
    pytest.importorskip('matplotlib')
    from matplotlib.axes import Axes

    write_comparison(tmp_path, [])
    source = tmp_path / 'summary.json'
    summary = json.loads(source.read_text(encoding='utf-8'))
    summary['motion_variation'] = motion
    source.write_text(json.dumps(summary), encoding='utf-8')

    def unexpected_bar(*args, **kwargs):
        raise AssertionError('Unavailable acceleration values must not become zero bars')

    monkeypatch.setattr(Axes, 'bar', unexpected_bar)
    outputs = report.render_comparison(tmp_path)
    assert 'motion_variation_chart' not in outputs
    assert not (tmp_path / 'motion_variation.png').exists()
    page = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert '暂无有效数据' in page
    assert 'src="motion_variation.png"' not in page

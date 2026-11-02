"""Render a portable, local report from a recorded smoothing comparison."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np


SERIES = (
    ('raw_tcp', 'Raw accepted TCP', '#6b7280'),
    ('old_camera_tcp', 'Before: previous display filter', '#d97706'),
    ('new_camera_tcp', 'After: rigid-pose filter', '#0284c7'),
)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_rows(path: Path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'elapsed_s', 'segment'} | {
            f'{prefix}_{axis}_m' for prefix, _, _ in SERIES for axis in 'xyz'
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path.name} is missing columns: {", ".join(sorted(missing))}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path.name} contains no samples')
    if any(_number(row['elapsed_s']) is None for row in rows):
        raise ValueError(f'{path.name}: elapsed_s must be finite')
    return rows


def segmented_series(rows, prefix):
    """Insert gaps without dropping valid samples or joining trace segments."""
    times, points = [], []
    previous_time = previous_segment = None
    for row in rows:
        timestamp = float(row['elapsed_s'])
        segment = row.get('segment')
        if previous_time is not None and (
            segment != previous_segment or timestamp <= previous_time or timestamp - previous_time > .2
        ):
            times.append(timestamp)
            points.append([np.nan] * 3)
        values = [_number(row.get(f'{prefix}_{axis}_m')) for axis in 'xyz']
        times.append(timestamp)
        points.append(values if all(value is not None for value in values) else [np.nan] * 3)
        previous_time, previous_segment = timestamp, segment
    return np.asarray(times), np.asarray(points, dtype=float) * 1000


def _windows(summary):
    windows = summary.get('static_windows', [])
    if not isinstance(windows, list):
        raise ValueError('summary.json: static_windows must be a list')
    return windows


def _window_rms(window):
    if _number(window.get('accepted_frames')) is None or float(window['accepted_frames']) < 2:
        return None
    values = [_number((window.get(key) or {}).get('rms_3d_mm')) for key in ('raw', 'old', 'new')]
    return values if all(value is not None and value >= 0 for value in values) else None


def _motion_rms(summary):
    motion = summary.get('motion_variation') or {}
    count = _number(motion.get('eligible_frame_triples'))
    if count is None or count < 1 or motion.get('units') != 'mm/s^2':
        return None
    values = [_number((motion.get(key) or {}).get('rms')) for key in ('raw', 'old', 'new')]
    return values if all(value is not None and value >= 0 for value in values) else None


def _pyplot():
    # Reporting dependencies must not be required to import the CLI module.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'Matplotlib is required to render comparison charts. '
            'Install reporting dependencies with: python -m pip install -e ".[reporting]"'
        ) from exc
    return plt


def _render_charts(rows, windows, directory, plt):
    xyz_path = directory / 'trajectory_xyz.png'
    rms_path = directory / 'window_rms.png'
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, dpi=150)
    for prefix, label, color in SERIES:
        times, points = segmented_series(rows, prefix)
        for index, axis in enumerate(axes):
            axis.plot(times, points[:, index], label=label, color=color,
                      linewidth=.8 if prefix == 'raw_tcp' else 1.1,
                      alpha=.55 if prefix == 'raw_tcp' else .9)
    for index, axis in enumerate(axes):
        axis.set_ylabel(f'TCP {"XYZ"[index]} (mm)')
        axis.grid(alpha=.2)
        for window in windows:
            start, end = _number(window.get('start_s')), _number(window.get('end_s'))
            if start is not None and end is not None and end > start:
                axis.axvspan(start, end, color='#94a3b8', alpha=.12)
    axes[0].set_title('Camera-frame TCP: same accepted raw poses, different display filters')
    axes[0].legend(fontsize=8, loc='best')
    axes[-1].set_xlabel('Elapsed time (s); shaded regions are selected windows')
    fig.tight_layout()
    fig.savefig(xyz_path)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8), dpi=150)
    valid = [(index, window, _window_rms(window)) for index, window in enumerate(windows)]
    valid = [(index, window, metrics) for index, window, metrics in valid if metrics is not None]
    if valid:
        locations = np.arange(len(valid), dtype=float)
        width = .24
        for column, (_, label, color) in enumerate(SERIES):
            bars = axis.bar(locations + (column - 1) * width,
                            [metrics[column] for _, _, metrics in valid],
                            width=width, label=label, color=color)
            axis.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
        labels = [f'Window {index + 1}\n{window.get("start_s")} - {window.get("end_s")} s\n'
                  f'n={window.get("accepted_frames")}' for index, window, _ in valid]
        axis.set_xticks(locations, labels)
        axis.set_ylabel('3D RMS around window mean (mm)')
        axis.legend(fontsize=8)
        axis.grid(alpha=.2, axis='y')
        axis.margins(y=.15)
    else:
        axis.text(.5, .5, 'No valid selected-window RMS metrics\nMissing data are not zero.',
                  transform=axis.transAxes, ha='center', va='center')
        axis.set_axis_off()
    axis.set_title('Selected-window variation; physical stillness is not independently verified')
    fig.tight_layout()
    fig.savefig(rms_path)
    plt.close(fig)
    return xyz_path, rms_path


def _escape(value):
    return html.escape(str(value), quote=True)


def _format(value, digits=3):
    number = _number(value)
    return f'{number:.{digits}f}' if number is not None else '暂无有效数据'


def _video(directory, filename, title, note):
    asset = directory / filename
    if not asset.is_file():
        return f'<article><h3>{_escape(title)}</h3><p>视频尚未生成：{_escape(filename)}</p></article>'
    url = quote(filename)
    return (f'<article><h3>{_escape(title)}</h3><video controls preload="metadata" '
            f'src="{url}"></video><p>{_escape(note)}</p>'
            f'<a href="{url}" download>下载视频</a></article>')


def _window_table(windows):
    if not windows:
        return '<p>没有指定窗口指标。缺失数据不会按 0 绘制，也不能据此判断静止效果。</p>'
    rows = []
    for index, window in enumerate(windows):
        metrics = _window_rms(window)
        values = [_format(value) for value in metrics] if metrics is not None else ['暂无有效数据'] * 3
        rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in (
            f'指定窗口 {index + 1}',
            f'{_format(window.get("start_s"), 2)}–{_format(window.get("end_s"), 2)} s',
            _escape(window.get('accepted_frames', '暂无有效数据')),
            *values,
        )) + '</tr>')
    return ('<div class="table-scroll"><table><thead><tr><th>窗口</th><th>时间</th>'
            '<th>有效帧</th><th>原始 RMS / mm</th><th>修改前 RMS / mm</th>'
            '<th>修改后 RMS / mm</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def _difference_table(summary):
    differences = summary.get('all_frames_raw_to_display_distance_mm') or {}
    if not differences:
        return ''
    rows = []
    for key, label in (('old', '修改前显示'), ('new', '修改后显示')):
        values = differences.get(key) or {}
        rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in (
            label, *(_format(values.get(metric)) for metric in ('median', 'p95', 'max')),
        )) + '</tr>')
    return ('<div class="table-scroll"><table><thead><tr><th>分支</th><th>距离中位 / mm</th>'
            '<th>距离 P95 / mm</th><th>最大距离 / mm</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


def _motion_section(summary):
    if 'motion_variation' not in summary:
        return ''
    motion = summary.get('motion_variation') or {}
    valid = _motion_rms(summary) is not None
    rows = []
    for key, label in (('raw', '原始测量'), ('old', '修改前显示'), ('new', '修改后显示')):
        values = motion.get(key) or {}
        cells = [_format(values.get(metric)) if valid else '暂无有效数据'
                 for metric in ('rms', 'median', 'p95')]
        rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in (label, *cells)) + '</tr>')
    chart = ('<a href="motion_variation.png"><img src="motion_variation.png" '
             'alt="原始、修改前和修改后 TCP 差分加速度 RMS"></a>') if valid else ''
    return ('<section class="card"><h2>运动中的短时变化</h2>'
            '<p>下表用同一轨迹段内连续 3 帧的位置差分估计加速度，排除失锁、断段和长间隔。'
            '有效三帧组数：' + _escape(motion.get('eligible_frame_triples', '暂无有效数据')) + '。</p>'
            '<p><strong>数值较低表示显示曲线的短时变化减少。</strong>该指标同时包含真实加速度、'
            '测量波动及滤波滞后，不能当成静止噪声、绝对定位精度或实际运动的加速度真值。</p>'
            '<div class="table-scroll"><table><thead><tr><th>分支</th><th>RMS / mm/s²</th>'
            '<th>中位 / mm/s²</th><th>P95 / mm/s²</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>' + chart + '</section>')


def _render_motion_chart(summary, directory, plt):
    values = _motion_rms(summary)
    if values is None:
        return None
    path = directory / 'motion_variation.png'
    fig, axis = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    bars = axis.bar(['Raw accepted TCP', 'Before display', 'After display'], values,
                    color=[color for _, _, color in SERIES], width=.6)
    axis.bar_label(bars, fmt='%.1f', fontsize=9, padding=4)
    axis.set_ylabel(r'RMS estimated acceleration (mm/s$^2$)')
    axis.set_title('Short-term TCP curve variation from consecutive frame triples')
    axis.text(.5, -.18, 'Includes actual acceleration and filtering lag; not positioning accuracy.',
              transform=axis.transAxes, ha='center', va='top', fontsize=9)
    axis.grid(alpha=.2, axis='y')
    axis.margins(y=.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _detail_section(directory, summary):
    detail = summary.get('detail_window')
    if not isinstance(detail, dict) or not (directory / 'detail_closeup.mp4').is_file():
        return ''
    note = (
        f'录制时间 {_format(detail.get("start_s"), 2)}–{_format(detail.get("end_s"), 2)} s，'
        f'{_escape(detail.get("frames", "暂无有效数据"))} 帧；'
        f'{_format(detail.get("slow_motion_factor"), 1)} 倍慢放。'
        '两侧使用相同裁剪、坐标原点和毫米标尺，保留完整录制中的滤波状态。'
    )
    chart = (
        '<a href="detail_timeseries.png"><img src="detail_timeseries.png" '
        'alt="放大片段的 TCP 时间曲线与波动对比"></a>'
    ) if (directory / 'detail_timeseries.png').is_file() else ''
    return (
        '<section class="card"><h2>同尺度放大与慢放</h2>'
        + _video(directory, 'detail_closeup.mp4', 'TCP 抖动细节', note)
        + '<p>该片段按角点和参考原点的低变化事后选取，约 1 秒，包含滤波收敛。'
        '窗口内波动不能代表充分静止后的稳定精度。</p>' + chart + '</section>'
    )


def _render_html(directory, summary, windows):
    source = summary.get('session', summary.get('source_session', '见 summary.json'))
    videos = ''.join([
        _video(directory, 'side_by_side.mp4', '同步并排对比', '左侧为修改前，右侧为修改后，使用同一录制帧。'),
        _video(directory, 'before.mp4', '修改前', 'RGB 叠加使用原始已接受 TCP 位姿；旧网页曲线使用旧显示滤波。'),
        _video(directory, 'after.mp4', '修改后', '先平滑参考 Marker 的完整位置与旋转，再组合固定 TCP 外参。'),
    ])
    warnings = summary.get('warnings', [])
    if isinstance(warnings, str):
        warnings = [warnings]
    warning_html = ''.join(f'<p>{_escape(warning)}</p>' for warning in warnings)
    filters = summary.get('new_filter') or {}
    coverage = _number(summary.get('pose_coverage'))
    sample_note = (
        f'总帧 {_escape(summary.get("frames", "暂无有效数据"))}；'
        f'已接受位姿 {_escape(summary.get("accepted_frames", "暂无有效数据"))}；'
        f'位姿覆盖率 {f"{coverage * 100:.1f}%" if coverage is not None else "暂无有效数据"}；'
        f'实际采样时长 {_format(summary.get("device_time_duration_s"), 2)} s。'
    )
    filter_note = '' if not filters else (
        f'<p>本次修改后滤波：位置最低截止频率 {_format(filters.get("position_cutoff_hz"), 2)} Hz，'
        f'旋转最低截止频率 {_format(filters.get("rotation_cutoff_hz"), 2)} Hz；运动时自适应提高带宽。</p>'
    )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TCP 平滑修改前后对比</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1160px;margin:auto;padding:32px 22px 60px}}h1{{font-size:30px;line-height:1.3}}h2{{margin-top:30px}}
.card,article{{background:white;padding:22px;border:1px solid #dde4ef;border-radius:12px;margin:18px 0}}
.notice{{background:#fff7e6;border-left:4px solid #d97706;padding:16px 20px}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.videos article:first-child{{grid-column:1/-1}}video,img{{width:100%;height:auto;border-radius:6px}}a{{color:#0369a1}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #dde4ef}}
.table-scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}.muted{{color:#526077}}@media(max-width:760px){{.videos{{grid-template-columns:1fr}}main{{padding:20px 14px}}}}
</style></head><body><main>
<h1>TCP 平滑修改前后对比</h1><p class="muted">数据集：<code>{_escape(source)}</code></p><p>{sample_note}</p>
<div class="notice"><strong>指定窗口不等于已确认静止。</strong>本报告没有独立的运动真值，也不能从录制时长确认用户执行了“静止 / 运动 / 静止”协议。窗口 RMS 表示相对窗口均值的波动；真实运动、漂移和测量噪声都会进入该值。只有实际保持静止的窗口，才能将它解释为静止重复性；静止重复性也不等于绝对 TCP 精度。{warning_html}</div>
<section class="card"><h2>比较方式</h2><p>两条处理分支共用同一录制、同一 RigTracker 已接受的原始位姿以及同一固定 TCP 外参。修改前的 RGB 轨迹与坐标轴使用原始位姿，网页 3D 曲线使用旧 RelativeTrace 滤波；修改后先平滑参考 Marker 的完整位姿，再组合 TCP 外参，RGB 与网页使用同一个平滑结果。原始测量数据保持独立。</p>
<p>曲线显示相机坐标系下的 TCP XYZ，修改前与修改后使用同一坐标系。轨迹失锁或断段后不连接。阴影为指定窗口。</p>{filter_note}</section>
{_detail_section(directory, summary)}
<h2>视频</h2><div class="videos">{videos}</div>
<section class="card"><h2>XYZ 时间曲线</h2><a href="trajectory_xyz.png"><img src="trajectory_xyz.png" alt="原始、修改前和修改后 TCP XYZ 时间曲线"></a></section>
<section class="card"><h2>指定窗口 RMS 对比</h2>{_window_table(windows)}<p>该值含窗口内真实移动。没有有效帧或指标时显示“暂无有效数据”，不会补成 0。</p><a href="window_rms.png"><img src="window_rms.png" alt="指定窗口 RMS 对比"></a></section>
{_motion_section(summary)}
<section class="card"><h2>运动中的显示滞后</h2><p>平滑会降低快速波动，并带来显示滞后。运动时原始与显示位置的差值同时包含滤波滞后和测量噪声，不能当成定位误差，也不能据此直接计算实际系统延迟。判断动态效果应同时查看并排视频、XYZ 曲线、原始测量和实际运动。</p>{_difference_table(summary)}<p class="muted">以上距离按整段有效帧统计，单位为距离，不是时间延迟。</p></section>
<p class="muted">离线分享：将 report.html、PNG、MP4、summary.json 与 comparison.csv 一起复制，保持它们位于同一目录。<a href="summary.json">查看完整指标</a> · <a href="comparison.csv">下载逐帧数据</a></p>
</main></body></html>'''


def render_comparison(comparison_dir: Path | str):
    directory = Path(comparison_dir)
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if not isinstance(summary, dict):
        raise ValueError('summary.json must contain an object')
    csv_path = directory / 'comparison.csv'
    if not csv_path.exists():
        csv_path = directory / 'trajectory_comparison.csv'
    rows = _load_rows(csv_path)
    windows = _windows(summary)
    plt = _pyplot()
    xyz_path, rms_path = _render_charts(rows, windows, directory, plt)
    motion_path = _render_motion_chart(summary, directory, plt)
    report_path = directory / 'report.html'
    report_path.write_text(_render_html(directory, summary, windows), encoding='utf-8')
    outputs = {'report': str(report_path), 'trajectory_chart': str(xyz_path), 'window_chart': str(rms_path)}
    if motion_path is not None:
        outputs['motion_variation_chart'] = str(motion_path)
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Render PNG charts and a local smoothing A/B report.')
    parser.add_argument('--comparison-dir', type=Path, required=True,
                        help='Directory containing summary.json and comparison.csv.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        outputs = render_comparison(args.comparison_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Cannot render smoothing comparison: {exc}') from exc
    print(json.dumps(outputs, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

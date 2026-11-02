#!/usr/bin/env python3
"""Build the full 2026-10-30 marker accuracy report and Feishu XML draft."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import zipfile
from pathlib import Path

import numpy as np


MARKERS = ("marker50", "marker60", "marker70")
SIDES = ("left", "right")
INTERVALS = (0.5, 1.0, 2.0, 5.0)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def vectors(rows: list[dict], prefix: str):
    times = np.asarray([float(row["gt_time_s"]) for row in rows])
    xyz = np.asarray([[float(row[f"{prefix}_{axis}_m"]) for axis in "xyz"] for row in rows])
    return times, xyz


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, rmse_mm=None, mae_mm=None, bias_mm=None, p95_abs_mm=None,
                    max_abs_mm=None, within3_pct=None)
    absolute = np.abs(values)
    return dict(n=int(len(values)), rmse_mm=float(np.sqrt(np.mean(values**2))),
                mae_mm=float(np.mean(absolute)), bias_mm=float(np.mean(values)),
                p95_abs_mm=float(np.percentile(absolute, 95)),
                max_abs_mm=float(np.max(absolute)),
                within3_pct=float(np.mean(absolute <= 3.0) * 100))


def pairwise_errors(camera: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pieces = []
    for index in range(len(camera) - 1):
        est = np.linalg.norm(camera[index + 1:] - camera[index], axis=1)
        truth = np.linalg.norm(gt[index + 1:] - gt[index], axis=1)
        pieces.append((est - truth) * 1000)
    return np.concatenate(pieces) if pieces else np.asarray([], dtype=float)


def interval_errors(times: np.ndarray, camera: np.ndarray, gt: np.ndarray, interval: float) -> np.ndarray:
    errors = []
    for index, start in enumerate(times):
        right = int(np.searchsorted(times, start + interval))
        candidates = [candidate for candidate in (right - 1, right)
                      if index < candidate < len(times)]
        if not candidates:
            continue
        end = min(candidates, key=lambda candidate: abs(times[candidate] - start - interval))
        if abs(times[end] - start - interval) > 0.035:
            continue
        errors.append((np.linalg.norm(camera[end] - camera[index]) -
                       np.linalg.norm(gt[end] - gt[index])) * 1000)
    return np.asarray(errors, dtype=float)


def write_detail_csvs(root: Path) -> dict:
    static_jitter, static_pairs, dynamic_tracks, intervals, range_sensitivity = [], [], [], [], []
    static_pair_arrays = {}
    for side in SIDES:
        ev = root / f"evaluation_{side}_filtered"
        for marker in ("marker50", "marker60"):
            points = read_json(ev / marker / "static_points.json")
            for point in points:
                if point["status"] != "ok":
                    continue
                static_jitter.append(dict(marker=marker, camera=side, distance_cm=point["distance_cm"],
                                          jitter_rms_mm=point["jitter_rms_mm"],
                                          jitter_p95_mm=point["jitter_p95_mm"],
                                          recording_coverage_pct=point["coverage"] * 100,
                                          frames=point["frames"], accepted=point["accepted"]))
            rows = read_csv(ev / marker / "static_pairs.csv")
            static_pair_arrays[(marker, side)] = rows
            for row in rows:
                static_pairs.append(dict(marker=marker, camera=side, from_cm=row["from_cm"],
                                         to_cm=row["to_cm"], estimate_mm=row["estimate_mm"],
                                         laser_mm=row["laser_mm"], error_mm=row["error_mm"],
                                         absolute_error_mm=abs(float(row["error_mm"])),
                                         within_3mm=abs(float(row["error_mm"])) <= 3.0))

            for endpoint in range(30, 101, 10):
                selected = np.asarray([float(row["error_mm"]) for row in rows
                                       if int(row["from_cm"]) >= 20 and int(row["to_cm"]) <= endpoint])
                s = stats(selected)
                range_sensitivity.append(dict(marker=marker, camera=side,
                                              range_basis="static_nominal_distance", range_unit="cm",
                                              range_start=20, range_end=endpoint,
                                              pair_count=s["n"], pair_rmse_mm=s["rmse_mm"],
                                              p95_abs_mm=s["p95_abs_mm"], max_abs_mm=s["max_abs_mm"],
                                              within_3mm_count=int(np.sum(np.abs(selected) <= 3.0)),
                                              within_3mm_pct=s["within3_pct"]))

        for marker, count in (("marker50", 4), ("marker60", 4), ("marker70", 1)):
            sequence_names = [(f"dynamic_track_{i:02d}", f"DT-{i}") for i in range(1, count + 1)]
            sequence_names.append(("continuous_static_20_100", "DJ-20-100"))
            for sequence, label in sequence_names:
                case = ev / marker / sequence
                summary = read_json(case / "summary.json")
                if summary.get("status") != "ok":
                    continue
                rows = read_csv(case / "paired_trajectories.csv")
                times, camera = vectors(rows, "camera")
                _, gt = vectors(rows, "gt")
                all_stats = stats(pairwise_errors(camera, gt))
                by_interval = {}
                for seconds in INTERVALS:
                    measured = interval_errors(times, camera, gt, seconds)
                    value = stats(measured)
                    by_interval[seconds] = value
                    intervals.append(dict(marker=marker, sequence=label, camera=side,
                                           interval_s=seconds, pair_count=value["n"],
                                           error_rmse_mm=value["rmse_mm"], error_mae_mm=value["mae_mm"],
                                           signed_bias_mm=value["bias_mm"], p95_abs_error_mm=value["p95_abs_mm"],
                                           max_abs_error_mm=value["max_abs_mm"],
                                           within_3mm_count=int(np.sum(np.abs(measured) <= 3.0)),
                                           within_3mm_pct=value["within3_pct"]))
                is_track = sequence.startswith("dynamic_track_")
                dynamic_tracks.append(dict(marker=marker, sequence=label, kind="DT" if is_track else "DJ",
                                           camera=side, paired_groups=summary["paired_groups"],
                                           eligible_gt_groups=summary["gt_groups_eligible"],
                                           paired_coverage_pct=summary["gt_group_coverage"] * 100,
                                           replay_frame_coverage_pct=summary["replay"].get("replay_frame_coverage", 0) * 100,
                                           time_offset_s=summary["time_offset_s"],
                                           all_pairs_rmse_mm=all_stats["rmse_mm"],
                                           all_pairs_p95_abs_mm=all_stats["p95_abs_mm"],
                                           all_pairs_within_3mm_pct=all_stats["within3_pct"],
                                           one_s_pair_count=by_interval[1.0]["n"],
                                           one_s_rmse_mm=by_interval[1.0]["rmse_mm"],
                                           one_s_p95_abs_mm=by_interval[1.0]["p95_abs_mm"],
                                           one_s_within_3mm_pct=by_interval[1.0]["within3_pct"],
                                           five_s_rmse_mm=by_interval[5.0]["rmse_mm"],
                                           aligned_trajectory_rmse_mm=summary["trajectory_rmse_mm"],
                                           quality_flags=";".join(summary.get("quality_flags", []))))

                if marker == "marker60" and sequence == "continuous_static_20_100":
                    displacement = np.linalg.norm(gt - gt[0], axis=1)
                    for limit in (0.4, 0.6, 0.8, 1.0, float(displacement.max())):
                        use = displacement <= limit + 1e-12
                        s = stats(pairwise_errors(camera[use], gt[use]))
                        range_sensitivity.append(dict(marker="marker60_continuous", camera=side,
                                                      range_basis="laser_displacement_from_first_paired_position",
                                                      range_unit="m", range_start=0, range_end=limit,
                                                      pair_count=s["n"], pair_rmse_mm=s["rmse_mm"],
                                                      p95_abs_mm=s["p95_abs_mm"], max_abs_mm=s["max_abs_mm"],
                                                      within_3mm_count=None, within_3mm_pct=s["within3_pct"]))

    out = root / "details"
    write_csv(out / "static_pose_jitter.csv", static_jitter)
    write_csv(out / "static_pair_errors_all.csv", static_pairs)
    write_csv(out / "dynamic_sequence_metrics.csv", dynamic_tracks)
    write_csv(out / "dynamic_interval_metrics.csv", intervals)
    write_csv(out / "range_sensitivity.csv", range_sensitivity)
    bundle = out / "accuracy_detail_tables_20261030.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("static_pose_jitter.csv", "static_pair_errors_all.csv", "dynamic_sequence_metrics.csv",
                     "dynamic_interval_metrics.csv", "range_sensitivity.csv"):
            archive.write(out / name, arcname=name)
        for name in ("evaluation_protocol.json",):
            archive.write(root / "evaluation_left_filtered" / name, arcname=f"provenance/{name}")
            archive.write(root / "evaluation_right_filtered" / name, arcname=f"provenance/right_{name}")
    return dict(static_jitter=static_jitter, static_pairs=static_pairs, dynamic=dynamic_tracks,
                intervals=intervals, ranges=range_sensitivity, bundle=bundle)


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def pct(value):
    return "—" if value is None else f"{float(value):.1f}%"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    assert all(len(row) == len(headers) for row in rows), headers
    result = ["| " + " | ".join(map(str, headers)) + " |",
              "| " + " | ".join(["---"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(result)


def xml_escape(value) -> str:
    return html.escape(str(value), quote=False)


def xml_table(headers: list[object], rows: list[list[object]]) -> str:
    assert all(len(row) == len(headers) for row in rows), headers
    def cell(value, tag):
        return f"<{tag}><p>{xml_escape(value)}</p></{tag}>"
    head = "<thead><tr>" + "".join(cell(value, "th") for value in headers) + "</tr></thead>"
    body = "<tbody>" + "".join("<tr>" + "".join(cell(value, "td") for value in row) + "</tr>" for row in rows) + "</tbody>"
    return f"<table>{head}{body}</table>"


def build_reports(root: Path, draft_xml: Path | None = None) -> Path:
    data = write_detail_csvs(root)
    static = {}
    for marker in ("marker50", "marker60"):
        for side in SIDES:
            ev = root / f"evaluation_{side}_filtered"
            static[(marker, side)] = dict(
                summary=read_json(ev / marker / "static_summary.json"),
                points=read_json(ev / marker / "static_points.json"),
                pairs=read_csv(ev / marker / "static_pairs.csv"))

    # Static jitter table by nominal position.
    jitter_rows = []
    for distance in range(20, 101, 10):
        row = [distance]
        for marker in ("marker50", "marker60"):
            for side in SIDES:
                point = next(p for p in static[(marker, side)]["points"] if p["distance_cm"] == distance)
                row.append(f"{fmt(point.get('jitter_rms_mm'))} / {fmt(point.get('jitter_p95_mm'))}")
        jitter_rows.append(row)

    # Static cumulative 20 cm-to-endpoint pair metrics.
    cum = {(r["marker"], r["camera"], int(r["range_end"])): r for r in data["ranges"]
           if r["marker"] in ("marker50", "marker60")}
    cumulative_tables = []
    for marker in ("marker50", "marker60"):
        rows = []
        for endpoint in range(30, 101, 10):
            l, r = cum[(marker, "left", endpoint)], cum[(marker, "right", endpoint)]
            rows.append([f"20–{endpoint}", l["pair_count"], fmt(l["pair_rmse_mm"]),
                         f"{int(l['within_3mm_count'])}/{l['pair_count']} ({pct(l['within_3mm_pct'])})",
                         fmt(r["pair_rmse_mm"]),
                         f"{int(r['within_3mm_count'])}/{r['pair_count']} ({pct(r['within_3mm_pct'])})"])
        cumulative_tables.append((marker, rows))

    # Static max range for two explicit definitions.
    range_rows = []
    for marker in ("marker50", "marker60"):
        for side in SIDES:
            values = [cum[(marker, side, endpoint)] for endpoint in range(30, 101, 10)]
            rmse_max = max((int(value["range_end"]) for value in values if value["pair_rmse_mm"] <= 3.0), default=None)
            every_max = 20
            for endpoint in range(30, 101, 10):
                value = cum[(marker, side, endpoint)]
                if value["max_abs_mm"] <= 3.0:
                    every_max = endpoint
                else:
                    break
            range_rows.append([marker, "左目" if side == "left" else "右目",
                               f"20–{rmse_max} cm" if rmse_max else "未达",
                               f"20–{every_max} cm"])

    # Dynamic sequence rows and interval pass records.
    by_track = {(r["marker"], r["sequence"], r["camera"]): r for r in data["dynamic"]}
    track_rows = []
    for marker, count in (("marker50", 4), ("marker60", 4), ("marker70", 1)):
        for i in range(1, count + 1):
            label = f"DT-{i}"
            left, right = by_track[(marker, label, "left")], by_track[(marker, label, "right")]
            track_rows.append([f"{marker} / {label}",
                               f"{fmt(left['paired_coverage_pct'],1)} / {fmt(right['paired_coverage_pct'],1)}",
                               f"{fmt(left['time_offset_s'],3)} / {fmt(right['time_offset_s'],3)}",
                               f"{fmt(left['all_pairs_rmse_mm'])} / {fmt(right['all_pairs_rmse_mm'])}",
                               f"{fmt(left['one_s_rmse_mm'])} / {fmt(right['one_s_rmse_mm'])}",
                               f"{pct(left['one_s_within_3mm_pct'])} / {pct(right['one_s_within_3mm_pct'])}"])

    continuous_rows = []
    for marker in MARKERS:
        left, right = by_track[(marker, "DJ-20-100", "left")], by_track[(marker, "DJ-20-100", "right")]
        continuous_rows.append([marker, f"{fmt(left['paired_coverage_pct'],1)} / {fmt(right['paired_coverage_pct'],1)}",
                                f"{fmt(left['replay_frame_coverage_pct'],1)} / {fmt(right['replay_frame_coverage_pct'],1)}",
                                f"{fmt(left['all_pairs_rmse_mm'])} / {fmt(right['all_pairs_rmse_mm'])}",
                                f"{fmt(left['one_s_rmse_mm'])} / {fmt(right['one_s_rmse_mm'])}",
                                f"{pct(left['one_s_within_3mm_pct'])} / {pct(right['one_s_within_3mm_pct'])}"])

    three_mm_dynamic = []
    for r in data["intervals"]:
        if (r["sequence"].startswith("DT-") and r["interval_s"] in (0.5, 1.0)
                and r["error_rmse_mm"] <= 3.0):
            three_mm_dynamic.append([r["marker"], r["sequence"], "左目" if r["camera"] == "left" else "右目",
                                     f"{r['interval_s']:.1f}s", fmt(r["error_rmse_mm"]),
                                     fmt(r["p95_abs_error_mm"]), pct(r["within_3mm_pct"])])

    # Continuous marker60 by laser displacement distance from first paired group.
    range60 = [r for r in data["ranges"] if r["marker"] == "marker60_continuous"]
    range60_rows = []
    for limit in (0.4, 0.6, 0.8, 1.0):
        left = next(r for r in range60 if r["camera"] == "left" and abs(float(r["range_end"]) - limit) < 1e-8)
        right = next(r for r in range60 if r["camera"] == "right" and abs(float(r["range_end"]) - limit) < 1e-8)
        range60_rows.append([f"≤{limit:.1f} m", f"{left['pair_count']} / {right['pair_count']}",
                             f"{fmt(left['pair_rmse_mm'])} / {fmt(right['pair_rmse_mm'])}",
                             f"{fmt(left['p95_abs_mm'])} / {fmt(right['p95_abs_mm'])}",
                             f"{pct(left['within_3mm_pct'])} / {pct(right['within_3mm_pct'])}"])

    # Time-scan sensitivity examples, using the recorded scan grid.
    time_sensitivity = []
    for marker, sequence, camera in (("marker50", "dynamic_track_03", "left"),
                                     ("marker70", "dynamic_track_01", "left"),
                                     ("marker60", "continuous_static_20_100", "left")):
        path = root / f"evaluation_{camera}_filtered" / marker / sequence / "time_offset_scan.csv"
        rows = read_csv(path)
        eligible = [r for r in rows if r.get("fit_rmse_mm") not in (None, "")]
        best = min(eligible, key=lambda r: float(r["fit_rmse_mm"]))
        best_offset = float(best["offset_s"])
        def nearest(delta):
            target = best_offset + delta
            row = min(eligible, key=lambda r: abs(float(r["offset_s"]) - target))
            return float(row["fit_rmse_mm"])
        time_sensitivity.append([marker, sequence.replace("dynamic_track_", "DT-").replace("continuous_static_20_100", "DJ"),
                                 f"{best_offset:.3f}", fmt(float(best["fit_rmse_mm"])),
                                 fmt(nearest(-0.05)), fmt(nearest(0.05)),
                                 "时移改变 ±50 ms 时的同段 SE(3) 拟合残差；不是同步精度"])

    # Gather frame coverage for every input recording, both cameras.
    replay_rows = []
    for marker in MARKERS:
        cases = []
        if marker in ("marker50", "marker60"):
            cases += [(f"JT-{d}", f"static_{d:03d}cm") for d in range(20, 101, 10)]
        cases += [("DJ-20-100", "continuous_static_20_100")]
        tracks = 4 if marker in ("marker50", "marker60") else 1
        cases += [(f"DT-{i}", f"dynamic_track_{i:02d}") for i in range(1, tracks + 1)]
        for label, seq in cases:
            pair = []
            for side in SIDES:
                summary_path = root / f"evaluation_{side}_filtered" / marker
                if seq.startswith("static_"):
                    point = next(p for p in read_json(summary_path / "static_points.json") if p["distance_cm"] == int(seq[7:10]))
                    cov = point.get("coverage", 0) * 100
                else:
                    summary = read_json(summary_path / seq / "summary.json")
                    cov = summary.get("replay", {}).get("replay_frame_coverage", 0) * 100
                pair.append(f"{cov:.1f}%")
            replay_rows.append([marker, label, *pair])

    # Main report (Markdown).
    report = []
    report += [
        "# 2026-10-30 Marker 全量精度评估",
        "",
        "**评估日期：2026-10-30；状态：本批数据的完整初步评估，仍需补做独立外参与重复性验证。**",
        "",
        "## 一、结论先看",
        "",
        "本批数据共 30 段录像，左右目共 60 次基线回放；另对 marker60 连续录像做了左右目各一次同 ID 干扰 marker 排除，共 62 次回放结果。使用用户提供的 Kalibr 相机内参后，大部分动态轨迹的相对位移误差 RMSE 为 5.25–10.59 mm；静态点位之间的全 20–100 cm 距离误差 RMSE 为 10.88–16.81 mm。**目前不能概括为整个系统达到 3 mm。**",
        "",
        "3 mm 条件依指标而变：静止时的相机位置抖动，在部分 marker / 距离可达到 RMS ≤3 mm；静态点位间距的累计误差 RMSE ≤3 mm 仅在靠近的一小段标称距离范围内成立；动态位移只有少数 0.5–1 秒的单目 / 单轨迹区间达到 RMSE ≤3 mm，且 P95 仍大于 3 mm，不能说全部采样都在 3 mm 内。",
        "",
        "marker60 连续走停序列实际激光首末位移约 1.293 m，相机估计深度延伸至约 1.66–1.67 m；因此该数据不能当作只覆盖 20–100 cm 的录像。全段动态结果单独列示，不并入近距离总体结论。",
        "",
        "## 二、3 mm 指标：达成条件与未达成部分",
        "",
        "| 评估对象 | 明确的 3 mm 判据 | 当前观测到的范围 / 条件 | 可下的结论 |",
        "| --- | --- | --- | --- |",
        "| 静止位置抖动 | 每个静态录像中，逐帧 Tag 中心到该段中心位置的径向误差 RMS ≤3 mm；另看 P95 | marker50：左目 20–70 cm、右目 20–80 cm 的 RMS ≤3；若要求 P95 ≤3，左目为20–60cm、右目为20–70cm。marker60：左目 20–90、右目 20–80 cm RMS ≤3；两目 P95 ≤3 为 20–80 cm | 是单段静止重复性；不代表绝对位置精度或跨点精度。marker50 左目 80 cm RMS 被少量跳点抬高 |",
        "| 静态点间距 | 以 20 cm 点为起点，对窗口内所有点对的激光 / 相机位移长度差计算 RMSE；严格判据另要求所有点对绝对误差 ≤3 mm | RMSE ≤3 mm 最远窗口：marker50 左 20–40、右 20–50；marker60 左 20–50、右 20–60 cm。若要求窗口内每个点对都 ≤3 mm：marker50 左右目均到 40 cm；marker60 左到 40、右到 50 cm | 20–100 cm 全范围不达 3 mm；全范围 RMSE 为 10.88–16.81 mm |",
        "| 动态轨迹，1 秒位移 | 每段录像中相隔约 1 秒的相对位移长度误差 RMSE ≤3 mm；同时看 P95 和 ≤3 的样本比例 | 9 段 DT 中，仅 marker60 DT-2 右目为 2.81 mm（P95 5.64 mm，71.6% 样本 ≤3）。marker50 / 60 / 70 的 DT 其余 17 个相机结果均 >3 mm | 单目、单段短时结果，不是整条轨迹达到 3 mm |",
        "| 动态轨迹，0.5 秒位移 | 相隔约 0.5 秒的相对位移长度误差 RMSE ≤3 mm | marker60 DT-1 左右目、DT-2 左右目、DT-3 右目共 5 个结果 RMSE ≤3 mm；其 P95 均 >3 mm | 轨迹、相机和时间间隔限定下的局部结果；不能推广到 marker60 的全部轨迹或更长位移 |",
        "| 连续走停录像 | 同上，按 1 秒间隔计算 | marker70 DJ 右目 2.93 mm（P95 4.42，84.8% ≤3）；左目 3.03 mm。marker50 DJ 为 5.23 / 6.61；marker60 DJ 全段为 12.58 / 14.20 mm | 仅 marker70 右目在这一统计口径下低于 3 mm RMSE，不代表全段所有时刻都低于 3 mm |",
        "",
        "**口径提醒：**“RMSE ≤3 mm”不等于“每次误差都 ≤3 mm”。即便 RMSE 达到 3 mm，以上达标的动态结果 P95 仍高于 3 mm；动态样本有时间重叠，百分比是描述性比例，不是独立抽样置信度。",
        "",
        "## 三、数据对应关系与覆盖",
        "",
        "| Marker | 黑码边长 / ID | 激光 GT 文件 | 录像目录与类别 |",
        "| --- | --- | --- | --- |",
        "| marker50 | 37.5 mm / ID 2 | `GT/marker_50mm/static.txt`、`dynamic1.txt`、`track1.txt`–`track4.txt` | `marker_50mm/marker50_JT-20`–`JT-100`（9 静态点）、`marker50_DJ-20-100`（连续走停）、`marker50_DT-1`–`DT-4`（4 动态轨迹） |",
        "| marker60 | 50 mm / ID 2 | `GT/marker_60mm/static.txt`、`dynamic1.txt`、`track1.txt`–`track4.txt` | `marker_60mm/marker_1030_JT-20`–`JT-100`（9 静态点）、`marker_1030_DJ-20-100`（连续走停）、`marker_1030_DT-1`–`DT-4`（4 动态轨迹） |",
        "| marker70 | 60 mm / ID 6 | `GT/marker_70mm/dynamic.txt`、`track.txt` | `marker_70mm/marker70-DJ-20-100`（连续走停）、`marker70-DT`（1 动态轨迹）；没有单独的 `static.txt` |",
        "",
        "所有 30 段原始录像均按左右相机回放。以下是各录像检测到有效位姿的帧覆盖率（左 / 右）；GT 配对覆盖率在动态结果表另列。",
        "",
        md_table(["Marker", "序列", "左目有效帧", "右目有效帧"], replay_rows),
        "",
        "marker60 连续录像里，图像左下区域出现了一个静止的同 ID 2 marker。基线回放覆盖率为左 34.7%、右 43.5%。对该序列单独使用图像中心排除框 `[0,700,300,1000]`（像素，X0/Y0/X1/Y1），排除位于左目约 (165,872)、右目约 (134,796) 的静止目标后，目标帧覆盖率提升至左 96.4%、右 99.6%；其余序列未应用排除框。",
        "",
        "## 四、评估参数、相机标定与指标定义",
        "",
        "左右相机分别使用 Kalibr 鱼眼等距畸变内参进行单目 PnP；本次没有使用双目基线进行三角化或双目融合。图像分辨率 1920×1080；检测最短边阈值 12 px、最大重投影误差 2.5 px。时间偏移网格步长 50 ms、搜索窗口 ±20 s；视觉轨迹插值到激光GT时刻，最大间隔 0.1 s，仅相邻有效视频帧之间插值；激光重复时间组要求 10 条样本且组内最大离散度不超过 60 mm。",
        "",
        "| 指标 | 算法与单位 | 解读边界 |",
        "| --- | --- | --- |",
        "| 静止位置抖动 | 各帧 Tag 中心相对该静态段位置中位数的三维径向距离，报告 RMS 和 P95，mm | 是单段重复性；对绝对位置偏差不敏感 |",
        "| 静态点间距 | 相机两静态点中心距离减激光两点距离；9 个点最多 36 个点对，报告误差 RMSE / MAE / 有符号偏差，mm | 不做尺度拟合或空间拟合；9 点各只有一次静态采集 |",
        "| 动态全点对位移 | 两个有效配对时刻之间相机位移长度减激光位移长度，遍历所有点对，mm | 时间偏移由同一段数据拟合；点对共用轨迹时刻，彼此相关；不做尺度拟合 |",
        "| 固定时间间隔位移 | 相隔 0.5 / 1 / 2 / 5 秒附近（±35 ms 容差）的位移长度误差；报告 RMSE、P95、≤3 mm 比例 | 按时间匹配后逐段计算；重叠窗口不是独立样本 |",
        "| 对齐轨迹 RMSE | 在同一段数据拟合时间偏移与刚体 SE(3) 后的位置残差，mm | 仅作事后对齐描述，不是独立绝对位置准确度；近直线运动会导致拟合退化 |",
        "",
        "### 用户提供的 Kalibr 相机参数",
        "",
        "内参按本批回放配置的均值使用；± 为 Kalibr 给出的不确定度。相机的重投影误差和 Aprilgrid 目标信息一并保留，供追溯。",
        "",
        "| 相机 | 投影 `[fx, fy, cx, cy]` | 等距畸变 `[k1,k2,k3,k4]` | 标定重投影误差 `[x,y]` px |",
        "| --- | --- | --- | --- |",
        "| cam0 左目 | `[834.92069, 834.51968, 929.07920, 585.15624]` ± `[0.03847,0.04145,0.17097,0.16018]` | `[-0.04227072,0.01086145,-0.01410186,0.00405435]` ± `[0.00115644,0.00277713,0.00259498,0.00083234]` | `[0.000001,0.000000]` ± `[0.225482,0.205627]` |",
        "| cam1 右目 | `[825.33512, 825.30373, 929.84660, 531.14802]` ± `[0.03715,0.04032,0.16570,0.15745]` | `[-0.03523133,-0.00846293,0.00478187,-0.00207974]` ± `[0.00117249,0.00294900,0.00287736,0.00096261]` | `[-0.000002,0.000000]` ± `[0.221889,0.204651]` |",
        "",
        "双目基线（本次未用于结果）：`q_1_0=[0.00070121,-0.00063618,-0.00186418,0.99999781]`；`t_1_0=[-0.08711653,0.00051678,0.00005623] m`。标定板为 6×6 Aprilgrid，tag size 0.1 m，spacing 0.03 m。",
        "",
        "## 五、静态点：逐距离抖动与点间精度",
        "",
        "下表每个单元格为 `抖动 RMS / P95 (mm)`。P95 表示该静态录像中 95% 的帧距中位位置不超过该径向误差。",
        "",
        md_table(["标称距离 cm", "marker50 左", "marker50 右", "marker60 左", "marker60 右"], jitter_rows),
        "",
        "静态点对误差偏正，说明估计的点间距离通常长于激光距离。20→100 cm 端到端距离误差为 marker50 左 / 右 +23.83 / +31.96 mm，marker60 +21.42 / +24.98 mm。全范围 36 个点对的带符号平均偏差为 marker50 +8.97 / +13.58 mm、marker60 +9.10 / +9.17 mm。",
        "",
        "### 从 20 cm 起累计点对误差",
        "",
        "每行把 20 cm 到该终点之间所有点对放入 RMSE；括号中为绝对误差 ≤3 mm 的点对数与比例。",
    ]
    for marker, rows in cumulative_tables:
        report += ["", f"#### {marker}", "", md_table(["距离窗口 cm", "点对数", "左目 RMSE mm", "左目 ≤3 mm 点对", "右目 RMSE mm", "右目 ≤3 mm 点对"], rows)]
    report += ["", "### 相邻 10 cm 段的距离误差", "", "正数表示相机估出的距离偏长；负数表示偏短。", ""]
    for marker in ("marker50", "marker60"):
        l = static[(marker, "left")]["pairs"]
        r = static[(marker, "right")]["pairs"]
        l_by = {(int(x["from_cm"]), int(x["to_cm"])): float(x["error_mm"]) for x in l if int(x["to_cm"]) - int(x["from_cm"]) == 10}
        r_by = {(int(x["from_cm"]), int(x["to_cm"])): float(x["error_mm"]) for x in r if int(x["to_cm"]) - int(x["from_cm"]) == 10}
        rows = [[f"{start}–{start+10}", f"{l_by[(start,start+10)]:+.2f}", f"{r_by[(start,start+10)]:+.2f}"] for start in range(20, 91, 10)]
        report += [f"#### {marker}", "", md_table(["点位段 cm", "左目误差 mm", "右目误差 mm"], rows), ""]
    report += [
        "静态抖动存在距离相关差异，也不能只看平均。marker50 左目 80 cm 录像 99 帧中，4 帧相对中位位置跳变超过 20 mm，最大约 54.0 mm；单看 RMS 为 10.72 mm，P95 为 6.63 mm，属于少量跳点拉高 RMS 的风险。相同点右目 RMS 为 2.67 mm、最大偏离约 6.33 mm。",
        "",
        "## 六、9 段动态轨迹：全点对与固定时间间隔",
        "",
        "以下值为左 / 右目。覆盖率是匹配到的激光时间组 / 合格 GT 时间组；时移单位秒；误差为 mm。全点对 RMSE 和 1 秒位移 RMSE 是两种不同基线，不应相互替代。",
        "",
        md_table(["Marker / 轨迹", "GT 配对覆盖 L/R", "拟合时移 L/R", "全点对 RMSE L/R", "1 秒 RMSE L/R", "1 秒误差 ≤3 mm 比例 L/R"], track_rows),
        "",
        "### 动态 3 mm 的短时条件",
        "",
        "下表列出本次 0.5 或 1 秒位移 RMSE ≤3 mm 的全部 DT 相机结果。P95 仍超过 3 mm，且样本比例列表示误差本身不超过 3 mm 的窗口比例。",
        "",
        md_table(["Marker", "轨迹", "相机", "间隔", "RMSE mm", "P95 绝对误差 mm", "窗口误差 ≤3 mm"], three_mm_dynamic),
        "",
        "覆盖率、时移、误差与 `≤3 mm` 比例来自每段配对后的有效时间窗口。时移扫描步长为 50 ms，输出的三位小数是网格采样结果，不代表微秒/毫秒级的时钟标定精度。",
        "",
        "## 七、3 段连续走停录像",
        "",
        "连续走停录制含停顿和移动段；完整录制中的停顿和运动均保留。表中配对覆盖为合格 GT 组覆盖，帧覆盖是视觉检测位姿的帧比例。",
        "",
        md_table(["Marker / DJ", "GT 配对覆盖 L/R", "有效帧覆盖 L/R", "全点对 RMSE L/R mm", "1 秒 RMSE L/R mm", "1 秒误差 ≤3 mm 比例 L/R"], continuous_rows),
        "",
        "### marker60 连续轨迹的超范围部分",
        "",
        "按首个已配对激光位置作为位移起点，首末位移约 1.293 m。下表的距离是激光位置的相对位移阈值，不是相机到 Marker 的量距；全段结果仍完整保留。",
        "",
        md_table(["距激光起点位移", "点对数 L/R", "所有点对 RMSE L/R mm", "P95 绝对误差 L/R mm", "所有点对中 ≤3 mm 比例 L/R"], range60_rows),
        "",
        "全段偏差明显为正：所有点对的平均带符号误差为左 +25.58 mm、右 +25.04 mm；P95 绝对误差为 64.27 / 63.14 mm。1 秒位移误差 RMSE 为 12.58 / 14.20 mm。该序列拟合时移约 −10.59 / −10.64 s，但其最佳偏移附近较宽范围内拟合残差变化很小，时移可辨识性较弱；不宜把该偏移视为准确的时钟差测量。",
        "",
        "## 八、时间对齐敏感性与报告解读",
        "",
        "时间扫描以同一序列的刚体对齐残差选择时移。下表给出最佳时移及偏移 ±50 ms 时的残差示例；每个偏移均重新拟合 SE(3)，有效配对样本也可能变化，因此不是固定公共重叠集上的纯时移敏感性比较。近直线运动会使时移和刚体变换部分互相补偿，所以对齐轨迹 RMSE 仅作辅助观察。",
        "",
        md_table(["Marker", "轨迹", "最佳时移 s", "最佳拟合 RMSE mm", "时移 −50 ms RMSE", "时移 +50 ms RMSE", "说明"], time_sensitivity),
        "",
        "marker50 DT-3 左目最佳拟合残差约 6.12 mm，偏移 ±50 ms 时变为约 12.46 / 11.77 mm；marker70 DT-1 左目由约 8.94 mm 变为 13.36 / 12.80 mm。多条连续轨迹接近直线，刚体拟合协方差条件较差；例如 50 / 60 / 70 连续序列的拟合第二/第一奇异值比约为 0.000364 / 0.000024 / 0.001961。不能用同段自由 SE(3) 拟合证明外参准确。",
        "",
        "## 九、整体判断与剩余缺口",
        "",
        "1. **3 mm 静止重复性有条件成立：** 近距离、静止、指定 marker 和单目下，多数点位姿抖动 RMS <3 mm；marker50 的 P95 门槛为左20–60、右20–70cm，marker60 大致为 20–80 cm。远端静态点抖动上升并有跳点。",
        "2. **3 mm 静态点间距仅限近距离子范围：** 使用所有点对的累计 RMSE 判据，最大范围为 marker50 左 20–40、右 20–50，marker60 左20–50、右20–60 cm；全点对都不超过 3 mm 的严格范围更短。完整 20–100 cm 结果约 11–17 mm RMSE。",
        "3. **3 mm 动态准确度尚未普遍达到：** 动态轨迹全点对 RMSE 为 5.25–10.59 mm；少数 0.5–1 秒窗的 RMSE 低于 3 mm，但 P95 仍超过 3 mm。连续走停的 marker60 明显更差，marker70 的右目 1 秒结果为 2.93 mm，但只有一条动态轨迹/一段连续数据支撑。",
        "4. **绝对位置和姿态精度仍未被本批 GT 证明：** 激光测量的是激光靶点，相机 PnP 输出 Tag 中心，两者固定外参/杠杆臂没有实测；激光只有 XYZ 没有姿态真值。动态旋转会把点位差带入位移对比。",
        "5. **置信界限不能从点对个数直接推导：** 静态每个标称距离仅一次采集；动态全点对复用大量相同时间点并高度相关。本报告报告观测 RMSE、P95 和比例，不以相关样本假定独立来构造置信区间。",
        "",
        "### 下一轮最有价值的补测",
        "",
        "- 在 20–100 cm 按每 10 cm 做多个独立静态重复，报告每次独立试验的误差分布，优先复查 marker50 左目 80 cm 偶发跳点及 90–100 cm 静态距离偏长。",
        "- 用标定板/测量夹具标定激光靶点到 Tag 中心的固定变换，并为目标提供姿态真值，才能把相对位移精度与绝对位姿/姿态精度分开。",
        "- 对时间偏移在最佳点附近以更细步长做局部扫描，并在固定公共重叠区间比较，特别是近直线/带停顿轨迹。",
        "- 如产品要求是“所有 1 秒窗口均 ≤3 mm”或“RMSE ≤3 mm”，需把判据写入验收规范，并以跨距离、跨轨迹、左右目都满足为整体验收条件。",
        "",
        "## 附件与复算入口",
        "",
        "本报告附有静态逐点对误差、逐距离抖动、动态序列指标、0.5/1/2/5 秒窗口误差和范围敏感性 CSV 压缩包；全部摘要与图表可在本地评估目录逐项回查。",
        "",
        "- 输入数据：`/home/dex/20261030/GT` 和 `/home/dex/20261030/marker_10_30`。",
        "- Kalibr 配置：`configs/calibration/uvc_20261030_camchain.yaml`。",
        "- 左 / 右目完整复核报告：`evaluation_left_filtered/report.md`、`evaluation_right_filtered/report.md`。",
        "- 左 / 右目汇总 JSON：`evaluation_left_filtered/summary.json`、`evaluation_right_filtered/summary.json`；每个数据段目录有 `paired_trajectories.csv`、`time_offset_scan.csv` 和配对图。",
        "- 本报告使用的是 marker60 连续序列过滤版：`manifest_filtered_left.json`、`manifest_filtered_right.json` 指向过滤版回放。",
        "- 汇总脚本：`tools/build_accuracy_20261030_report.py`；作图脚本：`tools/plot_accuracy_20261030_report.py`。",
    ]
    markdown = "\n".join(report) + "\n"
    md_path = root / "RESULTS.md"
    md_path.write_text(markdown, encoding="utf-8")

    # Include every interval and every static point pair in the published body.
    full_tables = []
    for marker in ("marker50", "marker60"):
        left = static[(marker, "left")]["pairs"]
        right = {(r["from_cm"], r["to_cm"]): r for r in static[(marker, "right")]["pairs"]}
        rows = [[f"{r['from_cm']}–{r['to_cm']}", fmt(float(r["laser_mm"])),
                 fmt(float(r["error_mm"])),
                 fmt(float(right[(r["from_cm"], r["to_cm"])]["error_mm"]))] for r in left]
        full_tables.append((f"{marker}：全部36组静态点对",
                            ["标称点位cm", "激光间距mm", "左目有符号误差mm", "右目有符号误差mm"], rows))
    for marker in MARKERS:
        rows = []
        for r in data["intervals"]:
            if r["marker"] == marker:
                rows.append([r["sequence"], "左" if r["camera"] == "left" else "右",
                             fmt(r["interval_s"], 1), r["pair_count"], fmt(r["error_rmse_mm"]),
                             fmt(r["p95_abs_error_mm"]), pct(r["within_3mm_pct"])])
        full_tables.append((f"{marker}：全部固定间隔位移结果",
                            ["序列", "相机", "间隔s", "窗口数", "RMSE mm", "P95 mm", "≤3mm比例"], rows))
    for marker in MARKERS:
        rows = [[r["sequence"], "左" if r["camera"] == "left" else "右",
                 f"{r['paired_groups']}/{r['eligible_gt_groups']}", fmt(r["all_pairs_rmse_mm"]),
                 fmt(r["all_pairs_p95_abs_mm"]), pct(r["all_pairs_within_3mm_pct"]),
                 fmt(r["aligned_trajectory_rmse_mm"])] for r in data["dynamic"] if r["marker"] == marker]
        full_tables.append((f"{marker}：全点对误差及对齐残差",
                            ["序列", "相机", "GT配对数", "点对RMSE mm", "点对P95 mm", "点对≤3mm", "对齐RMSE mm"], rows))
    with md_path.open("a", encoding="utf-8") as stream:
        stream.write("\n## 完整数值明细\n\n对齐RMSE为同段事后拟合值，不是绝对位置精度。\n")
        for title, headers, rows in full_tables:
            stream.write(f"\n### {title}\n\n" + md_table(headers, rows) + "\n")

    if draft_xml is not None:
        blocks = ['<title>2026-10-30 Marker 全量精度评估与 3 mm 指标边界</title>']
        def heading(level, text):
            blocks.append(f'<h{level} seq="auto">{xml_escape(text)}</h{level}>')
        def para(text):
            blocks.append(f'<p>{xml_escape(text)}</p>')
        def table(headers, rows):
            blocks.append(xml_table(headers, rows))

        heading(1, "结论先看")
        para("本批覆盖 30 段录像，左右目共 60 次基线回放；marker60 连续录像的同 ID 干扰 marker 另用排除框重跑左右目，共 62 次回放结果。使用本次提供的 Kalibr 内参后，大部分动态轨迹相对位移误差 RMSE 为 5.25–10.59 mm；静态点间距全 20–100 cm 的 RMSE 为 10.88–16.81 mm。整体尚不能称为达到 3 mm。")
        para("3 mm 有条件成立：部分静止距离的单段位置抖动 RMS 达标；静态点间累计距离 RMSE 达标范围仅约 20–40/50/60 cm；动态只有少量 0.5–1 秒窗口达到 RMSE≤3 mm，且 P95 仍大于 3 mm。")
        para("marker60 连续序列激光首末位移约 1.293 m，相机估计深度延伸到约 1.66–1.67 m，不应将全段按 20–100 cm 解读。")

        heading(1, "3 mm 指标：哪些条件下达到")
        table(["评估对象", "3 mm 判据", "本批达到条件", "结论"], [
            ["静止位置抖动", "单段逐帧位置相对中位数的径向 RMS≤3 mm", "marker50 左20–70cm、右20–80cm；marker60 左20–90cm、右20–80cm", "单段重复性，不代表绝对位置"],
            ["静止抖动 P95", "95%帧径向误差≤3 mm", "marker50 左20–60、右20–70cm；marker60 左右20–80cm", "同时关注尾部误差"],
            ["静态点间距", "20cm起窗口内所有点对 RMSE≤3mm", "marker50 左20–40右20–50；marker60 左20–50、右20–60cm", "全20–100cm不满足"],
            ["静态点对严格条件", "窗口中每个点对绝对误差≤3mm", "marker50左右20–40；marker60左20–40、右20–50cm", "严格范围更短"],
            ["动态1秒位移", "该间隔的位移长度误差RMSE≤3mm", "仅marker60 DT-2右目：2.81mm；P95=5.64mm、71.6%窗口≤3mm", "单目单轨迹，不是整段全部≤3mm"],
            ["动态0.5秒位移", "该间隔的位移长度误差RMSE≤3mm", "marker60 DT-1左右、DT-2左右、DT-3右目共5项", "轨迹与相机限定；P95均>3mm"],
            ["连续走停1秒位移", "1秒间隔位移RMSE≤3mm", "仅marker70右目2.93mm；左3.03mm", "右目P95仍4.42mm，84.8%窗口≤3mm"],
        ])
        para("RMSE≤3mm不等于每个样本都≤3mm。所有动态 3mm 达标例子的 P95 仍超过 3mm。")

        heading(1, "数据对应关系")
        table(["Marker", "黑码边长 / ID", "激光 GT 文件", "录像与类别"], [
            ["marker50", "37.5mm / ID 2", "marker_50mm/static.txt、dynamic1.txt、track1–4.txt", "JT-20…JT-100；DJ-20-100；DT-1…DT-4"],
            ["marker60", "50mm / ID 2", "marker_60mm/static.txt、dynamic1.txt、track1–4.txt", "JT-20…JT-100；DJ-20-100；DT-1…DT-4"],
            ["marker70", "60mm / ID 6", "marker_70mm/dynamic.txt、track.txt；无独立 static.txt", "DJ-20-100；一段 DT"],
        ])
        para("完整 ROS bag 目录映射：marker50 为 marker_50mm/marker50_JT-20…JT-100、marker50_DJ-20-100、marker50_DT-1…4；marker60 为 marker_60mm/marker_1030_JT-20…JT-100、marker_1030_DJ-20-100、marker_1030_DT-1…4；marker70 为 marker_70mm/marker70-DJ-20-100、marker70-DT。")
        table(["Marker", "序列", "左目有效帧覆盖", "右目有效帧覆盖"], replay_rows)
        para("marker60 连续录像内有一个静止的同 ID 2 marker，基线覆盖率左34.7%、右43.5%。对其单独排除像素框 [0,700,300,1000] 后，目标帧覆盖率为左96.4%、右99.6%；其余录像不使用该排除框。")

        heading(1, "评估方法与参数")
        table(["指标", "定义", "限制"], [
            ["静止位置抖动", "逐帧Tag中心相对单段中位位置的三维径向误差，RMS/P95，mm", "重复性，不是绝对位置"],
            ["静态点间距", "9点最多36组点对：相机距离减激光距离，不拟合尺度或空间变换", "每个距离仅一次静态采集"],
            ["动态全点对", "时间对齐后所有时刻对的位移长度误差，mm", "时间偏移在同段拟合；样本点对相关"],
            ["固定间隔位移", "0.5/1/2/5秒间隔（±35ms）的位移误差，报告RMSE/P95/≤3比例", "重叠窗口非独立样本"],
            ["对齐轨迹RMSE", "同段拟合时移与刚体SE(3)后的坐标残差", "事后拟合值，不代表绝对位置或外参精度"],
        ])
        para("使用1920×1080 Kalibr内参，左右目分别进行单目PnP；最短边阈值12px、最大重投影误差2.5px；激光组要求每时刻10条样本、组内离散度≤60mm；有效视频帧插值最大间隔0.1s；时移扫描±20s、步长50ms。未使用双目基线。")
        table(["相机", "投影参数 [fx,fy,cx,cy] ± 标准差", "等距畸变参数 ± 标准差", "标定重投影误差 px"], [
            ["cam0左目", "[834.92069,834.51968,929.07920,585.15624] ± [0.03847,0.04145,0.17097,0.16018]", "[-0.04227072,0.01086145,-0.01410186,0.00405435] ± [0.00115644,0.00277713,0.00259498,0.00083234]", "[0.000001,0.000000] ± [0.225482,0.205627]"],
            ["cam1右目", "[825.33512,825.30373,929.84660,531.14802] ± [0.03715,0.04032,0.16570,0.15745]", "[-0.03523133,-0.00846293,0.00478187,-0.00207974] ± [0.00117249,0.00294900,0.00287736,0.00096261]", "[-0.000002,0.000000] ± [0.221889,0.204651]"],
        ])
        para("未用于本次估计的双目基线：q=[0.00070121,-0.00063618,-0.00186418,0.99999781]，t=[-0.08711653,0.00051678,0.00005623]m。标定板6×6 Aprilgrid，tag size 0.1m、spacing 0.03m。")

        heading(1, "静态逐距离结果")
        para("单元格为静止位置抖动 RMS / P95，单位mm。")
        table(["标称 cm", "marker50左", "marker50右", "marker60左", "marker60右"], jitter_rows)
        para("静态点间距误差全范围 RMSE：marker50左11.48mm、右16.81mm；marker60左10.88mm、右11.89mm。20→100cm端到端距离偏差：marker50左+23.83mm、右+31.96mm；marker60左+21.42mm、右+24.98mm。")
        for marker, rows in cumulative_tables:
            heading(2, f"{marker}：20cm起累计点对误差")
            table(["窗口 cm", "点对数", "左RMSE mm", "左≤3mm", "右RMSE mm", "右≤3mm"], rows)
        para("完整36组静态点间距离误差、逐距离抖动和位姿覆盖明细见附件CSV包。")
        blocks.append('<img path="@./outputs/accuracy_20261030_kalibr/figures/static_jitter_and_displacement.png" caption="静态抖动与从20cm点起算的有符号位移长度误差；3mm线仅为目标参考。"/>')
        blocks.append('<img path="@./outputs/accuracy_20261030_kalibr/figures/static_range_3mm.png" caption="静态点间累计距离RMSE随纳入距离范围的变化；包含20cm至终点之间全部点对。"/>')

        heading(1, "9段动态轨迹逐项结果")
        table(["Marker/DT", "GT配对覆盖L/R", "时移L/R s", "全点对RMSE L/R mm", "1秒RMSE L/R mm", "1秒窗口≤3mm比例 L/R"], track_rows)
        table(["Marker", "轨迹", "相机", "间隔", "RMSE mm", "P95 mm", "窗口≤3mm"], three_mm_dynamic)
        blocks.append('<img path="@./outputs/accuracy_20261030_kalibr/figures/dynamic_displacement_rmse.png" caption="9段动态轨迹的1秒位移和全点对位移RMSE；虚线表示用户关注的3mm目标。"/>')

        heading(1, "连续走停序列与marker60长范围异常")
        table(["Marker/DJ", "GT配对覆盖L/R", "帧覆盖L/R", "全点对RMSE L/R mm", "1秒RMSE L/R mm", "1秒≤3mm比例 L/R"], continuous_rows)
        table(["marker60激光相对位移范围", "点对数L/R", "全点对RMSE L/R mm", "P95绝对误差L/R mm", "点对≤3mm比例L/R"], range60_rows)
        blocks.append('<img path="@./outputs/accuracy_20261030_kalibr/figures/continuous_displacement_rmse.png" caption="三组连续走停录制的1秒位移与全点对位移RMSE；marker60全段范围较长。"/>')
        blocks.append('<img path="@./outputs/accuracy_20261030_kalibr/figures/marker60_extended_range.png" caption="marker60连续录像中，相对激光起点位移与位移长度偏差；横轴不是相机距离。"/>')
        para("marker60全段所有点对平均带符号误差+25.58/+25.04mm，左/右；P95绝对误差64.27/63.14mm。该序列时移可辨识性弱，不把约−10.6秒当作精确时钟差。")

        heading(1, "完整数值明细：全部静态点对与动态间隔")
        para("以下包含每个静态点对，以及全部DT、DJ序列的左右目0.5/1/2/5秒结果。P95为绝对误差分位数；≤3mm比例按绝对误差计算。对齐RMSE是同段事后刚体拟合残差，不能当作绝对位置精度。CSV附件另含MAE、最大误差及带符号偏差。")
        for title, headers, rows in full_tables:
            heading(2, title)
            table(headers, rows)

        heading(1, "时间对齐敏感性")
        table(["Marker", "轨迹", "最佳时移s", "最佳残差mm", "−50ms残差mm", "+50ms残差mm", "含义"], time_sensitivity)
        para("时移网格步长为50ms，显示三位小数不代表毫秒级同步精度。±50ms示例在每个时移下都重新拟合SE(3)，有效配对样本可能变化，因此不是固定公共重叠集上的纯时移敏感性比较。近直线运动会使时移与空间变换互相补偿；连续50/60/70序列第二/第一奇异值比约0.000364/0.000024/0.001961，存在几何退化。")

        heading(1, "整体结论、边界与下一步")
        para("本批目前支持的3mm结论仅是：某些近距离静止段的单段位置抖动RMS达标；静态点间距在限定近距离窗口达标；少数短时动态窗口RMSE达标。整个20–100cm静态范围与9段动态轨迹都没有普遍达到3mm，连续marker60全段误差尤其明显。")
        para("激光真值是激光靶点位置，PnP结果是Tag中心位置，固定外参/杠杆臂未实测；激光无姿态真值，因此不能由当前数据声称绝对位置或6DoF姿态精度。动态比较还会受转动时两测量点不同影响。全点对和相邻时间窗大量共享样本，不能用其数量假设独立并构造置信区间。")
        para("建议：重复采集20–100cm静态点并复查marker50左目80cm跳点和90–100cm距离偏长；标定激光靶点至Tag中心的固定变换并补姿态真值；时移在最优点附近细扫并固定公共重叠区间；先明确验收的3mm是RMS门槛还是所有样本绝对误差门槛。")
        heading(1, "复算材料")
        para("本地原始输入目录：/home/dex/20261030/GT 与 /home/dex/20261030/marker_10_30。Kalibr配置：configs/calibration/uvc_20261030_camchain.yaml。左右目详细报告及每序列summary、paired trajectories、time offset scan和图像位于evaluation_left_filtered与evaluation_right_filtered。marker60连续序列使用manifest_filtered_left/right指向的过滤版回放。")
        blocks.append(f'<source path="@./{xml_escape(data["bundle"].resolve().relative_to(Path.cwd()).as_posix())}" name="20261030_accuracy_detail_tables.zip"/>')
        draft_xml.parent.mkdir(parents=True, exist_ok=True)
        draft_xml.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/accuracy_20261030_kalibr"))
    parser.add_argument("--draft-xml", type=Path)
    args = parser.parse_args()
    path = build_reports(args.root, args.draft_xml)
    print(f"report: {path}")
    if args.draft_xml:
        print(f"Feishu XML draft: {args.draft_xml}")


if __name__ == "__main__":
    main()

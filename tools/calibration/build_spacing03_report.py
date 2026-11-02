"""Build the measured-board calibration and three-model accuracy report."""

import csv
import html
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/d455_calibration_methods_spacing03"
METHODS = {"OpenCV": ROOT / "outputs/laser_spacing03_refined_20261020",
           "Kalibr": ROOT / "outputs/laser_spacing03_kalibr_20261020"}
GROUPS = {"L1": "47.5mm 单面", "L3": "47.5mm 双面采集",
          "R1": "37.5mm 单面", "R3_static": "37.5mm 双面采集"}
DISTANCES = [20, 40, 60, 80, 100, 120, 130, 140, 150]


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])
    def cells(row, tag):
        return "<tr>" + "".join(f"<{tag}><p>{html.escape(str(v))}</p></{tag}>" for v in row) + "</tr>"
    xml = "<table><thead>" + cells(headers, "th") + "</thead><tbody>" + "".join(cells(r, "td") for r in rows) + "</tbody></table>"
    return md, xml


def main():
    summaries = {m: read(p / "comparison_summary.json") for m, p in METHODS.items()}
    assert all(s["comparison_controlled"] and s["timestamp_matching_sessions"] == 36 for s in summaries.values())
    rows = {}
    for method, folder in METHODS.items():
        with (folder / "range_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
            rows[method] = {(r["series"], int(r["endpoint_cm"])): r for r in csv.DictReader(f)}
    for key in rows["OpenCV"]:
        assert rows["OpenCV"][key]["baseline_rmse_mm"] == rows["Kalibr"][key]["baseline_rmse_mm"]
    md, xml = [], []
    def heading(text):
        md.append("## " + text)
        xml.append("<h1>" + html.escape(text) + "</h1>")
    def paragraph(text):
        md.append(text)
        xml.append("<p>" + html.escape(text) + "</p>")
    def add_table(headers, data):
        a, b = table(headers, data)
        md.append(a)
        xml.append(b)
    md.append("# D455 正确板配置下的标定方式与定位精度对比")
    heading("结论")
    paragraph("实测板为 6×6、tag 黑框 10 cm、白色间隙 3 cm，所有本轮标定均采用 tagSpacing=0.3。已完成 OpenCV 与原生 Kalibr 标定，并以出厂参数为基准重跑 36 段激光静态数据和 11 段 Marker 视频。标定已跑通，但定位精度没有全面提升，不能据此声称精度范围扩大。")
    reductions = {m: [float(rows[m][s, 150]["rmse_reduction_pct"]) for s in GROUPS] for m in METHODS}
    paragraph("20～150 cm 累计 RMSE 相对出厂的改善率（正值改善、负值变差），按下表四组顺序：OpenCV 为 "
              + " / ".join(f"{v:+.1f}%" for v in reductions["OpenCV"]) + "；Kalibr 为 "
              + " / ".join(f"{v:+.1f}%" for v in reductions["Kalibr"]) + "。")
    heading("实际标定参数")
    calibration = read(ROOT / "outputs/d455_aprilgrid_1020_spacing03_refined/calibration_summary.json")
    factory = calibration["factory_camera_info"]["camera_matrix"]
    cv = yaml.safe_load((ROOT / "outputs/d455_aprilgrid_1020_spacing03_refined/camchain.yaml").read_text())["cam0"]
    kal = yaml.safe_load((ROOT / "kalibr_d455/camchain-spacing03.yaml").read_text())["cam0"]
    add_table(["项目", "出厂 CameraInfo", "OpenCV", "原生 Kalibr"],
              [[name, f"{[factory[0][0],factory[1][1],factory[0][2],factory[1][2]][i]:.6f}", f"{cv['intrinsics'][i]:.6f}", f"{kal['intrinsics'][i]:.6f}"]
               for i, name in enumerate(["fx / px", "fy / px", "cx / px", "cy / px"])]
              + [["模型", "radtan 5 参数", "radtan 5 参数", "radtan 4 参数"],
                 ["训练数据", "设备出厂", "三包联合", "前两包"]])
    paragraph("OpenCV 三包联合拟合 RMS=0.160514 px，120 个留出视角 p50/p95=0.141/0.228 px；仍有留出视角超过 1 px（最大 1.086 px）。Kalibr 原始报告 X/Y 残差标准差=0.272/0.265 px。两种算法检测与筛帧、畸变阶数不同，不能直接用训练误差排名。完整 K+D 及重复标定详见本地 d455_calibration_methods_20261020.md 和附件。")
    heading("统一的独立标定验证")
    common = read(OUT / "common_validation.json")["models"]
    add_table(["来源", "p50 / px", "p95 / px", "p95 相对出厂"],
              [[label, f"{common[key]['validation']['p50_px']:.4f}", f"{common[key]['validation']['p95_px']:.4f}",
                "基准" if key == "factory" else f"{100*(1-common[key]['validation']['p95_px']/common['factory']['validation']['p95_px']):+.1f}%"]
               for key, label in [("factory", "出厂"), ("opencv_first_two_bags", "OpenCV 前两包"), ("kalibr_first_two_bags", "Kalibr 前两包")]])
    paragraph("两种模型均只用前两包标定，再用第三包相同 30 个视角和修正后的 OpenCV 角点验证，不删视角。验证用 OpenCV 前两包 K=[645.265162,645.070936,642.309773,353.921511]；下方业务评估则用上表三包联合模型。此验证受角点和模型差异影响，只说明固定观测上的表现。")
    heading("20～150 cm 各距离的激光真值对比")
    paragraph("每格依次为：出厂 / OpenCV / Kalibr 的累计点间 RMSE（mm），越小越好。20 cm 只有一个点，无法计算；120 cm 包含 6 点、15 个点对，150 cm 包含 9 点、36 个点对。")
    data = []
    for d in DISTANCES:
        cells = []
        for s in GROUPS:
            a, b = rows["OpenCV"][s,d], rows["Kalibr"][s,d]
            cells.append("—" if d == 20 else " / ".join(f"{float(v):.2f}" for v in [a["baseline_rmse_mm"], a["candidate_rmse_mm"], b["candidate_rmse_mm"]]))
        data.append([f"{d}cm", *cells])
    add_table(["累计端点", *GROUPS.values()], data)
    bounds = []
    for s, label in GROUPS.items():
        a, b = (summaries[m]["range_boundaries"][s] for m in METHODS)
        bounds.append([label, a["baseline"]["all_earlier_prefixes_pass_through_cm"], a["candidate"]["all_earlier_prefixes_pass_through_cm"], b["candidate"]["all_earlier_prefixes_pass_through_cm"]])
    add_table(["累计 RMSE≤5 mm 的连续通过端点 / cm", "出厂", "OpenCV", "Kalibr"], bounds)
    paragraph("四组在 120 cm 的本轮两套标定结果均未达到累计 RMSE≤5 mm。通过端点只表示本批已测离散位置，不是连续工作距离保证。")
    heading("各距离位姿输出率")
    unchanged = all(r["baseline_point_valid_frames"] == r["candidate_point_valid_frames"] for values in rows.values() for r in values.values())
    assert unchanged, "Output count changed: report each method separately"
    paragraph("36 个点在三套参数下成功帧数相同，以下百分比同时代表三组。分母为该段全部彩色输入帧，低输出率没有因重新标定而改善。")
    add_table(["标签距离", *GROUPS.values()], [[f"{d}cm", *[f"{100*float(rows['OpenCV'][s,d]['baseline_point_output_rate']):.1f}%" for s in GROUPS]] for d in DISTANCES])
    heading("历史 Marker 视频的补充对比")
    paths = {"OpenCV": ROOT / "outputs/d455_aprilgrid_1020_spacing03_refined_ab",
             "Kalibr": ROOT / "outputs/d455_aprilgrid_1020_spacing03_kalibr_ab"}
    marker = {m: read(p / "fixed_windows_summary.json") for m,p in paths.items()}
    add_table(["位移 MAE / mm", "出厂", "OpenCV", "Kalibr"],
              [[label, f"{marker['OpenCV']['models']['factory'][key]['metrics_excluding_anchor']['mae_mm']:.3f}",
                *[f"{marker[m]['models']['candidate'][key]['metrics_excluding_anchor']['mae_mm']:.3f}" for m in METHODS]]
               for key,label in [("tape_single_face", "单面卷尺 20～80 cm"), ("tape_20cm", "多面场景固定 tag3，30～80 cm"), ("zdisp", "50 mm 步进，累计 200 mm")]])
    paragraph("两套新参数都未明显改善静态深度抖动。11 段共 7,878 帧，三组使用相同像素角点，卷尺采用相同冻结停点窗口，首点归零、仅估运动方向、不拟合尺度；MAE 排除被锚定的首点。")
    heading("评估口径与建议")
    paragraph("激光评估沿用 2026-10-15 的 36 段原始 RGB-D 记录、历史 rig 与 Marker4 外法线 25.5 mm 测量点，深度权重 0.2、尺寸门限 1 px；每点使用全部成功帧求三维均值，再比较所有点对长度和原始激光 XYZ 真值，不做空间或尺度拟合。")
    paragraph("出厂对照始终使用各段录制的原始参数；其焦距不一定等于上表标定 bag 的 CameraInfo。新模型同时用于 RGB 求解和深度投影，深度原生参数与外参保留。三组各 4,890 个输入彩色帧，36/36 段彩色与深度时间戳序列一致；旧表 36 格已独立复算一致。每位置只采集一次，结果不能推导绝对 XYZ、姿态或动态精度。")
    paragraph("当前建议：保留出厂参数作为默认，保存本轮两套标定供明确场景选用；暂不把重新标定视为解决中远距离误差的方法。板尺寸已实测，角点问题已修复，后续应重点排查远距离角点质量、rig/测量点几何与深度融合。")
    heading("结果与复现")
    paragraph("完整参数、逐距离指标、点对误差、共同验证及复算脚本见同目录附件和复核材料包。")
    plot(rows)
    md.insert(-2, "![三套参数逐距离误差](../outputs/d455_calibration_methods_spacing03/laser_comparison.png)")
    md.append("[完整标定参数与重复性](d455_calibration_methods_20261020.md)；[OpenCV 逐距离 CSV](../outputs/laser_spacing03_refined_20261020/range_comparison.csv)；[Kalibr 逐距离 CSV](../outputs/laser_spacing03_kalibr_20261020/range_comparison.csv)。")
    md.append("复算：`.venv/bin/python tools/calibration/recheck_spacing03.py laser-analyze`；`.venv/bin/python tools/calibration/recheck_spacing03.py laser-analyze --method kalibr`；`.venv/bin/python tools/calibration/build_spacing03_report.py`。")
    (ROOT / "docs/marker_intrinsics_comparison_20261020.md").write_text("\n\n".join(md) + "\n")
    zip_path = OUT / "spacing03_comparison_evidence.zip"
    files = [ROOT / "docs/d455_calibration_methods_20261020.md", ROOT / "docs/marker_intrinsics_comparison_20261020.md",
             ROOT / "kalibr_d455/camchain-spacing03.yaml", ROOT / "kalibr_d455/results-spacing03.txt",
             ROOT / "kalibr_d455/camchain-datad455_aprilgrid_1020_retry2.yaml", ROOT / "kalibr_d455/results-cam-datad455_aprilgrid_1020_retry2.txt",
             OUT / "common_validation.json", OUT / "laser_comparison.png"]
    for p in METHODS.values():
        files += [p / f for f in ["comparison_summary.json", "range_comparison.csv", "point_metrics.csv", "pair_metrics.csv", "protocol.json"]]
    for p in paths.values():
        files += [p / f for f in ["fixed_windows_summary.json", "stop_metrics.csv", "run_manifest.json", "camchain.yaml"]]
    files += [ROOT / "tools/calibration" / f for f in ["recheck_spacing03.py", "compare_calibration_methods.py", "build_spacing03_report.py"]]
    files += list((ROOT / "outputs/d455_aprilgrid_1020_spacing03_refined").glob("*.yaml"))
    for name in ["refined", "crossbag", "repeat"]:
        folder = ROOT / f"outputs/d455_aprilgrid_1020_spacing03_{name}"
        files += [folder / "calibration_summary.json", folder / "views.csv"]
    files += [ROOT / name for name in [
        "src/marker6dof/commands/calibration/calibrate_aprilgrid.py",
        "tests/test_aprilgrid_calibration.py", "kalibr_d455/april_6x6_03.yaml",
        "outputs/marker_intrinsics_recheck_20261020/analyze_run.py",
        "outputs/marker_intrinsics_recheck_20261020/run_manifest.json",
        "outputs/laser_intrinsics_recheck_20261020/run_laser.py",
        "outputs/laser_intrinsics_recheck_20261020/analyze_laser.py",
        "outputs/d455_aprilgrid_1019_ab/tape_single_factory.json",
        "outputs/d455_aprilgrid_1019_ab/tape_multi_factory.json"]]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(ROOT)))
    xml += [f'<img path="@./outputs/d455_calibration_methods_spacing03/laser_comparison.png" caption="实测板配置下，三套参数的累计位移长度 RMSE；5 mm 为参考线。"/>',
            f'<source path="@./outputs/d455_calibration_methods_spacing03/{zip_path.name}" name="spacing03_标定与精度对比_复核材料.zip"/>']
    (OUT / "report.xml").write_text("\n".join(xml))
    # Replace only the former calibration appendix in the historical document.
    intro = '<h1>2026-10-20 更新：实测板 spacing=0.3 的标定与精度对比</h1>'
    intro += '<p>实物黑框 10 cm、间隙 3 cm。此前图像尺寸判断受角点检测偏向黑框内部影响，现已修正；本节替代旧标定附录，原报告与证据保留在下方。</p>'
    intro += "\n".join(xml[:9])
    intro += '<p>三套参数的各距离激光真值结果、输出率与复核材料，见更新后的完整对比和复核材料包。</p>'
    (OUT / "original_appendix.xml").write_text(intro)
    print(json.dumps({"documents_built": True, "full_range_reductions_pct": reductions, "unchanged_output_counts": unchanged}, ensure_ascii=False))


def plot(rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, (s,label) in zip(axes.flat, GROUPS.items()):
        d = DISTANCES[1:]
        ax.plot(d, [float(rows["OpenCV"][s,x]["baseline_rmse_mm"]) for x in d], "o-", label="Factory", color="#777777")
        for method,color in [("OpenCV", "#176fa6"), ("Kalibr", "#d97706")]:
            ax.plot(d, [float(rows[method][s,x]["candidate_rmse_mm"]) for x in d], "o-", label=method, color=color)
        ax.axhline(5, ls="--", color="#b42318", label="5 mm")
        ax.set(title=s + (" (47.5 mm)" if s.startswith("L") else " (37.5 mm)"), xlabel="Cumulative endpoint label (cm)", ylabel="Pairwise length RMSE (mm)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("D455: measured AprilGrid spacing=0.3; same raw recordings and laser truth")
    fig.savefig(OUT / "laser_comparison.png", dpi=160)
    fig.savefig(OUT / "laser_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

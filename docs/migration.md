# 目录与命令迁移说明

当前实现统一放在 `src/marker6dof/`。旧 `scripts/*.py` 路径保留为启动器，现有录制流程、文档命令和评估测试可继续使用。核心库的 `marker6dof.localization`、`marker6dof.tracking` 等导入路径保持稳定。

## 实现位置

下表中的模块均位于 `src/marker6dof/commands/`，每项对应同名的旧 `scripts/<模块>.py`。

| 分类 | 模块 |
| --- | --- |
| `tracking/` | `run_video`、`live_dashboard` |
| `calibration/` | `build_relative_trajectory`、`cad_tcp_from_step`、`pivot_tcp_calibration` |
| `replay/` | `replay_d455`、`replay_rgbd`、`replay_dashboard_session` |
| `evaluation/` | `eval_static_noise`、`eval_known_displacement`、`eval_tape_static_accuracy`、`eval_tape_continuous_stops` |
| `demo/` | `generate_demo`、`evaluate_demo` |
| `reporting/` | `plot_accuracy`、`render_evidence_visuals`、`render_presentation_videos`、`render_report_rgb_panels` |

`scripts/record_d455.py`、`scripts/record_t265.py` 分别调用已有的 `marker6dof.d455.main` 和 `marker6dof.t265.main`。

原 `scripts/live_dashboard.html`、`scripts/session_review.html` 移到 `src/marker6dof/web/`。编辑网页请修改这里的文件；实时服务和会话重放自动从包内读取。两份 HTML 一并进入 wheel。

## 安装后的命令

既有五个命令保留：`marker-run-video`、`marker-build-relative`、`marker-record-d455`、`marker-record-t265`、`marker-live-dashboard`。

新增常用命令如下；实际注册列表以 `pyproject.toml` 的 `[project.scripts]` 为准。

| 命令 | 实现模块 |
| --- | --- |
| `marker-generate-demo` | `demo.generate_demo` |
| `marker-evaluate-demo` | `demo.evaluate_demo` |
| `marker-replay-d455` | `replay.replay_d455` |
| `marker-replay-rgbd` | `replay.replay_rgbd` |
| `marker-replay-sensor-bag` | `replay.replay_sensor_bag` |
| `marker-replay-dashboard` | `replay.replay_dashboard_session` |
| `marker-cad-tcp` | `calibration.cad_tcp_from_step` |
| `marker-pivot-tcp` | `calibration.pivot_tcp_calibration` |
| `marker-eval-static-noise` | `evaluation.eval_static_noise` |
| `marker-eval-displacement` | `evaluation.eval_known_displacement` |
| `marker-eval-tape-static` | `evaluation.eval_tape_static_accuracy` |
| `marker-eval-tape-stops` | `evaluation.eval_tape_continuous_stops` |
| `marker-eval-sensor-stops` | `evaluation.eval_sensor_stops` |

也可直接运行模块：

```bash
python -m marker6dof.commands.replay.replay_d455 --help
python scripts/replay_d455.py --help
marker-replay-d455 --help
```

原来通过脚本加载内部函数的自定义代码，应改为从上述包模块导入。薄启动器只提供命令兼容。两项历史报告脚本 `plot_accuracy` 与 `render_report_rgb_panels` 没有通用命令参数，继续从旧脚本路径或对应模块启动，并核对源文件的场景、字体和目录默认值。

## 升级与验证

在项目根目录激活原虚拟环境，然后重新安装以刷新入口：

```bash
python -m pip install -e ".[dev]"
make check
```

真实采集按设备选择 `.[dev,d455]` 或 `.[dev,t265]`，两个 SDK 环境独立。报告绘图使用 `.[reporting]`。三份 `requirements*.txt` 仍可从项目根目录安装，版本范围统一由 `pyproject.toml` 管理。

wheel 安装无需保留仓库里的 `scripts/` 或 HTML。合成配置、实测标定与录制数据仍是外部输入，在仓库之外运行时明确传入 `--config`、`--rig`、`--session` 等路径。迁移不移动本地录制、输出、虚拟环境或其他独立仓库，Git 边界见 [Git 工作流](git_workflow.md)。

本项目最初从工作区中的 D455 实验目录提取为独立 Python 仓库；旧工作区的绝对路径及临时 tar 包不再是安装步骤。

# Marker 6DoF Tracking

Python 算法仓库：从多面 ArUco 刚体的图像观测估计相机坐标系下的相对 6DoF 轨迹。支持离线视频、RealSense D455 RGB-D、T265 双鱼眼，以及浏览器实时可视化。

## 算法流程

```text
图像与相机模型
  -> ArUco 角点检测与细化
  -> 单面 IPPE 候选 / 多面刚体联合 PnP
  -> 可选 RGB-D 平面约束优化
  -> 几何和时序门控、失锁分段
  -> 相对位姿与轨迹
```

核心输出是相机坐标系内的相对刚体位姿。绝对世界位置和机器人 TCP 位姿需要另外提供相机世界变换或经过测量验证的刚体到 TCP 外参。重投影误差、平滑轨迹和合成测试不能单独证明实机绝对精度。

## 安装与验收

需要 Python 3.10 或更新版本。核心算法、离线处理、测试和合成演示不需要相机或 RealSense SDK。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make test
make demo
```

`make check` 会运行测试、合成闭环和 wheel 安装检查。合成数据闭环也可单独运行：

```bash
marker-generate-demo --config configs/demo_calibration.json --output-dir outputs/demo
marker-run-video --config configs/demo_calibration.json \
  --video outputs/demo/demo_input.avi --output-dir outputs/demo/result
marker-evaluate-demo --ground-truth outputs/demo/ground_truth.csv \
  --estimates outputs/demo/result/tcp_poses.csv \
  --output outputs/demo/result/evaluation.json
```

`configs/demo_calibration.json` 仅用于软件闭环，不是实机标定。依赖版本和可选功能统一在 `pyproject.toml` 管理：D455 安装 `.[dev,d455]`，T265 安装 `.[dev,t265]` 并使用独立环境，报告图表安装 `.[reporting]`，ROS bag 读取安装 `.[rosbag]`。

## 常用入口

```bash
marker-run-video --help
marker-record-d455 --help
marker-record-t265 --help
marker-live-dashboard --help
marker-replay-d455 --help
```

安装后的命令及实现映射见 [迁移说明](docs/migration.md)。录制和处理结果默认写入本地 `captures/`、`outputs/`，不会随 Python wheel 分发；输入标定通过参数显式传入。`configs/` 中的演示配置、复现基线和模板有来源说明，修改基线会改变复现结果。

## 仓库结构

```text
src/marker6dof/       算法核心、相机适配、命令和网页资源
tests/                算法、设备适配、CLI 与输出回归测试
configs/              合成配置、标定基线和模板
  calibration/        历史 sensor bag 相机标定示例
scripts/              旧命令路径的兼容启动器
tools/                wheel 检查与标定工具
  calibration/        AprilGrid 标定分析与复算
docs/                 架构、运行指南、SOP 与研究记录
kalibr_d455/          已版本化的 D455 Kalibr 标定结果
captures/             本机录制、标定 bag 和 sensor bag（Git 忽略）
outputs/              本机回放与评估结果（Git 忽略）
local/                本机文档草稿与研究评估脚本（Git 忽略）
```

开发时修改 `src/marker6dof/`；架构边界和模块职责见 [架构说明](docs/architecture.md)。新增算法行为应配套针对性测试，位姿字段须说明坐标系、变换方向和单位。

## 文档

- [文档索引](docs/README.md)：按任务查找运行说明、SOP 与实验记录。
- [算法管线](docs/pipeline.md)：坐标约定和处理阶段。
- [D455 运行参考](docs/d455_runtime_reference.md)：采集、重放、实时面板和 RGB-D。
- [T265 运行参考](docs/t265_runtime_reference.md)：双鱼眼运行和 SDK 边界。
- [录制与离线回放](docs/recording_guide.md)：数据采集和复验流程。
- [精度测试 SOP](docs/sop_accuracy_test.md)：真值、指标和验收边界。
- [配置说明](configs/README.md)：复现基线与本机配置管理。
- [本机数据目录](docs/local_data_layout.md)：录制、转换结果及旧路径映射。
- [参与开发](CONTRIBUTING.md)：开发环境、测试和提交要求。

研究文档记录特定数据、代码版本和实验条件下的观察，不等同于当前默认行为或普遍精度承诺；复现实验前应核对输入数据、配置与输出路径。

本机 ROS 2 bag、相机录制和生成结果由 Git 忽略。历史录制按设备和用途放在 `captures/calibration/`、`captures/sensor_bags/`；文档发布草稿放在 `local/document_drafts/`。克隆仓库后应自行提供实验输入；`kalibr_d455/` 保留已追踪的 Kalibr 配置与结果摘要，原始 bag 不随仓库发布。

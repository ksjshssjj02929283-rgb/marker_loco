# Marker 6DoF Tracking

版本：**D455 Python 初版**；当前分支：`release/d455-initial`。

[全部版本与选择指南](docs/versions.md) · [源码来源](docs/version_source.json) · [本次验证](docs/validation.md)

从 48 mm 多面 ArUco Marker 到 D455 相机坐标系下的相对 6DoF 轨迹，并提供实时浏览器 3D 可视化。

```text
48 mm 多面 Marker
        ↓
D455 RGB 图像与时间戳
        ↓
ArUco 角点（亚像素细化）
        ↓
单面 IPPE / PnP 候选解
        ↓
多面刚体标定（T_reference_tag）
        ↓
联合角点位姿（同一刚体的一次 PnP）
        ↓
时序门控、丢失处理与分段
        ↓
相对 6DoF：XYZ + 四元数 / Roll-Pitch-Yaw
        ↓
实时 3D 轨迹与姿态坐标轴
```

本分支保留 48 mm D455 初始 Python 工程。后续 T265、RGB-D 和 C++/SDK 功能见版本导航；录制数据和大体积媒体由使用者另行提供。

## 1. 快速开始

建议 Python 3.10+：

```bash
cd marker_loco
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,d455]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

等价快捷命令：`make venv` / `make install` / `make test` / `make demo`。默认安装 `opencv-contrib-python-headless`（无 GUI，适合 CI / 服务器 / 离线重放）；若需要实时预览窗口，换回完整版：`pip uninstall -y opencv-contrib-python-headless && pip install opencv-contrib-python`。

运行合成数据闭环（不需要 D455）：

```bash
python scripts/generate_demo.py --output-dir outputs/demo
python scripts/run_video.py \
  --config configs/demo_calibration.json \
  --video outputs/demo/demo_input.avi \
  --output-dir outputs/demo/result
python scripts/evaluate_demo.py \
  --ground-truth outputs/demo/ground_truth.csv \
  --estimates outputs/demo/result/tcp_poses.csv \
  --output outputs/demo/result/evaluation.json
```

合成配置仅用于验证代码逻辑，真实部署必须替换相机内参、多面角点和 TCP 外参。`evaluate_demo.py` 在出现非 MEASURED 帧或误差超过阈值（默认位置 15 mm、旋转 8°）时以非零退出码失败；也可直接 `make demo` 跑通上述三步。

## 2. D455 实时运行

安装硬件依赖：

```bash
python -m pip install -e ".[d455]"
```

先从一段录制中估计多面刚体关系：

```bash
marker-record-d455 \
  --serial <D455序列号> \
  --marker-size-mm 48 \
  --ids 2,3,4,5 \
  --duration 30 \
  --output-root outputs/captures

marker-build-relative \
  --poses outputs/captures/YYYYMMDD_HHMMSS/poses.csv \
  --output outputs/captures/YYYYMMDD_HHMMSS/relative_trajectory.csv \
  --reference-id 2 \
  --min-overlap 60 \
  --rig-output outputs/captures/YYYYMMDD_HHMMSS/rig_calibration_id2.json
```

### 2.1 离线重放（无需相机）

录制会同时保存干净彩色流 `color_raw.mp4`（不含坐标轴叠加）。可在不连接相机、不依赖 `pyrealsense2` 的情况下复验单面位姿：

```bash
python scripts/replay_d455.py \
  --session captures/d455_aruco/YYYYMMDD_HHMMSS \
  --output-dir outputs/replay \
  --min-coverage 0.5
```

输出 `replay_poses.csv` 与 `summary.json`（帧覆盖率、中位 / 95 分位重投影误差）；覆盖率低于 `--min-coverage` 时以非零退出码失败。

启动浏览器 3D 可视化：

```bash
marker-live-dashboard \
  --serial <D455序列号> \
  --marker-size-mm 48 \
  --rig outputs/captures/YYYYMMDD_HHMMSS/rig_calibration_id2.json \
  --port 8765
```

仓库的 `configs/rig_calibration_id2.json` 已包含此前高覆盖率录制生成的 2/3/4/5 号面关系，可直接用于本机复现；重新标定后再替换该文件。

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。停止服务在运行它的终端按 `Ctrl-C`；也可以从另一终端执行：

```bash
pkill -INT -f '[l]ive_dashboard.py'
pgrep -af '[l]ive_dashboard.py'
```

页面显示当前检测画面、使用的 Marker ID、重投影误差、相对 XYZ、四元数转换的 RPY 和分段 3D 轨迹。鼠标拖动旋转视角，滚轮缩放；轨迹中断后会保留分段，不会跨越失锁区间连线。

## 3. 坐标系和输出约定

- `T_camera_tag`：将 Tag 坐标系的点变换到 D455 彩色相机光学坐标系；+X 向右、+Y 向下、+Z 向前。
- `T_reference_tag`：由多帧重叠观测得到，把每个 Tag 坐标系变换到参考 Tag（默认 ID 2）坐标系。
- `T_camera_body`：联合所有当前可见面角点求得的刚体位姿。
- 相对轨迹以第一帧接受的刚体位姿为原点，输出平移单位为米。
- 四元数顺序为 `qx,qy,qz,qw`；页面的 RPY 由平滑四元数实时计算，单位为度。
- 该结果是相机系内的相对刚体轨迹。要得到 TCP 或世界系轨迹，还需测量 `T_body_tcp`、`T_reference_tcp` 或提供 `T_world_camera`。

## 4. 目录结构

```text
marker_6dof_tracking/
├── README.md                         # 安装、启动、停止和输出约定
├── pyproject.toml                    # 可安装包与命令行入口
├── requirements.txt                  # 开发/离线依赖
├── LICENSE
├── configs/
│   ├── demo_calibration.json         # 合成验证配置
│   ├── rig_calibration_id2.json      # 已标定的 2/3/4/5 多面相对关系
│   └── tcp_transform.template.json   # TCP 外参填写模板
├── src/marker6dof/
│   ├── localization.py               # ArUco 检测、单面 IPPE、联合多面 PnP
│   ├── tracking.py                   # 刚体假设、连续性门控、轨迹分段
│   ├── realsense.py                  # 录制/实时跟踪共用的 D455 与 ArUco 适配
│   ├── d455.py                       # 唯一的 RealSense 录制实现
│   └── cli.py                        # 安装后的命令行适配器
├── scripts/
│   ├── record_d455.py                # D455 录制兼容启动器
│   ├── replay_d455.py                # 录制的离线重放（无需相机）
│   ├── build_relative_trajectory.py  # 重叠图刚体标定与相对轨迹
│   ├── run_video.py                  # 离线视频定位
│   ├── live_dashboard.py             # 实时 HTTP 服务
│   ├── live_dashboard.html           # 3D 6DoF 浏览器界面
│   ├── generate_demo.py              # 合成闭环输入
│   └── evaluate_demo.py              # 合成真值评估
├── tests/                            # 几何、配置和联合 PnP 测试
└── docs/
    ├── pipeline.md                   # 从 Marker 到 6DoF 的完整技术说明
    ├── tcp_calibration.md            # Tag/刚体/TCP 外参测量
    └── d455_runtime_reference.md     # D455 录制操作参考
```

## 5. 工程边界

当前实现验证的是“刚体在 D455 相机坐标系内的相对 6DoF”。它不自动声称绝对世界坐标、机器人 TCP 精度或独立真值精度。生产使用前应完成真实内参、打印黑框尺寸、多面三维角点、刚体到 TCP 外参和独立参考系统的验收。

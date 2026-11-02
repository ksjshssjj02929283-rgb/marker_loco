# D455 多面 ArUco 6DoF 算法

版本：**精简 D455 Python 算法**；当前分支：`release/d455-algorithm`。

[全部版本与选择指南](docs/versions.md) · [源码来源](docs/version_source.json) · [本次验证](docs/validation.md)

基于初始已跑通版本 `faad044`，仅保留检测、位姿估计、多面联合 PnP、时序门控、相对轨迹及必要运行入口。

## 安装

Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[d455]"
```

## D455 实时算法

```bash
marker-track-d455 --rig configs/rig_calibration_id2.json --marker-size-mm 48
```

多台相机时增加 `--serial <D455序列号>`。逐帧输出 JSON，包含质量状态、使用的 Tag ID、相对 XYZ、四元数、重投影误差和轨迹分段。首个接受位姿为相对轨迹原点；平移单位米，四元数顺序 xyzw，相机光学坐标 +X 向右、+Y 向下、+Z 向前。丢失或突跳会断开轨迹，平滑仅影响显示用数据。

## 离线定位与刚体标定

```bash
marker-run-video --config <真实标定.json> --video <视频文件> --output-dir outputs/result --no-annotated-video
marker-build-relative --poses <单Tag位姿.csv> --output outputs/relative.csv --reference-id 2 --min-overlap 60 --rig-output outputs/rig.json
```

`configs/demo_calibration.json` 仅展示相机内参、多面角点和外参的配置格式，真实输入需要匹配的标定参数。`configs/rig_calibration_id2.json` 是原始 48 mm 多面 Marker 的标定关系，换 Marker 后应重新标定。

核心代码位于 `src/marker6dof/localization.py`、`tracking.py` 和 `realsense.py`。本分支沿用原始 Python 算法；已发布的 C++ 工具见 `release/d455-cpp`，C SDK 见 `dev/library`。

当前命令适配器从检出目录读取 `scripts/`，请使用上面的 editable 安装并保留源码目录；本分支不承诺独立 wheel 分发。

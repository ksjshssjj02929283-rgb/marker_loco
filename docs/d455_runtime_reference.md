# D455 运行参考

D455 路径通过 RealSense SDK 采集 RGB-D，可独立使用 RGB 跟踪，也可选用原始深度点云进行 RGB-D 局部位姿优化。核心离线算法不要求连接相机。

## 安装

使用 Python 3.10+，在项目虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,d455]"
```

默认 OpenCV 为 headless 版本。需要相机预览窗口时，另安装 `opencv-contrib-python` 并移除 `opencv-contrib-python-headless`，避免两个 OpenCV wheel 同时存在。

## 录制和建立多面刚体标定

使用实测 Marker 黑色外边长，ID 与配置应和打印目标相符：

```bash
marker-record-d455 --serial <SERIAL> --marker-size-mm 47.5 \
  --ids 2,3,4,5 --duration 30 --output-root captures/d455_aruco

marker-build-relative \
  --poses captures/d455_aruco/<SESSION>/poses.csv \
  --output captures/d455_aruco/<SESSION>/relative_trajectory.csv \
  --reference-id 2 --marker-size-mm 47.5 --min-overlap 60 \
  --rig-output captures/d455_aruco/<SESSION>/rig_calibration_id2.json
```

录制会话包含原始 `raw_realsense.db3`、干净与标注视频、逐帧观测、设备内参和摘要。`--no-raw` 可跳过原始 RGB-D 数据库。具体采集建议见[录制与回放指南](recording_guide.md)。

## 离线 RGB 重放

无需相机或 RealSense SDK，使用干净视频和会话元数据复验检测：

```bash
marker-replay-d455 --session captures/d455_aruco/<SESSION> \
  --output-dir outputs/replay --min-coverage 0.5
```

结果包含逐帧位姿和覆盖率、重投影误差摘要。旧录制的尺寸元数据错误时，可显式传入 `--marker-size-mm`，并写入新的输出目录。

## 实时跟踪

```bash
marker-live-dashboard --serial <SERIAL> --marker-size-mm 47.5 \
  --rig captures/d455_aruco/<SESSION>/rig_calibration_id2.json --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。添加 `--record-root captures/d455_aruco` 可同步录制；添加 `--tcp-transform <JSON>` 显示通过外参计算的 TCP；添加 `--use-depth` 启用 RGB-D 精修。

实时轨迹以首个接受位姿为相对原点。TCP 结果需要验证 `T_reference_TCP` 的来源、方向和米制单位；它仍处于相机坐标系。TCP 测量流程见 [TCP 标定](tcp_calibration.md)。

## RGB-D 回放与解释

```bash
marker-replay-rgbd \
  --recording captures/d455_aruco/<SESSION>/raw_realsense.db3 \
  --rig configs/rig_calibration_id2.json --marker-size-mm 47.5 \
  --output-dir outputs/rgbd_replay
```

回放使用原始深度点云和 SDK 深度到颜色相机的外参，按时间同步检查、Marker 内部平面筛选和角点联合优化；深度被拒绝时保留 RGB 位姿。没有独立真值时，重投影残差或深度接受率不能说明绝对精度。算法细节、参数和实测边界见 [RGB-D 项目梳理](project_review_rgbd.md)。

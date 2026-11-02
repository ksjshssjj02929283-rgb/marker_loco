# ROS 2 Sensor Bag 回放

`marker-replay-sensor-bag` 从 ROS 2 bag 中读取压缩图像并调用同一套离线 `RigTracker`，不要求安装或启动 ROS。输入 bag 是外部数据；仓库的 `configs/calibration/sensor_capture_1013-camchain-imucam.yaml` 是对应历史录制的 camchain 示例，不可直接用于其他设备。该仓库不包含可公开复现的完整原始录制。

## 安装和运行

```bash
python -m pip install -e ".[rosbag,reporting]"
marker-replay-sensor-bag \
  --session <BAG_DIRECTORY> \
  --camchain <CAMCHAIN.yaml> --camera-key cam0 \
  --marker-size-mm 47.5 --output-dir outputs/sensor_replay --video
marker-eval-sensor-stops \
  --replay-dir outputs/sensor_replay \
  --output-dir outputs/sensor_replay/evaluation
```

`--topic` 可覆盖 camchain 中的图像 topic。YAML 支持 `pinhole` 搭配 `equidistant`、`radtan` 或 `none`。`equidistant` 使用 OpenCV fisheye 去畸变和原图重投影，不应把其四项系数当普通径向/切向畸变传给 `solvePnP`。

本机历史会话位于 `captures/sensor_bags/sensor_capture_1014_1/` 和 `captures/sensor_bags/sensor_capture_1014_2/`；对应 camchain 示例位于 `configs/calibration/`。新的录制应使用自己的 camchain。

## 输出与评估边界

默认输出逐面单目位姿。提供匹配 Marker 尺寸的多面 rig 时，也可输出联合刚体结果。`poses.csv` 中的位置是 Tag 中心在所选相机光学坐标系中的坐标，单位为米；`Z` 是光轴深度，三维向量模长才是相机到 Tag 中心的直线距离。不同 Tag 面的中心不可混用。

图像 header 时间戳和 bag 存储时间戳都会保留，统计使用图像 header 时间戳。距离探索可降低 `--min-edge-px`；结果同时记录常规跟踪器尺寸门槛下通过的 ID。重复 ID 候选会按尺寸和边界筛选，证据保存在 `frames.jsonl`。

静止段评估根据观测运动提取停留区间；人工指定窗口时应先核对画面。相机视角变化处可用 `--sequence-end-s` 分段，避免跨视角比较位移。远距离统计只包含成功检测帧，不代表最大量程或丢失帧的深度。

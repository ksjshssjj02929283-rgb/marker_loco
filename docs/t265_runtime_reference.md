# T265 运行参考

T265 使用同步的左右灰度鱼眼图像跟踪 Marker。双目一致时合并到左鱼眼坐标系，一侧暂时不可用时允许单目继续输出。Marker 位姿由视觉重投影求解；设备 pose stream 可用于对照，但不会自动变成世界坐标轨迹。

## SDK 与安装

本项目的 T265 路径要求 `pyrealsense2>=2.50,<2.54`。D455 可使用较新的 SDK，因此应为 T265 和 D455 使用不同的虚拟环境，避免 SDK wheel 相互覆盖。

```bash
python3 -m venv .venv-t265
source .venv-t265/bin/activate
python -m pip install -e ".[dev,t265]"
```

## 录制和实时运行

```bash
marker-record-t265 --serial <SERIAL> --marker-size-mm 47.5 \
  --ids 2,3,4,5 --duration 30 --no-preview

marker-live-dashboard --camera t265 --serial <SERIAL> \
  --rig configs/rig_calibration_id2.json \
  --tcp-transform <VALIDATED_TCP_TRANSFORM.json> --port 8766
```

录制包含左右鱼眼视频、相机内外参、逐面观测、原生 pose stream 和会话摘要。实时页面以左鱼眼坐标系展示轨迹；配置 TCP 外参后，页面切换为 TCP 相对轨迹并在图像上显示 TCP 坐标轴。

录制其他 ArUco 字典时，使用 `--dictionary DICT_6X6_50` 等与打印源完全一致的名称；可选项见 `marker-record-t265 --help`。同一组数字 ID 可以分别用于不同字典，但要分开录制和标定。带预览录制时，左右鱼眼窗口会标出属于 `--ids` 的检测结果，便于检查各面共视；保存的鱼眼视频仍为原始画面。

`T_reference_TCP` 把 TCP 坐标中的点变换到参考 Marker 坐标系，单位为米。使用前应核对外参的测量来源、原点和轴向。没有相机世界姿态时，该结果仍是相机系相对运动。

## ROS 2 bag

ROS 2 原始录制可由设备侧工具完成，再通过项目的 sensor bag 命令读取图像数据。该流程不依赖 ROS 运行时，但需要安装 `.[rosbag]`；相机模型应使用与录制对应的 Kalibr camchain。命令和畸变模型说明见 [Sensor bag 回放](sensor_bag_replay.md)。

若设备未出现在 `pyrealsense2.context().query_devices()` 中，检查 USB 连接、RealSense udev 规则，并确认没有其他进程占用相机。

## 双夹爪实时轨迹

左右夹爪使用不同 ArUco 字典且重复 ID 2–5 时，分别加载两份 rig。当前实机标定示例：

```bash
source .venv-t265/bin/activate
python scripts/live_dual_t265.py \
  --left-rig configs/local/t265_left_5x5_54p6_rig.json \
  --right-rig configs/local/t265_right_6x6_56_rig.json \
  --serial 20422110550 --port 8767
```

打开 `http://127.0.0.1:8767`。网页左右并排显示同步鱼眼，绿色为左夹爪，紫色为右夹爪。默认 `--ego-mode static`：假设 T265 静止，两条 Marker 立方体中心轨迹直接使用左鱼眼坐标系，不启动 pose stream，也不显示 Ego 轨迹。相机移动时此假设不成立。需要恢复 VIO 世界系与 Ego 轨迹时，显式传入 `--ego-mode vio`。

`cube_center_reference_m` 是 ID 2 坐标系下的立方体中心位置，运行时会先将参考 Marker 位姿变换到该点，再绘制鱼眼坐标轴、轨迹与坐标读数。当前左右中心由多面 Marker 中心与法线估计，分别约为 `[1.3, -0.3, -36.2]` 和 `[0.5, 0.0, -33.5]` mm，未用实物尺寸复核；右侧 ID 5 的中心线与其余面有约 7.6 mm 偏差，故右侧估计使用 ID 2/3/4。Marker 丢失时轨迹断段；VIO 模式下 VIO 不同步时也会断段。`POST /reset` 或页面按钮可清空轨迹；VIO 模式还会重建共享原点。此入口仅做实时可视化，不录制数据，也不包含未标定的 TCP 偏移。

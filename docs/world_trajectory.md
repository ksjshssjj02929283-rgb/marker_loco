# 移动相机：VIO 世界系、夹爪轨迹与 Ego 轨迹

静态相机时，项目输出的是 `T_camera_TCP`；相机移动后，它不能直接表示夹爪在环境中的运动。
启用 T265 原生 VIO 后，系统在每个同步帧按同一条变换链输出：

```text
T_world_TCP = T_world_left_fisheye · T_left_TCP
T_world_ego = T_world_left_fisheye
```

其中 `T_left_TCP = T_left_reference @ T_reference_TCP` 来自多面 ArUco 和已标定的 TCP 外参。
`T_world_left_fisheye` 由 T265 pose stream 的 `T_world_pose_sensor` 与 SDK 给出的
`T_pose_sensor_left_fisheye` 合成；不能把 pose sensor frame 误当作左鱼眼光学坐标系。

## 实时使用

```bash
marker-live-dashboard --camera t265 --serial <T265_SERIAL> \
  --rig configs/rig_calibration_id2.json \
  --tcp-transform configs/tcp_transform.json \
  --world-trajectory \
  --world-output outputs/t265_world_poses.csv \
  --min-vio-confidence 2 \
  --vio-max-sync-ms 20 \
  --port 8766
```

`--world-trajectory` 保持为显式开关，因此既有静态相机和仅相机系轨迹流程不会变化。
`--world-output` 是可选的新文件路径；程序拒绝覆盖已有 CSV。每行使用一个鱼眼帧时间戳，
并同时写入 session-world 中的 `ego_*` 与 `gripper_*` 平移/四元数、VIO 时间戳、置信度和
Marker 状态。没有可接受测量的位姿列留空，以保留真实失锁/不同步信息。
启用后 `/state` 中：

- `trace` 是世界系下的夹爪 TCP（没有 `--tcp-transform` 时是参考 Marker）轨迹；
- `ego_trace` 是同一初始 VIO 坐标系下的左鱼眼/Ego 轨迹；
- `position_m` / `quaternion_xyzw` 是当前夹爪（或未配置 TCP 时的参考 Marker）位姿；
- `ego_position_m` / `ego_quaternion_xyzw` 是当前相机 Ego 位姿；
- `vio_status` 和 `vio_tracker_confidence` 说明本帧是否允许合成。

两个轨迹的原点统一设为重置后第一帧有效 VIO 相机位姿，而不是分别以各自首帧为零点；因此它们在同一坐标系中的相对位置和方向有物理含义。

## 同步与失效策略

T265 的鱼眼约 30 Hz、pose stream 约 200 Hz。每个鱼眼帧只接受时间差不超过
`--vio-max-sync-ms` 的 pose；同时默认要求 native `tracker_confidence >= 2`。若 pose 缺失、
超时、无效或置信度不足，Ego 与夹爪轨迹同时断段，绝不把旧相机姿态与新 Marker 观测相乘。
Marker 丢失时只断开夹爪轨迹；Ego 轨迹仍可继续。

T265 的 VIO 世界原点是会话启动后的局部地图，可能随重定位发生不连续，也没有全球尺度外的绝对基准。若需要跨会话或全局地图，请在进入
`WorldTrajectory.update()` 前把 SLAM 后端输出统一转换为同一个 `T_world_camera`；组合公式和时间/置信度门控保持不变。

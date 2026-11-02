# T265 双鱼眼联合定位与 VIO 运动补偿的双夹爪轨迹

## 实验设置

本阶段从 T265 左右鱼眼联合估计双夹爪 Marker 立方体中心，再利用录制的 T265 VIO 位姿把每帧相机坐标下的中心转换到**初始有效 VIO 左相机坐标系**。视频中的双鱼眼轨迹也按当前相机位姿重新投影，因此历史轨迹不再默认相机始终停在同一位置。

左夹爪使用 5×5 ArUco、边长 54.6 mm；右夹爪使用 6×6 ArUco、边长 56 mm。两侧均使用 ID 2–5 的多面标定几何。

## 时间对齐与坐标转换

录制文件中的 VIO 采样时间戳通常比对应鱼眼帧晚约 39 ms，因此不能直接把同一行的位姿用于该帧。这里使用相邻的已记录 VIO 样本，在鱼眼时间戳处进行平移线性插值和四元数球面插值；仅接受两端置信度均不低于 2、采样跨度不大于 80 ms 的结果。超出条件的帧保留缺口。

鱼眼到 VIO 的外参来自同一台序列号为 `20422110550` 的 T265 SDK 流配置，保存在 [vio_extrinsics.json](../../outputs/t265_stages/vio_extrinsics.json)。对每帧使用：

```text
T_initial_cube = T_initial_world · T_world_pose(t) · T_pose_left · T_left_cube(t)
```

初始坐标原点取第一帧有效 VIO 左相机位姿。录制文件只保存了每个鱼眼帧附近的一条 VIO 样本；插值不是完整频率的原始 VIO 流。

## 视频结果

上方为同步双鱼眼中的 Marker、立方体中心以及经过 ego 位姿补偿后重投影的轨迹；中间保留左、右夹爪各自的轨迹面板，底部在同一面板中叠加左夹爪、右夹爪和 T265 ego 三条轨迹。底部三条轨迹共用初始 VIO 左相机坐标系和同一绘图尺度，并显示当前 ego 位置或失锁提示。

| 操作 | 视频 | 总帧数 | VIO 有效 | 左有效 | 右有效 | 同帧双夹爪有效 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 打开盖子 | [MP4](../../outputs/t265_stages/03_stereo_vio/01_open_lid_153305/task_visualization.mp4) | 743 | 736 | 736 | 707 | 707（95.2%） |
| 折叠物品 | [MP4](../../outputs/t265_stages/03_stereo_vio/02_fold_item_153420/task_visualization.mp4) | 708 | 704 | 686 | 704 | 686（96.9%） |
| 整理桌面 | [MP4](../../outputs/t265_stages/03_stereo_vio/03_tidy_table_153506/task_visualization.mp4) | 876 | 873 | 813 | 860 | 803（91.7%） |

逐帧中心位姿和状态在各任务的 `dual_center_poses.csv`；对应相机位姿与 VIO 状态在 `ego_pose.csv`。第一段、第二段、第三段录制中，相机相对初始有效位姿的最大平移分别约为 **143、131、171 mm**。

## 与双目静止阶段对比

双夹爪同帧有效数从 **714 → 707**、**690 → 686**、**806 → 803**。减少的 7、4、3 帧来自 VIO 时间对齐门限，不是 Marker 观测本身减少。此阶段的轨迹已去除可观测的相机平移与旋转；同一任务中不能直接把两阶段的坐标数值视为相同参考系下的真值比较。

## 结果解释与边界

T265 鱼眼画面为灰度，彩色是可视化叠加。VIO 会有漂移和估计误差，且录制时的位姿流采样较稀疏；这些曲线仍需外部真值做精度评估。轨迹点是 Marker 立方体中心，**不是夹爪 TCP**，所以不能直接当作夹爪作用点标签。

## 复现

在仓库根目录执行：

```bash
PYTHONPATH=src .venv-t265/bin/python scripts/build_t265_vio_stage.py
PYTHONPATH=src .venv-t265/bin/python scripts/render_t265_vio_stereo.py
.venv-t265/bin/python scripts/export_t265_task_videos.py \
  --result-root outputs/t265_stages/03_stereo_vio
```

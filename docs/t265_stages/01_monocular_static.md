# T265 单目静止视角下的双夹爪 Marker 中心轨迹

## 实验设置

本阶段只使用 T265 **左鱼眼**录制画面进行定位。相机被视为静止，所有位置表达在左鱼眼光学坐标系中。左夹爪使用 5×5 ArUco、边长 54.6 mm；右夹爪使用 6×6 ArUco、边长 56 mm。两侧均使用 ID 2–5 的多面标定几何，输出点位于各自 Marker 立方体中心。

输入是三段同步录制数据；折叠物品统一取前 708 帧。T265 鱼眼为灰度图像，视频中的绿色和紫色为检测与轨迹叠加。

## 视频结果

每段视频左侧显示单个鱼眼中的 Marker 框、中心及投影轨迹；右侧分别显示两个夹爪随时间展开的空间轨迹。定位丢失时，轨迹断开并显示丢失状态。

| 操作 | 视频 | 总帧数 | 左有效 | 右有效 | 同帧双夹爪有效 |
| --- | --- | ---: | ---: | ---: | ---: |
| 打开盖子 | [MP4](../../outputs/t265_stages/01_monocular_static/01_open_lid_153305/task_visualization.mp4) | 743 | 743 | 697 | 697（93.8%） |
| 折叠物品 | [MP4](../../outputs/t265_stages/01_monocular_static/02_fold_item_153420/task_visualization.mp4) | 708 | 673 | 708 | 673（95.1%） |
| 整理桌面 | [MP4](../../outputs/t265_stages/01_monocular_static/03_tidy_table_153506/task_visualization.mp4) | 876 | 766 | 834 | 734（83.8%） |

逐帧数值见各任务目录中的 `dual_center_poses.csv`，其中包含原始与平滑位置、姿态、状态和轨迹段号。空白位姿代表该帧没有可用定位。

## 结果解释与边界

视频可以定性检查双夹爪是否被持续追踪、标记中心是否落在合理位置，以及遮挡处是否正确断轨。所示位置是 Marker 立方体中心，**不是夹爪 TCP**。若相机实际移动，静止相机坐标下的曲线会混入相机运动；本阶段也未用外部真值验证毫米级精度。因此这是一份可视化与数据链路检查结果，训练使用前仍需核对相机运动假设与末端参考点定义。

## 复现

在仓库根目录执行：

```bash
PYTHONPATH=src .venv-t265/bin/python scripts/build_t265_mono_stage.py
.venv-t265/bin/python scripts/export_t265_task_videos.py \
  --result-root outputs/t265_stages/01_monocular_static
```

已生成的轨迹目录会被第一条命令跳过；第二条命令可重新导出视频。

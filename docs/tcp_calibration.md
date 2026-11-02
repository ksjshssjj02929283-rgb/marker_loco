# Tag → TCP 外参测量（T_reference_TCP）

`relative_trajectory.py` 目前已经能从多张 ArUco 标签融合出**相对轨迹**（参考标签 ID 2 的初始系）。要把它变成**夹爪 TCP 的米制轨迹**，只差一个固定的静态外参：

```
T_camera_TCP = T_camera_ref  @  T_reference_TCP
```

`T_reference_TCP` 是"把 TCP 系下的点映射到参考标签（ID 2）系"的 4×4 刚体变换：

```
p_ref = T_reference_TCP @ p_tcp
```

Onshape 文档链接只提供 CAD 几何，不能自动推断项目里的 ArUco 参考坐标系。
从 Assembly/Part Studio 读取尺寸后，还需要把 CAD 坐标系对齐到 ID 2 的坐标系，
再将结果写入 `T_reference_TCP`。请明确记录：ID 2 的安装面和轴向、CAD 原点及轴向、
TCP 原点（夹爪闭合时两指尖中点还是其他点）、TCP 三个轴的方向，以及长度单位。
推荐在 CAD 中建立一个显式 Mate connector 作为 TCP，并用三个非共线基准点确认
CAD frame 到 Marker frame 的旋转；不能直接把 CAD 世界坐标矩阵当成
`T_reference_TCP`。

如果 CAD 导出的矩阵仍然使用独立 CAD frame，也可以在同一个 JSON 中提供
`T_reference_CAD` 和 `T_CAD_TCP`；程序按
`T_reference_TCP = T_reference_CAD @ T_CAD_TCP` 组合，避免把 Onshape 世界原点
误当成 ID 2 原点。

它只依赖夹爪的物理结构，不随时间变化，所以测一次即可。

## 1. 两个坐标系的定义

**参考标签系（ID 2，ArUco / solvePnP 约定）**
- 原点：黑色方块中心
- +X/+Y/+Z：由项目的 ArUco 四角顺序和标签安装面共同决定；不要只按
  “图像向右/向下”猜 CAD 轴向。用一张静态 RGB 图验证三个轴的投影方向。

**TCP 系（平行夹爪）**
- 原点：夹指完全闭合时，两指尖连线中点的中心（抓取点）
- +Z：夹爪接近方向（从腕部指向被抓物体 / 夹持时朝下）
- +X：夹指开合方向
- +Y：与 X、Z 构成右手系

## 2. 三种获取方式（按精度/成本排序）

| 方式 | 精度 | 成本 | 说明 |
|------|------|------|------|
| CAD 模型 | 高 | 低（若有模型） | 从夹爪 CAD 直接量取标签安装位到 TCP 的位姿 |
| 游标卡尺实测 | 中 | 中 | 下面第 3 节的步骤，适合无 CAD 时 |
| 手眼标定 | 高 | 高 | 把夹爪摆到多个已知位姿，解算外参，最稳但最费时 |

先做游标卡尺实测足够跑通链路；有 CAD 或验收精度时再换。

## 3. 游标卡尺实测步骤

1. 把夹爪固定在平整台面，夹指**完全闭合**，确定 TCP 原点。
2. 确定参考标签（ID 2）中心，用细笔在标签四个角连线标出中心与 +X、+Y 轴方向。
3. 测量三个分量（单位统一为**米**）：
   - `tx`：TCP 原点沿标签 +X 方向的偏移
   - `ty`：TCP 原点沿标签 +Y 方向的偏移
   - `tz`：TCP 原点沿标签 +Z 方向的偏移
4. 记录旋转：标签系相对 TCP 系的朝向。若标签平面与夹爪接近方向近似垂直，常见为绕某轴的 ±90° 翻转；若不垂直，逐轴测量角度。

### 4. 填写到 `tcp_transform.template.json`

把测量结果写成 4×4 矩阵。**平移部分**（单位 m）是 TCP 原点在标签系下的坐标：

```json
{
  "T_reference_TCP": [
    [ r00, r01, r02, tx ],
    [ r10, r11, r12, ty ],
    [ r20, r21, r22, tz ],
    [ 0.0, 0.0, 0.0, 1.0 ]
  ]
}
```

旋转 3×3 必须是正交矩阵（行列式 +1）。不熟悉矩阵时，用下面 Python 生成：

```python
import numpy as np
from scipy.spatial.transform import Rotation

# 例如：绕 X 轴转 -90°（把"标签 +Z"转到"夹爪 -Z"方向等，按实测填写）
R = Rotation.from_euler("xyz", [-90, 0, 0], degrees=True).as_matrix()
t = np.array([[tx], [ty], [tz]])          # TCP 原点在标签系坐标（m）
T = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
print(np.round(T, 6).tolist())
```

### 5. 生成米制 TCP 轨迹

```bash
marker-build-relative \
  --poses captures/d455_aruco/<SESSION>/poses.csv \
  --output captures/d455_aruco/<SESSION>/relative_trajectory.csv \
  --reference-id 2 --min-overlap 60 \
  --rig-output captures/d455_aruco/<SESSION>/rig_calibration_id2.json \
  --tcp-transform configs/local/tcp_transform.json \
  --tcp-output captures/d455_aruco/<SESSION>/tcp_poses.csv
```

生成的 `tcp_poses.csv` 是**相机系下的米制 TCP 位姿**（+X 右、+Y 下、+Z 前，单位 m，四元数 xyzw）。报告里 `metric_tcp_ready` 会变为 `true`。

## 6. D455 RGB 投影和深度融合

实时浏览器服务支持同一个外参文件：

```bash
marker-live-dashboard --rig <rig.json> \
  --tcp-transform <cad_tcp.json> --use-depth
```

服务会把 `T_camera_TCP = T_camera_reference @ T_reference_TCP` 的原点和 XYZ 轴
投影到 RGB 图像，并保留参考 Marker 和 TCP 的历史线段。投影使用 D455 报告的
RealSense 畸变模型。可选深度融合使用原始点云和真实 depth-to-color 外参，在每个
Marker 内部拟合平面，再联合优化 RGB 角点与深度点到面残差。单面也可融合；
`--depth-weight` 是目标函数权重，不是固定比例位姿插值。深度不可靠或融合后重投影
超限时保留 RGB，状态给出 `DEPTH_REJECTED_*` / `INSUFFICIENT_DEPTH` 等原因。
具体参数和验证边界见 `project_review_rgbd.md`。

> 注意：这仍是**相机系**位姿，不是世界系。要得到世界系轨迹，还需要后续接入 SLAM / 相机在世界系的外参（当前报告里 `high_reliability_ready` 仍为 `false`，因为尚缺独立的真值参考）。

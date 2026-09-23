# Tag → TCP 外参测量（T_reference_TCP）

`relative_trajectory.py` 目前已经能从多张 ArUco 标签融合出**相对轨迹**（参考标签 ID 2 的初始系）。要把它变成**夹爪 TCP 的米制轨迹**，只差一个固定的静态外参：

```
T_camera_TCP = T_camera_ref  @  T_reference_TCP
```

`T_reference_TCP` 是"把 TCP 系下的点映射到参考标签（ID 2）系"的 4×4 刚体变换：

```
p_ref = T_reference_TCP @ p_tcp
```

它只依赖夹爪的物理结构，不随时间变化，所以测一次即可。

## 1. 两个坐标系的定义

**参考标签系（ID 2，ArUco / solvePnP 约定）**
- 原点：黑色方块中心
- +X：标签图像朝右
- +Y：标签图像朝下
- +Z：垂直标签平面、指向相机（正面朝外）

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
.venv-hifi/bin/python experiments/d455_aruco/relative_trajectory.py \
  --poses captures/d455_aruco/20260921_151219/poses.csv \
  --output captures/d455_aruco/20260921_151219/relative_trajectory.csv \
  --reference-id 2 --min-overlap 60 \
  --rig-output captures/d455_aruco/20260921_151219/rig_calibration_id2.json \
  --tcp-transform experiments/d455_aruco/tcp_transform.json \
  --tcp-output captures/d455_aruco/20260921_151219/tcp_poses.csv
```

生成的 `tcp_poses.csv` 是**相机系下的米制 TCP 位姿**（+X 右、+Y 下、+Z 前，单位 m，四元数 xyzw）。报告里 `metric_tcp_ready` 会变为 `true`。

> 注意：这仍是**相机系**位姿，不是世界系。要得到世界系轨迹，还需要后续接入 SLAM / 相机在世界系的外参（当前报告里 `high_reliability_ready` 仍为 `false`，因为尚缺独立的真值参考）。

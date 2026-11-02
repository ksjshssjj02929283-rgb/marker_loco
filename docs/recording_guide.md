# 录制指南与离线回放

本文说明如何用 D455 / T265 录制带 Marker 的数据、如何离线回放调试，并记录当前已录制的数据集。原始数据位于 `captures/`（已 gitignore），本文仅留命令与说明。

## 1. 录制命令

在项目根目录激活与设备对应的虚拟环境：D455 使用 `.venv`，T265 使用单独的 `.venv-t265`。

### D455（RGB + 深度，用于 RGB / RGB-D 对比调试）

```bash
marker-record-d455 \
  --ids 2,3,4,5 \
  --marker-size-mm 47.5 \
  --duration 0 \
  --output-root captures/d455_aruco/<场景名>
```

- `--duration 0`：手动录制，按 **Q / Esc** 停止（干净结束，`interrupted=false`）。Ctrl-C 也能停，但会在 `summary.json` 记为 `interrupted: true`，数据仍完整落盘。
- **保留** `raw_realsense.db3`（默认，约 3GB/30s）才能做深度融合回放；仅调试 RGB 跟踪可加 `--no-raw` 省空间。
- 产物：`raw_realsense.db3`、`color_raw.mp4`、`annotated.mp4`、`poses.csv`、`frames.csv`、`metadata.json`、`summary.json`。

可通过 `--name` 为采集命名，并通过 `--duration` 设置秒数。例如：

```bash
marker-record-d455 \
  --name R1_P01 \
  --ids 2,3,4,5 \
  --marker-size-mm 37.5 \
  --duration 20 \
  --output-root captures/d455_aruco/tape_test
```

该示例使用 37.5 mm 黑框尺寸，按实物修改。数据保存到
`captures/d455_aruco/tape_test/R1_P01/YYYYMMDD_HHMMSS/`，名称同时写入
`metadata.json` 的 `session_name`。同名的后续采集会生成新的时间戳目录。
省略 `--name` 时继续直接保存到输出根目录下的时间戳目录。名称可以包含中文或空格
（命令中用引号包裹），不能包含路径分隔符。

#### 20–80 cm 卷尺静态精度快测

目标是补充一组**有卷尺参考的相对距离准度**数据，而不仅是静态抖动。将相机固定，Marker 正对相机并尽量只沿光轴移动；卷尺从同一个相机机身参考面量到同一个 Marker 平面。建议录 20、40、60、80 cm 四点，每点静止 5–10 秒。不要手持 Marker；用桌面、直尺槽或夹具固定，移动后等待画面稳定再开始录。

```bash
# 每个点单独录制；把 <距离> 替换为 20 / 40 / 60 / 80
marker-record-d455 \
  --ids 2,3,4,5 \
  --marker-size-mm 47.5 \
  --duration 0 \
  --no-raw \
  --output-root captures/d455_aruco/tape_<距离>cm

# 结束每个点后，填入真实时间戳目录；默认只使用稳定的参考面 2 和 PnP Z。
python scripts/eval_tape_static_accuracy.py \
  --point 20=20=captures/d455_aruco/tape_20cm/<timestamp> \
  --point 40=40=captures/d455_aruco/tape_40cm/<timestamp> \
  --point 60=60=captures/d455_aruco/tape_60cm/<timestamp> \
  --point 80=80=captures/d455_aruco/tape_80cm/<timestamp> \
  --output outputs/accuracy_test/tape_20_80cm.json
```

输出中的 `raw err` 是 PnP 距离与卷尺读数的直接差，包含 RGB 光心到你所选相机机身参考面的固定偏移；`zeroed err` 以第一个点（默认 20 cm）清除该固定偏移，检验 20–80 cm 的相对距离误差。脚本**不会**拟合比例或修改数据。若卷尺测的是相机到 Marker 中心的斜距而非沿光轴距离，改加 `--metric range`；通常正对固定安装时保持默认 `z` 更合适。

若已在**同一段**录制里按距离顺序移动，并在每个卷尺刻度停住至少 2 秒，可直接自动切分静止平台：

```bash
python scripts/eval_tape_continuous_stops.py \
  --session captures/d455_aruco/tape_20cm/<timestamp> \
  --distances-cm 30,40,50,60,70,80 \
  --tag-id 3 \
  --metric line \
  --output outputs/accuracy_test/tape_30_80cm.json
```

`--tag-id` 应选整段都持续可见的同一个面；手动移动难以保持纯光轴时使用 `--metric line`，它将三维位置投影到拟合出的实际运动线后再同卷尺行程比较。若有导轨并确认是纯光轴运动，使用默认 `--metric z`。脚本若没有恰好找出相同数量的平台会停止并打印候选帧区间，不会自行猜测或丢弃测点。

#### 实时跟踪 + 同步录制（D455）

需要边看实时 3D 轨迹、边把同一会话录成数据集时，用 `marker-live-dashboard` 的
`--record-root` 开关，一次相机会话同时产出网页 + 数据集：

```bash
source .venv/bin/activate
marker-live-dashboard \
  --serial <SERIAL> \
  --marker-size-mm 47.5 \
  --rig configs/rig_calibration_id2.json \
  --record-root captures/d455_aruco/<场景名> \
  --port 8765
```

- 网页在 http://127.0.0.1:8765 实时显示；`Ctrl-C` 停止时会先 finalize `.db3` 再退出。
- 产物与纯录制一致：`raw_realsense.db3`、`color_raw.mp4`、`annotated.mp4`，外加联合多面位姿
  `live_poses.csv`（每帧相对 XYZ + 四元数，与网页展示一致）、`frames.csv`、`metadata.json`、`summary.json`。
- 加 `--no-raw` 可跳过 `.db3`（仍写视频与 CSV），仅调试 RGB 时省空间。
- 该开关当前仅支持 D455；`--camera t265` 配 `--record-root` 会直接报错。

### T265（双鱼眼，用于 D455 vs T265 对比）

```bash
marker-record-t265 \
  --ids 2,3,4,5 \
  --marker-size-mm 47.5 \
  --duration 0 \
  --output-root captures/t265_aruco/<场景名>
```

产物：`fisheye_left.mp4`、`fisheye_right.mp4`、`poses.csv`、`t265_pose.csv`、`metadata.json`、`summary.json`。

双夹爪任务录制时，用 `--secondary-dictionary DICT_6X6_50 --secondary-marker-size-mm 56` 同时检测右夹爪；主字典设为左夹爪 `DICT_5X5_50 --marker-size-mm 54.6`。两套 ID 可同为 2–5，因为字典不同。新录制还会生成 `poses_secondary.csv`（右夹爪逐面位姿）和 `frames.csv`（鱼眼实际时间戳）。后处理用 `python scripts/replay_dual_t265.py --session <录制会话目录> --left-rig configs/local/t265_left_5x5_54p6_rig.json --right-rig configs/local/t265_right_6x6_56_rig.json --output-dir <输出目录>`，产出同一左鱼眼坐标系的 `dual_center_poses.csv`、`dual_trajectories.png` 和 `annotated_dual.mp4`。双夹爪轨迹按相机静止处理；移动相机时不能将其解读为世界轨迹。

## 2. 推荐场景

| 场景 | 动作描述 | 主要调试目标 |
|------|----------|-------------|
| `static` | 物体完全不动，当前朝向 1~2 个面可见 | 位姿噪声基线 |
| `z_sweep` | 沿光轴缓慢前后移动（0.3m ↔ 1.5m），往返 2 次 | 深度融合在不同距离的表现 |
| `rotation` | 慢速 roll/pitch/yaw，不同面交替出现 | 多面切换、姿态估计 |
| `occlusion` | 单面稳定 → 短暂完全遮挡 → 移开，重复 2~3 次 | 单面降级、重捕获 |
| `fast` | 较快平移，可加一次快速挥手遮挡 | 鲁棒性、轨迹分段 |

每个场景单独建目录（`--output-root captures/d455_aruco/<场景名>`），内部按时间戳分子目录。

## 3. 离线回放

录制完成后无需接相机即可复现同一输入：

```bash
# RGB 跟踪（读 color_raw.mp4 + metadata.json）
python3 scripts/replay_d455.py --session captures/d455_aruco/<场景>/<时间戳>

# RGB vs RGB-D 深度融合对比（读原生 db3）
python3 scripts/replay_rgbd.py \
  --recording captures/d455_aruco/<场景>/<时间戳>/raw_realsense.db3 \
  --rig configs/rig_calibration_id2.json \
  --output-dir outputs/rgbd_compare/<场景>
```

`replay_rgbd.py` 输出 `frames.jsonl` + `summary.json`，含 RGB 与 RGB-D 的覆盖率、重投影误差分布、深度融合接受/拒绝计数。

## 4. 已录制数据（2026-09-24）

| 场景 | Session | 帧数 | 覆盖率 | 中位重投影误差 | 中位 depth−PnP z |
|------|---------|------|--------|---------------|-----------------|
| static | `20260924_162904` | 135 | 100% | 0.29 px | +2.98 m |
| static | `20260924_162920` | 334 | 100% | 0.94 px | +0.33 m |
| static | `20260924_162943` | 193 | 96% | 0.90 px | +0.33 m |
| static | `20260924_162956` | 359 | 100% | 0.56 px | +0.33 m |
| z_sweep | `20260924_163018` | 421 | 100% | 0.16 px | +0.29 m |
| rotation | `20260924_163042` | 550 | 99.6% | 0.21 px | +0.17 m |
| occlusion | `20260924_163114` | 547 | 75% | 0.10 px | +0.34 m |
| fast | `20260924_163142` | 427 | 86% | 0.19 px | +0.29 m |

除 `fast` 外均以 Ctrl-C 结束（`interrupted: true`），数据完整。

## 5. 关键观察：黑色 Marker 深度不一致

所有 session 的 `depth_consistency_fraction_50mm ≈ 0`、`depth_diagnostic_reliable = false`，中位 `depth − PnP z ≈ +0.17 ~ +0.34 m`（static 首个 take 因录制极短出现 +2.98 m 的离群值）。即**深度读数系统性比 RGB PnP 深度远约 20~30 cm**。

原因：黑色 Marker 表面在 D455 深度图中大量返回 0（无效），`median_tag_depth_m` 在腐蚀后的多边形内取有效样本中位数，实际采到的是 marker 后方背景，故偏大。这正是 `d455.py` 结尾 WARNING 提示的情形，也是 RGB-D 深度融合里 `DEPTH_REJECTED_*` 门控要防住的情况。

调试建议：

- 用 `z_sweep` / `static` 对比 `replay_rgbd.py` 中 RGB 与 RGB-D 的接受率，确认深度融合在深度无效时正确回退到 RGB。
- 若想验证深度融合本身的收益，建议换用**非黑色、无镜面**的 Marker 表面单独录一组，作为深度有效的对照。

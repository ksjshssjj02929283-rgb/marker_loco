# 标定配置管理

这些已追踪 JSON 是合成演示、复现基线和填写模板。它们保留历史来源与坐标约定，修改会影响复现结果；新设备或新夹爪的标定使用独立本机文件，并通过命令参数指定。

| 文件 | 用途 |
| --- | --- |
| `demo_calibration.json` | 无硬件合成闭环，包含相机参数、刚体与 TCP 定义 |
| `rig_calibration_id2.json` | 47.5 mm Marker、参考面 ID 2、2/3/4/5 多面关系的录制基线 |
| `tcp_transform.template.json` | TCP 外参填写模板，单位为米，初始为单位矩阵 |
| `tcp_transform.json` | 物理估计外参，参考面 ID 2 的 -Z 方向 320 mm |
| `tcp_transform_cad.json` | STEP 几何得到的 TCP 外参，保留 CAD 来源与几何记录 |
| `tcp_transform_pivot.json` | pivot 录制拟合的 TCP 点，旋转仍是单位矩阵占位值 |

TCP 统一采用 `p_reference = T_reference_TCP @ p_TCP`，具体有效性与测量步骤见 [TCP 标定](../docs/tcp_calibration.md)。CAD、物理估计和 pivot 文件表达各自的历史测量，选择时检查文件内 `source`、坐标轴定义和残差。

`calibration/sensor_capture_1013-camchain-imucam.yaml` 是历史 ROS 2 sensor bag 的 Kalibr 相机标定示例；只与对应录制匹配，不应作为其他设备的默认内参。D455 AprilGrid 的 Kalibr 实验结果保留在 `kalibr_d455/`，原始 bag 放在 `captures/calibration/kalibr_d455/`，不纳入 Git。

## 本机配置

本机文件可放在已忽略的 `configs/local/`，或继续使用 `*_local.json`、`*_real.json` 命名。目录无需预先创建到 Git：
目前两份 UVC 候选标定 YAML 放在 `configs/local/`，待验证和记录来源后再决定是否发布为共享基线。

```bash
mkdir -p configs/local
cp configs/tcp_transform.template.json configs/local/tcp_transform.json
# 测量并填写 configs/local/tcp_transform.json 后再运行
marker-live-dashboard --rig configs/rig_calibration_id2.json \
  --tcp-transform configs/local/tcp_transform.json
```

录制生成的 rig 文件可以留在会话目录内。只有需要共享、来源清晰且经过复验的基线才加入 Git；复验说明与配置修改一起提交。摄像头序列号、录制和研究产物的管理见 [Git 工作流](../docs/git_workflow.md)。

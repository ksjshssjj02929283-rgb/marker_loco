# 本机数据目录

仓库只版本化源码、测试、配置基线和小型标定结果。录制、转换 bag、回放输出与文档发布草稿均留在本机，并由 Git 忽略。

| 目录 | 内容 |
| --- | --- |
| `captures/d455_aruco/`、`captures/t265_aruco/` | 常规设备录制会话 |
| `captures/calibration/d455/` | 历史 D455 AprilGrid ROS 2 原始会话与同名缓存 |
| `captures/calibration/kalibr_d455/` | 从上述会话转换的 Kalibr `.bag` |
| `captures/sensor_bags/` | 历史 sensor 图像录制与 2026-10-20 UVC 测试 |
| `outputs/` | 回放、统计、报告和演示结果 |
| `local/document_drafts/` | 本机文档发布草稿及其附件 |
| `local/evaluation/` | 依赖本机激光数据与生成结果的研究评估脚本 |
| `kalibr_d455/` | 已版本化的 AprilGrid 定义、camchain 和小型结果摘要 |

此次目录整理将旧的根目录 `1014/sensor_capture_1014_*`、`20261022/test_20261020*` 移至 `captures/sensor_bags/`；`d455_1019_*`、`d455_aprilgrid_1020_*` 移至 `captures/calibration/d455/`；`kalibr_d455/*.bag` 移至 `captures/calibration/kalibr_d455/`；`draft_*_folder` 移至 `local/document_drafts/`。同名会话内部文件保持原样。

旧 `outputs/` 中的 JSON、CSV、协议文件可能记录移动前的绝对输入路径。这些字段是当时实验的来源记录，没有被改写；重新运行实验时应将输入参数指向上表新路径。完整原始数据不随 Git 克隆获得。

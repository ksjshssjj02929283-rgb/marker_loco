# 文档索引

新用户从 [项目 README](../README.md) 开始；了解代码职责和依赖边界请读 [架构说明](architecture.md)，修改项目请读 [开发指南](../CONTRIBUTING.md)。

## 使用与开发

| 任务 | 文档 |
| --- | --- |
| 跑通无硬件算法闭环 | [算法跑通 SOP](sop_algorithm_run.md) |
| D455 安装、采集、实时跟踪和重放 | [D455 运行参考](d455_runtime_reference.md) |
| T265 双鱼眼安装与运行 | [T265 运行参考](t265_runtime_reference.md) |
| 录制数据、离线复验 | [录制与回放指南](recording_guide.md) |
| ROS 2 图像 bag 回放 | [Sensor bag 回放](sensor_bag_replay.md) |
| 评估精度并接入独立真值 | [精度测试 SOP](sop_accuracy_test.md) |
| 测量 Marker 到 TCP 的变换 | [TCP 标定](tcp_calibration.md) |
| 理解算法阶段与坐标系 | [算法管线](pipeline.md) |
| 运行命令迁移和安装入口 | [迁移说明](migration.md) |
| 管理分支、本机数据与忽略规则 | [Git 工作流](git_workflow.md) |
| 查找本机录制、转换 bag 与旧路径映射 | [本机数据目录](local_data_layout.md) |
| 选择标定配置 | [配置说明](../configs/README.md) |

## 算法与验证记录

| 主题 | 文档 |
| --- | --- |
| RGB-D 优化、门控、默认值和限制 | [RGB-D 项目梳理](project_review_rgbd.md)、[深度效果说明](depth_camera_effect.md) |
| D455 标定方法与结果 | [标定方法](d455_calibration_methods_20261020.md)、[标定评估](d455_calibration_evaluation_20261020.md) |
| Marker 尺寸与激光真值对比 | [内参与位移对比](marker_intrinsics_comparison_20261020.md) |
| 检测距离探索 | [D455 检测范围](d455_detection_range.md) |
| 项目阶段总结 | [结项摘要](project_closure_overview.md)、[技术说明](project_closure_technical.md)、[验证边界](project_closure_validation.md) |
| 自研内容与效果评估 | [项目自研说明](project_self_development_20261022.md) |
| 初期精度观察 | [初步精度报告](accuracy_preliminary.md) |

实验记录对应特定数据、配置和代码状态；复现实验前请确认输入数据可用，并检查文中参数是否仍适用于当前版本。当前命令和依赖以 `pyproject.toml` 为准。

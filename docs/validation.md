# 本次验证

日期：2026-11-02。Python 3.10；此记录对应本分支的当前代码快照。

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`：218 项测试及 18 项 subtests 通过。
- `make git-check`：索引中没有录制、本机数据或超过 10 MiB 的文件；推送前另检查新增文档及发布历史。
- `make demo`：60/60 帧为 MEASURED，位置最大误差 11.275 mm、旋转最大误差 6.527°，均通过合成阈值。
- `make package-check`：wheel 在隔离目录中成功安装，20 个无相机 SDK 命令的帮助入口通过，12 帧安装后合成闭环及错误退出码检查通过。
- 新版本说明、各分支 README、源码来源记录和 Git 工作流一起整理；算法沿用干净导入 `5ec6211`。

未连接 D455/T265，未重新采集实机真值。本次测试不代表硬件可用性或实机绝对精度。

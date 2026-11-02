# 参与开发

先阅读 [项目架构](docs/architecture.md) 和 [Git 工作流](docs/git_workflow.md)。后者说明两个远端的用途、发布分支和已有 worktree，修改前应确认自己所在的分支。

## 开发环境

核心算法、RGB 离线回放和 CI 不需要相机 SDK，原生 RGB-D 录制回放仍需要 SDK。默认开发环境只安装 `.[dev]`：

```bash
make venv
source .venv/bin/activate
make PYTHON=python install
```

需要连接 D455 时，在对应虚拟环境中执行 `make PYTHON=python install-d455`。T265 要求 librealsense / pyrealsense2 2.50–2.53，使用单独环境，避免与 D455 的 SDK 版本互相覆盖：

```bash
python3 -m venv .venv-t265
source .venv-t265/bin/activate
make PYTHON=python install-t265
```

默认 OpenCV 为 `opencv-contrib-python-headless`，适合 CI、服务器和离线处理。需要实时 GUI 预览时，按 README 的说明安装完整 OpenCV。

## 代码和数据边界

- 可复用算法和设备适配代码放在 `src/marker6dof/`。
- 命令实现按用途放在 `src/marker6dof/commands/`，通过已安装的 `marker-*` 命令或包模块调用。`scripts/` 保留旧命令路径的兼容入口；新增功能应进入包内实现。
- 页面资源随包发布，修改命令或页面时同时验证 wheel 安装后的调用。
- 对几何、门控或输出格式的行为修改，增加有针对性的回归测试；新增位姿字段应说明坐标系、变换方向和单位。
- 原始采集数据放在 `captures/`，生成结果放在 `outputs/`；本机标定覆盖放在 `configs/local/` 或使用已有 `*_local.json` / `*_real.json` 命名。
- `configs/` 中已追踪的 demo、rig 和 TCP 文件是复现基线。不要用本机新标定直接覆盖这些基线；确需更新时，应说明标定来源、坐标约定和验证结果。
- `.analysis_upstreams/` 和 `ego_umi_slam/` 是本地独立仓库，各自维护其 Git 历史，不作为本仓库源码提交。

## 提交前验证

```bash
make PYTHON=python git-check
make PYTHON=python test
make PYTHON=python demo
make PYTHON=python package-check
git diff --check
```

也可运行 `make PYTHON=python check` 完成全部检查。`git-check` 防止本机数据、录制文件、嵌套仓库或超过 10 MiB 的文件进入 Git 索引。`test` 禁用 pytest 插件自动加载，避免系统 ROS 插件干扰。`demo` 验证合成数据闭环，对非 `MEASURED` 帧及超出位置 15 mm / 旋转 8° 阈值的结果返回失败，结果保存在 `outputs/demo/`。`package-check` 离线构建 wheel，在临时安装环境中借用当前环境的基础依赖，检查导入来源、命令入口、页面资源、合成闭环与失败退出码，完成后自动清理临时文件。

需要验证真实设备时，用 `scripts/record_d455.py` 录制样本，再用 `scripts/replay_d455.py` 离线复核。相机连接、SDK 和实测数据的验证结果应在 PR 中单独说明。

提交前逐项检查暂存区，不提交录制数据、序列号、本机配置、虚拟环境或生成视频。PR 描述应说明触发场景、变化后的行为以及执行过的验证；提交和推送流程见 [Git 工作流](docs/git_workflow.md)。

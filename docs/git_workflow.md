# Git 工作流

统一 GitHub 发布仓库为 [marker_loco](https://github.com/ksjshssjj02929283-rgb/marker_loco)，完整分支用途和来源见 [版本导航](versions.md)。默认 `main` 是当前 Python 完整版，Python 功能线为 `feat/t265-dual-camera-support`；初版、精简 Python、C++ 和 C SDK 分别保留在独立分支。

## 获取与开发

```bash
git clone https://github.com/ksjshssjj02929283-rgb/marker_loco.git
cd marker_loco
git fetch origin
git branch -avv
git switch -c feat/my-change main
```

C++ 或 SDK 改动应从对应 `release/d455-cpp` / `dev/library` 出发。切换分支前检查 `git status`，保留未提交改动；多个版本并行使用独立 worktree、Python 环境和构建目录。

```bash
git worktree add ../marker_loco_cpp release/d455-cpp
git worktree list
```

## 验证与推送

Python 完整版按 [参与开发](../CONTRIBUTING.md) 执行 `make check`；历史 Python、C++ 与 SDK 按各自 README 的检查命令验收。提交前检查路径和内容，不使用 `git add .` 将本机数据混入提交。

```bash
git diff --check
git add <本次修改的具体文件>
git diff --cached --stat
git commit -m "docs: describe the change"
git push -u origin HEAD
```

本例适用于从 `marker_loco` 新克隆的目录。原项目 `/home/dex/marker_6dof_tracking` 中的 `origin` 仍指向旧 GitHub `Dex-loc`，`gitlab` 指向原 GitLab；不要在旧工作目录中照抄目标为 `origin` 的发布命令。此次整理专用目录为 `/home/dex/marker_loco`，其 `origin` 指向此仓库。

## 分支维护边界

- `main`：验证后的 Python 默认入口；本次与 Python 功能分支同步。
- `feat/t265-dual-camera-support`：继续维护当前 D455/T265 Python 功能。
- `release/d455-initial`：初版复现与必要修复。
- `release/d455-algorithm`：初始算法的精简 Python 入口。
- `release/d455-cpp`：独立 C++ 工具，不将 Python 新功能自动视为已移植。
- `dev/library`：C SDK 封装和集成，版本依赖与交付边界见该分支 README。

初版、精简 Python、C++、SDK 保留原始提交；当前 Python 使用剔除本机数据后的独立导入历史。跨语言或跨历史线同步应明确移植具体变更，核对接口及验证结果。禁止通过 `push --mirror`、覆盖 `main` 或 force push 来代替版本整理。

## 本机文件

`captures/`、`outputs/`、`local/`、`build/`、`dist/`、`.venv*`、`draft_*_folder/`、本机配置和凭证不进入源码提交。录制和打包产物留在本机。Python 完整版提供 `scripts/check_git_hygiene.py` 检查当前索引；发布历史分支前也应检查其全部可达提交中的文件。

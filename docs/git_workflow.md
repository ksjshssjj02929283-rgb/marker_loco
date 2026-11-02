# Git 工作流

本仓库维护 Marker 6DoF 的 Python 跟踪、离线工具和可视化。开发改动放在功能分支，历史发布线通过独立 worktree 保留。本项目的功能分支在 GitLab 协作；推送前应明确目标分支，避免与 GitLab 上的 C++ 主线混淆。

## 两个远端与当前分支

两个远端的 `main` 在当前仓库中代表不同的版本线，不应直接互相覆盖。

| 引用 | 本地引用快照（2026-10-28） | 用途 |
| --- | --- | --- |
| `origin/main` | `faad044`，初始 D455 Python 跟踪版本 | GitHub 远端的原始主线 |
| `gitlab/main` | `76a6a17`，D455 C++ 移植版本 | GitLab 远端的 C++ 发布线 |
| `gitlab/feat/t265-dual-camera-support` | `db3086b`，Python 功能分支 | 当前 Python 协作线 |
| `gitlab/dev/library` | `1063511`，从 C++ 主线延伸的库封装分支 | 独立的 C++ 开发线 |
| `release/d455-initial` | `faad044` | 初始 D455 版本快照 |
| `release/d455-algorithm` | `4ccf770` | 精简的 D455 算法版本 |
| `release/d455-cpp` | `76a6a17` | D455 C++ 发布版本 |

这张表是整理时读取的本地引用快照，不代表之后远端的实时状态。每次开发或推送前，重新检查目标；不要因为两个远端都叫 `main` 就认为它们应当一致。

```bash
git status --short --branch
git remote -v
git branch -vv
git worktree list
git log --all --oneline --decorate --graph -15
```

需要读取最新远端引用时，显式选择远端执行 `git fetch origin` 或 `git fetch gitlab`。`fetch` 更新远端引用，不自动合并当前开发分支。不要在未确认目标分支时直接运行 `git pull`。

## 功能开发与提交

沿用 `feat/<任务名>`、`fix/<问题名>`、`docs/<主题>` 等分支命名。已有未提交改动时先保留并检查，不用 reset 或清理命令抹去它们。工作区干净、确认当前版本适合作为起点后，可以创建新分支，例如：

```bash
git switch -c feat/project-layout
```

当前 Python 功能线包含初始 `main` 尚未具备的功能；创建后续 Python 开发分支时，应从实际要继续维护的开发版本出发。发布分支只接受属于该版本线的明确修复，合入开发功能前先检查差异和兼容性。

一项提交聚焦一个目的，例如目录迁移、包装修复、Git 配置或文档更新。可以使用 `refactor:`、`fix:`、`chore:`、`docs:` 前缀，后接具体变化。检查完成后按具体路径暂存，示例：

```bash
git diff --check
git diff --stat
git add .gitignore .gitattributes CONTRIBUTING.md docs/git_workflow.md
make PYTHON=python git-check
git diff --cached --stat
git diff --cached
git status --short
```

涉及算法、命令或页面的变动，提交前运行 CONTRIBUTING 中的检查。暂存区里只能包含本次已核对的改动；确认后再提交。`git-check` 检查索引中的本机数据目录、录制格式、嵌套仓库和超过 10 MiB 的 blob，GitLab CI 也执行它。它检查当前索引或已检出的提交，不能替代对未暂存、未追踪文件的检查。

## 显式选择推送目标

当前 Python 功能分支对应 GitLab 的同名远端分支。推送前先检查最新引用与目标，尤其要区分 Python 功能线和 GitLab `main` 的 C++ 版本线。新分支首次推送时先做 dry run；以下示例假设正在开发 `feat/project-layout`：

```bash
git push --dry-run gitlab feat/project-layout
git push -u gitlab feat/project-layout
```

`-u` 将本地分支的 upstream 设置为 GitLab 的同名分支；后续仍可用 `git push gitlab feat/project-layout` 清楚表达目标。创建 MR 时明确目标版本线，确认差异后再合入。GitHub `origin` 保留为另一远端，不用于本次同步。

不要把当前 Python 功能分支直接推到 `gitlab/main`，也不要使用 force push 同步两个不同的 `main`。发布、主线合并及改写共享历史应作为独立任务处理。

## 保留 release worktree

本地已有三个工作目录分别绑定三个发布分支：

| 工作目录 | 分支 |
| --- | --- |
| `../marker_6dof_tracking_d455_initial` | `release/d455-initial` |
| `../marker_6dof_tracking_d455_algorithm` | `release/d455-algorithm` |
| `../marker_6dof_tracking_d455_cpp` | `release/d455-cpp` |

需要检查某个发布版本时，在对应 worktree 内运行 `git status` 和验证命令。Git 不允许同一分支同时检出到多个普通 worktree，不能因此强制覆盖已有工作目录。保留它们的路径、分支和未提交数据；删除、移动或回收 worktree 时使用 Git 的 worktree 工具，并先确认工作区状态。

## 本地数据和独立仓库

根 `.gitignore` 定义代码仓库的边界：

- `captures/` 保存原始录制，其中 `calibration/d455/` 是 D455 AprilGrid 原始会话，`calibration/kalibr_d455/` 是 Kalibr 转换 bag，`sensor_bags/` 是其他 ROS 2 图像录制；`outputs/` 保存回放结果和演示。wheel 检查使用自动清理的临时目录。
- `local/document_drafts/` 保存本机文档发布草稿和证据副本，不作为项目源码分发。
- `configs/local/`、`configs/*_local.json`、`configs/*_real.json` 保存本机配置；两份 UVC 候选 YAML 也放在 `configs/local/`。`configs/` 中已追踪的标定文件继续作为复现基线保留。
- `.venv/`、`.venv-*/`、构建目录、测试缓存以及本机 `.claude/settings.local.json`、`.aws/`、`.codex/` 不提交。
- `graphify-out/`、`local/` 和登录二维码留在本地。
- `.analysis_upstreams/` 与 `ego_umi_slam/` 有各自的 `.git`，是独立 checkout。根仓库只忽略这两个精确路径，不把它们转换为 submodule，也不接管其历史。

需要检查某条忽略规则时，例如：

```bash
git check-ignore -v configs/local/example.json
git check-ignore -v .analysis_upstreams/open_vins/
git check-ignore -v ego_umi_slam/
```

忽略文件仍可能是不可替代的录制、标定或研究成果。不要使用 `git clean -fdx` 清理项目；该命令会删除被忽略的数据。整理目录时只处理已明确确认可再生成的具体文件。

`.gitattributes` 统一文本为 LF，并标记常见二进制格式。当前原始录制已经排除在 Git 外，不需要为本次整理引入 LFS 或改写历史。若未来确需版本化大型素材，再单独确定存储策略。

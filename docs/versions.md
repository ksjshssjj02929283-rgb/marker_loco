# 版本与分支导航

整理日期：2026-11-02。统一发布仓库为 [marker_loco](https://github.com/ksjshssjj02929283-rgb/marker_loco)。软件包的 `0.1.0` 被多条历史线沿用，不能据此判断代码版本；复现时请同时记录分支名和完整 Git 提交。

## 选择版本

| 分支 | 版本定位 | 核心能力 |
| --- | --- | --- |
| [`main`](https://github.com/ksjshssjj02929283-rgb/marker_loco/tree/main) | 当前 Python 完整版 / 默认入口 | D455、T265、RGB-D、回放、评估与浏览器面板 |
| [`feat/t265-dual-camera-support`](https://github.com/ksjshssjj02929283-rgb/marker_loco/tree/feat/t265-dual-camera-support) | Python 功能开发线 | 当前与 main 使用同一代码快照；后续设备功能开发从这里继续 |
| [`release/d455-initial`](https://github.com/ksjshssjj02929283-rgb/marker_loco/tree/release/d455-initial) | D455 Python 初版 | 48 mm 多面 ArUco、录制、相对轨迹、浏览器 3D |
| [`release/d455-algorithm`](https://github.com/ksjshssjj02929283-rgb/marker_loco/tree/release/d455-algorithm) | 精简 Python 算法版 | 保留检测、联合 PnP、门控、JSON 跟踪与刚体标定 |
| [`release/d455-cpp`](https://github.com/ksjshssjj02929283-rgb/marker_loco/tree/release/d455-cpp) | C++17 独立工具版 | 离线回放、D455 实时输入、刚体标定、核心测试 |
| [`dev/library`](https://github.com/ksjshssjj02929283-rgb/marker_loco/tree/dev/library) | C SDK 开发版 | libdfmarker.so、C API、C++ 示例、原生/Orin 构建脚本 |


需要完整 Python 工作流时使用 `main`；只研究初始 D455 算法时使用 `release/d455-algorithm`；需要独立 C++ 命令行时使用 `release/d455-cpp`；接入其他进程内的图像采集程序时查看 `dev/library`。C SDK 属于开发线，检查本分支 README 的交付边界后再集成。

## 获取与切换

```bash
git clone https://github.com/ksjshssjj02929283-rgb/marker_loco.git
cd marker_loco
git fetch origin
git branch -r
git switch --track origin/release/d455-cpp
```

已有同名本地分支时使用 `git switch release/d455-cpp`。各版本依赖不同，Python 环境和 CMake 构建目录应分开；也可通过 `git worktree add ../marker_loco_cpp release/d455-cpp` 建立独立目录。每个分支根 README 包含该版本的安装、运行入口及验证说明。

## 源码来源和演进

| 发布版本 | 原项目来源 | 原始提交 / 基线 | 本次整理方式 |
| --- | --- | --- | --- |
| D455 初版 | 原项目 `main`、`release/d455-initial` | `faad044195304dec08681228faf26e35c56b7848` | 保留历史，追加文档整理提交 |
| 精简 Python | 原项目 `release/d455-algorithm` | `4ccf770be7b9fb17713066cb1c2fa35417456673` | 保留历史，追加文档整理提交 |
| C++ 工具 | 原项目 `release/d455-cpp`、原 GitLab `main` | `76a6a172cf6946f0bcc9fe72fca9e0c2c6806d18` | 保留历史，整理文档并修正 OpenCV 构建模块选择 |
| C SDK | 原项目 `gitlab/dev/library` | `1063511` | 保留历史，整理文档并补齐漏提交的 C 示例 |
| 当前 Python | 原项目 `feat/t265-dual-camera-support` 的工作区 | 已提交基线 `db3086b`，另含后续工作区功能 | 沿用新仓库干净导入 `5ec6211`，追加文档整理提交 |

原始代码演进关系为：D455 初版 → 精简 Python → C++ 工具 → C SDK；另一路从 D455 初版演进为 D455/T265 完整 Python 工程。当前 Python 的干净导入历史与保留的 D455 发布历史独立，因此不应依赖 `git merge-base` 将两条线当作可直接合并的版本。Python 版本的源目录已提交基线并不代表完整工作区；`5ec6211` 是此次上传时实际代码快照的记录。

`main` 与 `feat/t265-dual-camera-support` 在本次整理完成时指向同一提交。后续功能分支可以独立前进，通过审查和验证后再合并到 `main`。`release/*` 保留对应版本的算法范围，修复通过单独提交维护。

原项目的 GitHub `origin/main` 是 Python 初版，而 GitLab `main` 是 C++ 版；在此仓库中分别通过 `release/d455-initial`、`release/d455-cpp` 表达，`main` 统一指向当前 Python 完整版。原仓库远端和工作区保留，未做覆盖同步。

每个分支的 [来源记录](version_source.json) 保存完整来源提交，具体检查见本分支 [验证记录](validation.md)。

## 数据与精度边界

原始 bag、录制视频、输出、环境、构建包、本机文档草稿不随核心分支上传。真实复现需自行提供采集数据和匹配的相机内参、Marker 几何及外参。初版 rig 对应 48 mm Marker；其他实验配置必须按各版本文档核对。合成闭环与接口测试用于软件验证，不能代替实机绝对精度验收。

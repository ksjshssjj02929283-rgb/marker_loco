# 从旧目录迁移到独立仓库

原始实时实现位于：

```text
/home/dex/1_Dex-work/location-live-dashboard-work/experiments/d455_aruco/
```

整理后的 GitHub 项目目录是：

```text
/home/dex/1_Dex-work/location-live-dashboard-work/marker_6dof_tracking/
```

它把原目录中的 `record_and_track.py`、`tracking_core.py`、`relative_trajectory.py`、`live_dashboard.py` 和 `live_dashboard.html` 分别归入采集、刚体追踪、轨迹构建和实时展示职责。旧 `experiments/d455_aruco/` 可以作为历史兼容入口保留；新开发和上传 GitHub 时只使用本目录。

如果当前终端对工作区有写权限，可执行：

```bash
tar -xzf /tmp/marker_6dof_tracking.tar.gz \
  -C /home/dex/1_Dex-work/location-live-dashboard-work
```

迁移后验证：

```bash
cd /home/dex/1_Dex-work/location-live-dashboard-work/marker_6dof_tracking
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make test
```

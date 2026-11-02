# 本次验证

日期：2026-11-02。Python 3.10。

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src python scripts/run_video.py --help
PYTHONPATH=src python scripts/build_relative_trajectory.py --help
PYTHONPATH=src python scripts/track_d455.py --help
```

编译检查和三个命令的帮助入口均通过。该精简来源提交未携带测试目录，未将其他版本的测试数算作本分支验证。算法源码保持 `4ccf770`。本分支安装使用 `pip install -e .`，命令适配器依赖检出目录中的 `scripts/`；未宣称可独立分发 wheel。未连接相机。

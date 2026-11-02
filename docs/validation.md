# 本次验证

日期：2026-11-02。GNU C++ 11.4、系统 OpenCV 4.5.4、nlohmann_json 3.10.5。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMARKER6DOF_ENABLE_REALSENSE=OFF
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

来源 `76a6a17` 的算法已包含 OpenCV 4.5/4.6 与 4.7+ 的 API 兼容路径，原 CMake 却限定 4.8+。本次按相同版本条件选择 aruco / objdetect，让构建配置与源码一致；未改算法实现。Release 构建通过；CTest 的 `marker_core_test` 通过（1/1 测试目标）。

未启用 RealSense，未重新回放历史录制；README 中的 834 帧/833 帧是历史结果，不是本次采集。未进行绝对精度验收。

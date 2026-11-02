# 本次验证

日期：2026-11-02。GNU C/C++ 11.4、OpenCV 4.5.4、nlohmann_json 3.10.5。

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX="$PWD/build/native/install" \
  -DMARKER6DOF_ENABLE_REALSENSE=OFF
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
cmake --install build/native
python3 scripts/validate_sdk.py --platform native
```

来源 `1063511` 的 CMake 和验证脚本引用了未提交的 `examples/c_api/`，干净检出无法完成默认测试构建。本次补齐 C11 最小调用示例及独立 CMake 配置，检查生命周期、时间戳和 LOST 帧回调；SDK 实现与算法不变。Release 构建和安装通过，3/3 CTest 目标通过；迁移安装目录后的外部 C11 / C++ 示例独立构建与调用均通过，导出符号、SONAME、依赖和 CMake 导出路径检查通过。验证脚本固定 `LC_ALL=C`，使其 ELF 输出解析在中文系统上也能工作。

未运行 Orin 交叉编译、SSH 板端验证、D455 采集或实机绝对精度验收。工具链、sysroot 和板端需另行提供。

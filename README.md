# D455 多面 ArUco 6DoF 算法（C++）

版本：**D455 C++17 工具**；当前分支：`release/d455-cpp`。

[全部版本与选择指南](docs/versions.md) · [源码来源](docs/version_source.json) · [本次验证](docs/validation.md)

从初始版本 `faad044` 移植的 C++17 算法：ArUco 检测、单面 IPPE、多面角点联合 PnP、刚体标定、位姿质量门控和分段相对轨迹。只保留算法、运行入口、必要配置与核心测试。

源码已包含新旧 ArUco API 的兼容路径；本次将 CMake 的模块选择与该路径对齐，算法源码保持来源版本。

## 编译

依赖 OpenCV 4.5.4+（4.5/4.6 使用 aruco，4.7+ 使用 objdetect，视频回放还需 videoio 及相应解码器）、nlohmann_json 3.2+、CMake 3.16+。D455 实时输入还需原生 librealsense2 SDK；离线回放不需要相机或 Python。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

已安装到系统路径时可省略 `OpenCV_DIR`。没有 RealSense SDK 时仍可编译回放与标定；也可显式设置 `-DMARKER6DOF_ENABLE_REALSENSE=OFF`。自定义 SDK 安装路径通过 `CMAKE_PREFIX_PATH` 或 `realsense2_DIR` 指定。

## 已有数据回放

```bash
./build/marker_tracker replay --session /path/to/d455_session --rig configs/rig_calibration_id2.json --marker-size-mm 48 --output-dir outputs/replay --min-coverage 0.95
```

录制目录包含 `color_raw.mp4`、`metadata.json` 和 `frames.csv`。程序读取实际相机内参、畸变模型及逐帧时间戳，输出 `poses.csv`、`trajectory.jsonl`、`summary.json`。设备时钟重复或倒退时，若记录的主机时间严格递增则使用该时间，并在结果中注明；没有时间戳时使用视频帧率。覆盖率不足、解码未完成或中断均以非零退出码结束。已知真内参及刚体角点时，也可用 `run-video --config <标定.json> --video <视频> --output-dir <目录>`。

## D455 实时算法与刚体标定

```bash
./build/marker_tracker live --rig configs/rig_calibration_id2.json --marker-size-mm 48
./build/marker_calibrate_rig --poses /path/to/poses.csv --output outputs/rig.json --reference-id 2 --min-overlap 60 --marker-size-mm 48
```

实时算法逐帧输出 JSON；多台相机时添加 `--serial <序列号>`。标定输入为逐 Tag 位姿 CSV，需含 `frame_index,tag_id,rx_rad,ry_rad,rz_rad,tx_m,ty_m,tz_m,reprojection_error_px`，不是回放输出的相对轨迹 CSV。弱重叠或不连通标定图会被拒绝。

相对轨迹以首个接受位姿为原点，单位米，四元数 xyzw。丢失或跳变会断轨；原始位姿与显示用平滑数据分别输出。相机光学坐标为 +X 向右、+Y 向下、+Z 向前。

`rig_calibration_id2.json` 对应原始 48 mm 多面 Marker；更换尺寸或刚体几何需重新标定。`demo_calibration.json` 仅作配置格式示例，不能用作真实内参。历史录制与验证产物只保留在本地。

已用现有 48 mm D455 录制验证：834 帧完整回放、833 帧接受（99.88%），各帧状态与使用的标签均与初版 Python 一致；9 组核心测试通过。验收范围为录制数据回放、算法对照与异常输入处理。

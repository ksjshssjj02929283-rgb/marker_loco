# dfmarker：多面 ArUco 6DoF C SDK

C++17 实现 ArUco 检测、单面 IPPE、多面角点联合 PnP、位姿质量门控及分段相对轨迹。公开 `dfmarker_*` C API 和 `libdfmarker.so`，调用方负责图像采集、视频解码和保存。SDK 不依赖 ROS 或 RealSense；原有 D455、TCP 定位和 rig 标定工具可选构建。

## 编译和安装

依赖 CMake 3.16+、C/C++ 编译器、OpenCV 4.5.4+ 和 nlohmann_json 3.2+。4.5 使用独立 `aruco` 模块，4.7 起使用 `objdetect`。本机使用 4.5.4，Orin 使用 4.8.0；同一进程中的 OpenCV 调用方与 SDK 应使用相同构建版本。

```bash
./scripts/build_native.sh
```

脚本执行构建、测试、安装和打包，默认只构建 SDK。安装前缀为 `build/native/install/`，交付包为 `build/native/dfmarker-0.1.0-x86_64.tar.gz`。当前机器复用 `../Dependence/3rd_party/nlohmann/include`；可通过 `NLOHMANN_JSON_INCLUDE_DIR` 指定头文件目录，或安装 nlohmann_json 的 CMake package。其他可覆盖的环境变量：`DFMARKER_OUTPUT_DIR`、`DFMARKER_BUILD_TYPE`、`DFMARKER_JOBS`、`DFMARKER_BUILD_TOOLS` 和 `DFMARKER_ENABLE_REALSENSE`。

标准 CMake 构建：

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/native/install" \
  -DNLOHMANN_JSON_INCLUDE_DIR=/path/to/nlohmann/include
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
cmake --install build/native
```

安装包包含唯一公开头文件 `include/dfmarker_api.h`、`lib/libdfmarker.so.0.1.0` 及 `.so.0`/`.so` 符号链接、`share/dfmarker/cmake`、配置、示例、许可证及 `build-info.txt`。OpenCV 为动态运行库依赖，不随 SDK 打包；JSON 仅用于内部编译。库仅导出七个 `dfmarker_*` 函数。版本和 SONAME 由 CMake 维护，不兼容 ABI 变更需升级主版本。

## 外部程序调用

公开头文件兼容 C11/C++，不暴露 OpenCV、JSON 或 STL 类型。纯 C 调用方仅需查找 SDK：

```cmake
find_package(dfmarker 0.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE dfmarker::dfmarker)
```

调用顺序：创建 → 注册回调 → 启动 → 输入图像 → 停止 → 释放。句柄必须初始化为 NULL，操作返回 1 成功、0 失败：

```c
DFMARKER_HANDLE handle = NULL;
dfmarker_create("sdk_config.json", &handle);
dfmarker_register_result_callback(handle, on_result, user_data);
dfmarker_start(handle);
/* 每帧填充 dfmarker_image_t，检查 dfmarker_feed_image(handle, &image) 的返回值。 */
dfmarker_stop(handle);
dfmarker_destroy(&handle);
```

输入支持 MONO8、RGB8、BGR8 和行填充，分辨率须匹配标定。时间戳为 uint64_t 纳秒，第一帧允许为零，之后严格递增；原始时间戳保留，算法使用相对首帧的时间差。图像数据在 feed 返回前须可读且不变，返回后可释放。

同一句柄串行调用，回调在输入线程同步执行，禁止重入或销毁当前句柄。回调中的数组、reason 和结果指针仅在回调期间有效，保留时须复制所需数据。每个有效输入帧回调一次，包括 LOST/WARN；这些帧仍返回成功。非法输入不消费帧号、时间或追踪状态；意外处理异常进入 ERROR，须销毁重建，诊断写入 stderr。重复 start/stop 幂等，stop 后无法重新 start；重复销毁已置空句柄成功。

`dfmarker_get_status` 查询生命周期 CREATED/RUNNING/STOPPED/ERROR；回调给出每帧质量 LOST/WARN/SINGLE/MULTI，两者独立。结果字段与有效标志的完整定义见 `dfmarker_api.h`：

- 位姿包含位置（米）和 Hamilton xyzw 四元数。
- `camera_pose` 将参考 Tag 坐标转换到相机光学坐标：+X 向右、+Y 向下、+Z 向前。
- `relative_pose` 表达在首个接受的参考 Tag 坐标系中，轴方向不一定与相机一致。`smooth_relative_pose` 仅用于显示。
- `has_pose` 为零时三个位姿都无效且清零；误差、角点的缺失也有有效标志。
- 丢失、拒绝和长时间间隔断轨，恢复后增加 segment 并保留最初原点；重建句柄可建立新原点。
- 检测角点、可见/使用/拒绝的 Tag ID 及逐 Tag 误差通过只读数组给出。

## SDK 配置与示例

参考 `configs/sdk_demo.json`。配置必填 `schema_version: 1`、`units: "m"`、`camera` 和 `rig_file`；camera 格式沿用现有相机参数，包含图像宽高、3x3 camera_matrix、distortion 和 distortion_model。支持 opencv、none、brown_conrady、inverse_brown_conrady；D455 inverse Brown 不能当作普通 OpenCV Brown 使用。

可选项默认为 `marker_size_m: 0.048`、`aruco.dictionary: "DICT_4X4_50"`、`quality.max_error_px: 2.5`、`quality.min_edge_px: 24`。rig 相对路径以 SDK 配置所在目录解析，格式沿用 reference_id、T_reference_tag、marker_size_m；尺寸与标定不一致时创建失败。示例相机内参是合成值，真实使用需替换；原始 rig 对应 48 mm 多面 Marker，换刚体几何需重新标定。

TCP 是夹爪实际工作点；rig 标定确定各 Tag 之间的固定几何。两者此次保留在工具中，公开 SDK 追踪参考 Tag。

```bash
python3 scripts/validate_sdk.py --platform native

cmake -S examples/c_api -B build/c-example -DCMAKE_PREFIX_PATH="$PWD/build/native/install"
cmake --build build/c-example
build/c-example/c_api_smoke configs/sdk_demo.json

cmake -S examples/cpp -B build/cpp-example -DCMAKE_PREFIX_PATH="$PWD/build/native/install"
cmake --build build/cpp-example
build/cpp-example/sdk_smoke configs/sdk_demo.json
build/cpp-example/video_demo /path/to/sdk_config.json /path/to/video.mp4
```

纯 C 示例使用 demo 的 640x480 配置，验证生命周期和无 Tag 时的回调。C++ smoke 生成 ID 2 图像，验证位姿和丢失。C++ 示例共享 OpenCV，使用 SDK 导出的 `dfmarker_OPENCV_VERSION` 选择一致版本；video_demo 另需 OpenCV highgui 和 calib3d。验证脚本迁移安装前缀后独立构建示例，检查导出符号、运行库、SONAME 和安装配置，日志保存在 `build/native/validation/`。

### PC 视频可视化验证

`video_demo` 通过公开 C API 输入图像。默认打开窗口，叠加 Tag 检测框（绿色：参与定位；橙色：被拒绝；灰色：其他检测），并绘制相机图像中的参考 Tag 坐标轴（X 红、Y 绿、Z 蓝）。右侧显示相对位置、RPY 角和 xyzw 四元数，以及表达在首个接受的参考 Tag 坐标系中的 XY 轨迹。图像坐标轴使用 `camera_pose`，轨迹和相对读数使用相对位姿；两者的坐标系不同。D455 inverse Brown 畸变按 SDK 相同的投影模型绘制。

按 Space 暂停或继续，N 或右箭头前进一帧，Q 或 Esc 退出。视频结束后默认保留最后画面，添加 `--exit-on-finish` 可自动退出。LOST/WARN 帧无当前有效位姿，不显示当前位姿轴；历史轨迹保留，并按 `segment` 断开，恢复后继续使用最初原点。

对完整离线测试包，可直接读取包内 rig、录制内参和真实逐帧时间戳：

```bash
PACKAGE="/home/dex/Projects/DexForce/d455_cpp_76a6a17_offline_test/test_package_76a6a17"

./build/cpp-example/video_demo --package "$PACKAGE" \
  --session fast_20260924_163142
```

本包实际 rig 文件是 `configs/rig_calibration_id2.json`，与当前仓库同名文件一致；其标定尺寸为 **48 mm**。录制 `metadata.json` 中写的是 47.5 mm，存在不一致；本次按 rig 的 48 mm 配置求解，不能将这份回放作为绝对尺度精度的证明。`--rig /path/to/rig.json` 可显式选择其他 rig，尺寸应与该 rig 标定一致。

无窗口导出带叠加画面的视频和逐帧相对位姿 CSV：

```bash
mkdir -p outputs/pc_fast_20260924_163142
./build/cpp-example/video_demo --package "$PACKAGE" \
  --session fast_20260924_163142 --no-display \
  --csv outputs/pc_fast_20260924_163142/sdk_poses.csv \
  --output-video outputs/pc_fast_20260924_163142/sdk_visualization.mp4
```

也可继续使用显式 SDK 配置和视频文件，添加 `--timestamps` 读取录制时间；未提供时按视频 FPS 生成时间。真实时间用于位姿突跳检查、平滑和轨迹分段，正式时序验证应优先使用它。设备时钟异常时，逐帧时间读取与 replay 工具相同，优先回退到严格递增的记录主机时间。

```bash
./build/cpp-example/video_demo /path/to/sdk_config.json /path/to/color_raw.mp4 \
  --timestamps /path/to/frames.csv --csv /path/to/poses.csv
```

`--csv`、`--output-video`、`--no-display` 和 `--exit-on-finish` 适用于两种输入方式。无窗口模式处理完视频后自动退出；MP4 按固定 FPS 保存画面，CSV 保留用于求解的逐帧时间戳。

## Orin 交叉编译与验证

```bash
./scripts/build_orin.sh
python3 scripts/validate_sdk.py --platform orin
# 若使用已登录的 SSH 复用会话：
python3 scripts/validate_sdk.py --platform orin --ssh-control /tmp/dfmarker-orin-ssh
```

复用 `../Dependence` 下的 GNU 11.3.1 工具链和 `orin-nx-sysroot/opencv48`；可通过 DFMARKER_TOOLCHAIN_ROOT、DFMARKER_SYSROOT、DFMARKER_DEPENDENCE_ROOT 和 DFMARKER_OUTPUT_DIR 覆盖路径。输出为 `build/orin/install/` 和 `build/orin/dfmarker-0.1.0-aarch64.tar.gz`，包内记录 sysroot manifest。

验证默认连接 dexforce@192.168.55.1，可用 `--host`、`--user` 或 `--ssh-control` 指定连接，也支持 DFMARKER_SSH_CONTROL 环境变量。脚本不存储密码，只上传到板端新建的 `/tmp/dfmarker-validation.*`，保留临时目录和本机日志便于复查，不安装或替换板端系统库。它运行外部 C/C++ 示例及核心/API 测试，检查 `.408` 依赖和未解析符号。交叉编译与板端运行分别验收。

## 原有工具构建

```bash
cmake -S . -B build/tools -DCMAKE_BUILD_TYPE=Release \
  -DMARKER6DOF_BUILD_TOOLS=ON -DNLOHMANN_JSON_INCLUDE_DIR=/path/to/nlohmann/include
cmake --build build/tools --parallel 2
```

工具另需 OpenCV videoio，D455 live 另需原生 librealsense2 SDK。无 RealSense SDK 时回放、TCP 定位和标定仍可用，也可显式设置 MARKER6DOF_ENABLE_REALSENSE=OFF。

## 已有数据回放

```bash
./build/tools/marker_tracker replay --session /path/to/d455_session --rig configs/rig_calibration_id2.json --marker-size-mm 48 --output-dir outputs/replay --min-coverage 0.95
```

录制目录包含 `color_raw.mp4`、`metadata.json` 和 `frames.csv`。程序读取实际相机内参、畸变模型及逐帧时间戳，输出 `poses.csv`、`trajectory.jsonl`、`summary.json`。设备时钟重复或倒退时，若记录的主机时间严格递增则使用该时间，并在结果中注明；没有时间戳时使用视频帧率。覆盖率不足、解码未完成或中断均以非零退出码结束。已知真内参及刚体角点时，也可用 `run-video --config <标定.json> --video <视频> --output-dir <目录>`。

## D455 实时算法与刚体标定

```bash
./build/tools/marker_tracker live --rig configs/rig_calibration_id2.json --marker-size-mm 48
./build/tools/marker_calibrate_rig --poses /path/to/poses.csv --output outputs/rig.json --reference-id 2 --min-overlap 60 --marker-size-mm 48
```

实时算法逐帧输出 JSON；多台相机时添加 `--serial <序列号>`。标定输入为逐 Tag 位姿 CSV，需含 `frame_index,tag_id,rx_rad,ry_rad,rz_rad,tx_m,ty_m,tz_m,reprojection_error_px`，不是回放输出的相对轨迹 CSV。弱重叠或不连通标定图会被拒绝。

相对轨迹以首个接受位姿为原点，单位米，四元数 xyzw。丢失或跳变会断轨；原始位姿与显示用平滑数据分别输出。

`rig_calibration_id2.json` 对应原始 48 mm 多面 Marker；更换尺寸或刚体几何需重新标定。`demo_calibration.json` 仅作配置格式示例，不能用作真实内参。历史录制与验证产物只保留在本地。

历史 48 mm D455 录制验证为 834 帧完整回放、833 帧接受（99.88%），各帧状态与使用标签与初版 Python 一致。当前 SDK 测试覆盖原有 9 组算法测试、输入/生命周期/异常处理、合成图像检测至位姿回调及外部 C/C++ 调用。合成数据验证接口和算法行为，不作为实机精度报告。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(DFMARKER_TOOLCHAIN_ROOT "" CACHE PATH "Arm GNU toolchain root")
set(DFMARKER_SYSROOT "" CACHE PATH "Prepared Orin OpenCV 4.8 sysroot")
if(NOT EXISTS "${DFMARKER_TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++" OR
   NOT EXISTS "${DFMARKER_SYSROOT}/usr/include")
    message(FATAL_ERROR "Set DFMARKER_TOOLCHAIN_ROOT and DFMARKER_SYSROOT")
endif()
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES DFMARKER_TOOLCHAIN_ROOT DFMARKER_SYSROOT)
set(CMAKE_C_COMPILER "${DFMARKER_TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${DFMARKER_TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++")
set(CMAKE_SYSROOT "${DFMARKER_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY OFF)
set(CMAKE_IGNORE_PREFIX_PATH /usr/local /opt/ros)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH "")
# Match the board's libc/start files and libstdc++, rather than the bundled
# toolchain runtime. Multiarch paths are not inferred by this GNU toolchain.
set(_multiarch "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
string(CONCAT _flags "-mcpu=cortex-a78 -B${_multiarch}/ "
    "-L${CMAKE_SYSROOT}/toolchain-libs -L${_multiarch} "
    "-isystem ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")
set(CMAKE_C_FLAGS_INIT "${_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_flags}")
string(CONCAT _link "-Wl,-rpath-link,${_multiarch} "
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib "
    "-Wl,-rpath-link,${_multiarch}/openblas-pthread")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_link} -Wl,--no-undefined")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_link}")

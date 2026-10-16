#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEPENDENCE_ROOT="${DFMARKER_DEPENDENCE_ROOT:-${PROJECT_ROOT}/../Dependence}"
TOOLCHAIN_ROOT="${DFMARKER_TOOLCHAIN_ROOT:-${DEPENDENCE_ROOT}/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu_mins/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu}"
SYSROOT="${DFMARKER_SYSROOT:-${DEPENDENCE_ROOT}/orin-nx-sysroot/opencv48}"
OUTPUT_DIR="${DFMARKER_OUTPUT_DIR:-${PROJECT_ROOT}/build/orin}"
BUILD_TYPE="${DFMARKER_BUILD_TYPE:-Release}"
JOBS="${DFMARKER_JOBS:-2}"
if (($#)); then echo "Set environment variables; this script takes no positional arguments" >&2; exit 2; fi
cmake -S "${PROJECT_ROOT}" -B "${OUTPUT_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/toolchain-aarch64-orin.cmake" \
    -DDFMARKER_TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}" -DDFMARKER_SYSROOT="${SYSROOT}" \
    -DOpenCV_DIR="${SYSROOT}/usr/lib/cmake/opencv4" -DDFMARKER_EXPECT_OPENCV_VERSION=4.8.0 \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}/install" \
    -DCMAKE_INSTALL_LIBDIR=lib -DMARKER6DOF_BUILD_TOOLS=OFF -DMARKER6DOF_ENABLE_REALSENSE=OFF \
    -DBUILD_TESTING=ON
cmake --build "${OUTPUT_DIR}" --parallel "${JOBS}"
if [[ -f "${SYSROOT}/../manifest.json" ]]; then DFMARKER_MANIFEST="${SYSROOT}/../manifest.json"; fi
source "${PROJECT_ROOT}/scripts/package_sdk.sh"
package_sdk "${OUTPUT_DIR}" "${OUTPUT_DIR}/install" "${OUTPUT_DIR}" aarch64
echo "Run scripts/validate_sdk.py --platform orin for board validation."

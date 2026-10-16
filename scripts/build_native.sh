#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# These defaults reuse local dependencies; override via environment on another machine.
DEPENDENCE_ROOT="${DFMARKER_DEPENDENCE_ROOT:-${PROJECT_ROOT}/../Dependence}"
OUTPUT_DIR="${DFMARKER_OUTPUT_DIR:-${PROJECT_ROOT}/build/native}"
BUILD_TYPE="${DFMARKER_BUILD_TYPE:-Release}"
JOBS="${DFMARKER_JOBS:-2}"
NLOHMANN_INCLUDE_DIR="${NLOHMANN_JSON_INCLUDE_DIR:-${DEPENDENCE_ROOT}/3rd_party/nlohmann/include}"
if (($#)); then echo "Set environment variables; this script takes no positional arguments" >&2; exit 2; fi
cmake -S "${PROJECT_ROOT}" -B "${OUTPUT_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}/install" \
    -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_TESTING=ON \
    -DNLOHMANN_JSON_INCLUDE_DIR="${NLOHMANN_INCLUDE_DIR}" \
    -DMARKER6DOF_BUILD_TOOLS="${DFMARKER_BUILD_TOOLS:-OFF}" \
    -DMARKER6DOF_ENABLE_REALSENSE="${DFMARKER_ENABLE_REALSENSE:-OFF}"
cmake --build "${OUTPUT_DIR}" --parallel "${JOBS}"
ctest --test-dir "${OUTPUT_DIR}" --output-on-failure
source "${PROJECT_ROOT}/scripts/package_sdk.sh"
package_sdk "${OUTPUT_DIR}" "${OUTPUT_DIR}/install" "${OUTPUT_DIR}" "$(uname -m)"

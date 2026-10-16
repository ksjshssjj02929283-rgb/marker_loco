#!/usr/bin/env bash
# Sourced by the build entry points. Stage into a fresh directory so incremental
# installation cannot accidentally put old binaries into the delivery package.
package_sdk() (
    set -euo pipefail
    local build_dir="$1" install_dir="$2" output_dir="$3" architecture="$4"
    local version staging package_name
    version=$(sed -n 's/^CMAKE_PROJECT_VERSION:STATIC=//p' "${build_dir}/CMakeCache.txt")
    [[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
    package_name="dfmarker-${version}"
    cmake --install "${build_dir}" --prefix "${install_dir}"
    staging=$(mktemp -d "${output_dir}/.package.XXXXXX")
    trap 'rm -rf -- "${staging}"' EXIT
    cmake --install "${build_dir}" --prefix "${staging}/${package_name}"
    if [[ -n "${DFMARKER_MANIFEST:-}" ]]; then
        cp "${DFMARKER_MANIFEST}" "${install_dir}/share/dfmarker/sysroot-manifest.json"
        cp "${DFMARKER_MANIFEST}" "${staging}/${package_name}/share/dfmarker/sysroot-manifest.json"
    fi
    tar -C "${staging}" -czf "${output_dir}/${package_name}-${architecture}.tar.gz" "${package_name}"
    echo "SDK: ${install_dir}"
    echo "Package: ${output_dir}/${package_name}-${architecture}.tar.gz"
)

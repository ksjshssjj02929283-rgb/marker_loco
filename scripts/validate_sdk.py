#!/usr/bin/env python3
"""Validate relocated C/C++ consumers locally or in an Orin temporary directory."""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
DEPENDENCE = PROJECT.parent / 'Dependence'


def run(*args, capture=False):
    print('+', shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          env={**os.environ, 'LC_ALL': 'C'}).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['native', 'orin'], default='native')
    parser.add_argument('--sdk-root', type=Path, help='Installed SDK prefix')
    parser.add_argument('--build-root', type=Path, help='SDK build containing core/API test binaries')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--toolchain-root', type=Path, default=DEPENDENCE /
                        'arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu_mins/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu')
    parser.add_argument('--sysroot', type=Path, default=DEPENDENCE / 'orin-nx-sysroot/opencv48')
    parser.add_argument('--host', default='192.168.55.1')
    parser.add_argument('--user', default='dexforce')
    parser.add_argument('--ssh-control', help='Existing SSH ControlMaster socket; no passwords are stored')
    args = parser.parse_args()
    build = (args.build_root or PROJECT / 'build' / args.platform).resolve()
    sdk = (args.sdk_root or build / 'install').resolve()
    work = (args.work_dir or build / 'validation').resolve()
    work.mkdir(parents=True, exist_ok=True)
    library = sdk / 'lib/libdfmarker.so'
    dynamic = run('readelf', '-d', library, capture=True)
    header = run('readelf', '-h', library, capture=True)
    symbols = run('nm', '-D', '--defined-only', library, capture=True)
    exports = {line.split()[-1] for line in symbols.splitlines() if line.split()}
    expected = {'dfmarker_' + name for name in ['create', 'start', 'stop', 'destroy', 'feed_image',
                                               'get_status', 'register_result_callback']}
    assert exports == expected, symbols
    assert 'RPATH' not in dynamic and 'RUNPATH' not in dynamic, dynamic
    version = re.search(r'^Version: (.+)$', (sdk / 'share/dfmarker/build-info.txt').read_text(), re.M).group(1)
    assert f'Library soname: [libdfmarker.so.{version.split(".")[0]}]' in dynamic, dynamic
    needed = re.findall(r'Shared library: \[(.*?)\]', dynamic)
    assert not any(re.search(r'videoio|realsense|rcl|ros|ament|cv_bridge', name) for name in needed), needed
    opencv = [name for name in needed if name.startswith('libopencv_')]
    assert opencv, needed
    if args.platform == 'orin':
        assert 'AArch64' in header and all('.408' in name for name in opencv), (header, needed)
    (work / 'elf.txt').write_text(header + dynamic + symbols)
    cmake_exports = '\n'.join(p.read_text() for p in (sdk / 'share/dfmarker/cmake').glob('*.cmake'))
    assert str(PROJECT) not in cmake_exports and str(DEPENDENCE) not in cmake_exports, cmake_exports
    relocated = work / 'relocated-sdk'
    # Repeated validation must replace symlinks and remove stale SDK files.
    if sdk == relocated or sdk in relocated.parents or relocated in sdk.parents:
        raise ValueError('SDK prefix and relocated output must not overlap')
    if relocated.is_symlink():
        raise ValueError('Relocated output must not be a symlink')
    if relocated.exists():
        shutil.rmtree(relocated)
    shutil.copytree(sdk, relocated, symlinks=True)
    common = [f'-Ddfmarker_DIR={relocated}/share/dfmarker/cmake', '-DCMAKE_BUILD_TYPE=Release']
    if args.platform == 'orin':
        common += [f'-DCMAKE_TOOLCHAIN_FILE={PROJECT}/cmake/toolchain-aarch64-orin.cmake',
                   f'-DDFMARKER_TOOLCHAIN_ROOT={args.toolchain_root}', f'-DDFMARKER_SYSROOT={args.sysroot}']
    for example in ['c_api', 'cpp']:
        options = common.copy()
        if args.platform == 'orin' and example == 'cpp':
            options += [f'-DOpenCV_DIR={args.sysroot}/usr/lib/cmake/opencv4']
        run('cmake', '-S', relocated / 'share/dfmarker/examples' / example, '-B', work / example, *options)
        run('cmake', '--build', work / example, '--parallel', '2')
    if args.platform == 'native':
        for binary in [work / 'c_api/c_api_smoke', work / 'cpp/sdk_smoke']:
            output = run(binary, relocated / 'share/dfmarker/configs/sdk_demo.json', capture=True)
            assert 'PASS:' in output, output
            (work / f'{binary.name}.txt').write_text(output)
        print('PASS: relocated native C/C++ SDK; logs:', work)
        return

    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10']
    control = args.ssh_control or os.environ.get('DFMARKER_SSH_CONTROL') or os.environ.get('DFSLAM_SSH_CONTROL')
    if control:
        ssh += ['-S', control]
    remote = f'{args.user}@{args.host}'

    def remote_run(command):
        return run(*ssh, remote, command, capture=True)

    stage = remote_run('mktemp -d /tmp/dfmarker-validation.XXXXXX').strip()
    (work / 'board-directory.txt').write_text(stage + '\n')
    run('rsync', '-a', '-e', shlex.join(ssh), str(relocated) + '/', f'{remote}:{stage}/sdk/')
    binaries = [work / 'c_api/c_api_smoke', work / 'cpp/sdk_smoke', build / 'marker_core_test', build / 'marker_api_test']
    for binary in binaries:
        run('rsync', '-a', '-e', shlex.join(ssh), binary, f'{remote}:{stage}/{binary.name}')
    prefix = 'env LD_LIBRARY_PATH=' + shlex.quote(stage + '/sdk/lib')
    config = shlex.quote(stage + '/sdk/share/dfmarker/configs/sdk_demo.json')
    for binary in binaries:
        executable = shlex.quote(stage + '/' + binary.name)
        deps = remote_run(f'{prefix} ldd -r {executable}')
        assert 'not found' not in deps and 'undefined symbol' not in deps, deps
        (work / f'{binary.name}-ldd.txt').write_text(deps)
        command = f'{prefix} {executable}'
        if binary.name in ['c_api_smoke', 'sdk_smoke']:
            command += ' ' + config
        output = remote_run(command)
        assert 'PASS:' in output, output
        (work / f'{binary.name}.txt').write_text(output)
    print('PASS: relocated Orin C/C++ SDK and algorithm/API tests; logs:', work, '; board:', stage)


if __name__ == '__main__':
    main()

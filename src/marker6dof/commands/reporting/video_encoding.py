"""Encode comparison artifacts as H.264 for browser and document previews."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def find_ffmpeg():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    return get_ffmpeg_exe()


def encode_comparison_videos(directory, filenames=('before.mp4', 'after.mp4', 'side_by_side.mp4')):
    """Replace only generated videos, after each conversion succeeds.

    Original recordings are not touched. FFmpeg is an optional reporting
    dependency; without it, retain OpenCV's MPEG-4 files and report the codec.
    """
    executable = find_ffmpeg()
    if executable is None:
        return {
            'codec': 'mpeg4', 'browser_compatible_encoding': False,
            'note': 'Install reporting dependencies for H.264 browser previews: python -m pip install -e ".[reporting]"',
        }
    directory = Path(directory)
    for filename in filenames:
        source = directory / filename
        temporary = source.with_name(source.stem + '.encoding.mp4')
        try:
            subprocess.run([
                executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                '-i', str(source), '-map', '0:v:0', '-an', '-c:v', 'libx264',
                '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', '-threads', '2', str(temporary),
            ], check=True, capture_output=True, text=True)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f'Encoder produced no video for {filename}')
            temporary.replace(source)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'H.264 conversion failed for {filename}: {error.stderr.strip()}') from error
        finally:
            temporary.unlink(missing_ok=True)
    return {'codec': 'h264', 'pixel_format': 'yuv420p', 'faststart': True, 'browser_compatible_encoding': True}

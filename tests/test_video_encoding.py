import subprocess

import pytest

from marker6dof.commands.reporting import video_encoding


def test_missing_optional_encoder_preserves_videos(tmp_path, monkeypatch):
    source = tmp_path / 'before.mp4'
    source.write_bytes(b'original')
    monkeypatch.setattr(video_encoding, 'find_ffmpeg', lambda: None)
    result = video_encoding.encode_comparison_videos(tmp_path, ('before.mp4',))
    assert result['codec'] == 'mpeg4'
    assert source.read_bytes() == b'original'


def test_failed_conversion_preserves_generated_video_and_cleans_temporary(tmp_path, monkeypatch):
    source = tmp_path / 'before.mp4'
    source.write_bytes(b'original')
    monkeypatch.setattr(video_encoding, 'find_ffmpeg', lambda: 'ffmpeg')

    def failure(command, **kwargs):
        (tmp_path / 'before.encoding.mp4').write_bytes(b'incomplete')
        raise subprocess.CalledProcessError(1, command, stderr='encoding error')

    monkeypatch.setattr(video_encoding.subprocess, 'run', failure)
    with pytest.raises(RuntimeError, match='encoding error'):
        video_encoding.encode_comparison_videos(tmp_path, ('before.mp4',))
    assert source.read_bytes() == b'original'
    assert not (tmp_path / 'before.encoding.mp4').exists()

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obolha import (
    compose_facecam,
    facecam_layout,
    build_facecam_ffmpeg_cmd,
    get_media_duration,
    get_video_encoder,
    clear_encoder_cache,
    video_encode_args,
)


@pytest.fixture(autouse=True)
def reset_encoder_cache():
    clear_encoder_cache()
    yield
    clear_encoder_cache()


def test_get_video_encoder_prefers_libopenh264_without_libx264():
    fake_encoders = " V..... libopenh264          OpenH264\n V.S..D mpeg4\n"
    with patch("obolha._ffmpeg_encoders_stdout", return_value=fake_encoders):
        assert get_video_encoder() == "libopenh264"
        assert video_encode_args("libopenh264") == ["-c:v", "libopenh264", "-b:v", "4M"]


def test_facecam_layout_9_16():
    layout = facecam_layout()
    assert layout["out_w"] == 1080
    assert layout["out_h"] == 1920
    assert layout["facecam_h"] == 768  # 40% of 1920
    assert layout["bottom_h"] == 1152


def test_build_facecam_ffmpeg_cmd_structure():
    clip = Path("/tmp/clip.mp4")
    facecam = Path("/tmp/cam.mp4")
    out = Path("/tmp/out.mp4")
    cmd = build_facecam_ffmpeg_cmd(clip, facecam, out, duration=60.0)

    assert cmd[0] == "ffmpeg"
    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=1080:1152" in fc
    assert "crop=1080:1152" in fc
    assert "scale=1080:768" in fc
    assert "vstack=inputs=2" in fc
    assert "overlay=" not in fc
    assert "-t" in cmd and "60.00" in cmd
    assert "-map" in cmd and "-1:a" in cmd
    assert "-c:v" in cmd
    assert str(clip) in cmd
    assert str(facecam) in cmd
    assert str(out) in cmd


def test_compose_facecam_missing_clip():
    with pytest.raises(FileNotFoundError, match="clip"):
        compose_facecam("/nonexistent/clip.mp4", "/tmp/cam.mp4")


def test_compose_facecam_missing_facecam():
    clip = Path("/tmp/clip.mp4")
    cam = Path("/nonexistent/cam.mp4")

    def fake_exists(self):
        return self == clip.resolve()

    with patch.object(Path, "exists", fake_exists):
        with pytest.raises(FileNotFoundError, match="facecam"):
            compose_facecam(clip, cam)


def test_compose_facecam_runs_ffmpeg():
    clip = Path("/tmp/clip.mp4")
    facecam = Path("/tmp/cam.mp4")
    out = Path("/tmp/out.mp4")

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stderr = ""

    with patch.object(Path, "exists", return_value=True), \
         patch("obolha.get_media_duration", return_value=45.0), \
         patch("obolha.get_video_encoder", return_value="libopenh264"), \
         patch("obolha.audio_encode_args", return_value=["-c:a", "aac", "-b:a", "192k"]), \
         patch.object(Path, "stat") as mock_stat, \
         patch("subprocess.run", mock_run):
        mock_stat.return_value.st_size = 5000000
        result = compose_facecam(clip, facecam, out)

    mock_run.assert_called_once()
    assert result == out


def test_get_media_duration_parses_ffprobe():
    mock_run = MagicMock()
    mock_run.return_value.stdout = "123.456\n"
    mock_run.return_value.returncode = 0

    with patch("subprocess.run", mock_run):
        assert get_media_duration(Path("/tmp/v.mp4")) == 123.456

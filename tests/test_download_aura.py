"""Tests for download_video and aura pipeline hardening."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obolha import (
    JobStatus,
    _format_yt_dlp_error,
    cut_aura_clips,
    download_video,
    ffmpeg_cut_segment,
    is_youtube_url,
    process_aura_video,
)


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/live/abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("/tmp/video.mp4")


def test_format_yt_dlp_error_adds_cookie_hint():
    msg = _format_yt_dlp_error("ERROR: The page needs to be reloaded.")
    assert "yt-dlp" in msg.lower()
    assert "cookies" not in msg.lower()


def test_yt_dlp_youtube_args_default_no_cookies(monkeypatch):
    monkeypatch.delenv("CLIPPER_YOUTUBE_USE_COOKIES", raising=False)
    from obolha import yt_dlp_youtube_args

    args = yt_dlp_youtube_args()
    assert "--no-cookies" in args
    assert "youtube:player_client=android,web" in args


def test_yt_dlp_youtube_args_with_cookies_opt_in(tmp_path, monkeypatch):
    from obolha import yt_dlp_youtube_args

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))
    monkeypatch.setenv("CLIPPER_YOUTUBE_USE_COOKIES", "1")

    args = yt_dlp_youtube_args()
    assert "--no-cookies" not in args
    assert "--cookies" in args
    assert "youtube:player_client=tv,tv_downgraded" in args


def test_download_video_works_without_cookies_file(tmp_path, monkeypatch):
    monkeypatch.setattr("obolha._cookie_source_path", lambda: tmp_path / "missing.txt")
    job = JobStatus(url="https://youtu.be/x")
    work = tmp_path / "work"
    work.mkdir()

    title_ok = MagicMock(returncode=0, stdout="Meu Discurso\n", stderr="")
    dl_ok = MagicMock(returncode=0, stderr="")
    (work / "video.mp4").write_bytes(b"\x00" * 2048)

    with patch("obolha.subprocess.run", side_effect=[title_ok, MagicMock(), dl_ok]):
        path, title, subs = download_video("https://youtu.be/x", work, job)

    assert title == "Meu Discurso"
    assert path.name == "video.mp4"
    assert subs == []


def test_download_video_fails_fast_on_bad_title(tmp_path, monkeypatch):
    monkeypatch.setattr("obolha._cookie_source_path", lambda: tmp_path / "cookies.txt")
    (tmp_path / "cookies.txt").write_text("# netscape\n")
    job = JobStatus(url="https://youtu.be/x")
    work = tmp_path / "work"
    work.mkdir()

    title_fail = MagicMock(returncode=1, stdout="", stderr="ERROR: bot check")

    with patch("obolha.subprocess.run", return_value=title_fail):
        with pytest.raises(RuntimeError, match="yt-dlp falhou"):
            download_video("https://youtu.be/x", work, job)


def test_ffmpeg_square_crop_vf_default_1080():
    from obolha import ffmpeg_square_crop_vf

    vf = ffmpeg_square_crop_vf()
    assert "scale=1080:1080" in vf
    assert "crop=1080:1080" in vf
    assert "force_original_aspect_ratio=increase" in vf


def test_ffmpeg_cut_segment_square_uses_crop_filter(tmp_path):
    from obolha import ffmpeg_cut_segment

    video = tmp_path / "src.mp4"
    video.write_bytes(b"\x00" * 2048)
    out = tmp_path / "out.mp4"
    encode_ok = MagicMock(returncode=0, stderr="")

    def side_effect(cmd, **kwargs):
        Path(out).write_bytes(b"ok" * 512)
        return encode_ok

    with patch("obolha.subprocess.run", side_effect=side_effect) as mock_run, \
         patch("obolha.get_video_encoder", return_value="libx264"):
        ffmpeg_cut_segment(video, 0.0, 5.0, out, square_size=1080)

    cmd = mock_run.call_args.args[0]
    assert "-vf" in cmd
    assert "crop=1080:1080" in cmd[cmd.index("-vf") + 1]
    assert "copy" not in cmd


def test_ffmpeg_cut_segment_reencodes_on_copy_failure(tmp_path):
    video = tmp_path / "src.mp4"
    video.write_bytes(b"\x00" * 2048)
    out = tmp_path / "out.mp4"

    copy_fail = MagicMock(returncode=1, stderr="copy failed")
    encode_ok = MagicMock(returncode=0, stderr="")

    def side_effect(cmd, **kwargs):
        if "-c" in cmd and "copy" in cmd:
            return copy_fail
        Path(out).write_bytes(b"ok" * 512)
        return encode_ok

    with patch("obolha.subprocess.run", side_effect=side_effect), \
         patch("obolha.get_video_encoder", return_value="libx264"):
        ffmpeg_cut_segment(video, 0.0, 5.0, out)

    assert out.is_file()


def test_cut_aura_clips_raises_when_all_fail(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"\x00" * 2048)
    job = JobStatus(url="file://" + str(video))
    clips = [{"start": 0, "end": 5, "titulo": "x", "score_final": 8}]

    with patch("obolha.ffmpeg_cut_segment", side_effect=RuntimeError("cut failed")):
        with pytest.raises(RuntimeError, match="Nenhum clip aura"):
            cut_aura_clips(video, clips, tmp_path / "out", "T", job)


def test_process_aura_video_fails_on_zero_results(tmp_path, monkeypatch):
    from obolha import CFG

    monkeypatch.setitem(CFG, "aura_dir", tmp_path / "aura")
    fake = tmp_path / "v.mp4"
    fake.write_bytes(b"x" * 2048)
    job = JobStatus(url=str(fake))

    with (
        patch("obolha.is_local_file", return_value=True),
        patch("obolha.load_local_video", return_value=(fake, "T", [])),
        patch("obolha.get_transcript", return_value=[{"start": 0, "end": 10, "text": "a"}] * 6),
        patch("obolha.analyze_with_llm", return_value=[{"start": 0, "end": 5, "titulo": "A", "score_final": 9}]),
        patch("obolha.cut_aura_clips", return_value=[]),
        patch("obolha.check_filter_deps"),
    ):
        process_aura_video(str(fake), job)

    assert job.status == "erro"
    assert "Nenhum clip aura" in job.error

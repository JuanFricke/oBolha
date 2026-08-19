import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obolha import (
    clear_yt_dlp_cookie_jar,
    fetch_latest_channel_short,
    fetch_top_channel_shorts,
    is_short_processed,
    list_channel_shorts,
    mark_short_processed,
    normalize_channel_shorts_url,
    short_source_path,
    yt_dlp_cookie_args,
    yt_dlp_video_format,
)


@pytest.fixture(autouse=True)
def reset_cookie_jar():
    clear_yt_dlp_cookie_jar()
    yield
    clear_yt_dlp_cookie_jar()


def test_normalize_channel_shorts_url_handle():
    assert normalize_channel_shorts_url("@MrBeast").endswith("/@MrBeast/shorts")
    assert normalize_channel_shorts_url("https://www.youtube.com/@MrBeast").endswith("/shorts")


def test_list_channel_shorts_sorts_by_views():
    payload = {
        "title": "MrBeast",
        "entries": [
            {"id": "a", "title": "Low", "view_count": 1000, "duration": 30},
            {"id": "b", "title": "High", "view_count": 5000000, "duration": 45},
            None,
        ],
    }
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = json.dumps(payload)
    mock_run.return_value.stderr = ""

    with patch("subprocess.run", mock_run):
        channel, shorts = list_channel_shorts("https://www.youtube.com/@MrBeast/shorts", scan_limit=50)

    assert channel == "MrBeast"
    assert len(shorts) == 2
    assert shorts[0]["title"] == "High"
    assert shorts[0]["view_count"] == 5000000
    assert shorts[0]["url"] == "https://youtu.be/b"


def test_fetch_top_channel_shorts_downloads_top_n(tmp_path):
    payload = {
        "title": "TestChannel",
        "entries": [
            {"id": "vid1", "title": "S1", "view_count": 100, "duration": 30},
            {"id": "vid2", "title": "S2", "view_count": 200, "duration": 30},
        ],
    }
    list_proc = MagicMock()
    list_proc.returncode = 0
    list_proc.stdout = json.dumps(payload)
    list_proc.stderr = ""

    dl_proc = MagicMock()
    dl_proc.returncode = 0
    dl_proc.stderr = ""

    def fake_run(cmd, **kwargs):
        if "-J" in cmd:
            return list_proc
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_bytes(b"fake" * 300)
        return dl_proc

    with patch("subprocess.run", fake_run), patch("obolha.get_media_duration", return_value=30.0):
        results = fetch_top_channel_shorts(
            "@TestChannel",
            top=1,
            output_dir=tmp_path / "out",
        )

    assert len(results) == 1
    assert results[0]["view_count"] == 200
    assert Path(results[0]["file"]).exists()


def test_yt_dlp_video_format_prefers_avc():
    fmt = yt_dlp_video_format()
    assert "vcodec^=avc" in fmt
    assert "bv*" in fmt


def test_download_shorts_uses_video_id_path(tmp_path):
    from obolha import download_shorts

    entry = {
        "id": "xyz99",
        "title": "My Title",
        "view_count": 42,
        "url": "https://youtu.be/xyz99",
    }
    expected = short_source_path("xyz99", tmp_path)
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        mock = MagicMock()
        mock.returncode = 0
        mock.stderr = ""
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_bytes(b"x" * 2048)
        return mock

    with patch("subprocess.run", fake_run), patch("obolha.get_media_duration", return_value=30.0):
        results = download_shorts([entry], tmp_path)

    assert Path(results[0]["file"]) == expected
    assert captured[0][captured[0].index("-f") + 1] == yt_dlp_video_format()


def test_download_shorts_cmd_uses_force_overwrites(tmp_path):
    from obolha import download_shorts

    entry = {
        "id": "abc",
        "title": "Test",
        "view_count": 1000,
        "url": "https://youtu.be/abc",
    }
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        mock = MagicMock()
        mock.returncode = 0
        mock.stderr = ""
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_bytes(b"x")
        return mock

    with patch("subprocess.run", fake_run):
        download_shorts([entry], tmp_path)

    assert captured
    assert "--force-overwrites" in captured[0]
    assert "-y" not in captured[0]


def test_list_channel_shorts_sort_latest_keeps_playlist_order():
    payload = {
        "title": "Renan",
        "entries": [
            {
                "id": "new",
                "title": "Novo",
                "view_count": 10,
                "duration": 20,
                "timestamp": 1700000000,
                "upload_date": "20260117",
            },
            {
                "id": "old",
                "title": "Velho",
                "view_count": 999999,
                "duration": 30,
                "timestamp": 1600000000,
                "upload_date": "20200101",
            },
        ],
    }
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = json.dumps(payload)
    mock_run.return_value.stderr = ""

    with patch("subprocess.run", mock_run):
        channel, shorts = list_channel_shorts("@renansantosmbl", scan_limit=5, sort="latest")

    assert channel == "Renan"
    assert [s["id"] for s in shorts] == ["new", "old"]
    assert shorts[0]["timestamp"] == 1700000000
    assert shorts[0]["upload_date"] == "20260117"


def test_fetch_latest_channel_short_downloads_first_playlist_item(tmp_path):
    payload = {
        "title": "Renan",
        "entries": [
            {"id": "abc", "title": "Latest", "view_count": 1, "duration": 15},
            {"id": "zzz", "title": "Older viral", "view_count": 9_000_000, "duration": 15},
        ],
    }
    list_proc = MagicMock()
    list_proc.returncode = 0
    list_proc.stdout = json.dumps(payload)
    list_proc.stderr = ""

    dl_proc = MagicMock()
    dl_proc.returncode = 0
    dl_proc.stderr = ""

    def fake_run(cmd, **kwargs):
        if "-J" in cmd:
            return list_proc
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_bytes(b"fake" * 300)
        return dl_proc

    with patch("subprocess.run", fake_run), patch("obolha.get_media_duration", return_value=30.0):
        result = fetch_latest_channel_short("@renansantosmbl", output_dir=tmp_path / "out")

    assert result["id"] == "abc"
    assert Path(result["file"]).exists()


def test_short_processed_dedup(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    assert is_short_processed("abc") is False
    mark_short_processed("abc")
    assert is_short_processed("abc") is True
    assert is_short_processed("other") is False


def test_yt_dlp_cookie_args_empty_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(tmp_path / "nope.txt"))
    assert yt_dlp_cookie_args() == []


def test_yt_dlp_cookie_args_when_file_exists(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nSID\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))
    args = yt_dlp_cookie_args()
    assert args[0] == "--cookies"
    work = Path(args[1])
    assert work != cookies
    assert work.read_text() == cookies.read_text()
    assert os.access(work, os.W_OK)


def test_yt_dlp_cookie_args_copy_is_writable_when_source_is_readonly(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    cookies.chmod(0o444)
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))
    work = Path(yt_dlp_cookie_args()[1])
    assert os.access(work, os.W_OK)
    work.write_text("mutated")


def test_cookie_work_jar_persists_rotated_cookies(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nSID=old\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))

    work = Path(yt_dlp_cookie_args()[1])
    work.write_text("# rotated by yt-dlp\nSID=rotated\n")

    work_again = Path(yt_dlp_cookie_args()[1])
    assert work_again == work
    assert work_again.read_text() == "# rotated by yt-dlp\nSID=rotated\n"


def test_cookie_work_jar_reseeds_when_source_refreshed(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nSID=old\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))

    work = Path(yt_dlp_cookie_args()[1])
    work.write_text("# stale rotated\nSID=dead\n")

    # Fresh export with the SAME mtime (docker cp / scp -p / rsync -a preserve it):
    # re-seed must be driven by content, not mtime.
    old_mtime = cookies.stat().st_mtime
    cookies.write_text("# Netscape HTTP Cookie File\nSID=fresh\n")
    os.utime(cookies, (old_mtime, old_mtime))

    work_after = Path(yt_dlp_cookie_args()[1])
    assert work_after == work
    assert work_after.read_text() == "# Netscape HTTP Cookie File\nSID=fresh\n"


def test_cookie_work_jar_reseeds_when_corrupt(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nSID=old\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))

    work = Path(yt_dlp_cookie_args()[1])
    work.write_text("")  # truncated mid-save by a crash / ENOSPC

    work_after = Path(yt_dlp_cookie_args()[1])
    assert work_after == work
    assert work_after.read_text() == cookies.read_text()


def test_cookie_work_jar_falls_back_when_dir_readonly(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    cookies = data / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nSID=x\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))
    data.chmod(0o500)  # read-only dir: cannot create the .work jar
    try:
        args = yt_dlp_cookie_args()
        assert args[0] == "--cookies"
        jar = Path(args[1])
        assert jar.read_text() == cookies.read_text()
        assert os.access(jar, os.W_OK)
    finally:
        data.chmod(0o700)


def test_cookie_work_jar_does_not_follow_symlink(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nSID=real\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))

    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch\n")
    work = cookies.with_name(cookies.name + ".work")
    work.symlink_to(victim)

    jar = Path(yt_dlp_cookie_args()[1])
    assert jar == work
    assert not work.is_symlink()
    assert victim.read_text() == "do-not-touch\n"
    assert work.read_text() == cookies.read_text()


def test_list_channel_shorts_passes_cookies(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))
    monkeypatch.setenv("CLIPPER_YOUTUBE_USE_COOKIES", "1")
    payload = {"title": "Renan", "entries": [{"id": "a", "title": "x", "view_count": 1, "duration": 10}]}
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = json.dumps(payload)
    mock_run.return_value.stderr = ""
    with patch("subprocess.run", mock_run):
        list_channel_shorts("@renansantosmbl", scan_limit=1)
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "yt-dlp"
    assert cmd[1] == "--cookies"
    assert Path(cmd[2]).read_text() == cookies.read_text()
    assert "-J" in cmd
    assert "--flat-playlist" in cmd
    assert "--extractor-args" in cmd


def test_download_shorts_passes_cookies(tmp_path, monkeypatch):
    from obolha import download_shorts

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))
    monkeypatch.setenv("CLIPPER_YOUTUBE_USE_COOKIES", "1")
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        mock = MagicMock()
        mock.returncode = 0
        mock.stderr = ""
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_bytes(b"x")
        return mock

    with patch("subprocess.run", fake_run):
        download_shorts(
            [{"id": "abc", "title": "Test", "view_count": 1, "url": "https://youtu.be/abc"}],
            tmp_path,
        )
    assert captured
    assert captured[0][1] == "--cookies"
    assert Path(captured[0][2]).read_text() == cookies.read_text()
    assert "--extractor-args" in captured[0]
    args_i = captured[0].index("--extractor-args")
    assert "player_client=tv" in captured[0][args_i + 1]

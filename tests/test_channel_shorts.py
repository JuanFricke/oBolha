import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obolha import (
    fetch_top_channel_shorts,
    list_channel_shorts,
    normalize_channel_shorts_url,
)


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
                Path(cmd[i + 1]).write_bytes(b"fake")
        return dl_proc

    with patch("subprocess.run", fake_run):
        results = fetch_top_channel_shorts(
            "@TestChannel",
            top=1,
            output_dir=tmp_path / "out",
        )

    assert len(results) == 1
    assert results[0]["view_count"] == 200
    assert Path(results[0]["file"]).exists()


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

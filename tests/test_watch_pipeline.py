from pathlib import Path
from unittest.mock import patch

import pytest

from obolha import CFG, run_latest_short_react, run_watch_once


@pytest.fixture(autouse=True)
def reset_cfg():
    orig = dict(CFG)
    yield
    CFG.clear()
    CFG.update(orig)


def test_run_latest_short_react_downloads_and_composes(tmp_path):
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"
    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "face.mp4").write_bytes(b"cam")

    clip = tmp_path / "clips" / "shorts" / "Renan" / "01_Latest_1.mp4"
    react_out = tmp_path / "reacts" / "shorts" / "Renan" / "01_Latest_1_facecam.mp4"

    short = {
        "id": "abc",
        "title": "Latest",
        "view_count": 1,
        "url": "https://youtu.be/abc",
        "description": "fala sobre crime",
        "file": str(clip),
    }

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])) as mock_list,
        patch("obolha.fetch_latest_channel_short", return_value=short) as mock_fetch,
        patch("obolha.compose_facecam", return_value=react_out) as mock_compose,
        patch("obolha.is_short_processed", return_value=False),
    ):
        result = run_latest_short_react("@renansantosmbl")

    mock_list.assert_called_once()
    mock_fetch.assert_called_once()
    mock_compose.assert_called_once()
    assert mock_compose.call_args[0][0] == Path(clip)
    assert mock_compose.call_args[0][1] == pool / "face.mp4"
    assert result["id"] == "abc"
    assert result["file"] == str(react_out)
    assert result["clip"] == str(clip)
    assert result["title"] == "Latest"


def test_run_latest_short_react_skips_processed(tmp_path):
    CFG["reacts_source_dir"] = tmp_path / "pool"
    (CFG["reacts_source_dir"]).mkdir()
    (CFG["reacts_source_dir"] / "face.mp4").write_bytes(b"cam")

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [{"id": "abc", "title": "x"}])),
        patch("obolha.is_short_processed", return_value=True),
        patch("obolha.fetch_latest_channel_short") as mock_fetch,
        patch("obolha.compose_facecam") as mock_compose,
    ):
        result = run_latest_short_react("@renansantosmbl")

    mock_fetch.assert_not_called()
    mock_compose.assert_not_called()
    assert result["skipped"] is True
    assert result["id"] == "abc"


def test_run_latest_short_react_requires_react_pool(tmp_path):
    CFG["reacts_source_dir"] = tmp_path / "empty_pool"
    CFG["reacts_source_dir"].mkdir()
    with pytest.raises(FileNotFoundError, match="react"):
        run_latest_short_react("@renansantosmbl")


def test_watch_cli_retries_when_react_pool_empty(monkeypatch):
    monkeypatch.setattr("sys.argv", ["obolha", "--interval", "30"])
    with (
        patch("obolha.check_download_deps"),
        patch("obolha.run_watch_once", side_effect=FileNotFoundError("No react videos")),
        patch("obolha.time.sleep", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            from obolha import run_watch_cli
            run_watch_cli()


def _react_result(tmp_path: Path) -> dict:
    video = tmp_path / "react.mp4"
    video.write_bytes(b"vid")
    return {
        "id": "abc",
        "title": "Latest",
        "description": "fala sobre crime",
        "file": str(video),
        "skipped": False,
    }


def test_watch_once_posts_and_marks_processed(tmp_path, monkeypatch):
    monkeypatch.setenv("POSTIZ_BASE_URL", "https://postiz.example/api/public/v1")
    monkeypatch.setenv("POSTIZ_API_KEY", "k")
    copy = {"titulo": "T", "caption": "C", "hashtags": ["#brasil"]}
    with (
        patch("obolha.run_latest_short_react", return_value=_react_result(tmp_path)),
        patch("obolha.generate_outside_bubble_copy", return_value=copy),
        patch("obolha.PostizClient") as mock_cls,
        patch("obolha.mark_short_processed") as mock_mark,
    ):
        mock_cls.return_value.publish_video.return_value = {"posted": True}
        result = run_watch_once("@renansantosmbl")
    mock_mark.assert_called_once_with("abc")
    assert result["posted"] is True
    assert result["copy"] == copy


def test_watch_once_keeps_video_if_postiz_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("POSTIZ_BASE_URL", "https://postiz.example/api/public/v1")
    monkeypatch.setenv("POSTIZ_API_KEY", "k")
    copy = {"titulo": "T", "caption": "C", "hashtags": []}
    with (
        patch("obolha.run_latest_short_react", return_value=_react_result(tmp_path)),
        patch("obolha.generate_outside_bubble_copy", return_value=copy),
        patch("obolha.PostizClient") as mock_cls,
        patch("obolha.mark_short_processed") as mock_mark,
    ):
        mock_cls.return_value.publish_video.side_effect = RuntimeError("tiktok 400")
        result = run_watch_once("@renansantosmbl")
    mock_mark.assert_not_called()
    assert result["posted"] is False
    assert "tiktok 400" in result["postiz_error"]
    assert Path(result["file"]).exists()


def test_watch_once_skips_without_posting(tmp_path):
    with (
        patch("obolha.run_latest_short_react", return_value={"id": "abc", "skipped": True}),
        patch("obolha.PostizClient") as mock_cls,
        patch("obolha.mark_short_processed") as mock_mark,
    ):
        result = run_watch_once("@renansantosmbl")
    mock_cls.assert_not_called()
    mock_mark.assert_not_called()
    assert result["skipped"] is True

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from obolha import (
    is_short_processed,
    load_video_stages,
    mark_short_processed,
    update_video_stage,
    write_json_atomic,
)


def test_write_json_atomic_fsync_and_no_leftover_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    target = tmp_path / "nested" / "state.json"
    write_json_atomic(target, {"ok": True})
    data = json.loads(target.read_text())
    assert data == {"ok": True}
    assert not list(tmp_path.rglob("*.tmp"))


def test_mark_short_processed_preserves_legacy_list_format(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    (tmp_path / "processed_shorts.json").write_text('["old1", "old2"]\n')
    mark_short_processed("new3")
    ledger = json.loads((tmp_path / "processed_shorts.json").read_text())
    assert ledger == {"ids": ["old1", "old2", "new3"]}
    assert is_short_processed("old1") is True
    assert is_short_processed("new3") is True


def test_read_json_non_utf8_returns_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    path = tmp_path / "video_stages.json"
    path.write_bytes(b"\xff\xfe")
    from obolha import _read_json

    data = _read_json(path, lambda: {"videos": {}})
    assert data == {"videos": {}}


def test_json_rmw_invokes_file_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    with patch("fcntl.flock") as mock_flock:
        update_video_stage("locked", downloaded=True)
    assert mock_flock.call_count >= 2


def test_mark_short_processed_stage_before_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    calls: list[str] = []
    real_update = update_video_stage
    real_write = write_json_atomic

    def track_stage(vid, **kw):
        calls.append("stage")
        return real_update(vid, **kw)

    def track_write(path, data):
        calls.append(f"write:{Path(path).name}")
        return real_write(path, data)

    monkeypatch.setattr("obolha.update_video_stage", track_stage)
    monkeypatch.setattr("obolha.write_json_atomic", track_write)
    mark_short_processed("vid9")

    assert calls[0] == "stage"
    assert calls[1].startswith("write:video_stages.json")
    assert any(c == "write:processed_shorts.json" for c in calls)


def test_corrupt_legacy_still_dedupes_when_stage_posted(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    (tmp_path / "processed_shorts.json").write_text("{not json")
    update_video_stage("vid1", posted=True)
    assert is_short_processed("vid1") is True


def test_concurrent_stage_updates_preserve_both_videos(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    update_video_stage("a", downloaded=True)
    update_video_stage("b", downloaded=True)
    stages = load_video_stages()
    assert stages["videos"]["a"]["downloaded"] is True
    assert stages["videos"]["b"]["downloaded"] is True

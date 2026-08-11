"""Web API tests for YouTube scheduling."""

import os
from datetime import datetime, timedelta, timezone

import pytest


def test_youtube_status_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "data"
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(data_dir))

    from webui import create_app

    client = TestClient(create_app())
    res = client.get("/api/youtube/status")
    assert res.status_code == 200
    body = res.json()
    assert body["connected"] is False
    assert "schedule" in body


def test_youtube_schedule_requires_connection(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "data"
    reacts = tmp_path / "reacts"
    reacts.mkdir()
    clip = reacts / "01_test_facecam.mp4"
    clip.write_bytes(b"fake")

    monkeypatch.setenv("CLIPPER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CLIPPER_REACTS_DIR", str(reacts))

    from obolha import CFG
    CFG["reacts_dir"] = reacts

    from webui import create_app

    client = TestClient(create_app())
    future = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M")
    res = client.post(
        "/api/youtube/schedule",
        data={
            "react_id": "01_test_facecam.mp4",
            "title": "Test title",
            "description": "desc",
            "publish_at": future,
        },
    )
    assert res.status_code == 400
    assert "Conecte" in res.json()["detail"]


def test_index_has_youtube_tab():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from webui import create_app

    client = TestClient(create_app())
    html = client.get("/").text
    assert "panel-youtube" in html
    assert "Agendar no YouTube" in html

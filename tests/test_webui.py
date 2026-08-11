from pathlib import Path

import pytest

from webui import list_available_clips, resolve_under_root, job_to_dict


def test_list_available_clips(tmp_path):
    folder = tmp_path / "My_Video"
    folder.mkdir()
    clip = folder / "01_test_score8.0.mp4"
    clip.write_bytes(b"fake")
    manifest = {
        "clips": [
            {"file": str(clip), "titulo": "Test Clip", "score_final": 8.0},
        ],
    }
    (folder / "manifest.json").write_text(__import__("json").dumps(manifest))

    clips = list_available_clips(tmp_path)
    assert len(clips) == 1
    assert clips[0]["titulo"] == "Test Clip"
    assert clips[0]["name"] == "01_test_score8.0.mp4"


def test_resolve_under_root_blocks_traversal(tmp_path):
    root = tmp_path / "clips"
    root.mkdir()
    safe = root / "video.mp4"
    safe.touch()

    assert resolve_under_root(safe, root) == safe.resolve()
    assert resolve_under_root(root / "../evil.mp4", root) is None


def test_job_to_dict():
    from obolha import JobStatus

    job = JobStatus(url="https://youtu.be/x", title="Test", status="cortando")
    job.clips_found = 3
    job.clips_done = 1
    d = job_to_dict(job, "job-1")
    assert d["id"] == "job-1"
    assert d["status"] == "cortando"
    assert d["clips_found"] == 3


def test_web_app_routes():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from webui import create_app

    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert "oBolha" in client.get("/").text
    assert client.get("/api/jobs").status_code == 200
    assert client.get("/api/clips").status_code == 200

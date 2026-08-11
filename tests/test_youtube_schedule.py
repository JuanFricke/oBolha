"""Tests for YouTube schedule upload module."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from youtube_schedule import (
    ScheduleStore,
    build_auth_url,
    default_title_from_path,
    parse_publish_at,
    publish_at_to_youtube_rfc3339,
    validate_schedule_input,
)


def test_default_title_from_path_strips_facecam_suffix():
    p = Path("reacts/shorts/ch/01_clip_score8.0_facecam.mp4")
    title = default_title_from_path(p)
    assert "_facecam" not in title
    assert "score8.0" in title or "01_clip" in title


def test_parse_publish_at_local_naive():
    dt = parse_publish_at("2026-08-11T18:30")
    assert dt.tzinfo is not None


def test_publish_at_to_youtube_rfc3339_utc():
    dt = datetime(2026, 8, 11, 18, 30, tzinfo=timezone.utc)
    assert publish_at_to_youtube_rfc3339(dt) == "2026-08-11T18:30:00.000Z"


def test_validate_schedule_input_requires_future_time():
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    err = validate_schedule_input("Title", past)
    assert err is not None
    assert "futuro" in err.lower() or "future" in err.lower()


def test_validate_schedule_input_ok():
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    err = validate_schedule_input("My clip title", future)
    assert err is None


def test_schedule_store_add_and_list(tmp_path):
    store = ScheduleStore(tmp_path / "schedule.json")
    item = store.add(
        video_rel="shorts/ch/01_test_facecam.mp4",
        title="Test",
        description="desc",
        publish_at=datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert item["id"]
    assert item["status"] == "pending"
    items = store.list_items()
    assert len(items) == 1
    assert items[0]["title"] == "Test"


def test_schedule_store_update(tmp_path):
    store = ScheduleStore(tmp_path / "schedule.json")
    item = store.add(
        video_rel="a.mp4",
        title="T",
        description="",
        publish_at=datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc),
    )
    store.update(item["id"], status="scheduled", youtube_video_id="vid123")
    updated = store.get(item["id"])
    assert updated["status"] == "scheduled"
    assert updated["youtube_video_id"] == "vid123"


def test_build_auth_url_contains_client_id():
    url = build_auth_url(
        client_id="test-client-id",
        redirect_uri="http://127.0.0.1:8765/callback",
        state="abc",
    )
    assert "test-client-id" in url
    assert "youtube.upload" in url
    assert "youtube.readonly" in url
    assert "abc" in url


def test_upload_video_scheduled_mock(tmp_path):
    from youtube_schedule import upload_video_youtube

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video-bytes")

    future = datetime.now(timezone.utc) + timedelta(days=1)

    with patch("youtube_schedule.httpx.post") as mock_post, patch(
        "youtube_schedule.httpx.put"
    ) as mock_put:
        mock_post.return_value.headers = {"Location": "https://upload.example/put"}
        mock_post.return_value.status_code = 200
        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {"id": "yt-video-id"}

        vid_id = upload_video_youtube(
            access_token="token",
            video_path=video,
            title="Scheduled clip",
            description="hello",
            publish_at=future,
        )
        assert vid_id == "yt-video-id"

        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert body["status"]["privacyStatus"] == "private"
        assert "publishAt" in body["status"]


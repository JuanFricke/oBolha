from pathlib import Path
from unittest.mock import MagicMock, patch

from postiz import (
    PostizClient,
    build_now_payload,
    filter_target_integrations,
    settings_for,
)


def test_filter_target_integrations_skips_disabled_and_unknown():
    rows = [
        {"id": "yt1", "identifier": "youtube", "disabled": False},
        {"id": "tt1", "identifier": "tiktok", "disabled": True},
        {"id": "ig1", "identifier": "instagram", "disabled": False},
        {"id": "x1", "identifier": "x", "disabled": False},
        {"id": "fb1", "identifier": "facebook", "disabled": False},
    ]
    got = filter_target_integrations(rows, ["youtube", "tiktok", "instagram", "facebook"])
    assert [g["id"] for g in got] == ["yt1", "ig1", "fb1"]


def test_settings_for_platforms():
    copy = {"titulo": "Crime no Brasil", "caption": "hook", "hashtags": ["#brasil"]}
    assert settings_for("youtube", copy) == {
        "__type": "youtube",
        "title": "Crime no Brasil",
        "type": "public",
        "selfDeclaredMadeForKids": "no",
    }
    tiktok = settings_for("tiktok", copy)
    assert tiktok["__type"] == "tiktok"
    assert tiktok["content_posting_method"] == "DIRECT_POST"
    assert tiktok["privacy_level"] == "PUBLIC_TO_EVERYONE"
    assert settings_for("instagram", copy)["post_type"] == "post"
    assert settings_for("facebook", copy) == {"__type": "facebook"}


def test_build_now_payload_one_post_per_integration():
    copy = {"titulo": "Titulo", "caption": "Caption #brasil", "hashtags": ["#brasil"]}
    media = {"id": "m1", "path": "https://postiz.example/uploads/v.mp4"}
    integrations = [
        {"id": "yt1", "identifier": "youtube"},
        {"id": "tt1", "identifier": "tiktok"},
    ]
    payload = build_now_payload(integrations, media, copy)
    assert payload["type"] == "now"
    assert payload["date"].endswith("Z") or "+" in payload["date"]
    assert len(payload["posts"]) == 2
    assert payload["posts"][0]["value"][0]["content"] == "Caption #brasil"
    assert payload["posts"][0]["value"][0]["image"][0]["path"] == media["path"]
    assert payload["posts"][0]["settings"]["__type"] == "youtube"
    assert payload["posts"][1]["settings"]["__type"] == "tiktok"


def test_build_now_payload_supports_draft_type():
    copy = {"titulo": "T", "caption": "C"}
    media = {"id": "m1", "path": "https://x/v.mp4"}
    integrations = [{"id": "yt1", "identifier": "youtube"}]
    payload = build_now_payload(integrations, media, copy, post_type="draft")
    assert payload["type"] == "draft"


def test_upload_and_create_now_posts(tmp_path):
    video = tmp_path / "react.mp4"
    video.write_bytes(b"mp4")
    copy = {"titulo": "Titulo", "caption": "Caption", "hashtags": ["#brasil"]}
    client = PostizClient("https://postiz.example/api/public/v1", "key-1")

    upload_resp = MagicMock()
    upload_resp.raise_for_status = MagicMock()
    upload_resp.json.return_value = {"id": "m1", "path": "https://postiz.example/uploads/v.mp4"}

    list_resp = MagicMock()
    list_resp.raise_for_status = MagicMock()
    list_resp.json.return_value = [
        {"id": "yt1", "identifier": "youtube", "disabled": False},
        {"id": "tt1", "identifier": "tiktok", "disabled": False},
    ]

    create_resp = MagicMock()
    create_resp.is_error = False
    create_resp.raise_for_status = MagicMock()
    create_resp.json.return_value = {"ok": True}

    with patch("postiz.httpx.Client") as mock_cls:
        session = mock_cls.return_value.__enter__.return_value
        session.post.side_effect = [upload_resp, create_resp]
        session.get.return_value = list_resp
        result = client.publish_video(video, copy, platforms=["youtube", "tiktok"])

    assert result["media"]["id"] == "m1"
    assert result["posted"] is True
    posts_call = session.post.call_args_list[1]
    assert str(posts_call.args[0]).endswith("/posts")
    body = posts_call.kwargs["json"]
    assert body["type"] == "now"
    assert len(body["posts"]) == 2

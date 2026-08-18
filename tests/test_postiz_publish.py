from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from postiz import (
    PostizAmbiguousPublishError,
    PostizClient,
    PostizPublishError,
    classify_postiz_transport_error,
    parse_response_json,
)


def test_classify_connect_timeout_is_definite_retryable():
    err = httpx.ConnectTimeout("connect")
    assert classify_postiz_transport_error(err) is PostizPublishError


def test_classify_remote_protocol_error_is_ambiguous():
    err = httpx.RemoteProtocolError("broken")
    assert classify_postiz_transport_error(err) is PostizAmbiguousPublishError


def test_parse_response_json_empty_on_decode_failure():
    resp = MagicMock()
    resp.json.side_effect = ValueError("bad json")
    assert parse_response_json(resp) == {}


def test_publish_video_connect_timeout_is_retryable(tmp_path):
    video = tmp_path / "react.mp4"
    video.write_bytes(b"mp4" * 400)
    client = PostizClient("https://postiz.example/api/public/v1", "key")

    list_resp = MagicMock()
    list_resp.raise_for_status = MagicMock()
    list_resp.json.return_value = [{"id": "yt1", "identifier": "youtube", "disabled": False}]

    with patch("postiz.httpx.Client") as mock_cls:
        session = mock_cls.return_value.__enter__.return_value
        session.get.return_value = list_resp
        session.post.side_effect = httpx.ConnectTimeout("connect")
        with pytest.raises(PostizPublishError):
            client.publish_video(video, {"titulo": "T", "caption": "C"}, platforms=["youtube"])


def test_publish_video_counts_2xx_with_bad_json_as_success(tmp_path):
    video = tmp_path / "react.mp4"
    video.write_bytes(b"mp4" * 400)
    copy = {"titulo": "T", "caption": "C", "hashtags": []}
    client = PostizClient("https://postiz.example/api/public/v1", "key")

    upload_resp = MagicMock()
    upload_resp.raise_for_status = MagicMock()
    upload_resp.json.return_value = {"id": "m1", "path": "https://x/v.mp4"}

    list_resp = MagicMock()
    list_resp.raise_for_status = MagicMock()
    list_resp.json.return_value = [{"id": "yt1", "identifier": "youtube", "disabled": False}]

    create_resp = MagicMock()
    create_resp.is_error = False
    create_resp.status_code = 200
    create_resp.json.side_effect = ValueError("bad")

    with patch("postiz.httpx.Client") as mock_cls:
        session = mock_cls.return_value.__enter__.return_value
        session.post.side_effect = [upload_resp, create_resp]
        session.get.return_value = list_resp
        result = client.publish_video(video, copy, platforms=["youtube"])

    assert result["posted"] is True
    assert result["response"] == {}


def test_publish_video_definite_http_error_is_retryable(tmp_path):
    video = tmp_path / "react.mp4"
    video.write_bytes(b"mp4" * 400)
    client = PostizClient("https://postiz.example/api/public/v1", "key")

    upload_resp = MagicMock()
    upload_resp.raise_for_status = MagicMock()
    upload_resp.json.return_value = {"id": "m1", "path": "https://x/v.mp4"}

    list_resp = MagicMock()
    list_resp.raise_for_status = MagicMock()
    list_resp.json.return_value = [{"id": "yt1", "identifier": "youtube", "disabled": False}]

    create_resp = MagicMock()
    create_resp.is_error = True
    create_resp.status_code = 400
    create_resp.text = "bad request"

    with patch("postiz.httpx.Client") as mock_cls:
        session = mock_cls.return_value.__enter__.return_value
        session.post.side_effect = [upload_resp, create_resp]
        session.get.return_value = list_resp
        with pytest.raises(PostizPublishError, match="400"):
            client.publish_video(video, {"titulo": "T", "caption": "C"}, platforms=["youtube"])


def test_publish_video_read_timeout_after_create_is_ambiguous(tmp_path):
    video = tmp_path / "react.mp4"
    video.write_bytes(b"mp4" * 400)
    client = PostizClient("https://postiz.example/api/public/v1", "key")

    upload_resp = MagicMock()
    upload_resp.raise_for_status = MagicMock()
    upload_resp.json.return_value = {"id": "m1", "path": "https://x/v.mp4"}

    list_resp = MagicMock()
    list_resp.raise_for_status = MagicMock()
    list_resp.json.return_value = [{"id": "yt1", "identifier": "youtube", "disabled": False}]

    with patch("postiz.httpx.Client") as mock_cls:
        session = mock_cls.return_value.__enter__.return_value
        session.post.side_effect = [
            upload_resp,
            httpx.ReadTimeout("read timeout"),
        ]
        session.get.return_value = list_resp
        with pytest.raises(PostizAmbiguousPublishError):
            client.publish_video(video, {"titulo": "T", "caption": "C"}, platforms=["youtube"])

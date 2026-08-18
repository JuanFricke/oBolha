"""Postiz public API client — upload a video and publish now."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_PLATFORMS = ["youtube", "tiktok", "instagram", "facebook"]

_INSTAGRAM_IDS = {"instagram", "instagram-standalone"}


class PostizPublishError(RuntimeError):
    """Definite HTTP failure — safe to retry publish."""


class PostizAmbiguousPublishError(RuntimeError):
    """Transport/read timeout after create may have committed — do not auto-republish."""


_AMBIGUOUS_TRANSPORT = (
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)
_DEFINITE_TRANSPORT = (
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
)


def classify_postiz_transport_error(exc: Exception) -> type[Exception]:
    if isinstance(exc, _DEFINITE_TRANSPORT):
        return PostizPublishError
    if isinstance(exc, _AMBIGUOUS_TRANSPORT):
        return PostizAmbiguousPublishError
    return PostizPublishError


def _reraise_postiz_transport(exc: Exception, *, phase: str) -> None:
    kind = classify_postiz_transport_error(exc)
    if kind is PostizAmbiguousPublishError:
        raise PostizAmbiguousPublishError(f"Postiz {phase} ambiguous: {exc}") from exc
    raise PostizPublishError(f"Postiz {phase} failed: {exc}") from exc


def parse_response_json(resp: httpx.Response) -> dict:
    try:
        data = resp.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def filter_target_integrations(
    rows: list[dict],
    platforms: list[str],
) -> list[dict]:
    wanted: set[str] = set()
    for p in platforms:
        if p == "instagram":
            wanted.update(_INSTAGRAM_IDS)
        else:
            wanted.add(p)
    out = []
    for row in rows:
        ident = row.get("identifier")
        if ident not in wanted:
            continue
        if row.get("disabled"):
            continue
        out.append(row)
    return out


def settings_for(identifier: str, copy: dict) -> dict[str, Any]:
    title = str(copy.get("titulo") or "Short")[:90]
    if len(title) < 2:
        title = (title + "  ")[:2]
    if identifier == "youtube":
        return {
            "__type": "youtube",
            "title": title,
            "type": "public",
            "selfDeclaredMadeForKids": "no",
        }
    if identifier == "tiktok":
        return {
            "__type": "tiktok",
            "title": title,
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "duet": False,
            "stitch": False,
            "comment": True,
            "autoAddMusic": "no",
            "brand_content_toggle": False,
            "brand_organic_toggle": False,
            "video_made_with_ai": False,
            "content_posting_method": "DIRECT_POST",
        }
    if identifier in _INSTAGRAM_IDS:
        return {
            "__type": identifier,
            "post_type": "post",
            "is_trial_reel": False,
        }
    if identifier == "facebook":
        return {"__type": "facebook"}
    return {"__type": identifier}


def build_now_payload(
    integrations: list[dict],
    media: dict,
    copy: dict,
) -> dict:
    caption = str(copy.get("caption") or copy.get("titulo") or "")
    image = [{"id": media["id"], "path": media["path"]}]
    posts = []
    for integ in integrations:
        ident = integ["identifier"]
        posts.append({
            "integration": {"id": integ["id"]},
            "value": [{"content": caption, "image": image}],
            "settings": settings_for(ident, copy),
        })
    return {
        "type": "now",
        "date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "shortLink": False,
        "tags": [],
        "posts": posts,
    }


class PostizClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    def publish_video(
        self,
        video: Path,
        copy: dict,
        platforms: list[str] | None = None,
    ) -> dict:
        platforms = platforms or list(DEFAULT_PLATFORMS)
        video = Path(video)
        with httpx.Client(timeout=180.0) as client:
            try:
                list_resp = client.get(f"{self.base_url}/integrations", headers=self._headers())
            except Exception as e:
                _reraise_postiz_transport(e, phase="integrations list")
            list_resp.raise_for_status()
            rows = list_resp.json()
            if isinstance(rows, dict):
                rows = rows.get("integrations") or rows.get("data") or []
            targets = filter_target_integrations(rows, platforms)
            if not targets:
                raise PostizPublishError(
                    "Nenhuma integração Postiz alvo conectada "
                    f"(pedido: {', '.join(platforms)})"
                )

            with video.open("rb") as fh:
                try:
                    upload_resp = client.post(
                        f"{self.base_url}/upload",
                        headers=self._headers(),
                        files={"file": (video.name, fh, "video/mp4")},
                    )
                except Exception as e:
                    _reraise_postiz_transport(e, phase="upload")
            upload_resp.raise_for_status()
            media = upload_resp.json()

            payload = build_now_payload(targets, media, copy)
            try:
                create_resp = client.post(
                    f"{self.base_url}/posts",
                    headers=self._headers(),
                    json=payload,
                )
            except Exception as e:
                _reraise_postiz_transport(e, phase="create")

            if create_resp.is_error:
                raise PostizPublishError(
                    f"Postiz {create_resp.status_code}: {create_resp.text[:500]}"
                )
            return {
                "media": media,
                "posted": True,
                "response": parse_response_json(create_resp),
            }

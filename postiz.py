"""Postiz public API client — upload a video and publish now."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

DEFAULT_PLATFORMS = ["youtube", "tiktok", "instagram", "facebook"]

_INSTAGRAM_IDS = {"instagram", "instagram-standalone"}


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
            list_resp = client.get(f"{self.base_url}/integrations", headers=self._headers())
            list_resp.raise_for_status()
            rows = list_resp.json()
            if isinstance(rows, dict):
                rows = rows.get("integrations") or rows.get("data") or []
            targets = filter_target_integrations(rows, platforms)
            if not targets:
                raise RuntimeError(
                    "Nenhuma integração Postiz alvo conectada "
                    f"(pedido: {', '.join(platforms)})"
                )

            with video.open("rb") as fh:
                upload_resp = client.post(
                    f"{self.base_url}/upload",
                    headers=self._headers(),
                    files={"file": (video.name, fh, "video/mp4")},
                )
            upload_resp.raise_for_status()
            media = upload_resp.json()

            payload = build_now_payload(targets, media, copy)
            create_resp = client.post(
                f"{self.base_url}/posts",
                headers=self._headers(),
                json=payload,
            )
            create_resp.raise_for_status()
            return {
                "media": media,
                "posted": True,
                "response": create_resp.json(),
            }

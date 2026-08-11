"""
YouTube OAuth + scheduled uploads via YouTube Data API v3 (publishAt).
"""

import json
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

YOUTUBE_SCOPES = " ".join(
    [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.readonly",
    ]
)
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
YOUTUBE_UPLOAD_INIT = "https://www.googleapis.com/upload/youtube/v3/videos"

_store_lock = threading.Lock()
_oauth_states: dict[str, float] = {}


def _require_httpx():
    if not HAS_HTTPX:
        raise RuntimeError("httpx required for YouTube uploads: uv sync --extra web")


def get_data_dir() -> Path:
    root = Path(os.getenv("CLIPPER_DATA_DIR", "./data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_tokens_path() -> Path:
    return get_data_dir() / "youtube_tokens.json"


def get_schedule_path() -> Path:
    return get_data_dir() / "youtube_schedule.json"


def oauth_client_config() -> tuple[str, str]:
    client_id = os.getenv("YOUTUBE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Defina YOUTUBE_CLIENT_ID e YOUTUBE_CLIENT_SECRET no .env "
            "(Google Cloud Console → APIs & Services → Credentials)"
        )
    return client_id, client_secret


def default_redirect_uri(host: str = "127.0.0.1", port: int = 8765) -> str:
    env = os.getenv("YOUTUBE_REDIRECT_URI")
    if env:
        return env
    return f"http://{host}:{port}/api/youtube/callback"


def build_auth_url(client_id: str, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": YOUTUBE_SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def register_oauth_state(state: str, ttl_seconds: int = 600) -> None:
    _oauth_states[state] = time.time() + ttl_seconds


def verify_oauth_state(state: str) -> bool:
    expiry = _oauth_states.pop(state, None)
    if expiry is None:
        return False
    return time.time() <= expiry


def exchange_code_for_tokens(code: str, redirect_uri: str) -> dict[str, Any]:
    _require_httpx()
    client_id, client_secret = oauth_client_config()
    resp = httpx.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OAuth token exchange failed: {resp.text[:300]}")
    data = resp.json()
    data["obtained_at"] = datetime.now(timezone.utc).isoformat()
    return data


def save_tokens(token_data: dict[str, Any]) -> None:
    path = get_tokens_path()
    with _store_lock:
        path.write_text(json.dumps(token_data, indent=2))


def load_tokens() -> Optional[dict[str, Any]]:
    path = get_tokens_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear_tokens() -> None:
    path = get_tokens_path()
    if path.exists():
        path.unlink()


def refresh_access_token(token_data: dict[str, Any]) -> dict[str, Any]:
    _require_httpx()
    refresh = token_data.get("refresh_token")
    if not refresh:
        raise RuntimeError("Sem refresh_token — reconecte a conta YouTube")

    client_id, client_secret = oauth_client_config()
    resp = httpx.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {resp.text[:300]}")

    new_data = resp.json()
    token_data["access_token"] = new_data["access_token"]
    if "refresh_token" in new_data:
        token_data["refresh_token"] = new_data["refresh_token"]
    token_data["expires_in"] = new_data.get("expires_in", 3600)
    token_data["obtained_at"] = datetime.now(timezone.utc).isoformat()
    save_tokens(token_data)
    return token_data


def get_valid_access_token() -> Optional[str]:
    data = load_tokens()
    if not data or not data.get("access_token"):
        return None

    obtained = data.get("obtained_at")
    expires_in = int(data.get("expires_in", 3600))
    if obtained:
        try:
            t = datetime.fromisoformat(obtained)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) >= t + timedelta(seconds=expires_in - 60):
                data = refresh_access_token(data)
        except (ValueError, TypeError):
            pass

    return data.get("access_token")


def fetch_channel_title(access_token: str) -> str:
    _require_httpx()
    try:
        resp = httpx.get(
            f"{YOUTUBE_API}/channels",
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
    except Exception:
        return ""
    if resp.status_code != 200:
        return ""
    items = resp.json().get("items", [])
    if not items:
        return ""
    return items[0].get("snippet", {}).get("title", "")


def is_youtube_connected() -> bool:
    return get_valid_access_token() is not None


def youtube_status() -> dict[str, Any]:
    tokens = load_tokens()
    if not tokens:
        return {"connected": False, "channel_title": ""}
    try:
        token = get_valid_access_token()
        if not token:
            return {"connected": False, "channel_title": ""}
        title = tokens.get("channel_title") or fetch_channel_title(token)
        if title and tokens.get("channel_title") != title:
            tokens["channel_title"] = title
            save_tokens(tokens)
        return {"connected": True, "channel_title": title}
    except Exception:
        return {"connected": False, "channel_title": "", "error": "token inválido"}


def default_title_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_facecam"):
        stem = stem[:-8]
    stem = stem.replace("_", " ").strip()
    return stem[:100] if stem else path.name[:100]


def parse_publish_at(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("Horário de publicação obrigatório")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def publish_at_to_youtube_rfc3339(dt: datetime) -> str:
    utc = dt.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def validate_schedule_input(title: str, publish_at: datetime) -> Optional[str]:
    title = (title or "").strip()
    if not title:
        return "Título obrigatório"
    if len(title) > 100:
        return "Título máximo 100 caracteres (YouTube)"
    min_time = datetime.now(timezone.utc) + timedelta(minutes=2)
    pub_utc = publish_at.astimezone(timezone.utc)
    if pub_utc < min_time:
        return "publish_at deve ser pelo menos 2 minutos no futuro"
    return None


class ScheduleStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or get_schedule_path()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"items": []}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"items": []}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    def list_items(self) -> list[dict[str, Any]]:
        with _store_lock:
            items = self._read().get("items", [])
            return sorted(items, key=lambda x: x.get("publish_at", ""), reverse=True)

    def get(self, item_id: str) -> Optional[dict[str, Any]]:
        for item in self.list_items():
            if item["id"] == item_id:
                return item
        return None

    def add(
        self,
        video_rel: str,
        title: str,
        description: str,
        publish_at: datetime,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        item = {
            "id": secrets.token_hex(8),
            "video_rel": video_rel,
            "title": title.strip()[:100],
            "description": (description or "").strip()[:5000],
            "publish_at": publish_at.isoformat(),
            "status": "pending",
            "youtube_video_id": None,
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        with _store_lock:
            data = self._read()
            data.setdefault("items", []).append(item)
            self._write(data)
        return item

    def update(self, item_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        with _store_lock:
            data = self._read()
            items = data.get("items", [])
            for item in items:
                if item["id"] == item_id:
                    for k, v in fields.items():
                        item[k] = v
                    item["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._write(data)
                    return item
        return None

    def delete(self, item_id: str) -> bool:
        with _store_lock:
            data = self._read()
            items = data.get("items", [])
            new_items = [i for i in items if i["id"] != item_id]
            if len(new_items) == len(items):
                return False
            data["items"] = new_items
            self._write(data)
            return True


def upload_video_youtube(
    access_token: str,
    video_path: Path,
    title: str,
    description: str,
    publish_at: datetime,
) -> str:
    _require_httpx()
    if not video_path.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    file_size = video_path.stat().st_size
    metadata = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publish_at_to_youtube_rfc3339(publish_at),
            "selfDeclaredMadeForKids": False,
        },
    }

    init_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size),
    }

    init_resp = httpx.post(
        YOUTUBE_UPLOAD_INIT,
        params={"uploadType": "resumable", "part": "snippet,status"},
        json=metadata,
        headers=init_headers,
        timeout=60,
    )
    if init_resp.status_code not in (200, 201):
        raise RuntimeError(f"YouTube upload init failed: {init_resp.text[:400]}")

    upload_url = init_resp.headers.get("Location")
    if not upload_url:
        raise RuntimeError("YouTube não retornou URL de upload")

    with video_path.open("rb") as f:
        video_bytes = f.read()

    put_resp = httpx.put(
        upload_url,
        content=video_bytes,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "video/mp4",
        },
        timeout=600,
    )
    if put_resp.status_code not in (200, 201):
        raise RuntimeError(f"YouTube upload failed: {put_resp.text[:400]}")

    body = put_resp.json()
    video_id = body.get("id")
    if not video_id:
        raise RuntimeError("YouTube upload sem video id na resposta")
    return video_id


def process_schedule_item(
    item: dict[str, Any],
    video_path: Path,
    store: Optional[ScheduleStore] = None,
) -> dict[str, Any]:
    store = store or ScheduleStore()
    item_id = item["id"]
    store.update(item_id, status="uploading", error="")

    token = get_valid_access_token()
    if not token:
        store.update(item_id, status="failed", error="YouTube não conectado")
        return store.get(item_id) or item

    publish_at = datetime.fromisoformat(item["publish_at"])
    if publish_at.tzinfo is None:
        publish_at = publish_at.replace(tzinfo=timezone.utc)

    try:
        video_id = upload_video_youtube(
            access_token=token,
            video_path=video_path,
            title=item["title"],
            description=item.get("description", ""),
            publish_at=publish_at,
        )
        return store.update(
            item_id,
            status="scheduled",
            youtube_video_id=video_id,
            error="",
        ) or item
    except Exception as e:
        store.update(item_id, status="failed", error=str(e))
        return store.get(item_id) or item


def run_pending_uploads(
    reacts_dir: Path,
    resolve_video: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Upload all pending schedule items (e.g. after server restart)."""
    store = ScheduleStore()
    results = []
    for item in store.list_items():
        if item.get("status") != "pending":
            continue
        rel = item.get("video_rel", "")
        if resolve_video:
            video_path = resolve_video(rel)
        else:
            video_path = reacts_dir / rel
        if not video_path or not Path(video_path).is_file():
            store.update(item["id"], status="failed", error="Arquivo de vídeo não encontrado")
            continue
        results.append(process_schedule_item(item, Path(video_path), store))
    return results

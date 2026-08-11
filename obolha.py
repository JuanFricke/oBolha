#!/usr/bin/env python3
"""
obolha — AI-powered video clip extractor
Parallel, multi-input, terminal-based.

CLI usage:
  python obolha.py <url_or_file> [url2 url3 ...]
  python obolha.py --file urls.txt
  python obolha.py --interactive
  python obolha.py facecam <clip.mp4> <facecam.mp4> [-o output.mp4]
  python obolha.py shorts @Canal --top 10
  python obolha.py auto-react [--force]
  python obolha.py web [--host 127.0.0.1] [--port 8765]

Python API (for AI agents / scripts):
  from obolha import clip_videos, compose_facecam
  results = clip_videos(["https://youtu.be/XYZ"])
  compose_facecam("clips/01_clip.mp4", "webcam.mp4")

Dependencies (managed via uv / pyproject.toml):
  uv sync   # installs everything
  # or: pip install groq faster-whisper rich python-dotenv yt-dlp

Environment variables (loaded from .env automatically):
  GROQ_API_KEY          — Groq API key (required, free at console.groq.com)
  CLIPPER_MODEL         — Groq model (default: llama-3.3-70b-versatile)
  CLIPPER_LANG          — transcription language (default: pt)
  CLIPPER_MAX_CLIPS     — max clips per video (default: 20)
  CLIPPER_MIN_DURATION  — min clip duration in seconds (default: 20)
  CLIPPER_MAX_DURATION  — max clip duration in seconds (default: 60)
  CLIPPER_OUTPUT_DIR    — output folder (default: ./clips)
  CLIPPER_WORKERS       — parallel workers (default: 3)
  CLIPPER_WHISPER_MODEL — whisper model (default: base)
"""

import os

# Load .env before reading any env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if not installed
import re
import sys
import json
import time
import shutil
import random
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from queue import Queue

# ──────────────────────────────────────────────────────────────────────────────
# Imports opcionais — avisa claramente se faltarem
# ──────────────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[WARN] 'rich' não instalado. Instale com: pip install rich")

try:
    import groq as groq_lib
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

try:
    import faster_whisper  # noqa: F401
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import anthropic as anthropic_lib
    HAS_ANTHROPIC = True
except ImportError:
    anthropic_lib = None
    HAS_ANTHROPIC = False

# Cheapest Anthropic model — always used when provider is anthropic
ANTHROPIC_HAIKU_MODEL = "claude-haiku-4-5"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}

# ──────────────────────────────────────────────────────────────────────────────
# Config via env
# ──────────────────────────────────────────────────────────────────────────────
_legacy_output = os.getenv("CLIPPER_OUTPUT_DIR")
CFG = {
    "provider":            os.getenv("CLIPPER_PROVIDER", "anthropic").lower(),
    "active_provider":     "anthropic",
    "groq_api_key":        os.getenv("GROQ_API_KEY", ""),
    "gemini_api_key":      os.getenv("GEMINI_API_KEY", "") or os.getenv("ANTIGRAVITY_API_KEY", ""),
    "antigravity_api_key": os.getenv("ANTIGRAVITY_API_KEY", ""),
    "anthropic_api_key":   (
        os.getenv("ANTHROPIC_API_KEY", "")
        or os.getenv("ANTROPIC_KEY", "")
        or os.getenv("antropic-key", "")
    ),
    "model":               os.getenv("CLIPPER_MODEL", ""),
    "lang":                os.getenv("CLIPPER_LANG", "pt"),
    "max_clips":           int(os.getenv("CLIPPER_MAX_CLIPS", "20")),
    "min_duration":        int(os.getenv("CLIPPER_MIN_DURATION", "20")),
    "max_duration":        int(os.getenv("CLIPPER_MAX_DURATION", "60")),
    "clips_dir":           Path(os.getenv("CLIPPER_CLIPS_DIR", _legacy_output or "./clips")),
    "reacts_dir":          Path(os.getenv("CLIPPER_REACTS_DIR", "./reacts")),
    "reacts_source_dir":   Path(os.getenv("CLIPPER_REACTS_SOURCE_DIR", "./reacts_pool")),
    "output_dir":          Path(os.getenv("CLIPPER_CLIPS_DIR", _legacy_output or "./clips")),
    "workers":             int(os.getenv("CLIPPER_WORKERS", "3")),
    "whisper_model":       os.getenv("CLIPPER_WHISPER_MODEL", "base"),
}


def get_clips_dir() -> Path:
    return Path(CFG["clips_dir"])


def get_reacts_dir() -> Path:
    return Path(CFG["reacts_dir"])


def get_reacts_source_dir() -> Path:
    return Path(CFG["reacts_source_dir"])


def _sync_output_dir_alias():
    CFG["output_dir"] = Path(CFG["clips_dir"])


def react_output_path_for_clip(clip_path: Path) -> Path:
    """Mirror clip path under reacts_dir with _facecam suffix."""
    clip = clip_path.resolve()
    clips_root = get_clips_dir().resolve()
    reacts_root = get_reacts_dir()
    try:
        rel = clip.relative_to(clips_root)
        return reacts_root / rel.parent / f"{rel.stem}_facecam.mp4"
    except ValueError:
        return reacts_root / f"{clip.stem}_facecam.mp4"


def list_video_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return files


def list_clip_files(clips_dir: Path | None = None) -> list[Path]:
    """All clip mp4s under clips_dir (excludes uploads and facecam outputs)."""
    root = clips_dir or get_clips_dir()
    clips = []
    for path in list_video_files(root):
        if "_uploads" in path.parts:
            continue
        if path.name.endswith(".raw.mp4") or "_facecam" in path.stem:
            continue
        clips.append(path)
    return clips


def list_react_sources(source_dir: Path | None = None) -> list[Path]:
    return list_video_files(source_dir or get_reacts_source_dir())

console = Console() if HAS_RICH else None

# ──────────────────────────────────────────────────────────────────────────────
# Estado de cada job
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class JobStatus:
    url: str
    title: str = "..."
    status: str = "aguardando"   # aguardando / baixando / transcrevendo / analisando / cortando / concluído / erro
    progress: int = 0
    clips_found: int = 0
    clips_done: int = 0
    error: str = ""
    start_time: float = field(default_factory=time.time)
    output_dir: Optional[Path] = None

    @property
    def elapsed(self):
        return int(time.time() - self.start_time)

    @property
    def status_color(self):
        colors = {
            "aguardando":   "dim",
            "baixando":     "cyan",
            "transcrevendo":"blue",
            "analisando":   "magenta",
            "cortando":     "yellow",
            "concluído":    "green",
            "erro":         "red",
        }
        return colors.get(self.status, "white")

# ──────────────────────────────────────────────────────────────────────────────
# Registry de jobs (thread-safe)
# ──────────────────────────────────────────────────────────────────────────────
_jobs_lock = threading.Lock()
_jobs: list[JobStatus] = []
_log_queue: Queue = Queue()

def log(msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"info": "●", "ok": "✓", "warn": "!", "err": "✗"}.get(level, "●")
    _log_queue.put(f"[{ts}] {prefix} {msg}")

def add_job(url: str) -> JobStatus:
    job = JobStatus(url=url)
    with _jobs_lock:
        _jobs.append(job)
    return job

# ──────────────────────────────────────────────────────────────────────────────
# Display (Rich live table)
# ──────────────────────────────────────────────────────────────────────────────
def build_status_table() -> Table:
    t = Table(show_header=True, header_style="bold white", box=None, padding=(0, 1))
    t.add_column("#",        width=3,  style="dim")
    t.add_column("Título",   width=32)
    t.add_column("Status",   width=14)
    t.add_column("Clips",    width=10, justify="center")
    t.add_column("Tempo",    width=8,  justify="right", style="dim")

    with _jobs_lock:
        for i, job in enumerate(_jobs, 1):
            status_text = Text(job.status, style=job.status_color)
            clips_text  = f"{job.clips_done}/{job.clips_found}" if job.clips_found else "-"
            t.add_row(
                str(i),
                job.title[:32],
                status_text,
                clips_text,
                f"{job.elapsed}s",
            )
    return t

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def ts_to_seconds(ts: str) -> float:
    """'HH:MM:SS' ou 'MM:SS' ou segundos float → float"""
    ts = str(ts).strip()
    if re.match(r"^\d+(\.\d+)?$", ts):
        return float(ts)
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    if len(parts) == 2:
        return parts[0]*60 + parts[1]
    return float(parts[0])

def seconds_to_ts(s: float) -> str:
    s = int(s)
    h, m = divmod(s, 3600)
    m, sec = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-_\. ]", "_", s)[:60].strip()

class MissingDependencyError(RuntimeError):
    """Raised when a required system tool or Python package is missing."""

class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not set."""

def check_deps(raise_on_error: bool = False):
    """
    Verify all required dependencies are available.

    Args:
        raise_on_error: If True, raises MissingDependencyError / MissingAPIKeyError
                        instead of calling sys.exit (use this from Python/agent code).
    """
    missing = []
    if not shutil.which("yt-dlp"):
        missing.append("yt-dlp  → pip install yt-dlp  (ou: sudo apt install yt-dlp)")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg  → sudo apt install ffmpeg")
    target_provider = CFG.get("provider", "auto").lower()

    if target_provider in ("gemini", "antigravity"):
        if not HAS_GENAI:
            missing.append("google-genai → pip install google-genai")
    elif target_provider == "anthropic":
        if not HAS_ANTHROPIC:
            missing.append("anthropic → pip install anthropic")
    elif target_provider == "groq":
        if not HAS_GROQ:
            missing.append("groq        → pip install groq")
    else:  # auto
        if not HAS_GROQ and not HAS_GENAI and not HAS_ANTHROPIC:
            missing.append(
                "groq, anthropic, or google-genai → pip install groq / anthropic / google-genai"
            )

    if not HAS_RICH:
        missing.append("rich        → pip install rich")
    if not HAS_WHISPER:
        missing.append("faster-whisper → pip install faster-whisper")
    if missing:
        msg = "Missing dependencies:\n" + "\n".join(f"  • {m}" for m in missing)
        if raise_on_error:
            raise MissingDependencyError(msg)
        print(f"\n[ERRO] {msg}\n")
        sys.exit(1)

    # Resolve active provider & validate API keys
    groq_key = CFG.get("groq_api_key", "")
    gemini_key = CFG.get("gemini_api_key", "") or CFG.get("antigravity_api_key", "")
    anthropic_key = CFG.get("anthropic_api_key", "")

    if target_provider == "anthropic":
        if not anthropic_key:
            msg = (
                "ANTHROPIC_API_KEY not set. Set ANTHROPIC_API_KEY=... "
                "(or ANTROPIC_KEY / antropic-key in .env)"
            )
            if raise_on_error:
                raise MissingAPIKeyError(msg)
            print(f"\n[ERRO] {msg}\n  export ANTHROPIC_API_KEY=sk-ant-...\n")
            sys.exit(1)
        CFG["active_provider"] = "anthropic"
        CFG["model"] = ANTHROPIC_HAIKU_MODEL

    elif target_provider in ("gemini", "antigravity"):
        if not gemini_key:
            msg = "GEMINI_API_KEY / ANTIGRAVITY_API_KEY not set. Set GEMINI_API_KEY=... or ANTIGRAVITY_API_KEY=..."
            if raise_on_error:
                raise MissingAPIKeyError(msg)
            print(f"\n[ERRO] {msg}\n  export GEMINI_API_KEY=...\n")
            sys.exit(1)
        CFG["active_provider"] = "gemini"
        if not CFG["model"]:
            CFG["model"] = "gemini-2.5-flash"

    elif target_provider == "groq":
        if not groq_key:
            msg = "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            if raise_on_error:
                raise MissingAPIKeyError(msg)
            print(f"\n[ERRO] {msg}\n  export GROQ_API_KEY=gsk_...\n")
            sys.exit(1)
        CFG["active_provider"] = "groq"
        if not CFG["model"]:
            CFG["model"] = "llama-3.3-70b-versatile"

    else:  # auto
        if anthropic_key and HAS_ANTHROPIC:
            CFG["active_provider"] = "anthropic"
            CFG["model"] = ANTHROPIC_HAIKU_MODEL
        elif groq_key and HAS_GROQ:
            CFG["active_provider"] = "groq"
            if not CFG["model"]:
                CFG["model"] = "llama-3.3-70b-versatile"
        elif gemini_key and HAS_GENAI:
            CFG["active_provider"] = "gemini"
            if not CFG["model"]:
                CFG["model"] = "gemini-2.5-flash"
        elif HAS_GENAI and not HAS_GROQ:
            if not gemini_key:
                msg = "GEMINI_API_KEY / ANTIGRAVITY_API_KEY not set."
                if raise_on_error:
                    raise MissingAPIKeyError(msg)
                print(f"\n[ERRO] {msg}\n  export GEMINI_API_KEY=...\n")
                sys.exit(1)
            CFG["active_provider"] = "gemini"
            if not CFG["model"]:
                CFG["model"] = "gemini-2.5-flash"
        else:
            msg = (
                "No LLM API key set. Set ANTHROPIC_API_KEY, GROQ_API_KEY, "
                "or GEMINI_API_KEY / ANTIGRAVITY_API_KEY."
            )
            if raise_on_error:
                raise MissingAPIKeyError(msg)
            print(f"\n[ERRO] {msg}\n")
            sys.exit(1)


def check_download_deps(raise_on_error: bool = False):
    """Verify yt-dlp + ffmpeg (no LLM keys required)."""
    missing = []
    if not shutil.which("yt-dlp"):
        missing.append("yt-dlp  → pip install yt-dlp  (ou: sudo apt install yt-dlp)")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg  → sudo apt install ffmpeg")
    if missing:
        msg = "Missing dependencies:\n" + "\n".join(f"  • {m}" for m in missing)
        if raise_on_error:
            raise MissingDependencyError(msg)
        print(f"\n[ERRO] {msg}\n")
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Etapa 1: Download (skipped for local files)
# ──────────────────────────────────────────────────────────────────────────────
def is_local_file(source: str) -> bool:
    p = Path(source)
    return p.exists() and p.is_file()


def load_local_video(source: str, job: JobStatus) -> tuple[Path, str, list]:
    """Use a local video file directly — no download needed."""
    video_path = Path(source).resolve()
    title = video_path.stem
    job.status = "transcrevendo"
    job.title = title[:50]
    log(f"Local file: {video_path.name}")
    return video_path, title, []  # no subtitle files
def download_video(url: str, work_dir: Path, job: JobStatus) -> tuple[Path, str]:
    """Baixa vídeo e retorna (caminho_video, titulo)"""
    job.status = "baixando"
    log(f"Baixando: {url}")

    # Pega título primeiro
    title_result = subprocess.run(
        ["yt-dlp", "--get-title", "--no-playlist", url],
        capture_output=True, text=True, timeout=30
    )
    title = title_result.stdout.strip() or url
    job.title = title[:50]
    log(f"Título: {title}")

    # Tenta baixar legendas existentes primeiro (mais rápido)
    subs_dir = work_dir / "subs"
    subs_dir.mkdir(exist_ok=True)

    subprocess.run([
        "yt-dlp",
        "--write-auto-sub", "--write-sub",
        "--sub-lang", f"{CFG['lang']},pt,pt-BR,en",
        "--sub-format", "vtt/srt/best",
        "--skip-download",
        "--no-playlist",
        "-o", str(subs_dir / "%(id)s.%(ext)s"),
        url
    ], capture_output=True, text=True, timeout=120)

    # Checa se baixou legenda
    sub_files = list(subs_dir.glob("*.vtt")) + list(subs_dir.glob("*.srt"))

    # Baixa vídeo (formato leve: menor resolução aceitável)
    video_path = work_dir / "video.mp4"
    dl_result = subprocess.run([
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(video_path),
        url
    ], capture_output=True, text=True, timeout=600)

    if dl_result.returncode != 0:
        raise RuntimeError(f"yt-dlp falhou: {dl_result.stderr[-300:]}")

    # Verifica arquivo baixado (nome pode diferir)
    candidates = list(work_dir.glob("video*.mp4"))
    if not candidates:
        candidates = list(work_dir.glob("*.mp4"))
    if not candidates:
        raise RuntimeError("Nenhum arquivo MP4 encontrado após download")

    video_path = candidates[0]
    return video_path, title, sub_files

# ──────────────────────────────────────────────────────────────────────────────
# Canal YouTube — top shorts por views
# ──────────────────────────────────────────────────────────────────────────────
def normalize_channel_shorts_url(channel: str) -> str:
    """Turn @handle, URL, or handle into the channel /shorts tab URL."""
    channel = channel.strip()
    if not channel.startswith("http"):
        handle = channel if channel.startswith("@") else f"@{channel}"
        channel = f"https://www.youtube.com/{handle}/shorts"
    else:
        channel = channel.rstrip("/")
        if "/shorts" not in channel:
            channel = channel + "/shorts"
    return channel


def channel_slug_from_url(url: str) -> str:
    match = re.search(r"youtube\.com/@([^/?]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"youtube\.com/channel/([^/?]+)", url)
    if match:
        return match.group(1)
    return safe_filename(url)


def list_channel_shorts(channel_url: str, scan_limit: int = 100) -> tuple[str, list[dict]]:
    """
    List shorts from a channel's /shorts tab, sorted by view_count (desc).
    Scans up to scan_limit entries via yt-dlp metadata (no download).
    """
    url = normalize_channel_shorts_url(channel_url)
    log(f"Listando shorts: {url} (scan={scan_limit})")

    result = subprocess.run(
        [
            "yt-dlp", "-J",
            "--playlist-end", str(scan_limit),
            url,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp shorts list failed: {result.stderr[-400:]}")

    data = json.loads(result.stdout)
    channel_title = data.get("title") or data.get("channel") or channel_slug_from_url(url)
    entries = data.get("entries") or []

    shorts: list[dict] = []
    for entry in entries:
        if not entry:
            continue
        vid = entry.get("id")
        if not vid:
            continue
        shorts.append({
            "id": vid,
            "title": entry.get("title") or vid,
            "view_count": int(entry.get("view_count") or 0),
            "duration": float(entry.get("duration") or 0),
            "url": entry.get("webpage_url") or f"https://youtu.be/{vid}",
        })

    shorts.sort(key=lambda s: s["view_count"], reverse=True)
    log(f"Shorts encontrados: {len(shorts)} em {channel_title}")
    return channel_title, shorts


def download_shorts(entries: list[dict], output_dir: Path) -> list[dict]:
    """Download short videos into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for i, entry in enumerate(entries, 1):
        title = safe_filename(entry["title"])
        views = entry["view_count"]
        out_path = output_dir / f"{i:02d}_{title}_{views}.mp4"

        if out_path.exists():
            log(f"Skip (exists): {out_path.name}", "info")
            results.append({**entry, "file": str(out_path), "skipped": True})
            continue

        log(f"Baixando short {i}/{len(entries)}: {entry['title']} ({views:,} views)")
        dl = subprocess.run(
            [
                "yt-dlp",
                "--force-overwrites",
                "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
                "--merge-output-format", "mp4",
                "--no-playlist",
                "-o", str(out_path),
                entry["url"],
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if dl.returncode != 0:
            err = (dl.stderr or "")[-300:]
            log(f"Erro download {entry['url']}: {err}", "warn")
            continue

        if not out_path.exists():
            candidates = list(output_dir.glob(f"{i:02d}_{title}*.mp4"))
            if candidates:
                out_path = candidates[0]
            else:
                log(f"Arquivo não encontrado após download: {entry['title']}", "warn")
                continue

        size_mb = out_path.stat().st_size / 1024 / 1024
        log(f"✓ {out_path.name} ({size_mb:.1f}MB)", "ok")
        results.append({**entry, "file": str(out_path), "size_mb": size_mb})

    return results


def fetch_top_channel_shorts(
    channel_url: str,
    top: int = 10,
    scan_limit: int = 100,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """
    Download the top N shorts from a YouTube channel (ranked by view count).

    Output: CLIPPER_CLIPS_DIR/shorts/<channel>/ by default.
    """
    channel_title, all_shorts = list_channel_shorts(channel_url, scan_limit=scan_limit)
    if not all_shorts:
        raise RuntimeError("Nenhum short encontrado no canal")

    top_shorts = all_shorts[:top]
    if output_dir is None:
        slug = safe_filename(channel_title) or channel_slug_from_url(
            normalize_channel_shorts_url(channel_url)
        )
        output_dir = get_clips_dir() / "shorts" / slug
    out = Path(output_dir)
    results = download_shorts(top_shorts, out)

    manifest = {
        "channel": channel_title,
        "channel_url": normalize_channel_shorts_url(channel_url),
        "top": top,
        "scan_limit": scan_limit,
        "processed_at": datetime.now().isoformat(),
        "shorts": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log(f"✓ {len(results)} shorts em {out}", "ok")
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Etapa 2: Transcrição
# ──────────────────────────────────────────────────────────────────────────────
def parse_vtt(vtt_path: Path) -> list[dict]:
    """Parseia VTT/SRT e retorna lista de {start, end, text}"""
    segments = []
    content = vtt_path.read_text(encoding="utf-8", errors="ignore")

    # Normaliza timestamps VTT/SRT → HH:MM:SS.mmm
    pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*\n(.*?)(?=\n\n|\Z)",
        re.DOTALL
    )
    for m in pattern.finditer(content):
        start_ts = m.group(1).replace(",", ".")
        end_ts   = m.group(2).replace(",", ".")
        text     = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        text     = re.sub(r"\n+", " ", text)
        if text and not text.startswith("WEBVTT"):
            segments.append({
                "start": ts_to_seconds(start_ts.split(".")[0]),
                "end":   ts_to_seconds(end_ts.split(".")[0]),
                "text":  text,
            })
    return segments

def transcribe_whisper(video_path: Path, job: JobStatus) -> list[dict]:
    """Transcreve com faster-whisper localmente"""
    job.status = "transcrevendo"
    log(f"Transcrevendo com Whisper ({CFG['whisper_model']}): {video_path.name}")

    from faster_whisper import WhisperModel
    model = WhisperModel(CFG["whisper_model"], device="cpu", compute_type="int8")

    segments_raw, _ = model.transcribe(
        str(video_path),
        language=CFG["lang"],
        beam_size=5,
        word_timestamps=False,
    )

    segments = []
    for seg in segments_raw:
        segments.append({
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text.strip(),
        })
    log(f"Transcrição: {len(segments)} segmentos")
    return segments

def get_transcript(video_path: Path, sub_files: list, job: JobStatus) -> list[dict]:
    """Usa legenda se disponível, senão transcreve"""
    if sub_files:
        log(f"Usando legenda existente: {sub_files[0].name}")
        job.status = "transcrevendo"
        segments = parse_vtt(sub_files[0])
        if segments:
            log(f"Legenda parseada: {len(segments)} segmentos")
            return segments
        log("Legenda vazia, usando Whisper", "warn")

    return transcribe_whisper(video_path, job)

# ──────────────────────────────────────────────────────────────────────────────
# Etapa 3: LLM Scoring
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Você é um editor sênior de cortes para redes sociais, especializado em conteúdo político brasileiro (discursos, entrevistas, podcasts, sabatinas, CPIs).

Sua função: receber a transcrição segmentada com timestamps de um vídeo longo e devolver os trechos com maior potencial de viralização em formato vertical (Reels / Shorts / TikTok), prontos para corte automatizado por ffmpeg.

## ENTRADA

Você recebe:
- METADADOS: título, participantes, contexto e duração do vídeo.
- SEGMENTOS: lista de segmentos no formato `[índice] start=<seg> end=<seg> speaker=<id>: texto`.

Os timestamps são em segundos (float). Você NUNCA inventa timestamps: todo `start` e `end` que você devolver deve corresponder ao `start` de um segmento existente e ao `end` de um segmento existente.

## O QUE FAZ UM CORTE VIRAL

Avalie cada candidato contra estes critérios:

1. HOOK (0-10): os primeiros 3 segundos precisam prender. Frase de impacto, pergunta direta, afirmação polêmica, número surpreendente ou início de história. Corte que começa com "então, como eu ia dizendo..." é lixo.
2. AUTOSSUFICIÊNCIA (0-10): o trecho se entende sozinho, sem o resto do vídeo. Se depende de uma pergunta feita 5 minutos antes, ou o corte inclui a pergunta, ou não serve.
3. CARGA EMOCIONAL (0-10): indignação, humor, confronto, emoção, revelação, ironia. Falas mornas e protocolares não performam.
4. CITABILIDADE (0-10): existe uma frase curta e cravada que funciona como legenda, thumbnail ou print? Quanto mais "printável", melhor.
5. TENSÃO / CONTROVÉRSIA (0-10): discordância, acusação, contradição, promessa ousada, crítica nominal a alguém ou a alguma política.
6. FECHAMENTO (0-10): o trecho termina em ponto de virada, punchline ou conclusão — não no meio de um raciocínio.

## REGRAS DE CORTE

- Duração alvo: {min_dur} a {max_dur} segundos. Aceite 20-30s só se o punchline for excepcional.
- Comece e termine em fronteira de frase. Nunca no meio de uma palavra ou oração.
- Prefira iniciar no segmento que contém a frase-gancho, não antes dele.
- Se a fala só faz sentido com a pergunta do entrevistador, inclua a pergunta.
- Sem sobreposição: dois cortes não podem compartilhar o mesmo intervalo. Se dois candidatos se sobrepõem, mantenha só o de maior score.
- Descarte: vinhetas, apresentações, publicidade, agradecimentos, leitura de regimento, papo burocrático, crosstalk ininteligível, silêncios longos.
- `hook_quote` e `transcript` devem ser verbatim da transcrição. Você não reescreve, não corrige e não parafraseia a fala do participante.

## RESPONSABILIDADE EDITORIAL

Conteúdo político descontextualizado desinforma. Para cada corte, avalie se o recorte inverte, exagera ou distorce o sentido do que a pessoa disse — por exemplo cortando um "não" inicial, uma condicional, uma hipótese que ela rejeita em seguida, ou uma citação de terceiro que soa como opinião própria.

Preencha `risco_descontextualizacao` como "baixo", "medio" ou "alto" e, quando não for "baixo", explique em `alerta` o que falta de contexto. Se o único jeito de o corte viralizar for distorcendo a fala, marque como "alto" — não maquie a avaliação para inflar o score.

## SAÍDA

Responda APENAS com JSON válido. Sem markdown, sem cercas de código, sem comentários, sem texto antes ou depois.

{{
  "clips": [
    {{
      "id": 1,
      "start": 412.88,
      "end": 468.20,
      "duration": 55.32,
      "segment_range": [87, 96],
      "tema": "reforma tributária",
      "hook_quote": "frase verbatim dos primeiros segundos",
      "transcript": "texto verbatim completo do corte",
      "titulo": "título de até 60 caracteres, sem clickbait mentiroso",
      "legenda": "legenda para o post, 1-2 frases",
      "hashtags": ["#exemplo", "#exemplo2"],
      "scores": {{
        "hook": 9,
        "autossuficiencia": 8,
        "emocao": 9,
        "citabilidade": 10,
        "tensao": 8,
        "fechamento": 7
      }},
      "score_final": 8.7,
      "motivo": "por que esse trecho performa, em 1-2 frases",
      "risco_descontextualizacao": "baixo",
      "alerta": null
    }}
  ],
  "resumo": "1-2 frases sobre o tom geral do vídeo e onde estão os melhores momentos"
}}

`score_final` é a média dos seis scores, arredondada em 1 decimal.
Ordene `clips` por `score_final` decrescente.
Devolva no máximo {max_clips} cortes com `score_final` >= 6.5. Se nenhum trecho atingir esse patamar, devolva `"clips": []` e explique em `resumo`.
"""


def chunk_transcript(segments: list[dict], max_chars: int = 12000) -> list[str]:
    """Divide transcript em chunks respeitando tamanho de contexto"""
    chunks = []
    current = []
    current_len = 0

    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        speaker = seg.get("speaker", "desconhecido")
        text = seg.get("text", "").strip()
        line = f"[{i}] start={start:.2f} end={end:.2f} speaker={speaker}: {text}"
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)

    if current:
        chunks.append("\n".join(current))
    return chunks


def _parse_llm_json(raw: str) -> tuple[list[dict], str]:
    """Extrai clips e resumo do retorno JSON do LLM."""
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                clips = data.get("clips", [])
                resumo = data.get("resumo", "")
                return clips, resumo
        except Exception:
            pass

    array_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if array_match:
        try:
            clips = json.loads(array_match.group())
            if isinstance(clips, list):
                return clips, ""
        except Exception:
            pass

    return [], ""


def validate_and_dedup_clips(
    clips: list[dict],
    max_clips: int,
    min_duration: int,
    max_duration: int,
) -> list[dict]:
    """
    Valida e desduplica clipes candidatos por sobreposição de tempo.
    Mantém apenas clipes com start < end e sem interseção de intervalo com clipes de maior score.
    """
    valid = []
    for c in clips:
        try:
            start = ts_to_seconds(c.get("start", 0))
            end = ts_to_seconds(c.get("end", 0))
            if start >= end:
                continue
            dur = end - start
            score = float(c.get("score_final", c.get("score_total", 0)))
            # Aceita 20s+ se punchline for excelente ou se estiver na faixa
            if dur < 20.0 or dur > (max_duration + 30):
                continue
            c["start"] = start
            c["end"] = end
            c["duration"] = dur
            c["score_final"] = score
            c["score_total"] = score
            valid.append(c)
        except Exception:
            continue

    # Ordena por score_final decrescente
    valid.sort(key=lambda x: x["score_final"], reverse=True)

    # Desduplicação por sobreposição de intervalo
    selected = []
    for cand in valid:
        c_start = cand["start"]
        c_end = cand["end"]
        overlap = False
        for s in selected:
            if max(c_start, s["start"]) < min(c_end, s["end"]):
                overlap = True
                break
        if not overlap:
            selected.append(cand)
            if len(selected) >= max_clips:
                break
    return selected


def analyze_with_llm(segments: list[dict], title: str, job: JobStatus) -> list[dict]:
    """Envia transcript pro LLM (Groq ou Gemini/Antigravity) e recebe timestamps dos melhores clips"""
    job.status = "analisando"
    active_prov = CFG.get("active_provider", CFG.get("provider", "anthropic"))
    log(f"Analisando com LLM ({active_prov}:{CFG['model']}): {len(segments)} segmentos")

    system = SYSTEM_PROMPT.format(
        min_dur=CFG["min_duration"],
        max_dur=CFG["max_duration"],
        max_clips=CFG["max_clips"],
    )

    chunks = chunk_transcript(segments)
    all_clips = []
    dur_total = segments[-1]["end"] if segments else 0

    if active_prov in ("gemini", "antigravity"):
        if not HAS_GENAI or genai is None:
            raise MissingDependencyError("google-genai package is required for Gemini/Antigravity provider")
        gemini_key = CFG.get("gemini_api_key") or CFG.get("antigravity_api_key")
        client = genai.Client(api_key=gemini_key or None)

        for i, chunk in enumerate(chunks):
            log(f"LLM chunk {i+1}/{len(chunks)}")
            user_msg = f"""METADADOS
titulo: "{title}"
participantes: "desconhecido"
contexto: "podcast / discurso / entrevista"
duracao_total_s: {dur_total:.2f}
max_cortes: {CFG['max_clips']}

SEGMENTOS
{chunk}"""

            try:
                response = client.models.generate_content(
                    model=CFG["model"],
                    contents=user_msg,
                    config={
                        "system_instruction": system,
                        "temperature": 0.3,
                        "max_output_tokens": 2000,
                    },
                )
                raw = response.text.strip()
                clips, _ = _parse_llm_json(raw)
                all_clips.extend(clips)
                log(f"LLM retornou {len(clips)} clips do chunk {i+1}")
            except Exception as e:
                log(f"Erro LLM chunk {i+1}: {e}", "warn")
                continue

    elif active_prov == "anthropic":
        if not HAS_ANTHROPIC or anthropic_lib is None:
            raise MissingDependencyError("anthropic package is required for Anthropic provider")
        client = anthropic_lib.Anthropic(api_key=CFG["anthropic_api_key"])

        for i, chunk in enumerate(chunks):
            log(f"LLM chunk {i+1}/{len(chunks)}")
            user_msg = f"""METADADOS
titulo: "{title}"
participantes: "desconhecido"
contexto: "podcast / discurso / entrevista"
duracao_total_s: {dur_total:.2f}
max_cortes: {CFG['max_clips']}

SEGMENTOS
{chunk}"""

            try:
                response = client.messages.create(
                    model=ANTHROPIC_HAIKU_MODEL,
                    max_tokens=2000,
                    temperature=0.3,
                    system=system,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = response.content[0].text.strip()
                clips, _ = _parse_llm_json(raw)
                all_clips.extend(clips)
                log(f"LLM retornou {len(clips)} clips do chunk {i+1}")
            except Exception as e:
                log(f"Erro LLM chunk {i+1}: {e}", "warn")
                continue

    else:  # Groq
        if not HAS_GROQ:
            raise MissingDependencyError("groq package is required for Groq provider")
        client = groq_lib.Groq(api_key=CFG["groq_api_key"])

        for i, chunk in enumerate(chunks):
            log(f"LLM chunk {i+1}/{len(chunks)}")
            user_msg = f"""METADADOS
titulo: "{title}"
participantes: "desconhecido"
contexto: "podcast / discurso / entrevista"
duracao_total_s: {dur_total:.2f}
max_cortes: {CFG['max_clips']}

SEGMENTOS
{chunk}"""

            try:
                response = client.chat.completions.create(
                    model=CFG["model"],
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )
                raw = response.choices[0].message.content.strip()
                clips, _ = _parse_llm_json(raw)
                all_clips.extend(clips)
                log(f"LLM retornou {len(clips)} clips do chunk {i+1}")
            except Exception as e:
                log(f"Erro LLM chunk {i+1}: {e}", "warn")
                continue

    result = validate_and_dedup_clips(
        all_clips,
        max_clips=CFG["max_clips"],
        min_duration=CFG["min_duration"],
        max_duration=CFG["max_duration"],
    )

    if not result and all_clips:
        # Fallback se a filtragem estrita remover tudo
        result = all_clips[:CFG["max_clips"]]

    if not result:
        raise RuntimeError("LLM não retornou nenhum clip válido")

    job.clips_found = len(result)
    log(f"Clips selecionados após validação/dedup: {len(result)}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Legendas automáticas (ASS — amarelo/preto, centro da tela)
# ──────────────────────────────────────────────────────────────────────────────
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,44,&H00FFFF00,&H00FFFF00,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    sec = seconds % 60
    sec_int = int(sec)
    cs = int(round((sec - sec_int) * 100))
    if cs >= 100:
        cs = 0
        sec_int += 1
    return f"{h}:{m:02d}:{sec_int:02d}.{cs:02d}"


def escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("\n", "\\N")


def segments_for_clip_range(
    segments: list[dict],
    start_padded: float,
    end_padded: float,
) -> list[dict]:
    """Transcript segments overlapping the clip, with clip-relative timestamps."""
    clip_segments = []
    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        if not text or seg_end <= start_padded or seg_start >= end_padded:
            continue
        rel_start = max(seg_start, start_padded) - start_padded
        rel_end = min(seg_end, end_padded) - start_padded
        if rel_end > rel_start:
            clip_segments.append({"start": rel_start, "end": rel_end, "text": text})
    return clip_segments


def write_ass_subtitles(ass_path: Path, segments: list[dict]) -> None:
    """Write ASS subtitles — yellow text, black outline, vertically centered."""
    lines = [ASS_HEADER]
    for seg in segments:
        start = seconds_to_ass_time(seg["start"])
        end = seconds_to_ass_time(seg["end"])
        text = escape_ass_text(seg["text"])
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
    ass_path.write_text("".join(lines), encoding="utf-8")


def _escape_ffmpeg_ass_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")


def build_burn_subtitles_cmd(
    video_in: Path,
    ass_path: Path,
    video_out: Path,
    encoder: str,
) -> list[str]:
    vf = f"ass='{_escape_ffmpeg_ass_path(ass_path)}'"
    return [
        "ffmpeg", "-y",
        "-i", str(video_in),
        "-vf", vf,
        *video_encode_args(encoder),
        "-c:a", "copy",
        str(video_out),
    ]


def burn_subtitles(video_in: Path, ass_path: Path, video_out: Path) -> None:
    """Re-encode video with burned-in ASS subtitles."""
    encoder = get_video_encoder()
    cmd = build_burn_subtitles_cmd(video_in, ass_path, video_out, encoder)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn failed: {result.stderr[-400:]}")


# ──────────────────────────────────────────────────────────────────────────────
# Etapa 4: Corte com ffmpeg
# ──────────────────────────────────────────────────────────────────────────────
def cut_clips(
    video_path: Path,
    clips: list[dict],
    output_dir: Path,
    title: str,
    job: JobStatus,
    segments: list[dict] | None = None,
):
    """Corta o vídeo nos timestamps indicados com padding"""
    job.status = "cortando"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_filename(title)
    results = []

    for i, clip in enumerate(clips, 1):
        try:
            start_s = ts_to_seconds(clip["start"])
            end_s   = ts_to_seconds(clip["end"])

            # Padding: -0.3s no início (mínimo 0.0) e +0.5s no fim
            start_padded = max(0.0, start_s - 0.3)
            end_padded   = end_s + 0.5
            duration     = end_padded - start_padded

            clip_title = safe_filename(clip.get("titulo", f"clip_{i}"))
            scores = clip.get("scores", {})
            score_final = float(clip.get("score_final", clip.get("score_total", 0)))

            filename = f"{i:02d}_{clip_title}_score{score_final:.1f}.mp4"
            out_path = output_dir / filename
            cut_target = out_path.with_suffix(".raw.mp4") if segments else out_path

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_padded:.2f}",
                "-to", f"{end_padded:.2f}",
                "-i", str(video_path),
                "-c", "copy",         # sem re-encode = rápido
                "-avoid_negative_ts", "make_zero",
                str(cut_target),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                log(f"ffmpeg erro clip {i}: {result.stderr[-200:]}", "warn")
                continue

            if segments and cut_target != out_path:
                ass_path = out_path.with_suffix(".ass")
                clip_segments = segments_for_clip_range(segments, start_padded, end_padded)
                try:
                    if clip_segments:
                        write_ass_subtitles(ass_path, clip_segments)
                        burn_subtitles(cut_target, ass_path, out_path)
                        log(f"Legendas: {len(clip_segments)} segmentos", "ok")
                    else:
                        cut_target.rename(out_path)
                except Exception as e:
                    log(f"Erro legendas clip {i}: {e}", "warn")
                    if cut_target.exists() and not out_path.exists():
                        cut_target.rename(out_path)
                finally:
                    cut_target.unlink(missing_ok=True)
                    ass_path.unlink(missing_ok=True)

            if not out_path.exists():
                log(f"Clip {i} não gerado", "warn")
                continue

            size_mb = out_path.stat().st_size / 1024 / 1024
            log(f"✓ Clip {i}: {filename} ({duration:.1f}s, {size_mb:.1f}MB)", "ok")
            results.append({
                    "file": str(out_path),
                    "titulo": clip.get("titulo"),
                    "resumo": clip.get("resumo"),
                    "legenda": clip.get("legenda"),
                    "hashtags": clip.get("hashtags", []),
                    "tema": clip.get("tema"),
                    "hook_quote": clip.get("hook_quote"),
                    "motivo": clip.get("motivo"),
                    "risco_descontextualizacao": clip.get("risco_descontextualizacao", "baixo"),
                    "alerta": clip.get("alerta"),
                    "scores": scores,
                    "score_final": score_final,
                    "score_total": score_final,
                    "duration": duration,
                    "size_mb": size_mb,
            })
            job.clips_done += 1
        except Exception as e:
            log(f"Erro cortando clip {i}: {e}", "warn")

    # Salva manifest JSON
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "video_title": title,
        "processed_at": datetime.now().isoformat(),
        "clips": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log(f"Manifest salvo: {manifest_path}", "ok")
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Facecam compose — 9:16 (TikTok / Shorts) layout
# ──────────────────────────────────────────────────────────────────────────────
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920
FACECAM_HEIGHT_RATIO = 0.4  # top strip height as fraction of canvas

# Prefer software encoders; skip vaapi (needs hw filters) unless nothing else works
_VIDEO_ENCODER_CANDIDATES = [
    "libx264",
    "libopenh264",
    "h264_nvenc",
    "h264_amf",
    "h264_v4l2m2m",
    "h264_qsv",
    "mpeg4",
]
_encoder_cache: Optional[str] = None
_encoders_stdout_cache: Optional[str] = None


def clear_encoder_cache():
    """Reset cached ffmpeg encoder detection (for tests)."""
    global _encoder_cache, _encoders_stdout_cache
    _encoder_cache = None
    _encoders_stdout_cache = None


def _ffmpeg_encoders_stdout() -> str:
    global _encoders_stdout_cache
    if _encoders_stdout_cache is not None:
        return _encoders_stdout_cache

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    _encoders_stdout_cache = result.stdout
    return _encoders_stdout_cache


def _list_ffmpeg_encoders(media_type: str = "video") -> set[str]:
    prefix = "V" if media_type == "video" else "A"
    encoders: set[str] = set()
    for line in _ffmpeg_encoders_stdout().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(prefix):
            encoders.add(parts[1])
    return encoders


def get_video_encoder() -> str:
    """Pick a working H.264/MPEG-4 encoder from the local ffmpeg build."""
    global _encoder_cache
    if _encoder_cache:
        return _encoder_cache

    available = _list_ffmpeg_encoders("video")
    for enc in _VIDEO_ENCODER_CANDIDATES:
        if enc in available:
            _encoder_cache = enc
            return enc

    raise MissingDependencyError(
        "No H.264/MPEG-4 video encoder in ffmpeg. "
        "Install full ffmpeg (e.g. rpmfusion: sudo dnf install ffmpeg)."
    )


def video_encode_args(encoder: str) -> list[str]:
    """ffmpeg video encoding flags for the chosen encoder."""
    if encoder == "libx264":
        return ["-c:v", encoder, "-preset", "fast", "-crf", "23"]
    if encoder == "libopenh264":
        return ["-c:v", encoder, "-b:v", "4M"]
    if encoder in ("h264_nvenc", "h264_amf"):
        return ["-c:v", encoder, "-preset", "fast", "-cq", "23"]
    if encoder == "mpeg4":
        return ["-c:v", encoder, "-q:v", "5"]
    return ["-c:v", encoder]


def audio_encode_args() -> list[str]:
    """ffmpeg audio encoding flags, or mute if no encoder available."""
    available = _list_ffmpeg_encoders("audio")
    if "aac" in available:
        return ["-c:a", "aac", "-b:a", "192k"]
    if "libopus" in available:
        return ["-c:a", "libopus", "-b:a", "192k"]
    return ["-an"]


def facecam_layout() -> dict:
    """Return pixel dimensions for the 9:16 facecam + clip layout."""
    facecam_h = int(VERTICAL_HEIGHT * FACECAM_HEIGHT_RATIO)
    bottom_h = VERTICAL_HEIGHT - facecam_h
    return {
        "out_w": VERTICAL_WIDTH,
        "out_h": VERTICAL_HEIGHT,
        "facecam_h": facecam_h,
        "bottom_h": bottom_h,
    }


def get_media_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-200:]}")
    return float(result.stdout.strip())


def build_facecam_ffmpeg_cmd(
    clip_path: Path,
    facecam_path: Path,
    output_path: Path,
    duration: float,
) -> list[str]:
    """
  Build ffmpeg command for 9:16 output:
  - facecam on top strip (full width)
  - clip below, starting at the pixel row after the facecam (no overlap)
  - both scaled/cropped to fill their region
    """
    layout = facecam_layout()
    w, h = layout["out_w"], layout["out_h"]
    fc_h = layout["facecam_h"]
    bottom_h = layout["bottom_h"]

    filter_complex = (
        f"[0:v]scale={w}:{bottom_h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{bottom_h},setpts=PTS-STARTPTS[main];"
        f"[1:v]scale={w}:{fc_h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{fc_h},setpts=PTS-STARTPTS[fc];"
        f"[fc][main]vstack=inputs=2[outv]"
    )

    vencoder = get_video_encoder()
    return [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-stream_loop", "-1",
        "-i", str(facecam_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
        "-map", "-1:a",  # mute facecam — clip audio only
        *video_encode_args(vencoder),
        *audio_encode_args(),
        "-t", f"{duration:.2f}",
        str(output_path),
    ]


def compose_facecam(
    clip_path: str | Path,
    facecam_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    Compose a clip with a facecam overlay for 9:16 vertical output.

    Layout (1080x1920):
      - Top: facecam strip (40% height, full width)
      - Bottom: clip fills remaining height, starts below facecam (vstack, no overlap)

    Args:
        clip_path: Path to the source clip (from cut_clips).
        facecam_path: Path to the facecam video file.
        output_path: Optional output path; defaults to <clip>_facecam.mp4.

    Returns:
        Path to the composed output file.

    Raises:
        FileNotFoundError: if clip or facecam does not exist.
        RuntimeError: if ffmpeg fails.
    """
    clip = Path(clip_path).resolve()
    facecam = Path(facecam_path).resolve()

    if not clip.exists():
        raise FileNotFoundError(f"clip not found: {clip}")
    if not facecam.exists():
        raise FileNotFoundError(f"facecam not found: {facecam}")

    if not shutil.which("ffmpeg"):
        raise MissingDependencyError("ffmpeg is required for facecam compose")

    if output_path is None:
        output_path = react_output_path_for_clip(clip)
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    duration = get_media_duration(clip)
    cmd = build_facecam_ffmpeg_cmd(clip, facecam, out, duration)

    log(f"Composing 9:16 facecam: {clip.name} + {facecam.name} (encoder: {get_video_encoder()})")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg facecam compose failed: {result.stderr[-400:]}")

    size_mb = out.stat().st_size / 1024 / 1024
    log(f"✓ Facecam compose: {out.name} ({duration:.1f}s, {size_mb:.1f}MB)", "ok")
    return out


def auto_compose_reacts(
    *,
    clips_dir: str | Path | None = None,
    reacts_dir: str | Path | None = None,
    reacts_source_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Compose every clip in clips_dir with a random react from reacts_source_dir.
    Outputs go to reacts_dir (mirroring the clips folder structure).
    """
    if clips_dir is not None:
        CFG["clips_dir"] = Path(clips_dir)
        _sync_output_dir_alias()
    if reacts_dir is not None:
        CFG["reacts_dir"] = Path(reacts_dir)
    if reacts_source_dir is not None:
        CFG["reacts_source_dir"] = Path(reacts_source_dir)

    sources = list_react_sources()
    if not sources:
        raise FileNotFoundError(
            f"No react videos in {get_reacts_source_dir()}. "
            "Set CLIPPER_REACTS_SOURCE_DIR to a folder of facecam mp4s."
        )

    get_clips_dir().mkdir(parents=True, exist_ok=True)
    get_reacts_dir().mkdir(parents=True, exist_ok=True)

    results = []
    for clip_path in list_clip_files():
        out_path = react_output_path_for_clip(clip_path)
        if skip_existing and out_path.exists():
            log(f"Skip (exists): {out_path.name}", "info")
            continue

        react_path = random.choice(sources)
        log(f"Auto-react: {clip_path.name} + {react_path.name}")
        try:
            out = compose_facecam(clip_path, react_path, out_path)
            results.append({
                "clip": str(clip_path),
                "react": str(react_path),
                "file": str(out),
            })
        except Exception as e:
            log(f"Erro auto-react {clip_path.name}: {e}", "err")

    log(f"Auto-react done: {len(results)} vídeo(s) em {get_reacts_dir()}", "ok")
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline completo para um vídeo
# ──────────────────────────────────────────────────────────────────────────────
def process_video(url: str, job: JobStatus):
    """Roda o pipeline completo para uma URL ou arquivo local."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = get_clips_dir() / f"tmp_{ts}_{abs(hash(url)) % 9999}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download (or use local file directly)
        if is_local_file(url):
            video_path, title, sub_files = load_local_video(url, job)
        else:
            video_path, title, sub_files = download_video(url, work_dir, job)
        job.title = title[:50]

        # 2. Transcrição
        segments = get_transcript(video_path, sub_files, job)

        if len(segments) < 5:
            raise RuntimeError(f"Transcrição insuficiente ({len(segments)} segmentos)")

        # 3. LLM Analysis
        clips = analyze_with_llm(segments, title, job)

        # 4. Corte
        out_dir = get_clips_dir() / safe_filename(title)
        results = cut_clips(video_path, clips, out_dir, title, job, segments=segments)

        job.output_dir = out_dir
        job.status = "concluído"
        log(f"✓ Concluído: {title} → {len(results)} clips em {out_dir}", "ok")

        # Limpa temporários (mantém apenas clips finais)
        for f in work_dir.glob("video*"):
            f.unlink(missing_ok=True)
        for f in (work_dir / "subs").glob("*") if (work_dir / "subs").exists() else []:
            f.unlink(missing_ok=True)
        try:
            (work_dir / "subs").rmdir()
            work_dir.rmdir()
        except OSError:
            pass  # não vazio, ok

    except Exception as e:
        job.status = "erro"
        job.error = str(e)
        log(f"ERRO [{job.title or url}]: {e}", "err")

# ──────────────────────────────────────────────────────────────────────────────
# Interface de terminal com Rich Live
# ──────────────────────────────────────────────────────────────────────────────
def print_summary():
    """Mostra resumo final"""
    if not HAS_RICH:
        print("\n=== Resumo ===")
        for j in _jobs:
            print(f"  {j.title}: {j.status} | {j.clips_done}/{j.clips_found} clips")
        return

    console.print()
    t = Table(title="[bold]Resumo Final[/bold]", show_header=True, header_style="bold white")
    t.add_column("Título",   width=35)
    t.add_column("Status",   width=14)
    t.add_column("Clips",    width=10, justify="center")
    t.add_column("Output",   width=40)
    t.add_column("Erro",     width=30)

    for job in _jobs:
        status_text = Text(job.status, style=job.status_color)
        clips_text  = f"{job.clips_done}/{job.clips_found}" if job.clips_found else "-"
        out_str     = str(job.output_dir) if job.output_dir else "-"
        err_str     = job.error[:30] if job.error else ""
        t.add_row(job.title[:35], status_text, clips_text, out_str, err_str)

    console.print(t)

def run_with_live_display(urls: list[str]):
    """Roda todos os jobs com display live atualizado"""
    get_clips_dir().mkdir(parents=True, exist_ok=True)

    # Cria jobs
    jobs = [add_job(url) for url in urls]

    log_lines = []
    max_log_lines = 12

    def update_logs():
        while not _log_queue.empty():
            log_lines.append(_log_queue.get_nowait())
        return log_lines[-max_log_lines:]

    if not HAS_RICH:
        # Fallback sem rich
        print(f"\n[clipper] Processando {len(urls)} vídeo(s) com {CFG['workers']} worker(s)")
        print(f"Output: clips={CFG['clips_dir']} reacts={CFG['reacts_dir']}\n")
        with ThreadPoolExecutor(max_workers=CFG["workers"]) as executor:
            futures = {executor.submit(process_video, url, job): job for url, job in zip(urls, jobs)}
            for future in as_completed(futures):
                pass
        print_summary()
        return

    # Com rich
    console.print(Panel(
        f"[bold cyan]obolha[/bold cyan] — {len(urls)} vídeo(s) | "
        f"{CFG['workers']} worker(s) | clips: [yellow]{CFG['clips_dir']}[/yellow] | reacts: [yellow]{CFG['reacts_dir']}[/yellow]",
        title="[bold]AI Video Clipper[/bold]",
    ))

    finished = threading.Event()

    def run_jobs():
        with ThreadPoolExecutor(max_workers=CFG["workers"]) as executor:
            futures = {executor.submit(process_video, url, job): job for url, job in zip(urls, jobs)}
            for future in as_completed(futures):
                pass
        finished.set()

    job_thread = threading.Thread(target=run_jobs, daemon=True)
    job_thread.start()

    from rich.console import Group
    with Live(console=console, refresh_per_second=4) as live:
        while not finished.is_set():
            lines = update_logs()
            log_text = "\n".join(lines) if lines else "[dim]aguardando...[/dim]"
            live.update(Group(
                build_status_table(),
                Panel(log_text, title="[dim]log[/dim]", border_style="dim"),
            ))
            time.sleep(0.25)

        # Drena logs restantes
        update_logs()

    job_thread.join()
    print_summary()

# ──────────────────────────────────────────────────────────────────────────────
# Modo interativo
# ──────────────────────────────────────────────────────────────────────────────
def interactive_mode():
    if HAS_RICH:
        console.print(Panel(
            "[bold cyan]AI Video Clipper[/bold cyan] — Modo Interativo\n\n"
            "Cole URLs do YouTube (uma por linha).\n"
            "Linha em branco para iniciar o processamento.\n"
            "Ctrl+C para cancelar.",
            title="clipper.py"
        ))
    else:
        print("\n=== AI Video Clipper — Modo Interativo ===")
        print("Cole URLs (uma por linha). Linha em branco para iniciar.\n")

    urls = []
    while True:
        try:
            line = input("URL> ").strip()
            if not line:
                if urls:
                    break
                continue
            urls.append(line)
            print(f"  ✓ adicionado ({len(urls)} total)")
        except (KeyboardInterrupt, EOFError):
            break

    if not urls:
        print("Nenhuma URL fornecida.")
        return

    run_with_live_display(urls)

# ──────────────────────────────────────────────────────────────────────────────
# Configurações customizáveis via argparse
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="AI Video Clipper — corta automaticamente os melhores momentos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python clipper.py https://youtu.be/XYZ
  python clipper.py https://youtu.be/ABC https://youtu.be/DEF
  python clipper.py --file urls.txt
  python clipper.py --interactive
  python clipper.py https://youtu.be/XYZ --clips 8 --min 60 --max 240 --output /tmp/clips
  GROQ_API_KEY=gsk_... python clipper.py https://youtu.be/XYZ
        """
    )
    p.add_argument("urls",           nargs="*",        help="URLs do YouTube")
    p.add_argument("--file",  "-f",  metavar="FILE",   help="Arquivo com URLs (uma por linha)")
    p.add_argument("--interactive", "-i", action="store_true", help="Modo interativo")
    p.add_argument("--provider",     choices=["auto", "groq", "gemini", "antigravity", "anthropic"], help=f"Provedor LLM (default: {CFG['provider']})")
    p.add_argument("--clips",        type=int,         help=f"Máx clips por vídeo (default: {CFG['max_clips']})")
    p.add_argument("--min",          type=int,         help=f"Duração mínima segundos (default: {CFG['min_duration']})")
    p.add_argument("--max",          type=int,         help=f"Duração máxima segundos (default: {CFG['max_duration']})")
    p.add_argument("--output", "-o", metavar="DIR",    help=f"Pasta de clips (default: {CFG['clips_dir']})")
    p.add_argument("--workers", "-w",type=int,         help=f"Workers paralelos (default: {CFG['workers']})")
    p.add_argument("--lang",         default=None,     help=f"Idioma transcrição (default: {CFG['lang']})")
    p.add_argument("--model",        default=None,     help="Modelo LLM (anthropic: sempre claude-haiku-4-5; groq/gemini: ver defaults)")
    p.add_argument("--whisper",      default=None,     help=f"Modelo Whisper (default: {CFG['whisper_model']})")
    p.add_argument("--check",        action="store_true", help="Verifica dependências e sai")
    return p.parse_args()


def parse_facecam_args():
    p = argparse.ArgumentParser(
        description="Compose a clip with facecam for 9:16 TikTok/Shorts output",
    )
    p.add_argument("clip", help="Path to the clip video")
    p.add_argument("facecam", help="Path to the facecam video")
    p.add_argument("-o", "--output", metavar="FILE", help="Output path (default: <clip>_facecam.mp4)")
    return p.parse_args()


def run_facecam_cli():
    args = parse_facecam_args()
    try:
        out = compose_facecam(args.clip, args.facecam, args.output)
        print(f"✓ Output: {out}")
    except (FileNotFoundError, MissingDependencyError, RuntimeError) as e:
        print(f"[ERRO] {e}")
        sys.exit(1)


def parse_shorts_args():
    p = argparse.ArgumentParser(
        description="Baixa os shorts mais viralizados de um canal YouTube (por views)",
    )
    p.add_argument(
        "channel",
        help="URL do canal, @handle, ou handle (ex: @MrBeast)",
    )
    p.add_argument(
        "--top", "-n",
        type=int,
        default=int(os.getenv("CLIPPER_SHORTS_TOP", "10")),
        help="Quantos shorts baixar (default: 10)",
    )
    p.add_argument(
        "--scan",
        type=int,
        default=int(os.getenv("CLIPPER_SHORTS_SCAN", "100")),
        help="Quantos shorts analisar para rankear (default: 100)",
    )
    p.add_argument("-o", "--output", metavar="DIR", help="Pasta de saída")
    return p.parse_args()


def run_shorts_cli():
    args = parse_shorts_args()
    check_download_deps()
    try:
        channel_title, found = list_channel_shorts(args.channel, scan_limit=args.scan)
        if not found:
            print(f"[ERRO] Nenhum short encontrado em {args.channel}")
            sys.exit(1)

        top_n = min(args.top, len(found))
        print(f"Canal: {channel_title} — {len(found)} shorts analisados, baixando top {top_n}…")

        results = fetch_top_channel_shorts(
            args.channel,
            top=args.top,
            scan_limit=args.scan,
            output_dir=args.output,
        )
        if not results:
            print("[ERRO] Downloads falharam. Rode com logs: verifique yt-dlp e ffmpeg.")
            sys.exit(1)
        print(f"✓ {len(results)} short(s) baixado(s)")
        for r in results:
            print(f"  {r['view_count']:,} views — {r.get('file', r['url'])}")
    except (MissingDependencyError, RuntimeError) as e:
        print(f"[ERRO] {e}")
        sys.exit(1)


def parse_auto_react_args():
    p = argparse.ArgumentParser(
        description="Compose clips with random reacts from CLIPPER_REACTS_SOURCE_DIR",
    )
    p.add_argument("--clips-dir", metavar="DIR", help=f"Clips folder (default: {CFG['clips_dir']})")
    p.add_argument("--reacts-dir", metavar="DIR", help=f"React output folder (default: {CFG['reacts_dir']})")
    p.add_argument("--reacts-source", metavar="DIR", help=f"React pool folder (default: {CFG['reacts_source_dir']})")
    p.add_argument("--force", action="store_true", help="Re-compose even if output exists")
    return p.parse_args()


def run_auto_react_cli():
    args = parse_auto_react_args()
    try:
        results = auto_compose_reacts(
            clips_dir=args.clips_dir,
            reacts_dir=args.reacts_dir,
            reacts_source_dir=args.reacts_source,
            skip_existing=not args.force,
        )
        print(f"✓ {len(results)} react(s) em {get_reacts_dir()}")
        for r in results:
            print(f"  {r['file']}")
    except (FileNotFoundError, MissingDependencyError, RuntimeError) as e:
        print(f"[ERRO] {e}")
        sys.exit(1)


def parse_web_args():
    p = argparse.ArgumentParser(description="oBolha web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def run_web_cli():
    from webui import run_web

    args = parse_web_args()
    run_web(host=args.host, port=args.port)

# ──────────────────────────────────────────────────────────────────────────────
# Clean Python API — for AI agents, scripts, and programmatic use
# ──────────────────────────────────────────────────────────────────────────────
def clip_videos(
    urls: list[str],
    *,
    provider: str | None = None,
    max_clips: int | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    output_dir: str | Path | None = None,
    workers: int | None = None,
    lang: str | None = None,
    model: str | None = None,
    whisper_model: str | None = None,
) -> list[dict]:
    """
    Process one or more video URLs and return clip metadata.

    This is the recommended entry point for AI agents and scripts.
    All options override the corresponding env vars for this call only.

    Args:
        urls:          List of YouTube (or any yt-dlp-supported) URLs.
        max_clips:     Maximum clips to extract per video.
        min_duration:  Minimum clip length in seconds.
        max_duration:  Maximum clip length in seconds.
        output_dir:    Where to save the clips (default: ./clips).
        workers:       Parallel workers.
        lang:          Transcription language code (e.g. "pt", "en").
        model:         Groq model name.
        whisper_model: faster-whisper model size (tiny/base/small/medium/large).

    Returns:
        List of clip dicts, each with keys:
          file, titulo, resumo, scores, score_total, duration, size_mb

    Raises:
        MissingDependencyError: if ffmpeg / yt-dlp / packages are missing.
        MissingAPIKeyError:     if GROQ_API_KEY is not set.
        ValueError:             if urls is empty.
    """
    if not urls:
        raise ValueError("urls must not be empty")

    # Apply per-call overrides without mutating the module-level CFG permanently
    orig = dict(CFG)
    try:
        if provider is not None:
            CFG["provider"] = provider.lower()
        if max_clips is not None:
            CFG["max_clips"] = max_clips
        if min_duration is not None:
            CFG["min_duration"] = min_duration
        if max_duration is not None:
            CFG["max_duration"] = max_duration
        if output_dir is not None:
            CFG["clips_dir"] = Path(output_dir)
            _sync_output_dir_alias()
        if workers is not None:
            CFG["workers"] = workers
        if lang is not None:
            CFG["lang"] = lang
        if model is not None:
            CFG["model"] = model
        if whisper_model is not None:
            CFG["whisper_model"] = whisper_model

        check_deps(raise_on_error=True)
        get_clips_dir().mkdir(parents=True, exist_ok=True)

        jobs = [add_job(url) for url in urls]

        with ThreadPoolExecutor(max_workers=CFG["workers"]) as executor:
            futures = {executor.submit(process_video, url, job): job for url, job in zip(urls, jobs)}
            for future in as_completed(futures):
                pass

        all_clips = []
        for job in jobs:
            if job.output_dir:
                manifest = job.output_dir / "manifest.json"
                if manifest.exists():
                    data = json.loads(manifest.read_text())
                    all_clips.extend(data.get("clips", []))
        return all_clips
    finally:
        CFG.update(orig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "facecam":
        sys.argv.pop(1)  # remove 'facecam' so parse_facecam_args sees clip/facecam
        run_facecam_cli()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "shorts":
        sys.argv.pop(1)
        run_shorts_cli()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "auto-react":
        sys.argv.pop(1)
        run_auto_react_cli()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "web":
        sys.argv.pop(1)
        run_web_cli()
        return

    args = parse_args()

    # Aplica overrides de CLI
    if args.provider:
        CFG["provider"] = args.provider.lower()
    if args.clips:
        CFG["max_clips"] = args.clips
    if args.min:
        CFG["min_duration"] = args.min
    if args.max:
        CFG["max_duration"] = args.max
    if args.output:
        CFG["clips_dir"] = Path(args.output)
        _sync_output_dir_alias()
    if args.workers:
        CFG["workers"] = args.workers
    if args.lang:
        CFG["lang"] = args.lang
    if args.model:
        CFG["model"] = args.model
    if args.whisper:
        CFG["whisper_model"] = args.whisper

    check_deps()

    if args.check:
        print("✓ Todas as dependências OK")
        return

    # Coleta URLs
    urls = list(args.urls)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERRO] Arquivo não encontrado: {args.file}")
            sys.exit(1)
        for line in file_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if args.interactive or not urls:
        interactive_mode()
        return

    run_with_live_display(urls)

if __name__ == "__main__":
    main()

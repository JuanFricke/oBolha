#!/usr/bin/env python3
"""
obolha — AI-powered video clip extractor
Parallel, multi-input, terminal-based.

CLI usage:
  python obolha.py <url_or_file> [url2 url3 ...]
  python obolha.py --file urls.txt
  python obolha.py --interactive

Python API (for AI agents / scripts):
  from obolha import clip_videos
  results = clip_videos(["https://youtu.be/XYZ"])
  # returns list of dicts with clip metadata

Dependencies (managed via uv / pyproject.toml):
  uv sync   # installs everything
  # or: pip install groq faster-whisper rich python-dotenv yt-dlp

Environment variables (loaded from .env automatically):
  GROQ_API_KEY          — Groq API key (required, free at console.groq.com)
  CLIPPER_MODEL         — Groq model (default: llama-3.3-70b-versatile)
  CLIPPER_LANG          — transcription language (default: pt)
  CLIPPER_MAX_CLIPS     — max clips per video (default: 5)
  CLIPPER_MIN_DURATION  — min clip duration in seconds (default: 30)
  CLIPPER_MAX_DURATION  — max clip duration in seconds (default: 180)
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
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich import print as rprint
    from rich.text import Text
    from rich.style import Style
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
    import faster_whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# ──────────────────────────────────────────────────────────────────────────────
# Config via env
# ──────────────────────────────────────────────────────────────────────────────
CFG = {
    "groq_api_key":    os.getenv("GROQ_API_KEY", ""),
    "model":           os.getenv("CLIPPER_MODEL", "llama-3.3-70b-versatile"),
    "lang":            os.getenv("CLIPPER_LANG", "pt"),
    "max_clips":       int(os.getenv("CLIPPER_MAX_CLIPS", "5")),
    "min_duration":    int(os.getenv("CLIPPER_MIN_DURATION", "30")),
    "max_duration":    int(os.getenv("CLIPPER_MAX_DURATION", "180")),
    "output_dir":      Path(os.getenv("CLIPPER_OUTPUT_DIR", "./clips")),
    "workers":         int(os.getenv("CLIPPER_WORKERS", "3")),
    "whisper_model":   os.getenv("CLIPPER_WHISPER_MODEL", "base"),
}

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
    if not HAS_GROQ:
        missing.append("groq    → pip install groq")
    if not HAS_RICH:
        missing.append("rich    → pip install rich")
    if not HAS_WHISPER:
        missing.append("faster-whisper → pip install faster-whisper")
    if missing:
        msg = "Missing dependencies:\n" + "\n".join(f"  • {m}" for m in missing)
        if raise_on_error:
            raise MissingDependencyError(msg)
        print(f"\n[ERRO] {msg}\n")
        sys.exit(1)
    if not CFG["groq_api_key"]:
        msg = "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        if raise_on_error:
            raise MissingAPIKeyError(msg)
        print(f"\n[ERRO] {msg}\n  export GROQ_API_KEY=gsk_...\n")
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Etapa 1: Download
# ──────────────────────────────────────────────────────────────────────────────
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

    subs_result = subprocess.run([
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
SYSTEM_PROMPT = """Você é um editor de vídeo especialista em criar clipes virais para redes sociais.
Analise a transcrição fornecida e identifique os melhores segmentos para recortar.

Critérios de avaliação (1-10):
- impacto: força emocional, reviravoltas, momentos marcantes
- viralidade: potencial de compartilhamento, ganchos, frases memoráveis
- absurdidade: momentos inesperados, humor, situações inusitadas
- engajamento: capacidade de prender atenção, narrativa completa

REGRAS:
- Cada segmento deve ter início e fim claros (início de raciocínio → conclusão)
- Duração mínima: {min_dur}s, máxima: {max_dur}s
- Retorne APENAS JSON válido, sem markdown, sem explicações fora do JSON
- Selecione os {max_clips} melhores segmentos, ordenados por score_total decrescente

Formato de resposta (array JSON):
[
  {{
    "start": "HH:MM:SS",
    "end": "HH:MM:SS",
    "titulo": "título curto do clip",
    "resumo": "o que acontece neste clip",
    "scores": {{
      "impacto": 8,
      "viralidade": 7,
      "absurdidade": 3,
      "engajamento": 9
    }},
    "score_total": 8.0
  }}
]"""

def chunk_transcript(segments: list[dict], max_chars: int = 12000) -> list[str]:
    """Divide transcript em chunks respeitando tamanho de contexto"""
    chunks = []
    current = []
    current_len = 0

    for seg in segments:
        line = f"[{seconds_to_ts(seg['start'])} → {seconds_to_ts(seg['end'])}] {seg['text']}"
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

def analyze_with_llm(segments: list[dict], title: str, job: JobStatus) -> list[dict]:
    """Envia transcript pro Groq e recebe timestamps dos melhores clips"""
    job.status = "analisando"
    log(f"Analisando com LLM ({CFG['model']}): {len(segments)} segmentos")

    client = groq_lib.Groq(api_key=CFG["groq_api_key"])

    system = SYSTEM_PROMPT.format(
        min_dur=CFG["min_duration"],
        max_dur=CFG["max_duration"],
        max_clips=CFG["max_clips"],
    )

    chunks = chunk_transcript(segments)
    all_clips = []

    for i, chunk in enumerate(chunks):
        log(f"LLM chunk {i+1}/{len(chunks)}")

        user_msg = f"""Título do vídeo: "{title}"

Transcrição (com timestamps):
{chunk}

Identifique os melhores clipes seguindo as regras do sistema."""

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

            # Extrai JSON mesmo se vier com markdown
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                clips = json.loads(json_match.group())
                all_clips.extend(clips)
                log(f"LLM retornou {len(clips)} clips do chunk {i+1}")
        except Exception as e:
            log(f"Erro LLM chunk {i+1}: {e}", "warn")
            continue

    if not all_clips:
        raise RuntimeError("LLM não retornou nenhum clip válido")

    # Ordena por score e limita
    all_clips.sort(key=lambda c: float(c.get("score_total", 0)), reverse=True)
    result = all_clips[:CFG["max_clips"]]

    job.clips_found = len(result)
    log(f"Clips selecionados: {len(result)}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Etapa 4: Corte com ffmpeg
# ──────────────────────────────────────────────────────────────────────────────
def cut_clips(video_path: Path, clips: list[dict], output_dir: Path, title: str, job: JobStatus):
    """Corta o vídeo nos timestamps indicados"""
    job.status = "cortando"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_title = safe_filename(title)
    results = []

    for i, clip in enumerate(clips, 1):
        try:
            start_s = ts_to_seconds(clip["start"])
            end_s   = ts_to_seconds(clip["end"])
            duration = end_s - start_s

            if duration < CFG["min_duration"] * 0.5:
                log(f"Clip {i} muito curto ({duration:.0f}s), pulando", "warn")
                continue

            clip_title = safe_filename(clip.get("titulo", f"clip_{i}"))
            scores = clip.get("scores", {})
            score_total = clip.get("score_total", 0)

            filename = f"{i:02d}_{clip_title}_score{score_total:.0f}.mp4"
            out_path = output_dir / filename

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_s),
                "-to", str(end_s),
                "-i", str(video_path),
                "-c", "copy",         # sem re-encode = rápido
                "-avoid_negative_ts", "make_zero",
                str(out_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                size_mb = out_path.stat().st_size / 1024 / 1024
                log(f"✓ Clip {i}: {filename} ({duration:.0f}s, {size_mb:.1f}MB)", "ok")
                results.append({
                    "file": str(out_path),
                    "titulo": clip.get("titulo"),
                    "resumo": clip.get("resumo"),
                    "scores": scores,
                    "score_total": score_total,
                    "duration": duration,
                    "size_mb": size_mb,
                })
                job.clips_done += 1
            else:
                log(f"ffmpeg erro clip {i}: {result.stderr[-200:]}", "warn")
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
# Pipeline completo para um vídeo
# ──────────────────────────────────────────────────────────────────────────────
def process_video(url: str, job: JobStatus):
    """Roda o pipeline completo para uma URL"""
    try:
        # Cria diretório de trabalho temporário
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = CFG["output_dir"] / f"tmp_{ts}_{abs(hash(url)) % 9999}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download
        video_path, title, sub_files = download_video(url, work_dir, job)
        job.title = title[:50]

        # 2. Transcrição
        segments = get_transcript(video_path, sub_files, job)

        if len(segments) < 5:
            raise RuntimeError(f"Transcrição insuficiente ({len(segments)} segmentos)")

        # 3. LLM Analysis
        clips = analyze_with_llm(segments, title, job)

        # 4. Corte
        out_dir = CFG["output_dir"] / safe_filename(title)
        results = cut_clips(video_path, clips, out_dir, title, job)

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
    CFG["output_dir"].mkdir(parents=True, exist_ok=True)

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
        print(f"Output: {CFG['output_dir']}\n")
        with ThreadPoolExecutor(max_workers=CFG["workers"]) as executor:
            futures = {executor.submit(process_video, url, job): job for url, job in zip(urls, jobs)}
            for future in as_completed(futures):
                pass
        print_summary()
        return

    # Com rich
    console.print(Panel(
        f"[bold cyan]clipper.py[/bold cyan] — {len(urls)} vídeo(s) | "
        f"{CFG['workers']} worker(s) | output: [yellow]{CFG['output_dir']}[/yellow]",
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

    with Live(console=console, refresh_per_second=4) as live:
        while not finished.is_set():
            lines = update_logs()
            log_text = "\n".join(lines) if lines else "[dim]aguardando...[/dim]"
            layout_content = (
                build_status_table(),
                Panel(log_text, title="[dim]log[/dim]", border_style="dim", height=max_log_lines + 2)
            )
            # Renderiza status + log
            from rich.columns import Columns
            live.update(Panel(
                "\n".join([
                    console.render_str(str(build_status_table())),
                ]),
                title="[bold cyan]AI Video Clipper[/bold cyan]"
            ))
            # Abordagem direta: usa Columns
            from rich.console import Group
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
    p.add_argument("--clips",        type=int,         help=f"Máx clips por vídeo (default: {CFG['max_clips']})")
    p.add_argument("--min",          type=int,         help=f"Duração mínima segundos (default: {CFG['min_duration']})")
    p.add_argument("--max",          type=int,         help=f"Duração máxima segundos (default: {CFG['max_duration']})")
    p.add_argument("--output", "-o", metavar="DIR",    help=f"Pasta de saída (default: {CFG['output_dir']})")
    p.add_argument("--workers", "-w",type=int,         help=f"Workers paralelos (default: {CFG['workers']})")
    p.add_argument("--lang",         default=None,     help=f"Idioma transcrição (default: {CFG['lang']})")
    p.add_argument("--model",        default=None,     help=f"Modelo Groq (default: {CFG['model']})")
    p.add_argument("--whisper",      default=None,     help=f"Modelo Whisper (default: {CFG['whisper_model']})")
    p.add_argument("--check",        action="store_true", help="Verifica dependências e sai")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Clean Python API — for AI agents, scripts, and programmatic use
# ──────────────────────────────────────────────────────────────────────────────
def clip_videos(
    urls: list[str],
    *,
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
        if max_clips is not None:     CFG["max_clips"]    = max_clips
        if min_duration is not None:  CFG["min_duration"] = min_duration
        if max_duration is not None:  CFG["max_duration"] = max_duration
        if output_dir is not None:    CFG["output_dir"]   = Path(output_dir)
        if workers is not None:       CFG["workers"]      = workers
        if lang is not None:          CFG["lang"]         = lang
        if model is not None:         CFG["model"]        = model
        if whisper_model is not None: CFG["whisper_model"]= whisper_model

        check_deps(raise_on_error=True)
        CFG["output_dir"].mkdir(parents=True, exist_ok=True)

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
    args = parse_args()

    # Aplica overrides de CLI
    if args.clips:   CFG["max_clips"]    = args.clips
    if args.min:     CFG["min_duration"] = args.min
    if args.max:     CFG["max_duration"] = args.max
    if args.output:  CFG["output_dir"]   = Path(args.output)
    if args.workers: CFG["workers"]      = args.workers
    if args.lang:    CFG["lang"]         = args.lang
    if args.model:   CFG["model"]        = args.model
    if args.whisper: CFG["whisper_model"]= args.whisper

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

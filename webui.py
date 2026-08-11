"""
oBolha web UI — clip extraction and facecam compose in the browser.
"""

import json
import os
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from obolha import (
    CFG,
    JobStatus,
    add_job,
    check_deps,
    compose_facecam,
    get_clips_dir,
    get_reacts_dir,
    list_clip_files,
    process_video,
    react_output_path_for_clip,
)
from youtube_schedule import (
    ScheduleStore,
    build_auth_url,
    clear_tokens,
    exchange_code_for_tokens,
    fetch_channel_title,
    oauth_client_config,
    parse_publish_at,
    process_schedule_item,
    register_oauth_state,
    run_pending_uploads,
    save_tokens,
    validate_schedule_input,
    verify_oauth_state,
    youtube_status,
)

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
    import uvicorn
    HAS_WEB = True
except ImportError:
    HAS_WEB = False
    FastAPI = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
_web_lock = threading.Lock()
_web_jobs: dict[str, JobStatus] = {}
_facecam_jobs: dict[str, dict] = {}
_job_started: dict[str, float] = {}


@dataclass
class WebPaths:
    clips_dir: Path
    reacts_dir: Path
    upload_dir: Path


def get_paths() -> WebPaths:
    clips = get_clips_dir().resolve()
    reacts = get_reacts_dir().resolve()
    upload = clips / "_uploads"
    upload.mkdir(parents=True, exist_ok=True)
    reacts.mkdir(parents=True, exist_ok=True)
    return WebPaths(clips_dir=clips, reacts_dir=reacts, upload_dir=upload)


def resolve_under_root(path: Path, root: Path) -> Optional[Path]:
    """Return resolved path only if it stays inside root."""
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def list_available_clips(clips_dir: Path) -> list[dict]:
    """Scan clips_dir for source clip mp4 files."""
    if not clips_dir.exists():
        return []

    clips = []
    for mp4 in list_clip_files(clips_dir):
        try:
            rel = mp4.relative_to(clips_dir)
        except ValueError:
            continue

        meta: dict = {}
        manifest_path = mp4.parent / "manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                for c in data.get("clips", []):
                    clip_file = c.get("file")
                    if clip_file and Path(clip_file).resolve() == mp4.resolve():
                        meta = c
                        break
            except (json.JSONDecodeError, OSError):
                pass

        clips.append({
            "id": str(rel),
            "path": str(mp4),
            "name": mp4.name,
            "folder": str(rel.parent) if rel.parent != Path(".") else "",
            "titulo": meta.get("titulo", mp4.stem),
            "score_final": meta.get("score_final", meta.get("score_total")),
            "is_facecam": False,
        })
    return clips


def list_available_reacts(reacts_dir: Path) -> list[dict]:
    if not reacts_dir.exists():
        return []
    items = []
    for mp4 in sorted(reacts_dir.rglob("*.mp4")):
        try:
            rel = mp4.relative_to(reacts_dir)
        except ValueError:
            continue
        items.append({
            "id": str(rel),
            "path": str(mp4),
            "name": mp4.name,
            "folder": str(rel.parent) if rel.parent != Path(".") else "",
            "titulo": mp4.stem,
        })
    return items


def job_to_dict(job: JobStatus, job_id: str) -> dict:
    return {
        "id": job_id,
        "kind": "clip",
        "url": job.url,
        "title": job.title,
        "status": job.status,
        "clips_found": job.clips_found,
        "clips_done": job.clips_done,
        "error": job.error,
        "output_dir": str(job.output_dir) if job.output_dir else None,
        "elapsed": job.elapsed,
        "started": _job_started.get(job_id, 0),
    }


def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            f.write(chunk)


def _youtube_redirect_uri(request: "Request") -> str:
    env = os.getenv("YOUTUBE_REDIRECT_URI")
    if env:
        return env
    return str(request.url_for("youtube_oauth_callback"))


def _queue_youtube_upload(item_id: str, video_path: Path) -> None:
    def run():
        store = ScheduleStore()
        item = store.get(item_id)
        if not item:
            return
        fresh = store.get(item_id)
        if fresh:
            process_schedule_item(fresh, video_path, store)

    threading.Thread(target=run, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# HTML (single page, two tabs)
# ──────────────────────────────────────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>oBolha</title>
  <style>
    :root {
      --bg: #0f1117;
      --surface: #1a1d27;
      --border: #2a2f3d;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --accent: #6c9eff;
      --accent-hover: #8ab4ff;
      --ok: #34d399;
      --err: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      min-height: 100vh;
      line-height: 1.5;
    }
    .wrap { max-width: 720px; margin: 0 auto; padding: 2rem 1.25rem; }
    h1 { font-size: 1.75rem; margin: 0 0 0.25rem; }
    .sub { color: var(--muted); margin-bottom: 1.5rem; }
    .tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
    .tab {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 0.5rem 1rem;
      border-radius: 8px;
      cursor: pointer;
    }
    .tab.active { color: var(--text); border-color: var(--accent); }
    .panel { display: none; }
    .panel.active { display: block; }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.35rem; }
    input, select {
      width: 100%;
      padding: 0.6rem 0.75rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      margin-bottom: 1rem;
    }
    input[type="file"] { padding: 0.45rem; }
    .row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; }
    button {
      background: var(--accent);
      color: #0f1117;
      border: none;
      padding: 0.65rem 1.25rem;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .jobs, .clips-list { margin-top: 2rem; }
    .jobs h2, .clips-list h2 { font-size: 1rem; margin-bottom: 0.75rem; }
    .job {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
    }
    .job .status { font-weight: 600; }
    .job.ok .status { color: var(--ok); }
    .job.err .status { color: var(--err); }
    .job a { color: var(--accent); }
    .clip-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5rem 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }
    .clip-item a { color: var(--accent); text-decoration: none; }
    .hint { font-size: 0.8rem; color: var(--muted); margin-top: -0.5rem; margin-bottom: 1rem; }
    #log { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>oBolha</h1>
    <p class="sub">AI clip extractor + 9:16 facecam compose</p>

    <div class="tabs">
      <button class="tab active" data-tab="clip">Clip</button>
      <button class="tab" data-tab="facecam">Facecam</button>
      <button class="tab" data-tab="youtube">YouTube</button>
    </div>

    <div id="panel-clip" class="panel active">
      <form id="clip-form">
        <label>YouTube URL</label>
        <input name="url" type="url" placeholder="https://youtu.be/...">
        <p class="hint">Or upload a local video file below (URL optional if file provided).</p>
        <label>Video file</label>
        <input name="video" type="file" accept="video/*">
        <div class="row">
          <div>
            <label>Max clips</label>
            <input name="max_clips" type="number" min="1" max="50" value="20">
          </div>
          <div>
            <label>Min sec</label>
            <input name="min_duration" type="number" min="10" value="20">
          </div>
          <div>
            <label>Max sec</label>
            <input name="max_duration" type="number" min="15" max="60" value="60">
          </div>
        </div>
        <button type="submit">Start clipping</button>
      </form>
    </div>

    <div id="panel-facecam" class="panel">
      <form id="facecam-form">
        <label>Clip</label>
        <select name="clip_id" id="clip-select" required>
          <option value="">— select a clip —</option>
        </select>
        <p class="hint">Or upload a clip file instead of selecting from the list.</p>
        <label>Clip file (optional)</label>
        <input name="clip_file" type="file" accept="video/*">
        <label>Facecam video</label>
        <input name="facecam" type="file" accept="video/*" required>
        <button type="submit">Compose 9:16</button>
      </form>
    </div>

    <div id="panel-youtube" class="panel">
      <div id="yt-status" class="job" style="margin-bottom:1rem"></div>
      <p class="hint" id="yt-hint">Conecte sua conta Google/YouTube para agendar Shorts.</p>
      <div style="display:flex;gap:0.5rem;margin-bottom:1rem">
        <button type="button" id="yt-connect">Conectar YouTube</button>
        <button type="button" id="yt-disconnect" style="background:var(--border);color:var(--text)">Desconectar</button>
      </div>
      <form id="youtube-form">
        <label>Vídeo (react)</label>
        <select name="react_id" id="yt-react-select" required>
          <option value="">— selecione um react —</option>
        </select>
        <label>Título</label>
        <input name="title" id="yt-title" type="text" maxlength="100" required>
        <label>Descrição</label>
        <input name="description" id="yt-description" type="text" placeholder="#shorts hashtags...">
        <label>Publicar em</label>
        <input name="publish_at" id="yt-publish-at" type="datetime-local" required>
        <button type="submit">Agendar no YouTube</button>
      </form>
      <div class="clips-list">
        <h2>Agendamentos</h2>
        <div id="yt-schedule"></div>
      </div>
    </div>

    <p id="log"></p>

    <div class="jobs">
      <h2>Jobs</h2>
      <div id="jobs"></div>
    </div>

    <div class="clips-list">
      <h2>Clips (sem react)</h2>
      <div id="clips"></div>
    </div>

    <div class="clips-list">
      <h2>Prontos com react</h2>
      <div id="reacts"></div>
    </div>
  </div>
  <script>
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');
    tabs.forEach(t => t.addEventListener('click', () => {
      tabs.forEach(x => x.classList.remove('active'));
      panels.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      document.getElementById('panel-' + t.dataset.tab).classList.add('active');
    }));

    function setLog(msg) { document.getElementById('log').textContent = msg; }

    async function refreshClips() {
      const res = await fetch('/api/clips');
      const clips = await res.json();
      const sel = document.getElementById('clip-select');
      const prev = sel.value;
      sel.innerHTML = '<option value="">— select a clip —</option>';
      clips.forEach(c => {
        const o = document.createElement('option');
        o.value = c.id;
        o.textContent = (c.folder ? c.folder + '/' : '') + c.name + (c.titulo ? ' — ' + c.titulo : '');
        sel.appendChild(o);
      });
      if (prev) sel.value = prev;

      const box = document.getElementById('clips');
      if (!clips.length) { box.innerHTML = '<p class="hint">No clips yet.</p>'; return; }
      box.innerHTML = clips.map(c =>
        `<div class="clip-item"><span>${c.titulo || c.name}</span>
         <a href="/files/clips/${encodeURIComponent(c.id)}" target="_blank">download</a></div>`
      ).join('');
    }

    async function refreshReacts() {
      const res = await fetch('/api/reacts');
      const reacts = await res.json();
      const box = document.getElementById('reacts');
      if (!reacts.length) { box.innerHTML = '<p class="hint">Nenhum react ainda.</p>'; return; }
      box.innerHTML = reacts.map(c =>
        `<div class="clip-item"><span>${c.titulo || c.name}</span>
         <a href="/files/reacts/${encodeURIComponent(c.id)}" target="_blank">download</a></div>`
      ).join('');

      const ytSel = document.getElementById('yt-react-select');
      if (ytSel) {
        const prev = ytSel.value;
        ytSel.innerHTML = '<option value="">— selecione um react —</option>';
        reacts.forEach(c => {
          const o = document.createElement('option');
          o.value = c.id;
          o.textContent = (c.folder ? c.folder + '/' : '') + c.name;
          o.dataset.title = c.titulo || c.name;
          ytSel.appendChild(o);
        });
        if (prev) ytSel.value = prev;
      }
    }

    function defaultPublishAtLocal() {
      const d = new Date(Date.now() + 3600000);
      d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
      return d.toISOString().slice(0, 16);
    }

    async function refreshYoutube() {
      const res = await fetch('/api/youtube/status');
      const data = await res.json();
      const st = document.getElementById('yt-status');
      if (data.connected) {
        st.innerHTML = `<span class="status" style="color:var(--ok)">Conectado</span> — ${data.channel_title || 'canal'}`;
        st.className = 'job ok';
      } else {
        st.innerHTML = '<span class="status">Não conectado</span>';
        st.className = 'job';
      }

      const list = document.getElementById('yt-schedule');
      const items = data.schedule || [];
      if (!items.length) {
        list.innerHTML = '<p class="hint">Nenhum agendamento.</p>';
        return;
      }
      const statusLabel = s => ({
        pending: 'aguardando', uploading: 'enviando', scheduled: 'agendado', failed: 'erro'
      }[s] || s);
      list.innerHTML = items.map(i => {
        const when = i.publish_at ? new Date(i.publish_at).toLocaleString() : '';
        const yt = i.youtube_video_id
          ? `<a href="https://youtu.be/${i.youtube_video_id}" target="_blank">ver</a>` : '';
        const retry = i.status === 'failed'
          ? `<button type="button" data-retry="${i.id}" class="yt-retry">retry</button>` : '';
        const del = ['pending','failed'].includes(i.status)
          ? `<button type="button" data-del="${i.id}" class="yt-del">remover</button>` : '';
        return `<div class="clip-item"><span><strong>${i.title}</strong><br>
          <span class="hint">${statusLabel(i.status)} · ${when}</span>
          ${i.error ? '<br><span style="color:var(--err)">' + i.error + '</span>' : ''}</span>
          <span>${yt} ${retry} ${del}</span></div>`;
      }).join('');

      list.querySelectorAll('.yt-retry').forEach(btn => {
        btn.addEventListener('click', async () => {
          await fetch('/api/youtube/schedule/' + btn.dataset.retry + '/retry', { method: 'POST' });
          refreshYoutube();
        });
      });
      list.querySelectorAll('.yt-del').forEach(btn => {
        btn.addEventListener('click', async () => {
          await fetch('/api/youtube/schedule/' + btn.dataset.del, { method: 'DELETE' });
          refreshYoutube();
        });
      });
    }

    async function refreshJobs() {
      const res = await fetch('/api/jobs');
      const jobs = await res.json();
      const box = document.getElementById('jobs');
      if (!jobs.length) { box.innerHTML = '<p class="hint">No jobs yet.</p>'; return; }
      box.innerHTML = jobs.map(j => {
        const cls = j.status === 'concluído' || j.status === 'done' ? 'ok' : (j.error ? 'err' : '');
        const extra = j.kind === 'clip'
          ? `${j.clips_done || 0}/${j.clips_found || '-'} clips`
          : (j.output ? `<a href="/files/reacts/${encodeURIComponent(j.output_rel)}">download</a>` : '');
        return `<div class="job ${cls}"><strong>${j.title || j.kind}</strong>
          <span class="status">${j.status}</span> — ${extra}
          ${j.error ? '<br><span style="color:var(--err)">' + j.error + '</span>' : ''}</div>`;
      }).join('');
      const running = jobs.some(j => !['concluído','done','erro','error'].includes(j.status));
      if (running) setTimeout(() => { refreshJobs(); refreshClips(); refreshReacts(); }, 2000);
    }

    document.getElementById('clip-form').addEventListener('submit', async e => {
      e.preventDefault();
      const fd = new FormData(e.target);
      setLog('Starting clip job…');
      const res = await fetch('/api/clip', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) { setLog(data.detail || 'Error'); return; }
      setLog('Job started: ' + data.id);
      refreshJobs();
    });

    document.getElementById('facecam-form').addEventListener('submit', async e => {
      e.preventDefault();
      const fd = new FormData(e.target);
      setLog('Starting facecam compose…');
      const res = await fetch('/api/facecam', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) { setLog(data.detail || 'Error'); return; }
      setLog('Job started: ' + data.id);
      refreshJobs();
    });

    document.getElementById('yt-connect').addEventListener('click', () => {
      window.location.href = '/api/youtube/auth';
    });
    document.getElementById('yt-disconnect').addEventListener('click', async () => {
      await fetch('/api/youtube/disconnect', { method: 'POST' });
      refreshYoutube();
    });
    document.getElementById('yt-react-select').addEventListener('change', e => {
      const opt = e.target.selectedOptions[0];
      if (opt && opt.dataset.title) document.getElementById('yt-title').value = opt.dataset.title;
    });
    document.getElementById('youtube-form').addEventListener('submit', async e => {
      e.preventDefault();
      const fd = new FormData(e.target);
      setLog('Agendando no YouTube…');
      const res = await fetch('/api/youtube/schedule', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) { setLog(data.detail || 'Erro'); return; }
      setLog('Agendado: ' + data.id);
      refreshYoutube();
    });

    const pub = document.getElementById('yt-publish-at');
    if (pub && !pub.value) pub.value = defaultPublishAtLocal();

    refreshClips();
    refreshReacts();
    refreshJobs();
    refreshYoutube();
    setInterval(() => { refreshJobs(); refreshClips(); refreshReacts(); refreshYoutube(); }, 5000);
  </script>
</body>
</html>
"""


def create_app() -> "FastAPI":
    if not HAS_WEB:
        raise ImportError("Install web dependencies: uv sync --extra web")

    app = FastAPI(title="oBolha")

    @app.on_event("startup")
    def _startup_youtube_pending():
        paths = get_paths()

        def run():
            run_pending_uploads(paths.reacts_dir)

        threading.Thread(target=run, daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/api/jobs")
    def api_jobs():
        with _web_lock:
            jobs = []
            for jid, job in _web_jobs.items():
                jobs.append(job_to_dict(job, jid))
            for jid, fj in _facecam_jobs.items():
                jobs.append(fj)
            jobs.sort(key=lambda x: x.get("started", 0), reverse=True)
            return jobs

    @app.get("/api/clips")
    def api_clips():
        paths = get_paths()
        return list_available_clips(paths.clips_dir)

    @app.get("/api/reacts")
    def api_reacts():
        paths = get_paths()
        return list_available_reacts(paths.reacts_dir)

    @app.post("/api/clip")
    async def api_clip(
        url: str = Form(""),
        max_clips: int = Form(20),
        min_duration: int = Form(20),
        max_duration: int = Form(60),
        video: UploadFile | None = File(None),
    ):
        paths = get_paths()
        source = url.strip()

        if video and video.filename:
            ext = Path(video.filename).suffix or ".mp4"
            dest = paths.upload_dir / f"{uuid.uuid4().hex}{ext}"
            _save_upload(video, dest)
            source = str(dest)

        if not source:
            raise HTTPException(400, "Provide a URL or video file")

        if max_clips:
            CFG["max_clips"] = max_clips
        if min_duration:
            CFG["min_duration"] = min_duration
        if max_duration:
            CFG["max_duration"] = max_duration

        job_id = secrets.token_hex(8)
        job = add_job(source)
        with _web_lock:
            _web_jobs[job_id] = job
            _job_started[job_id] = datetime.now().timestamp()

        def run():
            process_video(source, job)

        threading.Thread(target=run, daemon=True).start()
        return {"id": job_id, "status": job.status}

    @app.post("/api/facecam")
    async def api_facecam(
        clip_id: str = Form(""),
        clip_file: UploadFile | None = File(None),
        facecam: UploadFile = File(...),
    ):
        paths = get_paths()
        clip_path: Optional[Path] = None

        if clip_file and clip_file.filename:
            ext = Path(clip_file.filename).suffix or ".mp4"
            dest = paths.upload_dir / f"clip_{uuid.uuid4().hex}{ext}"
            _save_upload(clip_file, dest)
            clip_path = dest
        elif clip_id:
            clip_path = resolve_under_root(paths.clips_dir / clip_id, paths.clips_dir)
            if not clip_path or not clip_path.exists():
                raise HTTPException(400, "Invalid clip selection")
        else:
            raise HTTPException(400, "Select a clip or upload a clip file")

        if not facecam.filename:
            raise HTTPException(400, "Facecam video required")

        fc_dest = paths.upload_dir / f"facecam_{uuid.uuid4().hex}.mp4"
        _save_upload(facecam, fc_dest)

        job_id = secrets.token_hex(8)
        fj = {
            "id": job_id,
            "kind": "facecam",
            "title": f"Facecam: {clip_path.name}",
            "status": "processing",
            "error": "",
            "output": None,
            "output_rel": None,
            "started": datetime.now().timestamp(),
        }
        with _web_lock:
            _facecam_jobs[job_id] = fj

        def run():
            try:
                out_path = react_output_path_for_clip(clip_path)
                out = compose_facecam(clip_path, fc_dest, out_path)
                rel = out.relative_to(paths.reacts_dir)
                fj["status"] = "done"
                fj["output"] = str(out)
                fj["output_rel"] = str(rel)
            except Exception as e:
                fj["status"] = "error"
                fj["error"] = str(e)

        threading.Thread(target=run, daemon=True).start()
        return {"id": job_id, "status": "processing"}

    @app.get("/api/youtube/status")
    def api_youtube_status():
        status = youtube_status()
        status["schedule"] = ScheduleStore().list_items()
        return status

    @app.get("/api/youtube/auth")
    def youtube_auth(request: Request):
        try:
            client_id, _ = oauth_client_config()
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        state = secrets.token_urlsafe(16)
        register_oauth_state(state)
        redirect_uri = _youtube_redirect_uri(request)
        return RedirectResponse(build_auth_url(client_id, redirect_uri, state))

    @app.get("/api/youtube/callback", name="youtube_oauth_callback")
    def youtube_oauth_callback(
        request: Request,
        code: str = "",
        state: str = "",
        error: str = "",
    ):
        if error:
            return RedirectResponse(f"/?youtube_error={error}")
        if not code or not verify_oauth_state(state):
            return RedirectResponse("/?youtube_error=oauth_state")
        redirect_uri = _youtube_redirect_uri(request)
        try:
            tokens = exchange_code_for_tokens(code, redirect_uri)
            access = tokens.get("access_token")
            if access:
                try:
                    tokens["channel_title"] = fetch_channel_title(access)
                except Exception:
                    tokens["channel_title"] = ""
            save_tokens(tokens)
        except Exception as e:
            return RedirectResponse(f"/?youtube_error={str(e)[:120]}")
        return RedirectResponse("/?youtube=connected")

    @app.post("/api/youtube/disconnect")
    def youtube_disconnect():
        clear_tokens()
        return {"ok": True}

    @app.post("/api/youtube/schedule")
    async def api_youtube_schedule(
        react_id: str = Form(...),
        title: str = Form(...),
        description: str = Form(""),
        publish_at: str = Form(...),
    ):
        if not youtube_status().get("connected"):
            raise HTTPException(400, "Conecte YouTube primeiro")

        paths = get_paths()
        video_path = resolve_under_root(paths.reacts_dir / react_id, paths.reacts_dir)
        if not video_path or not video_path.is_file():
            raise HTTPException(400, "React inválido")

        try:
            pub_dt = parse_publish_at(publish_at)
        except ValueError as e:
            raise HTTPException(400, str(e))

        err = validate_schedule_input(title, pub_dt)
        if err:
            raise HTTPException(400, err)

        rel = str(video_path.relative_to(paths.reacts_dir))
        store = ScheduleStore()
        item = store.add(rel, title, description, pub_dt)
        _queue_youtube_upload(item["id"], video_path)
        return item

    @app.delete("/api/youtube/schedule/{item_id}")
    def youtube_delete_schedule(item_id: str):
        store = ScheduleStore()
        item = store.get(item_id)
        if not item:
            raise HTTPException(404, "Agendamento não encontrado")
        if item.get("status") not in ("pending", "failed"):
            raise HTTPException(400, "Só agendamentos pending/failed podem ser removidos")
        store.delete(item_id)
        return {"ok": True}

    @app.post("/api/youtube/schedule/{item_id}/retry")
    def youtube_retry_schedule(item_id: str):
        paths = get_paths()
        store = ScheduleStore()
        item = store.get(item_id)
        if not item:
            raise HTTPException(404, "Agendamento não encontrado")
        video_path = resolve_under_root(
            paths.reacts_dir / item["video_rel"], paths.reacts_dir,
        )
        if not video_path or not video_path.is_file():
            raise HTTPException(400, "Vídeo não encontrado")
        store.update(item_id, status="pending", error="")
        _queue_youtube_upload(item_id, video_path)
        return {"ok": True}

    @app.get("/files/clips/{file_path:path}")
    def serve_clip_file(file_path: str):
        paths = get_paths()
        resolved = resolve_under_root(paths.clips_dir / file_path, paths.clips_dir)
        if not resolved or not resolved.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(resolved)

    @app.get("/files/reacts/{file_path:path}")
    def serve_react_file(file_path: str):
        paths = get_paths()
        resolved = resolve_under_root(paths.reacts_dir / file_path, paths.reacts_dir)
        if not resolved or not resolved.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(resolved)

    @app.get("/files/{file_path:path}")
    def serve_file(file_path: str):
        """Legacy path — tries clips then reacts."""
        paths = get_paths()
        resolved = resolve_under_root(paths.clips_dir / file_path, paths.clips_dir)
        if not resolved:
            resolved = resolve_under_root(paths.reacts_dir / file_path, paths.reacts_dir)
        if not resolved or not resolved.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(resolved)

    return app


def run_web(host: str = "127.0.0.1", port: int = 8765):
    if not HAS_WEB:
        print("[ERRO] Web UI requires: uv sync --extra web")
        raise SystemExit(1)

    check_deps()
    paths = get_paths()
    paths.clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"oBolha web UI → http://{host}:{port}")
    print(f"Clips: {paths.clips_dir}")
    print(f"Reacts: {paths.reacts_dir}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def main():
    import argparse

    p = argparse.ArgumentParser(description="oBolha web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    run_web(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

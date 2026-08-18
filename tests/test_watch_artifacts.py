import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obolha import (
    CFG,
    MissingDependencyError,
    _clean_yt_dlp_artifacts,
    atomic_render_tmp_path,
    clear_postiz_ambiguous,
    clear_yt_dlp_cookie_jar,
    compose_facecam,
    compute_retry_backoff_seconds,
    clear_postiz_ambiguous,
    compute_retry_backoff_seconds,
    download_shorts,
    enrich_short_metadata,
    facecam_encode_args,
    get_video_stage,
    is_backoff_active,
    is_valid_media,
    load_video_stages,
    mark_short_processed,
    clear_yt_dlp_cookie_jar,
    resolve_facecam_path,
    run_latest_short_react,
    save_video_stages_atomic,
    select_facecam_for_video,
    short_render_path,
    short_source_path,
    update_video_stage,
    update_video_stage,
    yt_dlp_cookie_args,
    yt_dlp_video_format,
)


@pytest.fixture(autouse=True)
def reset_cfg():
    from obolha import clear_yt_dlp_cookie_jar

    clear_yt_dlp_cookie_jar()
    orig = dict(CFG)
    yield
    clear_yt_dlp_cookie_jar()
    CFG.clear()
    CFG.update(orig)


def test_resolve_facecam_reselects_when_persisted_missing(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    sources = [pool / "a.mp4", pool / "b.mp4"]
    for s in sources:
        s.write_bytes(b"x" * 100)
    stage = {"facecam_path": str(tmp_path / "gone.mp4")}
    picked = resolve_facecam_path("abc", sources, stage)
    assert picked in sources
    assert picked == select_facecam_for_video("abc", sources)


def test_yt_dlp_video_format_prefers_avc_with_fallback():
    fmt = yt_dlp_video_format(max_height=1080)
    assert "vcodec^=avc" in fmt
    assert "height<=1080" in fmt
    # graceful fallback when AVC unavailable
    assert fmt.index("vcodec^=avc") < fmt.index("bv*")


def test_short_paths_stable_by_video_id(tmp_path):
    out = tmp_path / "shorts" / "Renan"
    assert short_source_path("abc123", out) == out / "abc123_source.mp4"
    assert short_render_path("abc123", tmp_path / "reacts" / "Renan") == (
        tmp_path / "reacts" / "Renan" / "abc123_facecam.mp4"
    )


def test_download_shorts_uses_stable_id_path_despite_changing_metadata(tmp_path):
    entry = {
        "id": "vid42",
        "title": "Title v1",
        "view_count": 100,
        "url": "https://youtu.be/vid42",
    }
    expected = short_source_path("vid42", tmp_path)
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        mock = MagicMock()
        mock.returncode = 0
        mock.stderr = ""
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_bytes(b"x" * 2048)
        return mock

    with patch("subprocess.run", fake_run), patch("obolha.get_media_duration", return_value=30.0):
        results = download_shorts([entry], tmp_path)

    assert results[0]["file"] == str(expected)
    assert "-o" in captured[0]
    dl_arg = captured[0][captured[0].index("-o") + 1]
    assert dl_arg.endswith("_source.dl.mp4")
    assert expected.name not in dl_arg

    # retry with changed title/views reuses same path without yt-dlp
    entry2 = {**entry, "title": "Title v2", "view_count": 999999}
    mock_run = MagicMock(side_effect=fake_run)
    with patch("subprocess.run", mock_run), patch("obolha.get_media_duration", return_value=30.0):
        results2 = download_shorts([entry2], tmp_path)
    mock_run.assert_not_called()
    assert results2[0]["skipped"] is True
    assert results2[0]["file"] == str(expected)


def test_select_facecam_deterministic_per_video_id(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    sources = [pool / "b.mp4", pool / "a.mp4", pool / "c.mp4"]
    for s in sources:
        s.write_bytes(b"x")
    pick1 = select_facecam_for_video("abc123", sources)
    pick2 = select_facecam_for_video("abc123", sources)
    pick_other = select_facecam_for_video("other99", sources)
    assert pick1 == pick2
    assert pick1 in sources
    # different id may pick different facecam (not guaranteed, but usually)
    assert isinstance(pick_other, Path)


def test_is_valid_media_rejects_empty_or_corrupt(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert is_valid_media(empty) is False

    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"x")
    assert is_valid_media(tiny) is False

    with patch("obolha.get_media_duration", side_effect=RuntimeError("bad")):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"x" * 5000)
        assert is_valid_media(bad) is False

    with patch("obolha.get_media_duration", return_value=10.0):
        good = tmp_path / "good.mp4"
        good.write_bytes(b"x" * 5000)
        assert is_valid_media(good) is True


def test_atomic_render_tmp_path_keeps_mp4_suffix():
    out = Path("/data/vid_facecam.mp4")
    assert atomic_render_tmp_path(out) == Path("/data/vid_facecam.out.tmp.mp4")
    assert atomic_render_tmp_path(out).suffix == ".mp4"


def test_facecam_encode_args_explicit_fps_and_pix_fmt():
    args = facecam_encode_args("libx264")
    assert "-r" in args and "30" in args
    assert "-pix_fmt" in args and "yuv420p" in args


def test_yt_dlp_cookie_args_uses_persistent_writable_jar(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nsecret=value\n")
    monkeypatch.setenv("CLIPPER_YOUTUBE_COOKIES", str(cookies))
    args1 = yt_dlp_cookie_args()
    args2 = yt_dlp_cookie_args()
    work = Path(args1[1])
    assert work != cookies
    assert work == Path(args2[1])
    # Persistent jar lives beside the source (in the data volume), not a shared /tmp path.
    assert work.parent == cookies.parent
    assert work.name == cookies.name + ".work"
    assert stat.S_IMODE(work.stat().st_mode) == 0o600
    assert work.read_text() == cookies.read_text()


def test_is_valid_media_missing_ffprobe_raises(monkeypatch):
    monkeypatch.setattr("obolha.shutil.which", lambda cmd: None if cmd == "ffprobe" else "/usr/bin/ffmpeg")
    path = Path("/tmp/x.mp4")
    with pytest.raises(MissingDependencyError, match="ffprobe"):
        is_valid_media(path)


def test_is_valid_media_timeout_returns_false(tmp_path):
    path = tmp_path / "v.mp4"
    path.write_bytes(b"x" * 5000)
    with patch("obolha.get_media_duration", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)):
        assert is_valid_media(path) is False


def test_run_latest_short_react_valid_source_missing_render(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"
    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "face.mp4").write_bytes(b"cam" * 500)

    short_dir = CFG["clips_dir"] / "shorts" / "Renan"
    react_dir = CFG["reacts_dir"] / "shorts" / "Renan"
    short_dir.mkdir(parents=True)
    react_dir.mkdir(parents=True)
    source = short_source_path("abc", short_dir)
    source.write_bytes(b"src" * 1000)
    render = short_render_path("abc", react_dir)

    short = {
        "id": "abc",
        "title": "Latest",
        "view_count": 1,
        "url": "https://youtu.be/abc",
        "description": "desc",
    }

    def valid_only_source(path):
        return Path(path).resolve() == source.resolve()

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.download_shorts") as mock_dl,
        patch("obolha.compose_facecam", return_value=render) as mock_compose,
        patch("obolha.is_valid_media", side_effect=valid_only_source),
        patch("obolha.get_video_stage", return_value={}),
        patch("obolha.select_facecam_for_video", return_value=pool / "face.mp4"),
        patch("obolha.update_video_stage"),
    ):
        result = run_latest_short_react("@renansantosmbl")

    mock_dl.assert_not_called()
    mock_compose.assert_called_once()
    assert result["clip"] == str(source)
    assert result["file"] == str(render)


def test_run_latest_short_react_restart_after_compose_failure_reuses_source(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"
    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "face.mp4").write_bytes(b"cam" * 500)

    short_dir = CFG["clips_dir"] / "shorts" / "Renan"
    react_dir = CFG["reacts_dir"] / "shorts" / "Renan"
    short_dir.mkdir(parents=True)
    react_dir.mkdir(parents=True)
    source = short_source_path("abc", short_dir)
    source.write_bytes(b"src" * 1000)
    render = short_render_path("abc", react_dir)

    short = {
        "id": "abc",
        "title": "Latest",
        "view_count": 1,
        "url": "https://youtu.be/abc",
        "description": "desc",
    }

    def valid_only_source(path):
        return Path(path).resolve() == source.resolve()

    stage_after_fail = {
        "downloaded": True,
        "source_path": str(source),
        "facecam_path": str(pool / "face.mp4"),
        "retry_count": 1,
        "last_error": "ffmpeg failed",
    }

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.download_shorts") as mock_dl,
        patch("obolha.compose_facecam", return_value=render) as mock_compose,
        patch("obolha.is_valid_media", side_effect=valid_only_source),
        patch("obolha.get_video_stage", return_value=stage_after_fail),
        patch("obolha.update_video_stage"),
    ):
        result = run_latest_short_react("@renansantosmbl")

    mock_dl.assert_not_called()
    mock_compose.assert_called_once()
    assert result["clip"] == str(source)


def test_enrich_short_metadata_fetches_when_flat_description_empty():
    short = {"id": "abc", "title": "T", "url": "https://youtu.be/abc", "description": ""}
    with patch(
        "obolha.fetch_video_metadata",
        return_value={"description": "real description"},
    ) as mock_fetch:
        enriched = enrich_short_metadata(short)
    mock_fetch.assert_called_once_with("https://youtu.be/abc")
    assert enriched["description"] == "real description"


def test_enrich_short_metadata_skips_fetch_when_description_present():
    short = {"id": "abc", "title": "T", "url": "https://youtu.be/abc", "description": "already"}
    with patch("obolha.fetch_video_metadata") as mock_fetch:
        enriched = enrich_short_metadata(short)
    mock_fetch.assert_not_called()
    assert enriched["description"] == "already"


def test_compose_facecam_atomic_replace_and_temp_cleanup(tmp_path):
    clip = tmp_path / "clip.mp4"
    facecam = tmp_path / "cam.mp4"
    out = tmp_path / "out.mp4"
    clip.write_bytes(b"c" * 5000)
    facecam.write_bytes(b"f" * 5000)

    def fake_run(cmd, **kwargs):
        # ffmpeg writes to temp path passed in cmd
        out_arg = Path(cmd[-1])
        out_arg.write_bytes(b"rendered" * 500)
        mock = MagicMock()
        mock.returncode = 0
        mock.stderr = ""
        return mock

    with (
        patch("obolha.get_media_duration", return_value=30.0),
        patch("obolha.get_video_encoder", return_value="libx264"),
        patch("obolha.audio_encode_args", return_value=["-c:a", "aac"]),
        patch("subprocess.run", fake_run),
    ):
        result = compose_facecam(clip, facecam, out)

    assert result == out
    assert out.exists()
    assert not atomic_render_tmp_path(out).exists()
    temps = list(tmp_path.glob("*.out.tmp.mp4"))
    assert not temps


def test_clean_yt_dlp_artifacts_removes_source_dl_fragments(tmp_path):
    vid = "abc123"
    out = short_source_path(vid, tmp_path)
    out.write_bytes(b"x" * 2000)
    (tmp_path / f"{vid}_source.dl.f140.m4a").write_bytes(b"frag")
    (tmp_path / f"{vid}_source.dl.f397.mp4").write_bytes(b"frag")
    (tmp_path / f"{vid}_source.dl.mp4").write_bytes(b"partial")

    _clean_yt_dlp_artifacts(vid, tmp_path, keep=out)

    assert out.exists()
    assert not (tmp_path / f"{vid}_source.dl.f140.m4a").exists()
    assert not (tmp_path / f"{vid}_source.dl.f397.mp4").exists()
    assert not (tmp_path / f"{vid}_source.dl.mp4").exists()


def test_compute_retry_backoff_caps_exponential_delay():
    assert compute_retry_backoff_seconds(1) == 60
    assert compute_retry_backoff_seconds(3) == 240
    assert compute_retry_backoff_seconds(10) == 3600


def test_is_backoff_active_when_next_retry_in_future():
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    assert is_backoff_active({"next_retry_at": future}) is True
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    assert is_backoff_active({"next_retry_at": past}) is False


def test_run_latest_short_react_honors_backoff_without_processing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    CFG["reacts_source_dir"] = tmp_path / "pool"
    (CFG["reacts_source_dir"]).mkdir()
    (CFG["reacts_source_dir"] / "face.mp4").write_bytes(b"x" * 500)

    future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    short = {"id": "abc", "title": "Latest", "url": "https://youtu.be/abc"}

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.get_video_stage", return_value={"next_retry_at": future, "retry_count": 2}),
        patch("obolha.download_shorts") as mock_dl,
        patch("obolha.compose_facecam") as mock_compose,
        patch("obolha.enrich_short_metadata", side_effect=lambda s: s),
    ):
        result = run_latest_short_react("@renansantosmbl")

    mock_dl.assert_not_called()
    mock_compose.assert_not_called()
    assert result["backoff"] is True
    assert result["id"] == "abc"


def test_clear_postiz_ambiguous_clears_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    update_video_stage(
        "vid1",
        postiz_publish_ambiguous=True,
        postiz_publish_attempted=True,
    )
    clear_postiz_ambiguous("vid1")
    stage = get_video_stage("vid1")
    assert stage.get("postiz_publish_ambiguous") is False
    assert stage.get("postiz_publish_attempted") is False


def test_failure_handler_preserves_original_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"
    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "face.mp4").write_bytes(b"cam" * 500)

    short = {"id": "abc", "title": "T", "url": "https://youtu.be/abc", "description": ""}
    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.enrich_short_metadata", side_effect=lambda s: s),
        patch("obolha.download_shorts", side_effect=RuntimeError("download boom")),
        patch("obolha.is_valid_media", side_effect=[False, MissingDependencyError("ffprobe gone")]),
        patch("obolha.get_video_stage", return_value={}),
        patch("obolha.select_facecam_for_video", return_value=pool / "face.mp4"),
    ):
        with pytest.raises(RuntimeError, match="download boom"):
            run_latest_short_react("@renansantosmbl")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not available")
@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe not available")
def test_compose_facecam_integration_produces_valid_output(tmp_path):
    clip = tmp_path / "clip.mp4"
    facecam = tmp_path / "cam.mp4"
    out = tmp_path / "out.mp4"
    for target in (clip, facecam):
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=1",
                "-f", "lavfi", "-i", "sine=f=440:d=1",
                "-shortest", "-pix_fmt", "yuv420p", str(target),
            ],
            check=True,
            capture_output=True,
        )
    result = compose_facecam(clip, facecam, out, skip_if_valid=False)
    assert result == out
    assert is_valid_media(out)
    assert out.stat().st_size > 1024

    def ffprobe_field(field: str) -> str:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", f"stream={field}",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    assert ffprobe_field("width") == "720"
    assert ffprobe_field("height") == "1280"
    assert ffprobe_field("pix_fmt") == "yuv420p"
    assert ffprobe_field("r_frame_rate") == "30/1"


def test_compose_facecam_cleans_temp_on_failure(tmp_path):
    clip = tmp_path / "clip.mp4"
    facecam = tmp_path / "cam.mp4"
    out = tmp_path / "out.mp4"
    clip.write_bytes(b"c" * 5000)
    facecam.write_bytes(b"f" * 5000)

    mock_run = MagicMock()
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "encode failed"

    with (
        patch("obolha.get_media_duration", return_value=30.0),
        patch("obolha.get_video_encoder", return_value="libx264"),
        patch("obolha.audio_encode_args", return_value=["-c:a", "aac"]),
        patch("subprocess.run", mock_run),
    ):
        with pytest.raises(RuntimeError, match="failed"):
            compose_facecam(clip, facecam, out)

    assert not out.exists()
    assert not list(tmp_path.glob("*.out.tmp.mp4"))


def test_compose_facecam_reuses_valid_existing_output(tmp_path):
    clip = tmp_path / "clip.mp4"
    facecam = tmp_path / "cam.mp4"
    out = tmp_path / "out.mp4"
    clip.write_bytes(b"c" * 5000)
    facecam.write_bytes(b"f" * 5000)
    out.write_bytes(b"existing" * 500)

    with (
        patch("obolha.is_valid_media", return_value=True),
        patch("subprocess.run") as mock_run,
    ):
        result = compose_facecam(clip, facecam, out)
    mock_run.assert_not_called()
    assert result == out


def test_compose_facecam_rerenders_when_existing_output_corrupt(tmp_path):
    clip = tmp_path / "clip.mp4"
    facecam = tmp_path / "cam.mp4"
    out = tmp_path / "out.mp4"
    clip.write_bytes(b"c" * 5000)
    facecam.write_bytes(b"f" * 5000)
    out.write_bytes(b"bad")

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"ok" * 500)
        mock = MagicMock()
        mock.returncode = 0
        return mock

    mock_run = MagicMock(side_effect=fake_run)
    with (
        patch("obolha.get_media_duration", return_value=30.0),
        patch("obolha.get_video_encoder", return_value="libx264"),
        patch("obolha.audio_encode_args", return_value=["-c:a", "aac"]),
        patch("obolha.is_valid_media", side_effect=lambda p: str(p).endswith(".out.tmp.mp4")),
        patch("subprocess.run", mock_run),
    ):
        compose_facecam(clip, facecam, out)
    mock_run.assert_called_once()


def test_video_stage_atomic_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    update_video_stage(
        "vid1",
        downloaded=True,
        source_path="/data/vid1_source.mp4",
        last_error="timeout",
        retry_count=2,
    )
    stages = load_video_stages()
    assert stages["videos"]["vid1"]["downloaded"] is True
    assert stages["videos"]["vid1"]["retry_count"] == 2
    assert stages["videos"]["vid1"]["last_error"] == "timeout"
    # atomic write leaves no .tmp behind
    assert not list(tmp_path.glob("*.tmp"))


def test_video_stage_compatible_with_processed_shorts(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    mark_short_processed("posted1")
    update_video_stage("posted1", posted=True)
    stage = get_video_stage("posted1")
    assert stage.get("posted") is True
    from obolha import is_short_processed

    assert is_short_processed("posted1") is True


def test_run_latest_short_react_resumes_render_without_redownload(tmp_path):
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"
    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "face.mp4").write_bytes(b"cam" * 500)

    short_dir = CFG["clips_dir"] / "shorts" / "Renan"
    react_dir = CFG["reacts_dir"] / "shorts" / "Renan"
    short_dir.mkdir(parents=True)
    react_dir.mkdir(parents=True)
    source = short_source_path("abc", short_dir)
    source.write_bytes(b"src" * 1000)
    render = short_render_path("abc", react_dir)
    render.write_bytes(b"render" * 1000)

    short = {
        "id": "abc",
        "title": "Latest",
        "view_count": 1,
        "url": "https://youtu.be/abc",
        "description": "desc",
    }

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.download_shorts") as mock_dl,
        patch("obolha.compose_facecam") as mock_compose,
        patch("obolha.is_valid_media", return_value=True),
        patch("obolha.get_video_stage", return_value={
            "downloaded": True,
            "rendered": True,
            "source_path": str(source),
            "render_path": str(render),
            "facecam_path": str(pool / "face.mp4"),
        }),
    ):
        result = run_latest_short_react("@renansantosmbl")

    mock_dl.assert_not_called()
    mock_compose.assert_not_called()
    assert result["file"] == str(render)
    assert result["clip"] == str(source)


def test_run_latest_short_react_downloads_once_then_reuses_source(tmp_path):
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"
    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "face.mp4").write_bytes(b"cam" * 500)

    short = {
        "id": "abc",
        "title": "Latest",
        "view_count": 1,
        "url": "https://youtu.be/abc",
        "description": "desc",
    }
    short_dir = CFG["clips_dir"] / "shorts" / "Renan"
    short_dir.mkdir(parents=True)
    source = short_source_path("abc", short_dir)
    source.write_bytes(b"src" * 1000)
    react_out = short_render_path("abc", CFG["reacts_dir"] / "shorts" / "Renan")

    dl_result = [{**short, "file": str(source), "skipped": False}]

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.download_shorts", return_value=dl_result) as mock_dl,
        patch("obolha.compose_facecam", return_value=react_out) as mock_compose,
        patch("obolha.is_valid_media", return_value=False),
        patch("obolha.get_video_stage", return_value={}),
        patch("obolha.update_video_stage") as mock_update,
        patch("obolha.select_facecam_for_video", return_value=pool / "face.mp4"),
    ):
        run_latest_short_react("@renansantosmbl")

    mock_dl.assert_called_once()
    mock_compose.assert_called_once()
    assert mock_update.called


def test_run_latest_short_react_restart_persists_downloaded_after_compose_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"
    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "face.mp4").write_bytes(b"cam" * 500)

    short_dir = CFG["clips_dir"] / "shorts" / "Renan"
    react_dir = CFG["reacts_dir"] / "shorts" / "Renan"
    short_dir.mkdir(parents=True)
    react_dir.mkdir(parents=True)
    source = short_source_path("abc", short_dir)
    source.write_bytes(b"src" * 1000)

    short = {
        "id": "abc",
        "title": "Latest",
        "view_count": 1,
        "url": "https://youtu.be/abc",
        "description": "desc",
    }

    def valid_only_source(path):
        return Path(path).resolve() == source.resolve()

    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.download_shorts") as mock_dl,
        patch("obolha.compose_facecam", side_effect=RuntimeError("ffmpeg failed")),
        patch("obolha.is_valid_media", side_effect=valid_only_source),
        patch("obolha.is_backoff_active", return_value=False),
        patch("obolha.enrich_short_metadata", side_effect=lambda s: s),
        patch("obolha.select_facecam_for_video", return_value=pool / "face.mp4"),
    ):
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            run_latest_short_react("@renansantosmbl")
    mock_dl.assert_not_called()
    stage = get_video_stage("abc")
    assert stage.get("downloaded") is True
    assert stage.get("source_path") == str(source)

    render = short_render_path("abc", react_dir)
    with (
        patch("obolha.list_channel_shorts", return_value=("Renan", [short])),
        patch("obolha.is_short_processed", return_value=False),
        patch("obolha.download_shorts") as mock_dl2,
        patch("obolha.compose_facecam", return_value=render) as mock_compose,
        patch("obolha.is_valid_media", side_effect=valid_only_source),
        patch("obolha.is_backoff_active", return_value=False),
        patch("obolha.enrich_short_metadata", side_effect=lambda s: s),
    ):
        result = run_latest_short_react("@renansantosmbl")

    mock_dl2.assert_not_called()
    mock_compose.assert_called_once()
    assert result["clip"] == str(source)


def test_save_video_stages_atomic_uses_replace(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    data = {"videos": {"x": {"downloaded": True}}}
    save_video_stages_atomic(data)
    path = tmp_path / "video_stages.json"
    assert path.exists()
    assert json.loads(path.read_text()) == data
    assert not list(tmp_path.glob("*.tmp"))

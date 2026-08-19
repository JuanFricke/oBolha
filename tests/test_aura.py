"""Tests for aura clip category (Renan Santos discourse + jaguar sepia filter)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obolha import (
    CFG,
    AURA_SYSTEM_PROMPT,
    JobStatus,
    analyze_with_llm,
    cut_aura_clips,
    get_aura_dir,
    process_aura_video,
)


@pytest.fixture(autouse=True)
def reset_cfg():
    orig = dict(CFG)
    yield
    CFG.clear()
    CFG.update(orig)


def test_aura_cfg_defaults():
    assert "aura_dir" in CFG
    assert CFG["aura_min_duration"] >= 8
    assert CFG["aura_max_duration"] >= CFG["aura_min_duration"]
    assert CFG["aura_square_size"] == 1080
    assert get_aura_dir() == Path(CFG["aura_dir"])


def test_aura_system_prompt_mentions_renan_and_aura():
    assert "aura" in AURA_SYSTEM_PROMPT.lower()
    assert "renan" in AURA_SYSTEM_PROMPT.lower()
    assert "viraliza" in AURA_SYSTEM_PROMPT.lower()
    assert "proposta" in AURA_SYSTEM_PROMPT.lower()
    assert "ORDEM DE PRIORIDADE" in AURA_SYSTEM_PROMPT


def test_analyze_with_llm_accepts_custom_prompt():
    CFG["provider"] = "anthropic"
    CFG["active_provider"] = "anthropic"
    CFG["anthropic_api_key"] = "sk-ant-test"
    CFG["max_clips"] = 5
    CFG["min_duration"] = 30
    CFG["max_duration"] = 90

    job = JobStatus(url="https://youtu.be/test")
    segments = [
        {"start": 0.0, "end": 10.0, "text": "O Brasil precisa de coragem."},
        {"start": 10.0, "end": 70.0, "text": "Nós não vamos recuar diante da tirania."},
    ]

    fake_block = MagicMock()
    fake_block.text = """{
        "clips": [{
            "id": 1, "start": 10.0, "end": 70.0, "duration": 60.0,
            "titulo": "Declaração de princípios", "score_final": 9.0,
            "scores": {"hook": 9, "autossuficiencia": 8, "emocao": 10,
                "citabilidade": 9, "tensao": 8, "fechamento": 9}
        }],
        "resumo": "Momento de aura."
    }"""
    fake_response = MagicMock()
    fake_response.content = [fake_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    custom = "PROMPT_AURA_{min_dur}_{max_dur}_{max_clips}"

    with patch("obolha.anthropic_lib.Anthropic", return_value=mock_client), \
         patch("obolha.HAS_ANTHROPIC", True):
        clips = analyze_with_llm(
            segments,
            "Entrevista Renan Santos",
            job,
            system_prompt_template=custom,
            context="discurso Renan Santos",
        )

    assert len(clips) == 1
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "PROMPT_AURA_30_90_3" in call_kwargs["system"]
    assert "discurso Renan Santos" in mock_client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_cut_aura_clips_applies_jaguar_filter(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"\x00" * 2048)
    out_dir = tmp_path / "aura_out"
    job = JobStatus(url="file://" + str(video))

    clips = [{
        "start": 0.0,
        "end": 5.0,
        "titulo": "Momento épico",
        "score_final": 9.2,
        "scores": {},
    }]

    final_cut = out_dir / "01_Momento épico_score9.2.mp4"

    with (
        patch("obolha.ffmpeg_cut_segment") as mock_cut,
        patch("jaguar_sepia_filter.process_video") as mock_filter,
    ):
        mock_cut.side_effect = lambda _v, _s, _e, out, **kwargs: Path(out).write_bytes(b"x" * 2048)

        def fake_filter(inp, out, **kwargs):
            Path(out).write_bytes(b"filtered")

        mock_filter.side_effect = fake_filter

        results = cut_aura_clips(video, clips, out_dir, "Discurso", job, texture_path=None)

    assert len(results) == 1
    assert results[0]["category"] == "aura"
    mock_cut.assert_called_once()
    assert mock_cut.call_args.kwargs.get("square_size") == CFG["aura_square_size"]
    mock_filter.assert_called_once()
    assert Path(results[0]["file"]) == final_cut
    manifest = out_dir / "manifest.json"
    assert manifest.exists()
    import json
    data = json.loads(manifest.read_text())
    assert data["category"] == "aura"


def test_process_aura_video_pipeline(tmp_path, monkeypatch):
    monkeypatch.setitem(CFG, "aura_dir", tmp_path / "aura")

    job = JobStatus(url="/tmp/long_speech.mp4")

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"x" * 2048)

    with (
        patch("obolha.is_local_file", return_value=True),
        patch("obolha.load_local_video", return_value=(fake_video, "Discurso RS", [])),
        patch("obolha.get_transcript", return_value=[{"start": 0, "end": 120, "text": "fala"}] * 10),
        patch("obolha.analyze_with_llm", return_value=[{"start": 10, "end": 40, "titulo": "Aura", "score_final": 9}]),
        patch("obolha.cut_aura_clips", return_value=[{"file": str(tmp_path / "aura" / "01.mp4"), "category": "aura"}]) as mock_cut,
        patch("obolha.check_filter_deps"),
    ):
        process_aura_video("/tmp/long_speech.mp4", job)

    assert job.status == "concluído"
    mock_cut.assert_called_once()
    assert job.output_dir is not None

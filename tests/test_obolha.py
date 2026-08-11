from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obolha import (
    CFG,
    JobStatus,
    MissingAPIKeyError,
    analyze_with_llm,
    check_deps,
    cut_clips,
    parse_args,
    validate_and_dedup_clips,
)


@pytest.fixture(autouse=True)
def reset_cfg():
    orig = dict(CFG)
    yield
    CFG.clear()
    CFG.update(orig)


def test_provider_cfg_defaults():
    assert "provider" in CFG
    assert CFG["max_clips"] == 20
    assert CFG["max_duration"] == 60
    assert "clips_dir" in CFG
    assert "reacts_dir" in CFG
    assert "reacts_source_dir" in CFG
    assert "gemini_api_key" in CFG
    assert "antigravity_api_key" in CFG
    assert "anthropic_api_key" in CFG


def test_check_deps_anthropic_uses_haiku():
    CFG["provider"] = "anthropic"
    CFG["anthropic_api_key"] = "sk-ant-test"
    CFG["model"] = "claude-opus-4"
    with patch("obolha.shutil.which", return_value="/bin/dummy"), \
         patch("obolha.HAS_ANTHROPIC", True), \
         patch("obolha.HAS_RICH", True), \
         patch("obolha.HAS_WHISPER", True):
        check_deps(raise_on_error=True)
        assert CFG["active_provider"] == "anthropic"
        assert CFG["model"] == "claude-haiku-4-5"


def test_check_deps_missing_api_key_anthropic():
    CFG["provider"] = "anthropic"
    CFG["anthropic_api_key"] = ""
    with patch("obolha.shutil.which", return_value="/bin/dummy"), \
         patch("obolha.HAS_ANTHROPIC", True), \
         patch("obolha.HAS_RICH", True), \
         patch("obolha.HAS_WHISPER", True):
        with pytest.raises(MissingAPIKeyError, match="ANTHROPIC"):
            check_deps(raise_on_error=True)


def test_analyze_with_llm_anthropic():
    CFG["provider"] = "anthropic"
    CFG["active_provider"] = "anthropic"
    CFG["anthropic_api_key"] = "sk-ant-test"

    job = JobStatus(url="https://youtu.be/test")
    segments = [
        {"start": 0.0, "end": 10.0, "text": "Início da fala política"},
        {"start": 10.0, "end": 65.0, "text": "Discurso sobre reforma tributária importante"},
    ]

    fake_block = MagicMock()
    fake_block.text = """{
        "clips": [
            {
                "id": 1,
                "start": 10.0,
                "end": 65.0,
                "duration": 55.0,
                "titulo": "Reforma Tributária",
                "score_final": 8.5,
                "scores": {"hook": 9, "autossuficiencia": 8, "emocao": 9,
                    "citabilidade": 10, "tensao": 8, "fechamento": 7}
            }
        ],
        "resumo": "Momento central."
    }"""
    fake_response = MagicMock()
    fake_response.content = [fake_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("obolha.anthropic_lib.Anthropic", return_value=mock_client), \
         patch("obolha.HAS_ANTHROPIC", True):
        clips = analyze_with_llm(segments, "Discurso Político", job)

        assert len(clips) == 1
        assert clips[0]["titulo"] == "Reforma Tributária"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5"


def test_check_deps_missing_api_key_groq():
    CFG["provider"] = "groq"
    CFG["groq_api_key"] = ""
    with patch("obolha.shutil.which", return_value="/bin/dummy"), \
         patch("obolha.HAS_GROQ", True), \
         patch("obolha.HAS_RICH", True), \
         patch("obolha.HAS_WHISPER", True):
        with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
            check_deps(raise_on_error=True)


def test_check_deps_missing_api_key_gemini():
    CFG["provider"] = "gemini"
    CFG["gemini_api_key"] = ""
    CFG["antigravity_api_key"] = ""
    with patch("obolha.shutil.which", return_value="/bin/dummy"), \
         patch("obolha.HAS_GENAI", True), \
         patch("obolha.HAS_GROQ", True), \
         patch("obolha.HAS_RICH", True), \
         patch("obolha.HAS_WHISPER", True):
        with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
            check_deps(raise_on_error=True)


def test_check_deps_auto_provider_selects_gemini_if_no_groq():
    CFG["provider"] = "auto"
    CFG["groq_api_key"] = ""
    CFG["anthropic_api_key"] = ""
    CFG["gemini_api_key"] = "test_gemini_key"
    with patch("obolha.shutil.which", return_value="/bin/dummy"), \
         patch("obolha.HAS_GENAI", True), \
         patch("obolha.HAS_GROQ", True), \
         patch("obolha.HAS_RICH", True), \
         patch("obolha.HAS_WHISPER", True):
        check_deps(raise_on_error=True)
        assert CFG["active_provider"] == "gemini"


def test_analyze_with_llm_gemini_new_schema():
    CFG["provider"] = "gemini"
    CFG["active_provider"] = "gemini"
    CFG["model"] = "gemini-2.5-flash"
    CFG["gemini_api_key"] = "fake_gemini_key"

    job = JobStatus(url="https://youtu.be/test")
    segments = [
        {"start": 0.0, "end": 10.0, "text": "Início da fala política"},
        {"start": 10.0, "end": 65.0, "text": "Discurso sobre reforma tributária importante"},
    ]

    fake_response = MagicMock()
    fake_response.text = """{
        "clips": [
            {
                "id": 1,
                "start": 10.0,
                "end": 65.0,
                "duration": 55.0,
                "segment_range": [1, 2],
                "tema": "reforma tributária",
                "hook_quote": "Discurso sobre reforma tributária",
                "transcript": "Discurso sobre reforma tributária importante",
                "titulo": "Reforma Tributária e Impacto Social",
                "legenda": "Confira a declaração sobre a nova reforma.",
                "hashtags": ["#politica", "#reforma"],
                "scores": {
                    "hook": 9,
                    "autossuficiencia": 8,
                    "emocao": 9,
                    "citabilidade": 10,
                    "tensao": 8,
                    "fechamento": 7
                },
                "score_final": 8.5,
                "motivo": "Forte impacto e excelente citabilidade.",
                "risco_descontextualizacao": "baixo",
                "alerta": null
            }
        ],
        "resumo": "Momento central do discurso político."
    }"""

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response

    with patch("obolha.genai.Client", return_value=mock_client), \
         patch("obolha.HAS_GENAI", True):
        clips = analyze_with_llm(segments, "Discurso Político", job)

        assert len(clips) == 1
        assert clips[0]["titulo"] == "Reforma Tributária e Impacto Social"
        assert clips[0]["score_final"] == 8.5
        assert clips[0]["legenda"] == "Confira a declaração sobre a nova reforma."
        assert clips[0]["risco_descontextualizacao"] == "baixo"


def test_validate_and_dedup_clips_overlap():
    clips = [
        {
            "id": 1,
            "start": 10.0,
            "end": 70.0,
            "score_final": 9.0,
            "titulo": "Clip 1 Maior Score",
        },
        {
            "id": 2,
            "start": 30.0,  # overlaps with clip 1
            "end": 80.0,
            "score_final": 7.5,
            "titulo": "Clip 2 Menor Score (Sobreposto)",
        },
        {
            "id": 3,
            "start": 100.0,  # non-overlapping
            "end": 160.0,
            "score_final": 8.0,
            "titulo": "Clip 3 Sem Sobreposição",
        },
    ]

    filtered = validate_and_dedup_clips(clips, max_clips=5, min_duration=30, max_duration=300)
    assert len(filtered) == 2
    titles = [c["titulo"] for c in filtered]
    assert "Clip 1 Maior Score" in titles
    assert "Clip 3 Sem Sobreposição" in titles
    assert "Clip 2 Menor Score (Sobreposto)" not in titles


def test_cut_clips_padding():
    job = JobStatus(url="https://youtu.be/test")
    clips = [
        {
            "start": 10.0,
            "end": 70.0,
            "titulo": "Clip Com Padding",
            "score_final": 8.5,
            "scores": {"hook": 9},
        }
    ]

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stderr = ""

    out_dir = Path("/tmp/test_clips")
    with patch("subprocess.run", mock_run), \
         patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_size = 5000000
        cut_clips(Path("/tmp/video.mp4"), clips, out_dir, "Video Teste", job)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        # start_padded should be 10.0 - 0.3 = 9.7
        # end_padded should be 70.0 + 0.5 = 70.5
        assert "-ss" in cmd and "9.70" in cmd
        assert "-to" in cmd and "70.50" in cmd


def test_cli_parse_provider_arg():
    with patch("sys.argv", ["obolha.py", "--provider", "gemini", "--model", "gemini-2.5-flash", "https://youtu.be/xyz"]):
        args = parse_args()
        assert args.provider == "gemini"
        assert args.model == "gemini-2.5-flash"

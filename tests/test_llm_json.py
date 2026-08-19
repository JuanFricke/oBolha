"""Tests for LLM JSON parsing and salvage of truncated responses."""

from obolha import _parse_llm_json


def test_parse_llm_json_strips_markdown_fence():
    raw = """```json
{
  "clips": [{
    "id": 1, "start": 10.0, "end": 40.0, "duration": 30.0,
    "titulo": "Teste", "score_final": 8.0
  }],
  "resumo": "Um momento forte."
}
```"""
    clips, resumo = _parse_llm_json(raw)
    assert len(clips) == 1
    assert clips[0]["start"] == 10.0
    assert resumo == "Um momento forte."


def test_parse_llm_json_salvages_truncated_clips_array():
    raw = """```json
{
  "clips": [
    {
      "id": 1,
      "start": 149.44,
      "end": 167.44,
      "duration": 18.0,
      "titulo": "Um ano para destruir as facções",
      "score_final": 8.5,
      "scores": {"hook": 9, "autossuficiencia": 8, "emocao": 10,
        "citabilidade": 9, "tensao": 8, "fechamento": 9}
    },
    {
      "id": 2,
      "start": 200.0,
      "end": 230.0,
      "duration": 30.0,
      "titulo": "Corte incompleto",
      "score_final": 7.0,
      "hashtags": ["#renan"""
    clips, resumo = _parse_llm_json(raw)
    assert len(clips) == 1
    assert clips[0]["titulo"] == "Um ano para destruir as facções"
    assert clips[0]["score_final"] == 8.5

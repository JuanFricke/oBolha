from unittest.mock import patch

from obolha import generate_outside_bubble_copy, parse_outside_bubble_copy


def test_parse_outside_bubble_copy_extracts_json():
    raw = """
    aqui vai
    {"titulo": "Crime organizado no Brasil", "caption": "O Brasil não enfrenta o crime como deveria.", "hashtags": ["#brasil", "#noticias", "#shorts"]}
    """
    copy = parse_outside_bubble_copy(raw)
    assert copy["titulo"] == "Crime organizado no Brasil"
    assert "crime" in copy["caption"].lower()
    assert "#mbl" not in [h.lower() for h in copy["hashtags"]]
    assert "#brasil" in copy["hashtags"]


def test_parse_outside_bubble_copy_rejects_bubble_hashtags():
    raw = '{"titulo": "Renan fala", "caption": "hook", "hashtags": ["#MBL", "#Missao", "#brasil"]}'
    copy = parse_outside_bubble_copy(raw)
    assert copy["hashtags"] == ["#brasil"]


def test_generate_outside_bubble_copy_sends_source_and_prompt():
    raw = '{"titulo": "Título público", "caption": "Hook para quem não conhece o canal.", "hashtags": ["#brasil", "#politica"]}'
    with patch("obolha.llm_complete", return_value=raw) as mock_llm:
        copy = generate_outside_bubble_copy(
            title="Renan na CNN",
            description="fala sobre crime",
            captions="não enfrentamos o crime",
        )
    mock_llm.assert_called_once()
    system, user = mock_llm.call_args[0]
    assert "MBL" in system or "bolha" in system.lower()
    assert "Renan na CNN" in user
    assert copy["titulo"] == "Título público"
    assert copy["caption"]

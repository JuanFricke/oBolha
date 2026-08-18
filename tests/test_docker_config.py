from pathlib import Path


def test_dockerfile_pins_deno_version():
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    text = dockerfile.read_text()
    assert "DENO_VERSION=2.9.5" in text
    assert "deno --version" in text


def test_compose_passes_youtube_cookies_env():
    compose = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    text = compose.read_text()
    assert "CLIPPER_YOUTUBE_COOKIES" in text
    assert "pull_policy: always" in text

FROM python:3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY obolha.py webui.py youtube_schedule.py postiz.py ./

RUN uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1 \
    CLIPPER_CLIPS_DIR=/data/clips \
    CLIPPER_REACTS_DIR=/data/reacts \
    CLIPPER_REACTS_SOURCE_DIR=/data/reacts_pool \
    CLIPPER_DATA_DIR=/data/data

CMD ["/app/.venv/bin/obolha", "watch"]

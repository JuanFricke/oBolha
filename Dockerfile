FROM python:3.12-bookworm

ARG DENO_VERSION=2.9.5

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl unzip ca-certificates \
    && curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh -s v${DENO_VERSION} \
    && deno --version \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY obolha.py webui.py youtube_schedule.py postiz.py ./

RUN uv sync --frozen --no-dev \
    && ln -sf /app/.venv/bin/yt-dlp /usr/local/bin/yt-dlp

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    CLIPPER_CLIPS_DIR=/data/clips \
    CLIPPER_REACTS_DIR=/data/reacts \
    CLIPPER_REACTS_SOURCE_DIR=/data/reacts_pool \
    CLIPPER_DATA_DIR=/data/data

CMD ["/app/.venv/bin/obolha", "watch"]

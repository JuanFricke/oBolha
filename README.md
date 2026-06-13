# oBolha — AI Video Clipper

Automatically cuts the best moments from YouTube videos using an LLM.

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo>
cd oBolha
uv sync

# Set your API key
cp .env.example .env
# edit .env and add your GROQ_API_KEY (free at https://console.groq.com)
```

System dependencies: `ffmpeg` and `yt-dlp` must be on your PATH.
```bash
sudo apt install ffmpeg yt-dlp   # Debian/Ubuntu
brew install ffmpeg yt-dlp       # macOS
```

## CLI usage

```bash
# Single video
uv run obolha https://youtu.be/XYZ

# Multiple videos in parallel
uv run obolha https://youtu.be/ABC https://youtu.be/DEF

# From a file (one URL per line)
uv run obolha --file urls.txt

# Interactive mode
uv run obolha --interactive

# Full options
uv run obolha https://youtu.be/XYZ \
  --clips 8         \  # max clips per video   (default: 5)
  --min 60          \  # min clip seconds       (default: 30)
  --max 240         \  # max clip seconds       (default: 180)
  --output /tmp/clips  \  # output folder
  --workers 4       \  # parallel workers       (default: 3)
  --lang pt         \  # language (pt, en, es…)
  --whisper large      # Whisper model size

# Check dependencies
uv run obolha --check
```

## Python API (for AI agents / scripts)

```python
from obolha import clip_videos

clips = clip_videos(
    ["https://youtu.be/XYZ"],
    max_clips=3,
    min_duration=60,
    output_dir="/tmp/clips",
)

for clip in clips:
    print(clip["titulo"], clip["score_total"], clip["file"])
```

`clip_videos()` returns a list of dicts:

| Key           | Type  | Description                     |
|---------------|-------|---------------------------------|
| `file`        | str   | Absolute path to the clip file  |
| `titulo`      | str   | Short title from LLM            |
| `resumo`      | str   | Summary from LLM                |
| `scores`      | dict  | `{impacto, viralidade, absurdidade, engajamento}` |
| `score_total` | float | Weighted average score          |
| `duration`    | float | Clip duration in seconds        |
| `size_mb`     | float | File size in MB                 |

Raises `MissingDependencyError` or `MissingAPIKeyError` on missing setup — never calls `sys.exit()`.

## Environment variables

| Variable               | Default                    | Description                         |
|------------------------|----------------------------|-------------------------------------|
| `GROQ_API_KEY`         | (required)                 | Groq API key                        |
| `CLIPPER_MODEL`        | llama-3.3-70b-versatile    | Groq model for analysis             |
| `CLIPPER_LANG`         | pt                         | Transcription language              |
| `CLIPPER_MAX_CLIPS`    | 5                          | Max clips per video                 |
| `CLIPPER_MIN_DURATION` | 30                         | Min clip duration (seconds)         |
| `CLIPPER_MAX_DURATION` | 180                        | Max clip duration (seconds)         |
| `CLIPPER_OUTPUT_DIR`   | ./clips                    | Output folder                       |
| `CLIPPER_WORKERS`      | 3                          | Parallel workers                    |
| `CLIPPER_WHISPER_MODEL`| base                       | tiny / base / small / medium / large|

## Output

For each video a folder `./clips/<title>/` is created containing:
- `01_clip_title_score8.mp4`
- `02_clip_title_score7.mp4`
- `manifest.json` — scores, summaries, and metadata for every clip

## Notes

- If YouTube has auto-generated captions, they are used directly (faster). Otherwise transcription falls back to local Whisper.
- Clips are cut with `-c copy` (no re-encode) — fast and lossless.
- Groq free tier: ~14,400 tokens/min — enough for several concurrent videos.

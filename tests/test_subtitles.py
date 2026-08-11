from pathlib import Path

import pytest

from obolha import (
    build_burn_subtitles_cmd,
    segments_for_clip_range,
    seconds_to_ass_time,
    write_ass_subtitles,
)


def test_seconds_to_ass_time():
    assert seconds_to_ass_time(65.5) == "0:01:05.50"
    assert seconds_to_ass_time(0) == "0:00:00.00"


def test_segments_for_clip_range_relativizes():
    segments = [
        {"start": 0.0, "end": 5.0, "text": "antes"},
        {"start": 10.0, "end": 15.0, "text": "durante"},
        {"start": 20.0, "end": 25.0, "text": "depois"},
    ]
    out = segments_for_clip_range(segments, 9.7, 15.5)
    assert len(out) == 1
    assert out[0]["text"] == "durante"
    assert out[0]["start"] == pytest.approx(0.3)
    assert out[0]["end"] == pytest.approx(5.3)


def test_write_ass_subtitles_yellow_middle(tmp_path):
    ass_path = tmp_path / "sub.ass"
    write_ass_subtitles(ass_path, [{"start": 1.0, "end": 3.0, "text": "Olá mundo"}])
    content = ass_path.read_text(encoding="utf-8")
    assert "00FFFF00" in content
    assert ",1,4,0,5," in content
    assert "Dialogue:" in content
    assert "Olá mundo" in content


def test_build_burn_subtitles_cmd():
    cmd = build_burn_subtitles_cmd(
        Path("/tmp/clip.mp4"),
        Path("/tmp/sub.ass"),
        Path("/tmp/out.mp4"),
        encoder="libopenh264",
    )
    assert "-vf" in cmd
    assert "ass=" in cmd[cmd.index("-vf") + 1]
    assert "-c:a" in cmd and "copy" in cmd

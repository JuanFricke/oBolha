from pathlib import Path
from unittest.mock import patch

import pytest

from obolha import (
    CFG,
    auto_compose_reacts,
    list_clip_files,
    react_output_path_for_clip,
)


@pytest.fixture(autouse=True)
def reset_cfg():
    orig = dict(CFG)
    yield
    CFG.clear()
    CFG.update(orig)


def test_react_output_path_mirrors_clips_tree(tmp_path):
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    clip = CFG["clips_dir"] / "Video_Title" / "01_clip_score8.0.mp4"
    out = react_output_path_for_clip(clip)
    assert out == CFG["reacts_dir"] / "Video_Title" / "01_clip_score8.0_facecam.mp4"


def test_list_clip_files_skips_uploads_and_facecam(tmp_path):
    clips_root = tmp_path / "clips"
    folder = clips_root / "My_Video"
    folder.mkdir(parents=True)
    (folder / "01_clip.mp4").write_bytes(b"x")
    (folder / "01_clip_facecam.mp4").write_bytes(b"x")
    (clips_root / "_uploads").mkdir(parents=True)
    (clips_root / "_uploads" / "up.mp4").write_bytes(b"x")

    found = list_clip_files(clips_root)
    assert len(found) == 1
    assert found[0].name == "01_clip.mp4"


def test_auto_compose_reacts_picks_random_react(tmp_path):
    CFG["clips_dir"] = tmp_path / "clips"
    CFG["reacts_dir"] = tmp_path / "reacts"
    CFG["reacts_source_dir"] = tmp_path / "pool"

    clip_dir = CFG["clips_dir"] / "Vid"
    clip_dir.mkdir(parents=True)
    clip = clip_dir / "01_test.mp4"
    clip.write_bytes(b"clip")

    pool = CFG["reacts_source_dir"]
    pool.mkdir()
    (pool / "react_a.mp4").write_bytes(b"a")
    (pool / "react_b.mp4").write_bytes(b"b")

    with patch("obolha.compose_facecam") as mock_compose:
        mock_compose.return_value = CFG["reacts_dir"] / "Vid" / "01_test_facecam.mp4"
        results = auto_compose_reacts()

    assert len(results) == 1
    mock_compose.assert_called_once()
    assert mock_compose.call_args[0][0] == clip
    assert mock_compose.call_args[0][1].parent == pool

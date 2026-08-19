"""Tests for jaguar_sepia_filter module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from jaguar_sepia_filter import (
    apply_jaguar_sepia_filter,
    build_gradient_lut,
    build_tone_maps,
    generate_procedural_leopard_texture,
    load_or_tile_texture,
    overlay_blend,
    process_video,
    shadow_mask,
    smoothstep,
)


@pytest.fixture(autouse=True)
def mock_video_encoder(monkeypatch):
    monkeypatch.setattr("obolha.get_video_encoder", lambda: "libx264")


def test_build_gradient_lut_shape_and_endpoints():
    lut = build_gradient_lut((20, 30, 45), (40, 120, 180), (140, 210, 245))
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    np.testing.assert_array_equal(lut[0], [20, 30, 45])
    np.testing.assert_array_equal(lut[255], [140, 210, 245])


def test_build_tone_maps_has_at_most_16_colors():
    lut = build_gradient_lut((0, 0, 0), (42, 168, 238), (72, 248, 255))
    gray_to_bgr = build_tone_maps(lut, max_tones=16)
    assert gray_to_bgr.shape == (256, 3)
    assert len(np.unique(gray_to_bgr.reshape(-1, 3), axis=0)) <= 16


def test_smoothstep_edges():
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = smoothstep(x, 0.25, 0.75)
    assert y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(0.0)
    assert y[3] == pytest.approx(1.0)
    assert y[4] == pytest.approx(1.0)
    assert 0.0 < y[2] < 1.0


def test_shadow_mask_dark_pixels_high():
  l = np.array([[0.0, 0.5, 1.0]])
  mask = shadow_mask(l, low_threshold=60, high_threshold=140, feather_radius=0)
  assert mask[0, 0] == pytest.approx(1.0, abs=0.05)
  assert mask[0, 2] == pytest.approx(0.0, abs=0.05)


def test_overlay_blend_midpoint():
    base = np.full((1, 1, 3), 128, dtype=np.uint8)
    blend = np.full((1, 1, 3), 64, dtype=np.uint8)
    out = overlay_blend(base, blend)
    assert out.shape == base.shape
    assert out.dtype == np.uint8


def test_absolute_black_mask_tight():
    from jaguar_sepia_filter import absolute_black_mask

    l = np.array([[0.0, 0.04, 0.5, 1.0]])
    mask = absolute_black_mask(l, black_level=18, feather_radius=0)
    assert mask[0, 0] == pytest.approx(1.0, abs=0.05)
    assert mask[0, 2] == pytest.approx(0.0, abs=0.05)
    assert mask[0, 3] == pytest.approx(0.0, abs=0.05)


def test_texture_does_not_alter_bright_areas():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :16] = 210
    frame[:, 16:] = 10
    texture = np.full((32, 32, 3), (30, 40, 50), dtype=np.uint8)
    with_tex = apply_jaguar_sepia_filter(frame, texture, enable_texture=True, low_threshold=10)
    no_tex = apply_jaguar_sepia_filter(frame, texture, enable_texture=False, low_threshold=10)
    np.testing.assert_array_equal(with_tex[:, :16], no_tex[:, :16])

    l = frame[:, :16, 0].astype(np.float32) / 255.0
    from jaguar_sepia_filter import absolute_black_mask
    mask = absolute_black_mask(l, black_level=10)
    assert float(mask.max()) == pytest.approx(0.0, abs=0.05)


def test_quantize_luma_index_limits_levels():
    from jaguar_sepia_filter import _quantize_luma_index

    gray = np.arange(256, dtype=np.uint8)
    q = _quantize_luma_index(gray, max_tones=16)
    assert len(np.unique(q)) <= 16


def test_apply_jaguar_sepia_filter_max_16_tones():
    frame = np.random.default_rng(1).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    texture = generate_procedural_leopard_texture(128, 128, seed=3)
    out = apply_jaguar_sepia_filter(frame, texture, max_tones=16, enable_texture=True)
    unique = np.unique(out.reshape(-1, 3), axis=0)
    assert len(unique) <= 16


def test_apply_jaguar_sepia_filter_changes_frame():
    frame = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    texture = generate_procedural_leopard_texture(64, 64, seed=42)
    out = apply_jaguar_sepia_filter(frame, texture)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)


def test_procedural_texture_is_reproducible():
    a = generate_procedural_leopard_texture(32, 32, seed=7)
    b = generate_procedural_leopard_texture(32, 32, seed=7)
    np.testing.assert_array_equal(a, b)


def test_load_or_tile_texture_procedural_when_no_path():
    tiled = load_or_tile_texture(None, 100, 80)
    assert tiled.shape == (80, 100, 3)
    assert tiled.dtype == np.uint8


def test_open_video_writer_uses_software_codec(tmp_path):
    from jaguar_sepia_filter import _open_video_writer

    out = tmp_path / "silent.mp4"
    writer = _open_video_writer(str(out), 30.0, 64, 64)
    try:
        assert writer.isOpened()
    finally:
        writer.release()


def test_process_video_filters_unknown_filter_params():
    from jaguar_sepia_filter import _filter_kwargs

    got = _filter_kwargs({"contrast": 1.2, "show_progress": False, "texture_path": "x"})
    assert got == {"contrast": 1.2}
    assert "show_progress" not in got


def _fake_ffmpeg_pipe():
    fake_proc = MagicMock()
    fake_proc.stdin = MagicMock()
    fake_proc.stderr = io.BytesIO(b"")
    fake_proc.wait.return_value = 0
    return fake_proc


def test_normalize_frame_resizes_mismatch():
    from jaguar_sepia_filter import _normalize_frame

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = _normalize_frame(frame, 1080, 1080)
    assert out.shape == (1080, 1080, 3)


def test_ffmpeg_lut_filter_vf_quotes_paths(tmp_path):
    from jaguar_sepia_filter import _ffmpeg_lut_filter_vf

    vf = _ffmpeg_lut_filter_vf(tmp_path)
    assert "lutrgb=r='" in vf
    assert ":g='" in vf
    assert ":b='" in vf


def test_write_frame_to_pipe_raises_on_broken_pipe():
    from jaguar_sepia_filter import _write_frame_to_pipe

    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write.side_effect = BrokenPipeError()
    proc.stdin.closed = False
    proc.wait.return_value = 1

    with pytest.raises(RuntimeError, match="ffmpeg pipe encode failed"):
        _write_frame_to_pipe(proc, np.zeros((4, 4, 3), dtype=np.uint8), 4, 4)


def test_ffmpeg_video_encode_args_uses_detected_encoder(monkeypatch):
    from jaguar_sepia_filter import _ffmpeg_video_encode_args

    monkeypatch.setattr("obolha.get_video_encoder", lambda: "libopenh264")
    args = _ffmpeg_video_encode_args()
    assert "libopenh264" in args
    assert "libx264" not in args
    assert "yuv420p" in args


def test_process_video_ffmpeg_lut_fast_path(tmp_path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"fake")

    def fake_run(cmd, **kwargs):
        out = Path(cmd[-1])
        out.write_bytes(b"processed")
        return MagicMock(returncode=0, stderr="")

    with patch("jaguar_sepia_filter.subprocess.run", side_effect=fake_run):
        process_video(str(input_path), str(output_path), enable_texture=False, show_progress=False)

    assert output_path.is_file()


def test_process_video_pipe_remux_error_propagates(tmp_path):
    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    inp.write_bytes(b"\x00" * 2048)

    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.get.side_effect = lambda prop: {
        getattr(cv2, "CAP_PROP_FRAME_WIDTH"): 4,
        getattr(cv2, "CAP_PROP_FRAME_HEIGHT"): 4,
        getattr(cv2, "CAP_PROP_FPS"): 30.0,
        getattr(cv2, "CAP_PROP_FRAME_COUNT"): 1,
        getattr(cv2, "CAP_PROP_FOURCC"): 0,
    }.get(prop, 0)
    fake_cap.read.side_effect = [
        (True, np.zeros((4, 4, 3), dtype=np.uint8)),
        (False, None),
    ]

    def fake_popen(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"silent")
        return _fake_ffmpeg_pipe()

    with (
        patch("jaguar_sepia_filter._process_video_ffmpeg_lut", side_effect=RuntimeError("skip")),
        patch("jaguar_sepia_filter.cv2.VideoCapture", return_value=fake_cap),
        patch("jaguar_sepia_filter.subprocess.Popen", side_effect=fake_popen),
        patch("jaguar_sepia_filter._remux_audio", side_effect=RuntimeError("boom")),
        patch("jaguar_sepia_filter.tqdm", side_effect=lambda x, **_: x),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            process_video(str(inp), str(out), enable_texture=True, show_progress=False)


def test_process_video_pipe_writes_frames(tmp_path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    input_path.write_bytes(b"fake")

    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.get.side_effect = lambda prop: {
        getattr(cv2, "CAP_PROP_FRAME_WIDTH"): 4,
        getattr(cv2, "CAP_PROP_FRAME_HEIGHT"): 4,
        getattr(cv2, "CAP_PROP_FPS"): 30.0,
        getattr(cv2, "CAP_PROP_FRAME_COUNT"): 2,
        getattr(cv2, "CAP_PROP_FOURCC"): int.from_bytes(b"mp4v", "little"),
    }.get(prop, 0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    fake_cap.read.side_effect = [(True, frame.copy()), (True, frame.copy()), (False, None)]

    def fake_popen(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"silent")
        return _fake_ffmpeg_pipe()

    def fake_remux(_src: str, _silent: str, final: str) -> None:
        Path(final).write_bytes(b"processed")

    with (
        patch("jaguar_sepia_filter._process_video_ffmpeg_lut", side_effect=RuntimeError("skip")),
        patch("jaguar_sepia_filter.cv2.VideoCapture", return_value=fake_cap),
        patch("jaguar_sepia_filter.subprocess.Popen", side_effect=fake_popen) as popen,
        patch("jaguar_sepia_filter._remux_audio", side_effect=fake_remux) as remux,
        patch("jaguar_sepia_filter.tqdm", side_effect=lambda x, **_: x),
    ):
        process_video(str(input_path), str(output_path), enable_texture=True, show_progress=False)

    assert popen.called
    assert remux.called
    assert output_path.is_file()

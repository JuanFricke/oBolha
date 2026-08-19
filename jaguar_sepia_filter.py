"""Jaguar sepia duotone video filter — gradient map + leopard texture in shadows."""

from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

log = logging.getLogger(__name__)

_FILTER_PARAM_KEYS = frozenset({
    "shadow_color", "midtone_color", "highlight_color",
    "low_threshold", "high_threshold", "black_level", "feather_radius",
    "texture_opacity", "contrast", "brightness", "use_subject_mask", "enable_texture",
    "max_tones", "lut",
})
_LUT_PARAM_KEYS = frozenset({"shadow_color", "midtone_color", "highlight_color"})


def _filter_kwargs(filter_params: dict) -> dict:
    return {k: v for k, v in filter_params.items() if k in _FILTER_PARAM_KEYS}


def _open_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """OpenCV VideoWriter — never reuse input H.264 fourcc (breaks on many Linux setups)."""
    for codec in ("mp4v", "avc1", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not open VideoWriter (tried mp4v, avc1, XVID, MJPG)")


def build_gradient_lut(
    shadow_color: tuple[int, int, int],
    midtone_color: tuple[int, int, int],
    highlight_color: tuple[int, int, int],
) -> np.ndarray:
    """Build a 256-entry BGR LUT interpolating shadow → midtone → highlight."""
    indices = np.arange(256, dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.float32)
    for ch in range(3):
        low = np.where(
            indices <= 128,
            shadow_color[ch] + (midtone_color[ch] - shadow_color[ch]) * (indices / 128.0),
            midtone_color[ch]
            + (highlight_color[ch] - midtone_color[ch]) * ((indices - 128.0) / 127.0),
        )
        lut[:, ch] = low
    return np.clip(lut, 0, 255).astype(np.uint8)


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Hermite smoothstep between edge0 and edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _quantize_luma_index(index: np.ndarray, max_tones: int = 16) -> np.ndarray:
    """Posterize luminance index to at most max_tones discrete levels."""
    levels = int(np.clip(max_tones, 2, 256))
    if levels >= 256:
        return index
    q = np.floor(index.astype(np.float32) / 256.0 * levels)
    q = np.clip(q, 0, levels - 1)
    return (q / (levels - 1) * 255.0).astype(np.uint8)


def _palette_from_lut(lut: np.ndarray, max_tones: int) -> np.ndarray:
    indices = np.linspace(0, 255, int(np.clip(max_tones, 2, 256)), dtype=np.uint8)
    palette = lut[indices]
    return np.unique(palette, axis=0)


def build_tone_maps(
    lut: np.ndarray,
    max_tones: int = 16,
    contrast: float = 1.65,
) -> np.ndarray:
    """Precompute gray (0-255) → BGR poster color for fast per-frame lookup."""
    gray_to_bgr = np.zeros((256, 3), dtype=np.uint8)
    for g in range(256):
        idx = _quantize_luma_index(
            np.array([_prepare_luma_index(np.array([g], dtype=np.uint8), contrast=contrast)[0]]),
            max_tones=max_tones,
        )[0]
        gray_to_bgr[g] = lut[idx]
    return gray_to_bgr


def _write_ffmpeg_lut_file(path: Path, channel: np.ndarray) -> None:
    path.write_text("\n".join(str(int(v)) for v in channel) + "\n")


def _remap_posterized(frame_bgr: np.ndarray, gray_to_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray_to_bgr[gray]


def _snap_to_palette(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map each pixel to the nearest BGR color in the palette."""
    if palette.shape[0] <= 1:
        return image
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)
    pal = palette.astype(np.float32)
    dist = ((pixels[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    nearest = dist.argmin(axis=1)
    return pal[nearest].reshape(h, w, 3).astype(np.uint8)


def _prepare_luma_index(gray: np.ndarray, contrast: float = 1.65, pivot: float = 0.36) -> np.ndarray:
    """Stretch luminance before LUT for a punchy duotone."""
    g = gray.astype(np.float32) / 255.0
    g = np.clip((g - pivot) * contrast + pivot, 0.0, 1.0)
    return (g * 255.0).astype(np.uint8)


def absolute_black_mask(
    luminance: np.ndarray,
    black_level: int = 18,
    feather_radius: float = 0,
) -> np.ndarray:
    """Mask = 1 only in near-absolute black (gray < black_level)."""
    level = black_level / 255.0
    mask = 1.0 - smoothstep(luminance, level * 0.35, level)
    if feather_radius > 0:
        mask = gaussian_filter(mask, sigma=feather_radius)
    return np.clip(mask, 0.0, 1.0)


def shadow_mask(
    luminance: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    feather_radius: float,
) -> np.ndarray:
    """Shadow mask: 1 in dark areas, 0 in highlights, with optional feather."""
    low = low_threshold / 255.0
    high = high_threshold / 255.0
    mask = 1.0 - smoothstep(luminance, low, high)
    if feather_radius > 0:
        mask = gaussian_filter(mask, sigma=feather_radius)
    return np.clip(mask, 0.0, 1.0)


def overlay_blend(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Photoshop-style overlay blend on uint8 BGR images."""
    b = base.astype(np.float32) / 255.0
    s = blend.astype(np.float32) / 255.0
    out = np.where(s < 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def generate_procedural_leopard_texture(
    width: int,
    height: int,
    seed: int = 42,
) -> np.ndarray:
    """Tileable-ish leopard spot texture on an earth-tone base (static per video)."""
    rng = np.random.default_rng(seed)
    base_color = np.array([38, 58, 82], dtype=np.uint8)  # warm dark BGR base
    texture = np.tile(base_color, (height, width, 1))

    n_spots = max(8, (width * height) // 800)
    for _ in range(n_spots):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        rx = int(rng.integers(max(4, width // 20), max(8, width // 8)))
        ry = int(rng.integers(max(3, height // 25), max(6, height // 10)))
        angle = float(rng.uniform(0, 180))
        spot = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(spot, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
        ring = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(ring, (cx, cy), (rx + 2, ry + 2), angle, 0, 360, 255, max(1, rx // 6))
        ring = cv2.subtract(ring, spot)
        texture[spot > 0] = (6, 8, 12)
        texture[ring > 0] = (3, 4, 8)

    return texture


def load_or_tile_texture(
    texture_path: Optional[str | Path],
    frame_width: int,
    frame_height: int,
    procedural_seed: int = 42,
) -> np.ndarray:
    """Load a texture file or generate procedural spots, tiled to frame size."""
    if texture_path:
        path = Path(texture_path)
        if not path.is_file():
            raise FileNotFoundError(f"Texture not found: {path}")
        tile = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tile is None:
            raise ValueError(f"Could not read texture image: {path}")
    else:
        tile = generate_procedural_leopard_texture(
            max(64, frame_width // 4),
            max(64, frame_height // 4),
            seed=procedural_seed,
        )

    th, tw = tile.shape[:2]
    reps_y = int(np.ceil(frame_height / th))
    reps_x = int(np.ceil(frame_width / tw))
    tiled = np.tile(tile, (reps_y, reps_x, 1))
    return tiled[:frame_height, :frame_width]


def _apply_gradient_map(
    frame: np.ndarray,
    lut: np.ndarray,
    contrast: float = 1.65,
    max_tones: int = 16,
) -> np.ndarray:
    """Remap frame colors using contrast-stretched, posterized luminance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    index = _prepare_luma_index(gray, contrast=contrast)
    index = _quantize_luma_index(index, max_tones=max_tones)
    return lut[index]


def _luminance(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _optional_subject_mask(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return a 0–1 mask protecting skin/face when mediapipe is available."""
    try:
        import mediapipe as mp
    except ImportError:
        return None

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = seg.process(rgb)
        if result.segmentation_mask is None:
            return None
        # Protect person (foreground) from heavy texture / sepia on skin
        return result.segmentation_mask.astype(np.float32)


def apply_jaguar_sepia_filter(
    frame: np.ndarray,
    texture: np.ndarray,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    midtone_color: tuple[int, int, int] = (42, 168, 238),
    highlight_color: tuple[int, int, int] = (72, 248, 255),
    low_threshold: int = 10,
    high_threshold: int = 10,
    black_level: int | None = None,
    feather_radius: float = 0,
    texture_opacity: float = 0.85,
    contrast: float = 1.65,
    brightness: int = 0,
    use_subject_mask: bool = False,
    enable_texture: bool = True,
    max_tones: int = 16,
    lut: Optional[np.ndarray] = None,
    gray_to_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply high-contrast gold duotone; leopard texture only in absolute black."""
    if lut is None:
        lut = build_gradient_lut(shadow_color, midtone_color, highlight_color)
    if gray_to_bgr is None:
        gray_to_bgr = build_tone_maps(lut, max_tones=max_tones, contrast=contrast)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    l = gray.astype(np.float32) / 255.0
    frame_gm = gray_to_bgr[gray]

    black_cutoff = black_level if black_level is not None else low_threshold
    frame_final = frame_gm

    if enable_texture and texture_opacity > 0:
        tex = texture if texture.shape[:2] == frame.shape[:2] else cv2.resize(
            texture, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        textured = (frame_gm.astype(np.float32) * tex.astype(np.float32) / 255.0)
        mask = absolute_black_mask(l, black_level=black_cutoff, feather_radius=feather_radius)
        mask_3 = mask[..., np.newaxis]
        frame_final = (
            frame_gm.astype(np.float32) * (1.0 - mask_3 * texture_opacity)
            + textured * (mask_3 * texture_opacity)
        )
        frame_final = np.clip(frame_final, 0, 255).astype(np.uint8)
        frame_final = _remap_posterized(frame_final, gray_to_bgr)

    if use_subject_mask:
        subj = _optional_subject_mask(frame)
        if subj is not None:
            subj_3 = subj[..., np.newaxis]
            frame_final = (
                frame_final.astype(np.float32) * (1.0 - subj_3 * 0.6)
                + frame.astype(np.float32) * (subj_3 * 0.6)
            ).astype(np.uint8)

    if brightness:
        return cv2.convertScaleAbs(frame_final, alpha=1.0, beta=brightness)
    return frame_final


def _even_dim(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def _normalize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Ensure frame matches encoder size and memory layout."""
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _ffmpeg_lut_filter_vf(tmp_path: Path) -> str:
    """Build lutrgb filter with quoted absolute paths."""
    r = (tmp_path / "r.txt").resolve()
    g = (tmp_path / "g.txt").resolve()
    b = (tmp_path / "b.txt").resolve()
    return f"hue=s=0,lutrgb=r='{r}':g='{g}':b='{b}'"


def _pipe_encode_failure(proc: subprocess.Popen, detail: str) -> RuntimeError:
    if proc.stdin and not proc.stdin.closed:
        try:
            proc.stdin.close()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    code = proc.returncode
    msg = detail if detail else f"exit code {code}"
    return RuntimeError(f"ffmpeg pipe encode failed: {msg}")


def _write_frame_to_pipe(
    proc: subprocess.Popen,
    frame: np.ndarray,
    width: int,
    height: int,
) -> None:
    assert proc.stdin is not None
    data = _normalize_frame(frame, width, height).tobytes()
    try:
        proc.stdin.write(data)
    except BrokenPipeError as exc:
        raise _pipe_encode_failure(proc, str(exc)) from exc


def _ffmpeg_video_encode_args() -> list[str]:
    from obolha import get_video_encoder, video_encode_args

    return [*video_encode_args(get_video_encoder()), "-pix_fmt", "yuv420p"]


def _remux_audio(source_video: str, processed_video: str, output_video: str) -> None:
    """Copy filtered video stream and map audio from source (no video re-encode)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", source_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_video,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio remux failed: {(result.stderr or '')[-400:]}")


def _process_video_ffmpeg_lut(
    input_path: Path,
    output_path: Path,
    gray_to_bgr: np.ndarray,
) -> Path:
    """Apply poster LUT in a single ffmpeg pass (fast path, no texture)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_ffmpeg_lut_file(tmp_path / "b.txt", gray_to_bgr[:, 0])
        _write_ffmpeg_lut_file(tmp_path / "g.txt", gray_to_bgr[:, 1])
        _write_ffmpeg_lut_file(tmp_path / "r.txt", gray_to_bgr[:, 2])
        vf = _ffmpeg_lut_filter_vf(tmp_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf,
            *_ffmpeg_video_encode_args(),
            "-c:a", "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg lut filter failed: {(result.stderr or '')[-800:]}")
    return output_path


def _encode_frames_via_ffmpeg_pipe(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    show_progress: bool,
    read_frame,
) -> None:
    """Encode filtered frames from read_frame() -> ndarray | None."""
    width = _even_dim(width)
    height = _even_dim(height)
    silent_path = output_path.with_suffix(".silent.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
        "-i", "pipe:0",
        *_ffmpeg_video_encode_args(),
        str(silent_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    written = 0
    pbar = tqdm(desc="jaguar_sepia", unit="frame") if show_progress else None
    try:
        while True:
            frame = read_frame()
            if frame is None:
                break
            _write_frame_to_pipe(proc, frame, width, height)
            written += 1
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        try:
            proc.stdin.close()
        except Exception:
            pass

    if proc.wait() != 0 or not silent_path.is_file() or written == 0:
        silent_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg pipe encode failed: exit={proc.returncode} frames={written}"
        )

    try:
        _remux_audio(str(input_path), str(silent_path), str(output_path))
    finally:
        silent_path.unlink(missing_ok=True)


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    texture_path: Optional[str | Path] = None,
    show_progress: bool = True,
    **filter_params,
) -> Path:
    """Process a full video and preserve original audio."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    frame_kwargs = _filter_kwargs(filter_params)
    lut_params = {k: filter_params[k] for k in _LUT_PARAM_KEYS if k in filter_params}
    default_lut = (
        (0, 0, 0),
        (42, 168, 238),
        (72, 248, 255),
    )
    lut = build_gradient_lut(
        lut_params.get("shadow_color", default_lut[0]),
        lut_params.get("midtone_color", default_lut[1]),
        lut_params.get("highlight_color", default_lut[2]),
    )
    max_tones = int(frame_kwargs.get("max_tones", 16))
    contrast = float(frame_kwargs.get("contrast", 1.65))
    gray_to_bgr = build_tone_maps(lut, max_tones=max_tones, contrast=contrast)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enable_texture = bool(frame_kwargs.get("enable_texture", True))
    use_subject_mask = bool(frame_kwargs.get("use_subject_mask", False))
    if not enable_texture and not use_subject_mask:
        try:
            return _process_video_ffmpeg_lut(input_path, output_path, gray_to_bgr)
        except Exception as exc:
            log.warning("ffmpeg LUT fast path failed, falling back to pipe: %s", exc)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = _even_dim(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = _even_dim(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    texture = load_or_tile_texture(texture_path, width, height)

    def read_frame() -> np.ndarray | None:
        ok, frame = cap.read()
        if not ok:
            return None
        return apply_jaguar_sepia_filter(
            frame,
            texture,
            lut=lut,
            gray_to_bgr=gray_to_bgr,
            **frame_kwargs,
        )

    try:
        _encode_frames_via_ffmpeg_pipe(
            input_path,
            output_path,
            width,
            height,
            fps,
            show_progress,
            read_frame,
        )
    finally:
        cap.release()

    return output_path


def preview_frame(
    input_path: str | Path,
    output_path: str | Path,
    frame_index: int = 0,
    texture_path: Optional[str | Path] = None,
    **filter_params,
) -> Path:
    """Apply the filter to a single frame and save as PNG for quick tuning."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {input_path}")
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {input_path}")

    h, w = frame.shape[:2]
    texture = load_or_tile_texture(texture_path, w, h)
    out = apply_jaguar_sepia_filter(frame, texture, **filter_params)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    return output_path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Jaguar sepia duotone video filter")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("-t", "--texture", default=None, help="Leopard texture image (optional)")
    parser.add_argument("--preview-frame", type=int, metavar="N", help="Preview single frame as PNG")
    parser.add_argument("--low-threshold", type=int, default=10, help="Gray level below which texture applies")
    parser.add_argument("--high-threshold", type=int, default=18, help=argparse.SUPPRESS)
    parser.add_argument("--texture-opacity", type=float, default=0.9)
    parser.add_argument("--contrast", type=float, default=1.65)
    parser.add_argument("--no-texture", action="store_true", help="Disable leopard texture overlay")
    parser.add_argument("--max-tones", type=int, default=16, help="Max posterized color tones (default: 16)")
    parser.add_argument("--subject-mask", action="store_true", help="Protect face/skin (mediapipe)")
    args = parser.parse_args()

    params = {
        "low_threshold": args.low_threshold,
        "high_threshold": args.high_threshold,
        "texture_opacity": args.texture_opacity,
        "contrast": args.contrast,
        "max_tones": args.max_tones,
        "use_subject_mask": args.subject_mask,
        "enable_texture": not args.no_texture,
    }

    if args.preview_frame is not None:
        preview_path = Path(args.output)
        if preview_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            preview_path = preview_path.with_suffix(".png")
        out = preview_frame(args.input, preview_path, args.preview_frame, args.texture, **params)
        print(f"Preview saved: {out}")
    else:
        out = process_video(args.input, args.output, args.texture, **params)
        print(f"Video saved: {out}")


if __name__ == "__main__":
    _cli()

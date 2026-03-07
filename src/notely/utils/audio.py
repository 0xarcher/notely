"""
Audio processing utilities.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def extract_audio(
    video_path: Path | str,
    output_path: Path | str | None = None,
    audio_format: str = "wav",
    sample_rate: int = 16000,
) -> Path:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to the video file.
        output_path: Path for the output audio file. If None, creates a temp file.
        audio_format: Output audio format (wav, mp3, etc.)
        sample_rate: Sample rate for the audio.

    Returns:
        Path to the extracted audio file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=f".{audio_format}"))
    else:
        output_path = Path(output_path)

    # Use ffmpeg to extract audio
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-i",
        str(video_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le" if audio_format == "wav" else "copy",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",  # Mono channel
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}") from e

    return output_path


def get_audio_duration(audio_path: Path | str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    from notely.utils.common import get_media_duration

    return get_media_duration(audio_path)


def normalize_audio(
    audio_path: Path | str,
    output_path: Path | str | None = None,
    target_db: float = -20.0,
) -> Path:
    """
    Normalize audio volume.

    Args:
        audio_path: Path to the audio file.
        output_path: Path for the output file. If None, modifies in place.
        target_db: Target dB level.

    Returns:
        Path to the normalized audio file.
    """
    audio_path = Path(audio_path)

    if output_path is None:
        output_path = audio_path.with_suffix(".normalized.wav")
    else:
        output_path = Path(output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-af",
        f"loudnorm=I={target_db}:TP=-1.5:LRA=11",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to normalize audio: {e.stderr.decode()}") from e

    return output_path


def split_audio(
    audio_path: Path | str,
    chunk_duration: int = 600,  # 10 minutes
    output_dir: Path | str | None = None,
) -> list[Path]:
    """
    Split audio into chunks for processing long recordings.

    Args:
        audio_path: Path to the audio file.
        chunk_duration: Duration of each chunk in seconds.
        output_dir: Directory for output chunks. If None, creates temp dir.

    Returns:
        List of paths to audio chunks.
    """
    audio_path = Path(audio_path)
    duration = get_audio_duration(audio_path)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for i, start in enumerate(range(0, int(duration), chunk_duration)):
        output_path = output_dir / f"chunk_{i:04d}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-t",
            str(chunk_duration),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            chunks.append(output_path)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to split audio: {e.stderr.decode()}") from e

    return chunks

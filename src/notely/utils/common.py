"""
Common utility functions.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def temp_file(suffix: str = "", prefix: str = "notely_") -> Generator[Path, None, None]:
    """
    Context manager for creating a temporary file.

    Args:
        suffix: File suffix/extension.
        prefix: File prefix.

    Yields:
        Path to the temporary file.
    """
    fd, filepath = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    try:
        os.close(fd)
        yield Path(filepath)
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def get_media_duration(media_path: Path | str) -> float:
    """
    Get the duration of a media file (audio or video) in seconds.

    Uses ffprobe to extract duration information.

    Args:
        media_path: Path to the media file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If duration cannot be determined.
    """
    media_path = Path(media_path)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get media duration: {e}") from e


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS or MM:SS.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(timestamp: str) -> float:
    """
    Parse a timestamp string to seconds.

    Args:
        timestamp: Time string (HH:MM:SS or MM:SS).

    Returns:
        Time in seconds.
    """
    parts = timestamp.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    raise ValueError(f"Invalid timestamp format: {timestamp}")

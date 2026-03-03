"""
Video processing utilities.
"""

from __future__ import annotations

from typing import Union

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class KeyFrame:
    """A key frame extracted from a video."""

    path: Path
    timestamp: float  # seconds
    width: int
    height: int

    @property
    def dimensions(self) -> tuple[int, int]:
        """Get frame dimensions as (width, height)."""
        return self.width, self.height


def get_video_duration(video_path: Union[Path, str]) -> float:
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds.
    """
    from notely.utils.common import get_media_duration

    return get_media_duration(video_path)


def get_video_info(video_path: Union[Path, str]) -> dict:
    """
    Get detailed video information.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with video metadata.
    """
    video_path = Path(video_path)

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to get video info: {e}") from e


def extract_frame(
    video_path: Union[Path, str], timestamp: float, output_path: Union[Path, str, None] = None
) -> Path:
    """
    Extract a single frame from a video at a specific timestamp.

    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds.
        output_path: Path for the output image. If None, creates a temp file.

    Returns:
        Path to the extracted frame.
    """
    video_path = Path(video_path)

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".png"))
    else:
        output_path = Path(output_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-vframes",
        "1",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract frame: {e.stderr.decode()}") from e

    return output_path


def extract_frames(
    video_path: Union[Path, str],
    interval_seconds: float = 1.0,
    output_dir: Union[Path, str, None] = None,
) -> list[Path]:
    """
    Extract frames from a video at regular intervals.

    Args:
        video_path: Path to the video file.
        interval_seconds: Interval between frames in seconds.
        output_dir: Directory for output frames. If None, creates temp dir.

    Returns:
        List of paths to extracted frames.
    """
    video_path = Path(video_path)
    duration = get_video_duration(video_path)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, timestamp in enumerate(np.arange(0, duration, interval_seconds).tolist()):
        output_path = output_dir / f"frame_{i:04d}.png"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            frames.append(output_path)
        except subprocess.CalledProcessError:
            continue  # Skip failed frames

    return frames


def extract_key_frames(
    video_path: Union[Path, str],
    interval_seconds: float = 5.0,
    min_similarity: float = 0.85,
    output_dir: Union[Path, str, None] = None,
) -> list[KeyFrame]:
    """
    Extract key frames from a video, filtering out similar frames.

    This is useful for extracting presentation slides from lecture videos.
    Similar consecutive frames (e.g., static slides) are deduplicated.

    Args:
        video_path: Path to the video file.
        interval_seconds: Base interval for frame extraction.
        min_similarity: Minimum similarity threshold for deduplication.
        output_dir: Directory for output frames. If None, creates temp dir.

    Returns:
        List of KeyFrame objects.
    """
    video_path = Path(video_path)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # First, extract all frames at interval
    all_frames = extract_frames(video_path, interval_seconds, output_dir)

    # Then, filter similar frames
    key_frames = []
    prev_image = None

    for i, frame_path in enumerate(all_frames):
        current_image = Image.open(frame_path).convert("RGB")
        current_array = np.array(current_image)

        timestamp = i * interval_seconds

        # Check similarity with previous frame
        if prev_image is not None:
            similarity = _compute_image_similarity(prev_image, current_array)
            if similarity >= min_similarity:
                # Skip similar frame
                frame_path.unlink(missing_ok=True)
                continue

        # This is a key frame
        key_frames.append(
            KeyFrame(
                path=frame_path,
                timestamp=timestamp,
                width=current_image.width,
                height=current_image.height,
            )
        )
        prev_image = current_array

    return key_frames


def _compute_image_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute similarity between two images using histogram comparison.

    Args:
        img1: First image as numpy array.
        img2: Second image as numpy array.

    Returns:
        Similarity score between 0 and 1.
    """
    # Resize images to same size if needed
    if img1.shape != img2.shape:
        from PIL import Image

        img2_pil = Image.fromarray(img2)
        img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]))
        img2 = np.array(img2_pil)

    # Compute histograms
    hist1 = _compute_histogram(img1)
    hist2 = _compute_histogram(img2)

    # Compare histograms using correlation
    correlation = float(np.corrcoef(hist1.flatten(), hist2.flatten())[0, 1])

    return max(0.0, min(1.0, (correlation + 1) / 2))  # Normalize to [0, 1]


def _compute_histogram(image: np.ndarray) -> np.ndarray:
    """Compute color histogram for an image.

    Returns:
        np.ndarray:
    """
    hist = np.zeros((8, 8, 8))

    # Downsample colors to 8 bins per channel
    downsampled = (image // 32).astype(np.int32)

    # Count pixels in each bin
    for i in range(downsampled.shape[0]):
        for j in range(downsampled.shape[1]):
            r, g, b = downsampled[i, j]
            hist[r, g, b] += 1

    # Normalize
    hist = hist / hist.sum()

    return hist


class VideoProcessor:
    """
    Video processor for extracting frames and audio.

    This class provides a high-level interface for video processing
    operations commonly needed for lecture transcription.
    """

    def __init__(self, video_path: Union[Path, str]):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._info: Union[dict, None] = None

    @property
    def info(self) -> dict:
        """Get video information (cached)."""
        if self._info is None:
            self._info = get_video_info(self.video_path)
        return self._info

    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        return float(self.info.get("format", {}).get("duration", 0))

    @property
    def width(self) -> int:
        """Get video width."""
        for stream in self.info.get("streams", []):
            if stream.get("codec_type") == "video":
                return int(stream.get("width", 0))
        return 0

    @property
    def height(self) -> int:
        """Get video height."""
        for stream in self.info.get("streams", []):
            if stream.get("codec_type") == "video":
                return int(stream.get("height", 0))
        return 0

    def extract_audio(self, output_path: Union[Path, str, None] = None) -> Path:
        """Extract audio from the video."""
        from notely.utils.audio import extract_audio

        return extract_audio(self.video_path, output_path)

    def extract_key_frames(
        self,
        interval_seconds: float = 5.0,
        min_similarity: float = 0.85,
        output_dir: Union[Path, str, None] = None,
    ) -> list[KeyFrame]:
        """Extract key frames from the video."""
        return extract_key_frames(
            self.video_path,
            interval_seconds,
            min_similarity,
            output_dir,
        )

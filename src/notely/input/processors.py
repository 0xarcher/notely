"""
Input processors for different file formats.

This module provides unified processing functions for various input formats
(video, audio, PDF). Each processor extracts relevant content and returns
a standardized ProcessedInput object.
"""

from __future__ import annotations

from pathlib import Path

from notely.utils.audio import extract_audio, get_audio_duration
from notely.utils.video import extract_key_frames, get_video_duration

from .types import ProcessedInput


def process_video(
    path: Path,
    key_frame_interval: float = 5.0,
    min_similarity: float = 0.85,
) -> ProcessedInput:
    """
    Process video input: extract audio and key frames.

    Args:
        path: Path to the video file.
        key_frame_interval: Interval in seconds for extracting key frames.
        min_similarity: Minimum similarity threshold for frame deduplication.

    Returns:
        ProcessedInput with audio and frames.

    Raises:
        FileNotFoundError: If video file doesn't exist.
        RuntimeError: If extraction fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    # Extract audio
    audio_path = extract_audio(path)

    # Extract key frames
    frames = extract_key_frames(
        path,
        interval_seconds=key_frame_interval,
        min_similarity=min_similarity,
    )

    # Get metadata
    duration = get_video_duration(path)

    return ProcessedInput(
        audio_path=audio_path,
        frames=frames,
        metadata={
            "source_type": "video",
            "source_path": str(path),
            "duration": duration,
            "frame_count": len(frames),
        },
    )


def process_audio(path: Path) -> ProcessedInput:
    """
    Process audio input: use audio file directly.

    Args:
        path: Path to the audio file.

    Returns:
        ProcessedInput with audio path.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Get metadata
    duration = get_audio_duration(path)

    return ProcessedInput(
        audio_path=path,
        metadata={
            "source_type": "audio",
            "source_path": str(path),
            "duration": duration,
        },
    )


def process_pdf(path: Path) -> ProcessedInput:
    """
    Process PDF input: extract text and images from pages.

    Args:
        path: Path to the PDF file.

    Returns:
        ProcessedInput with PDF pages as OCR results.

    Raises:
        FileNotFoundError: If PDF file doesn't exist.
        RuntimeError: If PDF processing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    from .pdf import extract_pdf_pages

    # Extract pages
    pdf_pages = extract_pdf_pages(path)

    return ProcessedInput(
        pdf_pages=pdf_pages,
        metadata={
            "source_type": "pdf",
            "source_path": str(path),
            "page_count": len(pdf_pages),
        },
    )

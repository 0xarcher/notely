"""
Data types for input processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from notely.utils.video import KeyFrame


@dataclass
class ProcessedInput:
    """
    Unified result from processing different input formats.

    This data class encapsulates all extracted content from various input sources
    (video, audio, PDF, images) into a standardized format for downstream processing.

    Attributes:
        audio_path: Path to extracted or original audio file (for ASR).
        frames: List of key frames extracted from video (for OCR).
        pdf_pages: List of OCR results from PDF pages.
        metadata: Additional metadata about the input (duration, page count, etc.).
    """

    audio_path: Union[Path, None] = None
    frames: list[KeyFrame] = field(default_factory=list)
    pdf_pages: list[Any] = field(default_factory=list)  # OCRResult, avoid circular import
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_audio(self) -> bool:
        """Check if audio content is available."""
        return self.audio_path is not None and self.audio_path.exists()

    def has_visual_content(self) -> bool:
        """Check if visual content (frames or PDF pages) is available."""
        return len(self.frames) > 0 or len(self.pdf_pages) > 0

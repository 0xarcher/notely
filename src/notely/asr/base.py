"""
Base classes for ASR backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


@dataclass
class TranscriptSegment:
    """A single segment of transcribed text with timing information."""

    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float = 1.0
    speaker_id: Union[str, None] = None
    words: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
            "duration": self.duration,
        }


@dataclass
class ASRResult:
    """Result from ASR transcription."""

    segments: list[TranscriptSegment]
    language: str = "unknown"
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get the full transcribed text."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def text_with_timestamps(self) -> str:
        """Get text with timestamp markers."""
        lines = []
        for seg in self.segments:
            start = self._format_time(seg.start_time)
            lines.append(f"[{start}] {seg.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "full_text": self.full_text,
            "language": self.language,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def transcribe(self, audio_path: Union[Path, str]) -> ASRResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ASRResult containing the transcription.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and properly configured."""
        raise NotImplementedError

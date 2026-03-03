"""
OpenAI Whisper backend for ASR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from notely.asr.base import ASRBackend, ASRResult, TranscriptSegment


class WhisperBackend(ASRBackend):
    """
    OpenAI Whisper backend for ASR.

    Whisper provides robust ASR across many languages with:
    - Good accuracy on noisy audio
    - Multilingual support
    - Timestamp support
    - No internet required (local model)

    Args:
        model: Whisper model size ("tiny", "base", "small", "medium", "large")
        device: Device to use ("cuda", "cpu", "auto")
        language: Language code (e.g., "zh", "en"). None for auto-detect.
    """

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        language: Union[str, None] = None,
    ):
        self.model_size = model
        self.device = device
        self.language = language
        self._model = None

    def _load_model(self) -> Any:
        """Lazy load the Whisper model."""
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "Whisper is not installed. Install it with: pip install openai-whisper"
            )
        return whisper.load_model(self.model_size, device=self.device)

    @property
    def model(self) -> Any:
        """Get the model (lazy loading)."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def transcribe(self, audio_path: Union[Path, str]) -> ASRResult:
        """
        Transcribe audio using Whisper.

        Args:
            audio_path: Path to audio file.

        Returns:
            ASRResult with transcribed segments.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    text=seg["text"].strip(),
                    start_time=seg["start"],
                    end_time=seg["end"],
                    confidence=seg.get("avg_logprob", 0.0),
                    words=seg.get("words", []),
                )
            )

        return ASRResult(
            segments=segments,
            language=result.get("language", "unknown"),
            duration=max(seg.end_time for seg in segments) if segments else 0.0,
            metadata={"backend": "whisper", "model": self.model_size},
        )

    def is_available(self) -> bool:
        """Check if Whisper is available."""
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False


class WhisperAPIBackend(ASRBackend):
    """
    OpenAI Whisper API backend.

    Uses OpenAI's hosted Whisper API instead of local model.
    Requires OPENAI_API_KEY environment variable.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        model: Model to use ("whisper-1")
    """

    def __init__(self, api_key: Union[str, None] = None, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self) -> Any:
        """Get OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required")

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def transcribe(self, audio_path: Union[Path, str]) -> ASRResult:
        """Transcribe using OpenAI API."""
        audio_path = Path(audio_path)

        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        for seg in result.segments or []:
            segments.append(
                TranscriptSegment(
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                )
            )

        return ASRResult(
            segments=segments,
            language=result.language or "unknown",
            duration=result.duration or 0.0,
            metadata={"backend": "whisper-api", "model": self.model},
        )

    def is_available(self) -> bool:
        """Check if API key is configured."""
        import os

        return bool(self.api_key or os.getenv("OPENAI_API_KEY"))

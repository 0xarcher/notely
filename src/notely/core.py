"""
Core module for Notely - Multimodal lecture to Markdown transformation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from notely.asr import ASRBackend, ASRResult, FunASRBackend, WhisperBackend
from notely.formatter import MarkdownFormatter
from notely.llm import LLMBackend, OpenAIBackend
from notely.ocr import OCRBackend, OCRResult, PaddleOCRBackend
from notely.prompts import NoteTemplate, get_default_template
from notely.utils import (
    ensure_dir,
    extract_audio,
    extract_key_frames,
)


# Provider defaults
PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
    },
}


@dataclass
class LectureInput:
    """Input data for lecture processing."""

    video_path: Union[Path, None] = None
    audio_path: Union[Path, None] = None
    pdf_paths: list[Path] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    subtitle_path: Union[Path, None] = None  # SRT/VTT file if available
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert strings to Path objects
        if isinstance(self.video_path, str):
            self.video_path = Path(self.video_path)
        if isinstance(self.audio_path, str):
            self.audio_path = Path(self.audio_path)
        if isinstance(self.subtitle_path, str):
            self.subtitle_path = Path(self.subtitle_path)
        self.pdf_paths = [Path(p) if isinstance(p, str) else p for p in self.pdf_paths]
        self.image_paths = [Path(p) if isinstance(p, str) else p for p in self.image_paths]


@dataclass
class NotelyResult:
    """Result of lecture processing."""

    markdown: str
    thinking_process: str
    transcript: ASRResult
    ocr_results: list[OCRResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_path: Union[Path, str]) -> None:
        """Save the markdown result to a file."""
        path = Path(output_path)
        ensure_dir(path.parent)
        path.write_text(self.markdown, encoding="utf-8")


class Notely:
    """
    Notely - Transform multimodal lectures into structured Markdown notes.

    This SDK provides a complete pipeline for:
    1. Extracting audio from video
    2. Transcribing speech (ASR)
    3. Extracting text from slides (OCR)
    4. Fusing multimodal information
    5. Generating beautiful, structured Markdown notes

    Examples:
        # Zero-config (reads OPENAI_API_KEY from environment)
        >>> notely = Notely()
        >>> result = notely.process("lecture.mp4")

        # Specify API key
        >>> notely = Notely(api_key="sk-xxx")

        # Switch provider
        >>> notely = Notely(provider="zhipu", model="glm-4")

        # Full configuration
        >>> notely = Notely(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     asr_backend="funasr",
        ...     ocr_backend="paddleocr",
        ... )
    """

    def __init__(
        self,
        # LLM configuration (core)
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4o",
        base_url: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        # ASR configuration
        asr_backend: str = "funasr",
        asr_device: str = "cuda",
        asr_model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        # OCR configuration
        ocr_backend: str = "paddleocr",
        ocr_lang: str = "ch",
        # Processing settings
        key_frame_interval_seconds: float = 5.0,
        min_frame_similarity: float = 0.85,
        # Other settings
        template: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize Notely SDK.

        Args:
            api_key: LLM API key (required)
            provider: LLM provider ("openai", "zhipu", "anthropic", etc.)
            model: Model name (e.g., "gpt-4o", "glm-4")
            base_url: API base URL (auto-set if not provided)
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum tokens for LLM response
            asr_backend: ASR backend ("funasr", "whisper")
            asr_device: Device for ASR ("cuda", "cpu")
            asr_model: ASR model name
            ocr_backend: OCR backend ("paddleocr")
            ocr_lang: OCR language ("ch", "en")
            key_frame_interval_seconds: Interval for key frame extraction
            min_frame_similarity: Minimum similarity for frame deduplication
            template: Note template name
            verbose: Enable verbose logging
        """
        # LLM configuration
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self._get_base_url(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # ASR/OCR configuration
        self.asr_backend = asr_backend
        self.asr_device = asr_device
        self.asr_model = asr_model
        self.ocr_backend = ocr_backend
        self.ocr_lang = ocr_lang

        # Processing settings
        self.key_frame_interval_seconds = key_frame_interval_seconds
        self.min_frame_similarity = min_frame_similarity

        # Other settings
        self.template_name = template
        self.verbose = verbose

        # Lazy-initialized backends
        self._asr: Optional[ASRBackend] = None
        self._ocr: Optional[OCRBackend] = None
        self._llm: Optional[LLMBackend] = None
        self._formatter = MarkdownFormatter()

    @staticmethod
    def _get_base_url(provider: str) -> str:
        """Get default base URL for provider."""
        provider_config = PROVIDER_DEFAULTS.get(provider, {})
        return provider_config.get("base_url", "")

    @property
    def asr(self) -> ASRBackend:
        """Get ASR backend (lazy initialization)."""
        if self._asr is None:
            if self.asr_backend == "funasr":
                self._asr = FunASRBackend(
                    model=self.asr_model,
                    device=self.asr_device,
                )
            elif self.asr_backend == "whisper":
                self._asr = WhisperBackend(device=self.asr_device)
            else:
                raise ValueError(f"Unknown ASR backend: {self.asr_backend}")
        return self._asr

    @property
    def ocr(self) -> OCRBackend:
        """Get OCR backend (lazy initialization)."""
        if self._ocr is None:
            if self.ocr_backend == "paddleocr":
                self._ocr = PaddleOCRBackend(lang=self.ocr_lang)
            else:
                raise ValueError(f"Unknown OCR backend: {self.ocr_backend}")
        return self._ocr

    @property
    def llm(self) -> LLMBackend:
        """Get LLM backend (lazy initialization)."""
        if self._llm is None:
            self._llm = OpenAIBackend(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm

    def process(
        self,
        video_path: Union[Path, str, None] = None,
        audio_path: Union[Path, str, None] = None,
        pdf_paths: Union[list[Union[Path, str]], None] = None,
        template: Union[NoteTemplate, None] = None,
        **kwargs: Any,
    ) -> NotelyResult:
        """
        Process a lecture and generate Markdown notes.

        Args:
            video_path: Path to the lecture video file.
            audio_path: Path to the audio file (if no video).
            pdf_paths: Paths to PDF slides/handouts.
            template: Custom note template for formatting.
            **kwargs: Additional metadata (title, instructor, date, etc.)

        Returns:
            NotelyResult containing the generated Markdown and metadata.
        """
        # Prepare input
        input_data = LectureInput(
            video_path=Path(video_path) if video_path else None,
            audio_path=Path(audio_path) if audio_path else None,
            pdf_paths=[Path(p) for p in pdf_paths] if pdf_paths else [],
            metadata=kwargs,
        )

        # Use default template if not provided
        if template is None:
            template = get_default_template()

        # Step 1: Extract audio from video if needed
        audio_to_process = input_data.audio_path
        if audio_to_process is None and input_data.video_path:
            audio_to_process = extract_audio(input_data.video_path)

        # Step 2: ASR - Speech to text with timestamps
        if audio_to_process:
            transcript = self.asr.transcribe(audio_to_process)
        else:
            # Create empty transcript if no audio
            transcript = ASRResult(segments=[], duration=0.0)

        # Step 3: OCR - Extract text from video frames and PDFs
        ocr_results = []

        # Extract key frames from video
        if input_data.video_path:
            key_frames = extract_key_frames(
                input_data.video_path,
                interval_seconds=self.key_frame_interval_seconds,
                min_similarity=self.min_frame_similarity,
            )
            for frame in key_frames:
                ocr_result = self.ocr.recognize(frame.path)
                ocr_results.append(ocr_result)

        # Process PDF files
        # TODO: Implement PDF processing

        # Step 4: LLM - Generate structured notes
        notes, thinking_process = self.llm.generate_notes(
            transcript=transcript,
            ocr_results=ocr_results,
            template=template,
            metadata=input_data.metadata,
        )

        # Step 5: Format and beautify
        formatted_notes = self._formatter.beautify(notes)

        return NotelyResult(
            markdown=formatted_notes,
            thinking_process=thinking_process,
            transcript=transcript,
            ocr_results=ocr_results,
            metadata=input_data.metadata,
        )

    def process_audio(
        self,
        audio_path: Union[Path, str],
        template: Union[NoteTemplate, None] = None,
        **kwargs: Any,
    ) -> NotelyResult:
        """
        Process an audio file and generate notes.

        Args:
            audio_path: Path to the audio file.
            template: Custom note template.
            **kwargs: Additional metadata.

        Returns:
            NotelyResult containing the generated notes.
        """
        return self.process(audio_path=audio_path, template=template, **kwargs)

    def process_pdf(
        self,
        pdf_path: Union[Path, str],
        template: Union[NoteTemplate, None] = None,
        **kwargs: Any,
    ) -> NotelyResult:
        """
        Process a PDF file and generate notes.

        Args:
            pdf_path: Path to the PDF file.
            template: Custom note template.
            **kwargs: Additional metadata.

        Returns:
            NotelyResult containing the generated notes.
        """
        return self.process(pdf_paths=[pdf_path], template=template, **kwargs)


# Backward compatibility: keep NotelyConfig for existing code
class NotelyConfig:
    """
    Deprecated: Use Notely() constructor parameters instead.

    This class is kept for backward compatibility.
    """

    def __init__(self, **kwargs):
        import warnings

        warnings.warn(
            "NotelyConfig is deprecated. Use Notely() constructor parameters instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.__dict__.update(kwargs)

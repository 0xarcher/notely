"""
Data models for Notely SDK.

This module defines all data structures used in the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notely.asr.base import ASRResult
from notely.ocr.base import OCRResult


@dataclass
class NotelyResult:
    """
    Result of lecture processing.

    This class encapsulates all outputs from the Notely processing pipeline,
    including the generated Markdown notes, transcript, OCR results, and metadata.

    Attributes:
        markdown: Generated Markdown notes
        thinking_process: Processing metadata and logs
        transcript: ASR transcription result
        ocr_results: OCR results from slides/images
        metadata: Additional metadata (title, date, etc.)

    Example:
        from notely import Notely, NotelyConfig, EnhancerConfig, LLMConfig

        config = NotelyConfig(
            enhancer=EnhancerConfig(
                llm=LLMConfig(api_key="sk-xxx", model="gpt-4o")
            )
        )
        notely = Notely(config)
        result = await notely.process("lecture.mp4")
        print(f"Generated {len(result.markdown)} characters")
        result.save("output/notes.md")
    """

    markdown: str
    thinking_process: str
    transcript: ASRResult
    ocr_results: list[OCRResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_path: Path | str) -> None:
        """
        Save the markdown result to a file.

        Args:
            output_path: Path to output file

        Example:
            result.save("output/notes.md")
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.markdown, encoding="utf-8")

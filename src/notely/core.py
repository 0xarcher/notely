"""
Core module for Notely - Configuration container and processing orchestrator.

This module provides the main Notely class which acts as a configuration
container and coordinates the ASR → OCR → Enhancer pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from notely.asr import ASRBackend, FunASRBackend
from notely.config import NotelyConfig
from notely.enhancer.enhancer import ThreeLayerEnhancer
from notely.formatter import MarkdownFormatter
from notely.models import NotelyResult
from notely.ocr import OCRBackend, OCRResult, PaddleOCRBackend
from notely.utils import extract_audio, extract_key_frames
from notely.utils.language import detect_transcript_language

logger = logging.getLogger(__name__)


class Notely:
    """
    Notely - Configuration container and processing orchestrator.

    This class acts as a configuration container that holds all settings
    for ASR, OCR, and Enhancer. It coordinates the processing pipeline:
    ASR → OCR → Enhancer → Markdown output.

    The Notely class does NOT make processing decisions (like choosing
    between simple/enhanced processing). Instead, it simply orchestrates
    the configured components.

    Example:
        from notely import Notely, NotelyConfig, EnhancerConfig, LLMConfig

        # From configuration object
        config = NotelyConfig(
            enhancer=EnhancerConfig(
                llm=LLMConfig(api_key="sk-xxx", model="gpt-4o")
            )
        )
        notely = Notely(config)
        result = await notely.process("lecture.mp4")

        # From YAML file
        notely = Notely.from_yaml("config.yaml")
        result = await notely.process("lecture.mp4")

        # From dictionary
        notely = Notely.from_dict({
            "llm": {"api_key": "sk-xxx", "model": "gpt-4o"}
        })
        result = await notely.process("lecture.mp4")
    """

    def __init__(self, config: NotelyConfig) -> None:
        """
        Initialize Notely with configuration.

        Args:
            config: Notely configuration object
        """
        self.config = config

        # Lazy-initialized backends
        self._asr: ASRBackend | None = None
        self._ocr: OCRBackend | None = None
        self._enhancer: ThreeLayerEnhancer | None = None
        self._formatter = MarkdownFormatter()

        logger.info("Initialized Notely with configuration")

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Notely:
        """
        Create Notely from YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Notely instance

        Example:
            notely = Notely.from_yaml("config.yaml")
        """
        config = NotelyConfig.from_yaml(yaml_path)
        return cls(config)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Notely:
        """
        Create Notely from configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Notely instance

        Example:
            notely = Notely.from_dict({
                "llm": {"api_key": "sk-xxx", "model": "gpt-4o"}
            })
        """
        config = NotelyConfig.from_dict(config_dict)
        return cls(config)

    @property
    def asr(self) -> ASRBackend:
        """
        Get ASR backend (lazy initialization).

        Returns:
            ASR backend instance
        """
        if self._asr is None:
            if self.config.asr.backend == "funasr":
                self._asr = FunASRBackend(
                    model=self.config.asr.model,
                )
            else:
                raise ValueError(f"Unsupported ASR backend: {self.config.asr.backend}")

            logger.info(f"Initialized ASR backend: {self.config.asr.backend}")

        return self._asr

    @property
    def ocr(self) -> OCRBackend:
        """
        Get OCR backend (lazy initialization).

        Returns:
            OCR backend instance
        """
        if self._ocr is None:
            if self.config.ocr.backend == "paddleocr":
                self._ocr = PaddleOCRBackend(
                    lang=self.config.ocr.language,
                    use_gpu=self.config.ocr.use_gpu,
                )
            else:
                raise ValueError(f"Unsupported OCR backend: {self.config.ocr.backend}")

            logger.info(f"Initialized OCR backend: {self.config.ocr.backend}")

        return self._ocr

    @property
    def enhancer(self) -> ThreeLayerEnhancer:
        """
        Get Enhancer (lazy initialization).

        Returns:
            ThreeLayerEnhancer instance
        """
        if self._enhancer is None:
            self._enhancer = ThreeLayerEnhancer.from_config(self.config.enhancer)
            logger.info("Initialized Enhancer")

        return self._enhancer

    async def process(
        self,
        input_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> NotelyResult:
        """
        Process input file and generate notes.

        This is the main processing pipeline:
        1. Extract audio (if video)
        2. ASR - Transcribe speech
        3. OCR - Extract text from slides (if video)
        4. Enhancer - Generate structured notes
        5. Format - Beautify Markdown

        Args:
            input_path: Path to input file (audio/video)
            metadata: Optional metadata (title, date, etc.)

        Returns:
            NotelyResult with generated notes

        Example:
            result = await notely.process("lecture.mp4")
            result.save("output/notes.md")
        """
        input_path = Path(input_path)
        metadata = metadata or {}

        logger.info("=" * 80)
        logger.info(f"Processing: {input_path}")
        logger.info("=" * 80)

        # Step 1: Extract audio from video if needed
        audio_path = input_path
        if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            logger.info("Extracting audio from video...")
            audio_path = extract_audio(input_path)
            logger.info(f"✓ Audio extracted: {audio_path}")

        # Step 2: ASR - Speech to text
        logger.info("Transcribing audio...")
        transcript = self.asr.transcribe(str(audio_path))
        logger.info(f"✓ Transcription completed: {len(transcript.segments)} segments")

        # Step 2.5: Detect language if not explicitly set
        if self.config.enhancer.language is None:
            asr_hint = getattr(transcript, "language", None)
            detected_lang = detect_transcript_language(
                transcript.full_text,
                asr_hint=asr_hint,
            )
            # Temporarily override for this processing
            self.config.enhancer.language = detected_lang
            logger.info(f"✓ Auto-detected language: {detected_lang}")
        else:
            logger.info(f"Using configured language: {self.config.enhancer.language}")

        # Step 3: OCR - Extract text from slides (if video)
        ocr_results: list[OCRResult] = []
        if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            logger.info("Extracting key frames...")
            key_frames = extract_key_frames(str(input_path))
            logger.info(f"✓ Extracted {len(key_frames)} key frames")

            if key_frames:
                logger.info("Performing OCR on key frames...")
                for i, frame in enumerate(key_frames):
                    ocr_result = self.ocr.recognize(frame.path)
                    ocr_results.append(ocr_result)
                    logger.info(
                        f"  Frame {i + 1}/{len(key_frames)}: {len(ocr_result.full_text)} chars"
                    )
                logger.info(f"✓ OCR completed: {len(ocr_results)} frames")

        # Step 4: Enhancer - Generate notes (唯一入口)
        logger.info("Generating notes with Enhancer...")
        markdown = await self.enhancer.process(
            transcript=transcript,
            ocr_results=ocr_results,
            metadata=metadata,
        )
        logger.info(f"✓ Notes generated: {len(markdown)} characters")

        # Step 5: Format and beautify
        logger.info("Formatting Markdown...")
        formatted_markdown = self._formatter.beautify(markdown)
        logger.info("✓ Formatting completed")

        # Create result
        result = NotelyResult(
            markdown=formatted_markdown,
            thinking_process=f"Processed with 3-Layer Enhancer: {self.enhancer.metrics}",
            transcript=transcript,
            ocr_results=ocr_results,
            metadata=metadata,
        )

        logger.info("=" * 80)
        logger.info("✓ Processing completed successfully")
        logger.info("=" * 80)

        return result

    def process_sync(
        self,
        input_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> NotelyResult:
        """
        Synchronous version of process().

        Args:
            input_path: Path to input file
            metadata: Optional metadata

        Returns:
            NotelyResult with generated notes
        """
        return asyncio.run(self.process(input_path, metadata))

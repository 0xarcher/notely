"""
3-Layer Pipeline Enhancer - Main Orchestrator

This is the main entry point for the enhancer system, coordinating
all three layers of the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tiktoken

from .comprehension import ComprehensionAgent
from .models import (
    ProcessingMetrics,
    ProcessingStage,
    SemanticChunk,
)
from .structuring import StructuringAgent

if TYPE_CHECKING:
    from notely.asr.base import ASRResult
    from notely.config import EnhancerConfig
    from notely.llm.client import LLMClient
    from notely.ocr.base import OCRResult

logger = logging.getLogger(__name__)


class ThreeLayerEnhancer:
    """
    3-Layer Pipeline Enhancer - Main Orchestrator.

    This class coordinates the entire enhancement pipeline:
    - Layer 1: Capture (ASR transcription - done externally)
    - Layer 2: Comprehension (parallel semantic extraction)
    - Layer 3: Structuring (organize into structured notes)

    Key Features:
    - Semantic chunking with overlap
    - Parallel processing in Layer 2
    - Caching for idempotency
    - Progress tracking
    - Graceful degradation
    - Comprehensive metrics

    Attributes:
        llm: LLM client
        chunk_size: Maximum chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        cache_dir: Directory for caching results
        language: Output language ('zh' or 'en', auto-detected if None)
        progress_callback: Optional callback for progress updates
    """

    def __init__(
        self,
        llm: LLMClient,
        chunk_size: int = 4000,
        chunk_overlap: int = 800,
        cache_dir: Path | None = None,
        language: str | None = None,
        max_concurrent: int = 5,
        progress_callback: Callable[[ProcessingMetrics], None] | None = None,
    ) -> None:
        """
        Initialize 3-Layer Enhancer.

        Args:
            llm: LLM client
            chunk_size: Maximum chunk size in tokens (default: 4000)
            chunk_overlap: Overlap between chunks in tokens (default: 800)
            cache_dir: Cache directory (default: .cache/enhancer)
            language: Output language ('zh' or 'en', auto-detected if None)
            max_concurrent: Maximum concurrent API calls (default: 5)
            progress_callback: Optional callback(metrics)
        """
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = cache_dir or Path(".cache/enhancer")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback

        # Initialize metrics
        self.metrics = ProcessingMetrics()

        # Agents will be initialized with detected language
        self.comprehension_agent: ComprehensionAgent | None = None
        self.structuring_agent: StructuringAgent | None = None

        # Initialize tiktoken encoder for accurate token counting
        try:
            self.encoding = tiktoken.encoding_for_model(llm.model)
        except KeyError:
            # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo)
            logger.warning(f"Model {llm.model} not found in tiktoken, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def from_config(cls, config: EnhancerConfig) -> ThreeLayerEnhancer:
        """
        Create Enhancer from configuration.

        Args:
            config: Enhancer configuration

        Returns:
            ThreeLayerEnhancer instance
        """
        from notely.llm.client import LLMClient

        llm = LLMClient(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )

        return cls(
            llm=llm,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            cache_dir=config.cache_dir,
            language=config.language,
            max_concurrent=config.max_concurrent,
        )

    async def process(
        self,
        transcript: ASRResult | None,
        ocr_results: list[OCRResult] | None = None,
        metadata: dict[Any, Any] | None = None,
    ) -> str:
        """
        Process transcript and generate structured notes.

        Args:
            transcript: ASR transcription result (None for PDF/image processing)
            ocr_results: Optional OCR results
            metadata: Optional metadata (title, date, etc.)

        Returns:
            Markdown formatted notes

        Raises:
            ValueError: If both transcript and ocr_results are None
            Exception: If processing fails completely
        """
        logger.info("=" * 80)
        logger.info("3-Layer Pipeline Enhancer - Starting")
        logger.info("=" * 80)

        # Reset metrics
        self.metrics = ProcessingMetrics()
        self._update_progress(ProcessingStage.CHUNKING)

        try:
            # Prepare input
            full_text, metadata = self._prepare_input(transcript, ocr_results, metadata)

            # Detect language if not specified
            if self.language is None:
                self.language = self._detect_language(full_text)
            logger.info(f"Language: {self.language}")

            # Initialize agents with detected language
            self._initialize_agents()

            # Check cache
            cache_key = self._generate_cache_key(full_text, metadata)
            if cached := self._load_cache(cache_key):
                logger.info("✓ Loaded from cache")
                self._update_progress(ProcessingStage.COMPLETED)
                return cached

            # Layer 1: Semantic Chunking
            chunks = self._semantic_chunking(transcript, full_text)
            self.metrics.chunks_total = len(chunks)
            logger.info(f"Layer 1: Created {len(chunks)} semantic chunks")
            self._update_progress(ProcessingStage.COMPREHENSION)

            # Layer 2: Comprehension (parallel)
            if self.comprehension_agent is None:
                raise RuntimeError("Comprehension agent not initialized")
            comprehensions = await self.comprehension_agent.process_chunks(chunks)
            logger.info(f"Layer 2: Extracted {len(comprehensions)} comprehension results")
            self._update_progress(ProcessingStage.STRUCTURING)

            # Layer 3: Structuring
            if self.structuring_agent is None:
                raise RuntimeError("Structuring agent not initialized")
            structured_note = await self.structuring_agent.structure(comprehensions, metadata)
            logger.info(
                f"Layer 3: Created structured note with {len(structured_note.sections)} sections"
            )

            # Convert to Markdown
            markdown = structured_note.to_markdown(self.language)

            # Save cache
            self._save_cache(cache_key, markdown)

            # Complete
            self._update_progress(ProcessingStage.COMPLETED)
            logger.info("=" * 80)
            logger.info(f"✓ Processing completed - {self.metrics}")
            logger.info("=" * 80)

            return markdown

        except Exception as e:
            logger.error(f"✗ Processing failed: {e}", exc_info=True)
            self.metrics.errors.append(str(e))
            self._update_progress(ProcessingStage.FAILED)
            raise

    def _prepare_input(
        self,
        transcript: ASRResult | None,
        ocr_results: list[OCRResult] | None,
        metadata: dict[Any, Any] | None,
    ) -> tuple[str, dict[Any, Any]]:
        """Prepare input text and metadata."""
        # Start with transcript or empty
        full_text = transcript.full_text if transcript else ""

        # Add OCR content
        if ocr_results:
            ocr_text = "\n\n".join([ocr.full_text for ocr in ocr_results if ocr.full_text])
            if ocr_text:
                if full_text:
                    full_text += f"\n\n## OCR Content\n\n{ocr_text}"
                else:
                    full_text = ocr_text

        # Validate we have content to process
        if not full_text.strip():
            raise ValueError("No content to process: both transcript and OCR results are empty")

        # Prepare metadata
        final_metadata = metadata or {}
        final_metadata.setdefault("title", "Course Notes")
        final_metadata.setdefault("source", "document" if not transcript else "audio")

        # Add duration if available
        if transcript and transcript.segments:
            duration_sec = transcript.segments[-1].end_time
            final_metadata["duration"] = f"{int(duration_sec // 60)}min"

        return full_text, final_metadata

    def _detect_language(self, text: str) -> str:
        """Detect text language."""
        sample = text[:1000]
        chinese_chars = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff")
        return "zh" if chinese_chars / len(sample) > 0.3 else "en"

    def _initialize_agents(self) -> None:
        """Initialize agents with detected language."""
        if self.comprehension_agent is None:
            self.comprehension_agent = ComprehensionAgent(
                self.llm,
                language=self.language or "en",
                max_concurrent=self.max_concurrent,
                progress_callback=self._on_chunk_processed,
            )

        if self.structuring_agent is None:
            self.structuring_agent = StructuringAgent(self.llm, language=self.language or "en")

    def _semantic_chunking(
        self, transcript: ASRResult | None, full_text: str
    ) -> list[SemanticChunk]:
        """
        Perform semantic chunking with intelligent boundary detection.

        Strategy:
        - Split by semantic boundaries (paragraphs, sentences)
        - Add 20% overlap between chunks for context preservation
        - Add previous/next context for cross-chunk information
        - Use tiktoken for accurate token counting

        For documents (no transcript): chunk by paragraphs/sentences
        For audio/video (with transcript): chunk by transcript segments

        Args:
            transcript: ASR result with segments (None for documents)
            full_text: Full text content to chunk

        Returns:
            List of semantic chunks
        """
        # Handle document processing (no transcript)
        if transcript is None:
            return self._chunk_text_only(full_text)

        # Handle audio/video processing (with transcript)
        return self._chunk_transcript(transcript)

    def _chunk_transcript(self, transcript: ASRResult) -> list[SemanticChunk]:
        """Chunk transcript by segments."""
        chunks: list[SemanticChunk] = []
        current_text: list[str] = []
        current_start: float = 0.0
        current_tokens = 0

        for segment in transcript.segments:
            segment_text = segment.text
            # Use tiktoken for accurate token counting
            segment_tokens = len(self.encoding.encode(segment_text))

            if current_tokens + segment_tokens > self.chunk_size:
                # Find semantic boundary
                full_text = "\n".join(current_text)
                split_pos = self._find_semantic_boundary(full_text, self.chunk_size)

                # Create chunk with text up to boundary
                chunk_text = full_text[:split_pos]
                remaining = full_text[split_pos:]

                # Only create chunk if text is not empty
                if chunk_text.strip():
                    chunks.append(
                        SemanticChunk(
                            text=chunk_text,
                            start_time=current_start,
                            end_time=segment.start_time,
                            index=len(chunks),
                        )
                    )

                # Start new chunk with overlap based on tokens
                overlap_tokens = int(self.chunk_overlap)
                overlap_text = self._get_text_for_tokens(chunk_text, overlap_tokens)

                current_text = (
                    [overlap_text, remaining, segment_text]
                    if remaining
                    else [overlap_text, segment_text]
                )
                current_start = segment.start_time
                # Recalculate tokens for new chunk
                current_tokens = int(sum(len(self.encoding.encode(t)) for t in current_text))
            else:
                current_text.append(segment_text)
                current_tokens += segment_tokens

        # Save last chunk
        if current_text:
            chunks.append(
                SemanticChunk(
                    text="\n".join(current_text),
                    start_time=current_start,
                    end_time=transcript.segments[-1].end_time if transcript.segments else 0,
                    index=len(chunks),
                )
            )

        # Add context for cross-chunk information preservation
        self._add_chunk_context(chunks)

        return chunks

    def _chunk_text_only(self, full_text: str) -> list[SemanticChunk]:
        """
        Chunk text content without transcript segments.

        Used for document processing (PDF/images).

        Args:
            full_text: Full text content to chunk.

        Returns:
            List of SemanticChunk objects.
        """
        chunks: list[SemanticChunk] = []

        # Split by paragraphs (double newlines)
        paragraphs = full_text.split("\n\n")

        current_text: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = len(self.encoding.encode(para))

            if current_tokens + para_tokens > self.chunk_size and current_text:
                # Create chunk
                chunk_text = "\n\n".join(current_text)
                if chunk_text.strip():
                    chunks.append(
                        SemanticChunk(
                            text=chunk_text,
                            start_time=0.0,
                            end_time=0.0,
                            index=chunk_index,
                        )
                    )
                    chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_text_for_tokens(chunk_text, self.chunk_overlap)
                current_text = [overlap_text, para] if overlap_text else [para]
                current_tokens = sum(len(self.encoding.encode(t)) for t in current_text)
            else:
                current_text.append(para)
                current_tokens += para_tokens

        # Save last chunk
        if current_text:
            chunk_text = "\n\n".join(current_text)
            if chunk_text.strip():
                chunks.append(
                    SemanticChunk(
                        text=chunk_text,
                        start_time=0.0,
                        end_time=0.0,
                        index=chunk_index,
                    )
                )

        # Add context for cross-chunk information preservation
        self._add_chunk_context(chunks)

        logger.info(f"Created {len(chunks)} chunks from document text")
        return chunks

    def _get_text_for_tokens(self, text: str, target_tokens: int) -> str:
        """
        Get text that approximately matches target token count from the end.

        Args:
            text: Source text
            target_tokens: Target token count

        Returns:
            Text substring with approximately target_tokens tokens
        """
        # Split by sentences for better boundaries
        sentences = text.split("。")
        if not sentences:
            return ""

        # Start from the end and accumulate sentences
        result: list[str] = []
        current_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = len(self.encoding.encode(sentence + "。"))
            if current_tokens + sentence_tokens <= target_tokens:
                result.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break

        return "。".join(result) + "。" if result else ""

    def _find_semantic_boundary(self, text: str, target_pos: int) -> int:
        """
        Find the best semantic boundary near target position.

        Priority:
        1. Paragraph boundary (\\n\\n)
        2. Sentence boundary (。！？.!?)
        3. Phrase boundary (，；,;)
        4. Word boundary (space)

        Args:
            text: Text to search in
            target_pos: Target split position

        Returns:
            Best boundary position
        """

        # Search within ±200 chars of target
        search_start = max(0, target_pos - 200)
        search_end = min(len(text), target_pos + 200)
        search_text = text[search_start:search_end]
        relative_target = target_pos - search_start

        # 1. Paragraph boundary
        if "\n\n" in search_text:
            pos = search_text.rfind("\n\n", 0, relative_target)
            if pos != -1:
                return search_start + pos + 2

        # 2. Sentence boundary
        sentence_delimiters = ["。", "！", "？", ".", "!", "?"]
        for delimiter in sentence_delimiters:
            pos = search_text.rfind(delimiter, 0, relative_target)
            if pos != -1:
                return search_start + pos + 1

        # 3. Phrase boundary
        phrase_delimiters = ["，", "；", ",", ";"]
        for delimiter in phrase_delimiters:
            pos = search_text.rfind(delimiter, 0, relative_target)
            if pos != -1:
                return search_start + pos + 1

        # 4. Word boundary
        pos = search_text.rfind(" ", 0, relative_target)
        if pos != -1:
            return search_start + pos + 1

        # 5. Fallback to target position
        return target_pos

    def _add_chunk_context(self, chunks: list[SemanticChunk]) -> None:
        """
        Add previous/next context to chunks for continuity.

        Args:
            chunks: List of chunks to add context to
        """

        for i, chunk in enumerate(chunks):
            # Add previous context (last 3 sentences from previous chunk)
            if i > 0:
                prev_text = chunks[i - 1].text
                chunk.previous_context = self._extract_last_sentences(prev_text, n=3)

            # Add next preview (first 3 sentences from next chunk)
            if i < len(chunks) - 1:
                next_text = chunks[i + 1].text
                chunk.next_preview = self._extract_first_sentences(next_text, n=3)

    @staticmethod
    def _extract_last_sentences(text: str, n: int = 3) -> str:
        """
        Extract last N sentences from text.

        Args:
            text: Text to extract from
            n: Number of sentences

        Returns:
            Last N sentences
        """
        import re

        # Split by sentence delimiters
        sentences = re.split(r"[。！？.!?]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        # Return last N sentences
        last_sentences = sentences[-n:] if len(sentences) >= n else sentences
        return "。".join(last_sentences) + "。"

    @staticmethod
    def _extract_first_sentences(text: str, n: int = 3) -> str:
        """
        Extract first N sentences from text.

        Args:
            text: Text to extract from
            n: Number of sentences

        Returns:
            First N sentences
        """
        import re

        # Split by sentence delimiters
        sentences = re.split(r"[。！？.!?]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        # Return first N sentences
        first_sentences = sentences[:n] if len(sentences) >= n else sentences
        return "。".join(first_sentences) + "。"

    def _on_chunk_processed(self, current: int, total: int) -> None:
        """Callback when a chunk is processed."""
        self.metrics.chunks_processed = current
        self._update_progress(ProcessingStage.COMPREHENSION)

    def _update_progress(self, stage: ProcessingStage) -> None:
        """Update processing stage and notify callback."""
        self.metrics.stage = stage

        if self.progress_callback:
            self.progress_callback(self.metrics)

    def _generate_cache_key(self, text: str, metadata: dict[Any, Any]) -> str:
        """Generate cache key from text and metadata."""
        content = f"{text}:{metadata.get('title', '')}:{self.language}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_cache(self, cache_key: str) -> str | None:
        """Load cached result."""
        cache_file = self.cache_dir / f"{cache_key}.md"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def _save_cache(self, cache_key: str, content: str) -> None:
        """Save result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.md"
        cache_file.write_text(content, encoding="utf-8")
        logger.debug(f"Saved to cache: {cache_file}")

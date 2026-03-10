"""
Layer 2: Comprehension Agent

Responsible for extracting semantic information from transcript chunks.
This layer preserves all key information without compression.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .models import ComprehensionResult, SemanticChunk
from .utils import parallel_with_limit

if TYPE_CHECKING:
    from notely.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ComprehensionError(Exception):
    """Exception raised when comprehension fails."""

    pass


class ComprehensionAgent:
    """
    Comprehension Agent - Layer 2 of the 3-Layer Pipeline.

    This agent extracts semantic information from transcript chunks:
    - Detailed summaries (preserving all key information)
    - Core concepts and definitions
    - Important examples
    - Related questions

    The agent does NOT compress information - it extracts and structures it.

    Attributes:
        llm: LLM client for generation
        language: Output language ('zh' or 'en')
        temperature: LLM temperature (default: 0.3 for consistency)
        max_retries: Maximum number of retries on failure
        max_concurrent: Maximum concurrent API calls
        progress_callback: Optional callback for progress updates
    """

    def __init__(
        self,
        llm: LLMClient,
        language: str = "zh",
        temperature: float = 0.3,
        max_retries: int = 3,
        max_concurrent: int = 5,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Initialize Comprehension Agent.

        Args:
            llm: LLM client
            language: Output language ('zh' or 'en')
            temperature: LLM temperature (lower = more consistent)
            max_retries: Maximum retries on failure
            max_concurrent: Maximum concurrent API calls (default: 5)
            progress_callback: Optional callback(current, total)
        """
        self.llm = llm
        self.language = language
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback

    async def process_chunks(self, chunks: list[SemanticChunk]) -> list[ComprehensionResult]:
        """
        Process multiple chunks in parallel with concurrency limit.

        Args:
            chunks: List of semantic chunks

        Returns:
            List of comprehension results

        Raises:
            ComprehensionError: If processing fails for all chunks
        """
        logger.info(
            f"Processing {len(chunks)} chunks in parallel (max_concurrent={self.max_concurrent})"
        )

        # Process with concurrency limit and retry
        results: list[ComprehensionResult] = await parallel_with_limit(
            items=chunks,
            process_func=self._process_single_chunk,
            max_concurrent=self.max_concurrent,
            max_retries=self.max_retries,
        )

        # Update progress for all chunks
        if self.progress_callback:
            self.progress_callback(len(chunks), len(chunks))

        return results

    async def _process_single_chunk(self, chunk: SemanticChunk) -> ComprehensionResult:
        """
        Process a single chunk.

        Args:
            chunk: Semantic chunk to process

        Returns:
            Comprehension result

        Raises:
            ComprehensionError: If processing fails
        """
        # Build prompt
        prompt = self._build_prompt(chunk)

        # Call LLM
        response = await asyncio.to_thread(self.llm.generate, prompt, temperature=self.temperature)

        # Parse response
        result_dict = self._parse_json_response(response)

        # Validate and create result
        try:
            return ComprehensionResult(**result_dict)
        except Exception as e:
            logger.warning(f"Failed to parse response as ComprehensionResult: {e}")
            # Try to salvage what we can
            return ComprehensionResult(
                summary=result_dict.get("summary", chunk.text),
                key_concepts=result_dict.get("key_concepts", []),
                examples=result_dict.get("examples", []),
                questions=result_dict.get("questions", []),
            )

    def _build_prompt(self, chunk: SemanticChunk) -> str:
        """
        Build comprehension prompt for a chunk.

        Args:
            chunk: Semantic chunk

        Returns:
            Formatted prompt string
        """
        from notely.prompts.registry import PromptRegistry

        template = PromptRegistry.get("comprehension")
        return template.format(transcript_text=chunk.text, language=self.language)

    def _parse_json_response(self, response: str) -> dict[Any, Any]:
        """
        Parse JSON response from LLM.

        Args:
            response: LLM response string

        Returns:
            Parsed dictionary

        Raises:
            ComprehensionError: If parsing fails
        """
        # Strip whitespace
        response = response.strip()

        # Try to strip markdown code blocks first
        if response.startswith("```"):
            # Remove opening ```json or ```
            lines = response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        # Try direct parsing
        try:
            return json.loads(response)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        # Try to extract from code block (fallback)
        code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        # Last resort: try to fix common JSON errors
        try:
            # Remove trailing commas
            fixed = re.sub(r",\s*([}\]])", r"\1", response)
            return json.loads(fixed)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Last resort: treat entire response as summary (for simpler LLMs)
        logger.warning(
            f"Failed to parse JSON, treating response as raw summary: {response[:200]}..."
        )
        return {
            "summary": response,
            "key_concepts": [],
            "examples": [],
            "questions": [],
        }

    def _create_fallback_result(self, chunk: SemanticChunk, error: str) -> ComprehensionResult:
        """
        Create fallback result when processing fails.

        Args:
            chunk: Original chunk
            error: Error message

        Returns:
            Fallback comprehension result
        """
        logger.warning(f"Creating fallback result for {chunk}: {error}")

        return ComprehensionResult(
            summary=chunk.text,  # Use original text as summary
            key_concepts=[],
            examples=[],
            questions=[],
            metadata={"fallback": True, "error": error, "chunk_index": chunk.index},
        )

    def estimate_tokens(self, chunks: list[SemanticChunk]) -> tuple[int, int]:
        """
        Estimate token usage for processing chunks.

        Args:
            chunks: List of chunks to process

        Returns:
            Tuple of (input_tokens, estimated_output_tokens)
        """
        # Rough estimation: 1 token ≈ 0.75 characters
        input_chars = sum(len(chunk.text) for chunk in chunks)
        prompt_overhead = 500 * len(chunks)  # Prompt template overhead

        input_tokens = int((input_chars + prompt_overhead) / 0.75)

        # Estimate output: assume 50% compression
        output_tokens = int(input_tokens * 0.5)

        return input_tokens, output_tokens

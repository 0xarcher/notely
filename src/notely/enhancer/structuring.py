"""
Layer 3: Structuring Agent

Responsible for organizing comprehension results into structured notes.
This layer applies formatting rules and generates the final Markdown output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import ComprehensionResult, NoteSection, StructuredNote

if TYPE_CHECKING:
    from notely.llm.client import LLMClient

logger = logging.getLogger(__name__)


class StructuringError(Exception):
    """Exception raised when structuring fails."""

    pass


class StructuringAgent:
    """
    Structuring Agent - Layer 3 of the 3-Layer Pipeline.

    This agent organizes comprehension results into structured notes:
    - Merges multiple comprehension results
    - Organizes content by topic (not chronological order)
    - Applies formatting rules
    - Generates executive summary
    - Creates hierarchical structure

    Attributes:
        llm: LLM client for generation
        language: Output language ('zh' or 'en')
        temperature: LLM temperature (default: 0.5 for creativity)
        max_retries: Maximum number of retries on failure
    """

    def __init__(
        self,
        llm: LLMClient,
        language: str = "zh",
        temperature: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize Structuring Agent.

        Args:
            llm: LLM client
            language: Output language ('zh' or 'en')
            temperature: LLM temperature (higher = more creative)
            max_retries: Maximum retries on failure
        """
        self.llm = llm
        self.language = language
        self.temperature = temperature
        self.max_retries = max_retries

    async def structure(
        self, comprehensions: list[ComprehensionResult], metadata: dict[Any, Any]
    ) -> StructuredNote:
        """
        Structure comprehension results into a final note.

        Args:
            comprehensions: List of comprehension results
            metadata: Note metadata (title, date, etc.)

        Returns:
            Structured note

        Raises:
            StructuringError: If structuring fails
        """
        logger.info(f"Structuring {len(comprehensions)} comprehension results")

        try:
            # Try structured generation
            note = await self._structure_with_retry(comprehensions, metadata)
            logger.info(f"✓ Structured note created: {note}")
            return note

        except Exception as e:
            logger.error(f"✗ Structuring failed: {e}")
            # Fallback to simple structure
            return self._create_fallback_note(comprehensions, metadata, str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def _structure_with_retry(
        self, comprehensions: list[ComprehensionResult], metadata: dict[Any, Any]
    ) -> StructuredNote:
        """
        Structure with retry logic.

        Args:
            comprehensions: Comprehension results
            metadata: Note metadata

        Returns:
            Structured note

        Raises:
            StructuringError: If structuring fails
        """
        # Build prompt
        prompt = self._build_prompt(comprehensions, metadata)

        # Call LLM
        response = await asyncio.to_thread(self.llm.generate, prompt, temperature=self.temperature)

        # Parse response
        result_dict = self._parse_json_response(response)

        # Validate and create note
        try:
            return StructuredNote(**result_dict)
        except Exception as e:
            logger.warning(f"Failed to parse response as StructuredNote: {e}")
            # Try to salvage what we can
            return self._salvage_note(result_dict, metadata)

    def _build_prompt(
        self, comprehensions: list[ComprehensionResult], metadata: dict[Any, Any]
    ) -> str:
        """
        Build structuring prompt.

        Args:
            comprehensions: Comprehension results
            metadata: Note metadata

        Returns:
            Formatted prompt string
        """
        from notely.prompts.registry import PromptRegistry

        # Merge all summaries
        combined_summaries = "\n\n---\n\n".join(
            [f"**Segment {i + 1}**:\n{c.summary}" for i, c in enumerate(comprehensions)]
        )

        # Collect all concepts and examples
        all_concepts = []
        for comp in comprehensions:
            all_concepts.extend(comp.key_concepts)

        all_examples = []
        for comp in comprehensions:
            all_examples.extend(comp.examples)

        # Detect cross-chunk patterns
        cross_chunk_hints = self._detect_cross_chunk_patterns(comprehensions)

        template = PromptRegistry.get("structuring")
        return template.format(
            combined_summaries=combined_summaries,
            concept_count=len(all_concepts),
            concepts_list="\n".join(f"- {c}" for c in all_concepts[:20]),
            example_count=len(all_examples),
            examples_list="\n".join(f"- {e}" for e in all_examples[:10]),
            cross_chunk_hints=cross_chunk_hints,
            duration=metadata.get("duration", ""),
            date=metadata.get("date", ""),
            language=self.language,
        )

    def _detect_cross_chunk_patterns(self, comprehensions: list[ComprehensionResult]) -> str:
        """
        Detect patterns that span multiple chunks.

        Args:
            comprehensions: List of comprehension results

        Returns:
            Hints about cross-chunk patterns
        """
        hints = []

        for i in range(len(comprehensions) - 1):
            curr = comprehensions[i]
            next_comp = comprehensions[i + 1]

            # Pattern 1: Mentions quantity but no enumeration
            # e.g., "有三种类型" but no list
            if re.search(
                r"有[一二三四五六七八九十\d]+[种个类]", curr.summary
            ) and not self._has_enumeration(curr.summary):
                hints.append(
                    f"Segment {i + 1} mentions a quantity, segment {i + 2} may have the enumeration"
                )

            # Pattern 2: Mentions "例如" "比如" but no examples
            if (
                re.search(r"(例如|比如|包括|such as|for example|including)", curr.summary)
                and len(curr.examples) == 0
            ):
                hints.append(
                    f"Segment {i + 1} mentions examples, segment {i + 2} may have the details"
                )

            # Pattern 3: Concept overlap (likely same topic)
            common_concepts = set(curr.key_concepts) & set(next_comp.key_concepts)
            if common_concepts:
                concepts_str = ", ".join(list(common_concepts)[:3])
                hints.append(
                    f"Segments {i + 1} and {i + 2} both mention: {concepts_str} (likely same topic)"
                )

        return "\n".join(hints) if hints else "No cross-chunk patterns detected"

    def _has_enumeration(self, text: str) -> bool:
        """
        Check if text contains enumeration.

        Args:
            text: Text to check

        Returns:
            True if enumeration found
        """
        # Check for numbered lists
        if re.search(r"[1-9]\.|[一二三四五六七八九十]、", text):
            return True

        # Check for bullet points
        if re.search(r"[•\-\*]\s", text):
            return True

        # Check for multiple items separated by commas/semicolons
        return text.count("、") >= 2 or text.count("，") >= 2

    def _parse_json_response(self, response: str) -> dict[Any, Any]:
        """
        Parse JSON response from LLM.

        Args:
            response: LLM response string

        Returns:
            Parsed dictionary

        Raises:
            StructuringError: If parsing fails
        """
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

        # Try to extract from code block
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
            # Fix unescaped newlines in strings
            return json.loads(fixed)  # type: ignore[no-any-return]
            fixed = re.sub(r'(?<!\\)"([^"]*)\n([^"]*)"', r'"\1\\n\2"', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        logger.error(f"Failed to parse JSON response: {response[:500]}...")
        raise StructuringError("Failed to parse JSON response from LLM")

    def _salvage_note(
        self, result_dict: dict[Any, Any], metadata: dict[Any, Any]
    ) -> StructuredNote:
        """
        Try to salvage a note from partial JSON.

        Args:
            result_dict: Partial result dictionary
            metadata: Note metadata

        Returns:
            Salvaged structured note
        """
        logger.warning("Attempting to salvage note from partial JSON")

        # Extract what we can
        title = result_dict.get("title", metadata.get("title", "Course Notes"))
        summary = result_dict.get("summary", "Summary generation failed.")
        key_concepts = result_dict.get("key_concepts", [])
        sections_data = result_dict.get("sections", [])

        # Convert sections
        sections = []
        for i, section_data in enumerate(sections_data):
            if isinstance(section_data, dict):
                try:
                    sections.append(NoteSection(**section_data))
                except Exception as e:
                    logger.warning(f"Failed to parse section {i}: {e}")
                    # Create minimal section
                    sections.append(
                        NoteSection(
                            title=section_data.get("title", f"Section {i + 1}"),
                            emoji=section_data.get("emoji", "📝"),
                            content=section_data.get("content", "Content unavailable"),
                        )
                    )

        # Ensure at least one section
        if not sections:
            sections.append(
                NoteSection(
                    title="Content",
                    emoji="📝",
                    content="Note generation encountered an error.",
                )
            )

        return StructuredNote(
            title=title or "Untitled Note",
            summary=summary,
            key_concepts=key_concepts,
            sections=sections,
            metadata=metadata,
        )

    def _create_fallback_note(
        self, comprehensions: list[ComprehensionResult], metadata: dict[Any, Any], error: str
    ) -> StructuredNote:
        """
        Create fallback note when structuring fails.

        Args:
            comprehensions: Comprehension results
            metadata: Note metadata
            error: Error message

        Returns:
            Fallback structured note
        """
        logger.warning(f"Creating fallback note: {error}")

        # Create sections from comprehensions
        sections = []
        for i, comp in enumerate(comprehensions):
            sections.append(
                NoteSection(
                    title=f"Section {i + 1}",
                    emoji="📝",
                    content=comp.summary,
                    metadata={"fallback": True, "comprehension_index": i},
                )
            )

        # Collect all concepts
        all_concepts = []
        for comp in comprehensions:
            all_concepts.extend(comp.key_concepts)

        return StructuredNote(
            title=metadata.get("title", "Course Notes"),
            summary="Note generation encountered an error. Showing raw summaries.",
            key_concepts=list(set(all_concepts))[:10],  # Deduplicate and limit
            sections=sections,
            metadata={**metadata, "fallback": True, "error": error},
        )

    def estimate_tokens(self, comprehensions: list[ComprehensionResult]) -> tuple[int, int]:
        """
        Estimate token usage for structuring.

        Args:
            comprehensions: Comprehension results

        Returns:
            Tuple of (input_tokens, estimated_output_tokens)
        """
        # Rough estimation: 1 token ≈ 0.75 characters
        input_chars = sum(len(c.summary) for c in comprehensions)
        prompt_overhead = 1000  # Prompt template overhead

        input_tokens = int((input_chars + prompt_overhead) / 0.75)

        # Estimate output: structured note is usually 30-40% of input
        output_tokens = int(input_tokens * 0.35)

        return input_tokens, output_tokens

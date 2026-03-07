"""
Data models for the 3-Layer Pipeline Enhancer.

This module defines all data structures used in the enhancer pipeline,
using Pydantic for type safety and validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProcessingStage(StrEnum):
    """Processing stage enumeration."""

    CHUNKING = "chunking"
    COMPREHENSION = "comprehension"
    STRUCTURING = "structuring"
    COMPLETED = "completed"
    FAILED = "failed"


class SemanticChunk(BaseModel):
    """
    A semantic chunk of text from the transcript.

    Attributes:
        text: The chunk content
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        speaker: Speaker identifier (if available)
        index: Chunk index in the sequence
        metadata: Additional metadata
        previous_context: Context from previous chunk for continuity
        next_preview: Preview of next chunk for continuity
    """

    text: str = Field(..., min_length=1, description="Chunk content")
    start_time: float = Field(default=0.0, ge=0, description="Start timestamp (seconds)")
    end_time: float = Field(default=0.0, ge=0, description="End timestamp (seconds)")
    speaker: str = Field(default="", description="Speaker identifier")
    index: int = Field(default=0, ge=0, description="Chunk index")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Context for cross-chunk information preservation
    previous_context: str = Field(default="", description="Context from previous chunk")
    next_preview: str = Field(default="", description="Preview of next chunk")

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, v: float, info: Any) -> float:
        """Ensure end_time >= start_time."""
        if "start_time" in info.data and v < info.data["start_time"]:
            raise ValueError("end_time must be >= start_time")
        return v

    @property
    def duration(self) -> float:
        """Get chunk duration in seconds."""
        return self.end_time - self.start_time

    def __str__(self) -> str:
        """String representation."""
        return f"Chunk[{self.index}]({self.start_time:.1f}s-{self.end_time:.1f}s, {len(self.text)} chars)"


class ComprehensionResult(BaseModel):
    """
    Result from the Comprehension layer.

    This represents the semantic understanding extracted from a chunk,
    preserving all key information without compression.

    Attributes:
        summary: Detailed summary preserving all key information
        key_concepts: List of core concepts and definitions
        examples: Important examples and demonstrations
        questions: Related questions or discussion points
        metadata: Additional metadata (e.g., confidence scores)
    """

    summary: str = Field(..., min_length=10, description="Detailed summary")
    key_concepts: list[str] = Field(
        default_factory=list, description="Core concepts and definitions"
    )
    examples: list[str] = Field(default_factory=list, description="Important examples")
    questions: list[str] = Field(default_factory=list, description="Related questions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("key_concepts", "examples", "questions")
    @classmethod
    def validate_lists(cls, v: list[str]) -> list[str]:
        """Remove empty strings from lists."""
        return [item.strip() for item in v if item.strip()]

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ComprehensionResult("
            f"summary={len(self.summary)} chars, "
            f"concepts={len(self.key_concepts)}, "
            f"examples={len(self.examples)})"
        )


class NoteSection(BaseModel):
    """
    A section in the structured note.

    Attributes:
        title: Section title
        emoji: Emoji icon for the section
        content: Section content in Markdown format
        subsections: Optional nested subsections
        metadata: Additional metadata
    """

    title: str = Field(..., min_length=1, description="Section title")
    emoji: str = Field(default="📝", description="Emoji icon")
    content: str = Field(..., min_length=1, description="Section content (Markdown)")
    subsections: list[NoteSection] = Field(default_factory=list, description="Nested subsections")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("emoji")
    @classmethod
    def validate_emoji(cls, v: str) -> str:
        """Ensure emoji is a single character."""
        if len(v) > 2:  # Allow for multi-byte emojis
            return "📝"
        return v

    def to_markdown(self, level: int = 2) -> str:
        """
        Convert section to Markdown format.

        Args:
            level: Heading level (2 for ##, 3 for ###, etc.)

        Returns:
            Markdown formatted section
        """
        lines = []

        # Section heading
        heading_prefix = "#" * level
        lines.append(f"{heading_prefix} {self.emoji} {self.title}\n")

        # Section content
        lines.append(self.content)

        # Subsections
        for subsection in self.subsections:
            lines.append("\n" + subsection.to_markdown(level + 1))

        return "\n".join(lines)


class StructuredNote(BaseModel):
    """
    The final structured note output.

    Attributes:
        title: Note title
        summary: Executive summary (1-2 paragraphs)
        key_concepts: List of core concepts
        sections: List of note sections
        metadata: Additional metadata (source, duration, etc.)
        created_at: Creation timestamp
    """

    title: str = Field(..., min_length=1, description="Note title")
    summary: str = Field(..., min_length=50, description="Executive summary")
    key_concepts: list[str] = Field(default_factory=list, description="Core concepts")
    sections: list[NoteSection] = Field(default_factory=list, description="Note sections")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v: list[NoteSection]) -> list[NoteSection]:
        """Ensure at least one section exists."""
        if not v:
            raise ValueError("At least one section is required")
        return v

    def to_markdown(self, language: str = "zh") -> str:
        """
        Convert structured note to Markdown format.

        Args:
            language: Output language ('zh' or 'en')

        Returns:
            Complete Markdown document
        """
        lines = []

        # Title
        lines.append(f"# {self.title}\n")

        # Executive summary
        summary_heading = "📋 执行摘要" if language == "zh" else "📋 Executive Summary"
        lines.append(f"## {summary_heading}\n")
        lines.append(f"{self.summary}\n")

        # Key concepts
        if self.key_concepts:
            concepts_heading = "🔑 核心概念" if language == "zh" else "🔑 Key Concepts"
            lines.append(f"## {concepts_heading}\n")
            for concept in self.key_concepts:
                lines.append(f"- {concept}")
            lines.append("")

        # Sections
        for section in self.sections:
            lines.append(section.to_markdown(level=2))
            lines.append("")

        # Metadata footer
        if self.metadata:
            lines.append("---\n")
            lines.append(f"*Generated at: {self.created_at.strftime('%Y-%m-%d %H:%M')}*")
            if "source" in self.metadata:
                lines.append(f"*Source: {self.metadata['source']}*")
            if "duration" in self.metadata:
                lines.append(f"*Duration: {self.metadata['duration']}*")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"StructuredNote("
            f"title='{self.title}', "
            f"sections={len(self.sections)}, "
            f"concepts={len(self.key_concepts)})"
        )


class ProcessingMetrics(BaseModel):
    """
    Metrics for monitoring processing performance.

    Attributes:
        stage: Current processing stage
        chunks_total: Total number of chunks
        chunks_processed: Number of chunks processed
        start_time: Processing start time
        end_time: Processing end time
        llm_calls: Number of LLM API calls
        tokens_input: Total input tokens
        tokens_output: Total output tokens
        errors: List of errors encountered
    """

    stage: ProcessingStage = Field(default=ProcessingStage.CHUNKING, description="Current stage")
    chunks_total: int = Field(default=0, ge=0, description="Total chunks")
    chunks_processed: int = Field(default=0, ge=0, description="Processed chunks")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time")
    end_time: datetime | None = Field(default=None, description="End time")
    llm_calls: int = Field(default=0, ge=0, description="LLM API calls")
    tokens_input: int = Field(default=0, ge=0, description="Input tokens")
    tokens_output: int = Field(default=0, ge=0, description="Output tokens")
    errors: list[str] = Field(default_factory=list, description="Errors")

    @property
    def progress(self) -> float:
        """Calculate processing progress (0.0 to 1.0)."""
        if self.chunks_total == 0:
            return 0.0
        return min(self.chunks_processed / self.chunks_total, 1.0)

    @property
    def duration(self) -> float:
        """Calculate processing duration in seconds."""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    @property
    def estimated_cost(self) -> float:
        """
        Estimate processing cost in USD (based on GPT-4 pricing).

        Pricing (as of 2026):
        - Input: $0.01 per 1K tokens
        - Output: $0.03 per 1K tokens
        """
        input_cost = (self.tokens_input / 1000) * 0.01
        output_cost = (self.tokens_output / 1000) * 0.03
        return input_cost + output_cost

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Metrics("
            f"stage={self.stage.value}, "
            f"progress={self.progress:.1%}, "
            f"duration={self.duration:.1f}s, "
            f"cost=${self.estimated_cost:.3f})"
        )

"""
Base classes for LLM backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class LLMResult:
    """Result from LLM generation."""

    content: str
    thinking: str = ""
    tokens_used: int = 0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Union[str, None] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Returns:
            LLMResult containing the generated text.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_notes(
        self,
        transcript: Any,
        ocr_results: list[Any],
        template: Any,
        metadata: Union[dict[str, Any], None] = None,
    ) -> tuple[str, str]:
        """
        Generate structured notes from transcript and OCR results.

        Args:
            transcript: ASR transcript.
            ocr_results: List of OCR results.
            template: Note template.
            metadata: Additional metadata.

        Returns:
            Tuple of (markdown_content, thinking_process).
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        raise NotImplementedError

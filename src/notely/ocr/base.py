"""
Base classes for OCR backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


@dataclass
class TextBlock:
    """A block of text detected in an image."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    block_type: str = "text"  # "text", "title", "formula", "table"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "block_type": self.block_type,
        }


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text_blocks: list[TextBlock]
    source_path: str = ""
    timestamp: Union[float, None] = None  # For video frames
    page_number: Union[int, None] = None  # For PDF pages
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get all text concatenated."""
        return "\n".join(block.text for block in self.text_blocks)

    @property
    def average_confidence(self) -> float:
        """Get average confidence score."""
        if not self.text_blocks:
            return 0.0
        return sum(b.confidence for b in self.text_blocks) / len(self.text_blocks)

    @property
    def titles(self) -> list[str]:
        """Get detected titles."""
        return [b.text for b in self.text_blocks if b.block_type == "title"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "full_text": self.full_text,
            "source_path": self.source_path,
            "timestamp": self.timestamp,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


class OCRBackend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def recognize(self, image_path: Union[Path, str]) -> OCRResult:
        """
        Recognize text in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult containing detected text blocks.
        """
        raise NotImplementedError

    @abstractmethod
    def recognize_pdf(self, pdf_path: Union[Path, str]) -> list[OCRResult]:
        """
        Recognize text in a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of OCRResult, one per page.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        raise NotImplementedError

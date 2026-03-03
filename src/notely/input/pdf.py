"""
PDF processing utilities.

This module provides functions to extract text and images from PDF files
using pdfplumber (MIT license).
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from notely.ocr.base import OCRResult, TextBlock


def extract_pdf_pages(pdf_path: Path) -> list[OCRResult]:
    """
    Extract text from PDF pages, returning OCRResult list.

    For each page, attempts to extract text directly. If text extraction
    succeeds, returns an OCRResult with the text. If extraction fails
    (e.g., scanned PDF), converts the page to an image for later OCR processing.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of OCRResult objects, one per page.

    Raises:
        RuntimeError: If PDF cannot be opened or processed.
    """
    results = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Try to extract text directly
                text = page.extract_text()

                if text and text.strip():
                    # Successfully extracted text
                    result = OCRResult(
                        text_blocks=[
                            TextBlock(
                                text=text,
                                confidence=1.0,  # Direct extraction has perfect confidence
                                bbox=(0, 0, int(page.width), int(page.height)),
                            )
                        ],
                        page_number=i + 1,
                        metadata={
                            "source": "pdf_text_extraction",
                            "width": page.width,
                            "height": page.height,
                        },
                    )
                else:
                    # Cannot extract text - mark for OCR processing
                    # Save page as image for later OCR
                    import tempfile

                    img = page.to_image(resolution=150)
                    img_path = Path(tempfile.mktemp(suffix=f"_page_{i + 1}.png"))
                    img.save(str(img_path))

                    result = OCRResult(
                        text_blocks=[],
                        page_number=i + 1,
                        metadata={
                            "source": "pdf_image",
                            "image_path": str(img_path),
                            "needs_ocr": True,
                            "width": page.width,
                            "height": page.height,
                        },
                    )

                results.append(result)

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}") from e

    return results

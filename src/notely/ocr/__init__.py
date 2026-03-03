"""
OCR (Optical Character Recognition) module for Notely.
"""

from notely.ocr.base import OCRBackend, OCRResult, TextBlock
from notely.ocr.paddle import PaddleOCRBackend

__all__ = [
    "OCRBackend",
    "OCRResult",
    "PaddleOCRBackend",
    "TextBlock",
]

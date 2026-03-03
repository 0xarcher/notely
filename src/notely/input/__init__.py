"""
Input processing module for Notely.

This module provides unified processing for different input formats:
- Video files (.mp4, .avi, .mov, etc.)
- Audio files (.wav, .mp3, .m4a, etc.)
- PDF files (.pdf)

Each processor extracts relevant content (audio, frames, text) and returns
a standardized ProcessedInput object for downstream processing.
"""

from .processors import process_audio, process_pdf, process_video
from .types import ProcessedInput

__all__ = [
    "ProcessedInput",
    "process_audio",
    "process_pdf",
    "process_video",
]

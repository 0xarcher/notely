"""
Notely - Transform multimodal lectures into beautiful, structured Markdown notes.

A powerful SDK for converting video/audio lectures with slides into
high-quality, readable Markdown documents.
"""

from notely.core import Notely, NotelyConfig
from notely.formatter import MarkdownFormatter
from notely.prompts import NoteTemplate

__version__ = "0.1.0"
__all__ = [
    "MarkdownFormatter",
    "NoteTemplate",
    "Notely",
    "NotelyConfig",
]

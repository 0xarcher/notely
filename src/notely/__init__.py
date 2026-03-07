"""
Notely - Transform multimodal lectures into beautiful, structured Markdown notes.

A powerful SDK for converting video/audio lectures with slides into
high-quality, readable Markdown documents.
"""

from notely.config import (
    ASRConfig,
    EnhancerConfig,
    LLMConfig,
    NotelyConfig,
    OCRConfig,
)
from notely.core import Notely
from notely.formatter import MarkdownFormatter
from notely.models import NotelyResult

# Enhancer (3-Layer Pipeline)
try:
    from notely.enhancer import (
        ComprehensionAgent,
        ProcessingMetrics,
        StructuringAgent,
        ThreeLayerEnhancer,
    )

    _has_enhancer = True
except ImportError:
    _has_enhancer = False

__version__ = "0.1.0"
__all__ = [
    "ASRConfig",
    "EnhancerConfig",
    "LLMConfig",
    "MarkdownFormatter",
    "Notely",
    "NotelyConfig",
    "NotelyResult",
    "OCRConfig",
]

# Add Enhancer exports if available
if _has_enhancer:
    __all__.extend(
        [
            "ComprehensionAgent",
            "ProcessingMetrics",
            "StructuringAgent",
            "ThreeLayerEnhancer",
        ]
    )

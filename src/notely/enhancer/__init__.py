"""
Enhancer - 3-Layer Pipeline Architecture

A production-grade enhancer based on top AI products' design philosophy.

Architecture:
- Layer 1: Capture (ASR transcription)
- Layer 2: Comprehension (parallel semantic extraction)
- Layer 3: Structuring (organize into structured notes)

Key Features:
- Structured output with Pydantic validation
- Graceful degradation on failures
- Caching for idempotency
- Progress tracking
- Comprehensive metrics
- Type-safe throughout

Example:
    >>> from notely import Notely, NotelyConfig, EnhancerConfig, LLMConfig
    >>> from notely.enhancer import ThreeLayerEnhancer
    >>>
    >>> config = NotelyConfig(enhancer=EnhancerConfig(llm=LLMConfig(api_key="sk-xxx", model="gpt-4o")))
    >>> notely = Notely(config)
    >>>
    >>> # result = await notely.process("audio.wav")
"""

# Register all prompts
import notely.prompts.comprehension
import notely.prompts.structuring

from .comprehension import ComprehensionAgent, ComprehensionError
from .enhancer import ThreeLayerEnhancer
from .models import (
    ComprehensionResult,
    NoteSection,
    ProcessingMetrics,
    ProcessingStage,
    SemanticChunk,
    StructuredNote,
)
from .structuring import StructuringAgent, StructuringError

__all__ = [
    "ComprehensionAgent",
    "ComprehensionError",
    "ComprehensionResult",
    "NoteSection",
    "ProcessingMetrics",
    "ProcessingStage",
    "SemanticChunk",
    "StructuredNote",
    "StructuringAgent",
    "StructuringError",
    "ThreeLayerEnhancer",
]

__version__ = "3.0.0"

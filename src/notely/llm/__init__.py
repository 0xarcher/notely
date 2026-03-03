"""
LLM (Large Language Model) module for Notely.
"""

from notely.llm.base import LLMBackend, LLMResult
from notely.llm.openai import OpenAIBackend

__all__ = [
    "LLMBackend",
    "LLMResult",
    "OpenAIBackend",
]

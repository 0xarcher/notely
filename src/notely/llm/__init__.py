"""
LLM (Large Language Model) module for Notely.

This module provides a simple LLM client for OpenAI-compatible APIs.
"""

from notely.llm.client import LLMClient
from notely.llm.zhipu import ZhipuLLMBackend

__all__ = [
    "LLMClient",
    "ZhipuLLMBackend",
]

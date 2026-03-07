"""
LLM Client - Pure API capability layer.

This module provides a simple, clean interface to OpenAI-compatible LLM APIs.
It has no knowledge of business logic (transcripts, notes, enhancers, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Universal LLM client for OpenAI-compatible APIs.

    This is a pure API wrapper with no business logic. It simply calls
    the LLM API and returns the response.

    Attributes:
        client: OpenAI client instance
        model: Model identifier
        temperature: Default sampling temperature
        max_tokens: Default maximum tokens
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        """
        Initialize LLM client.

        Args:
            base_url: Base URL for API endpoint
            api_key: API key for authentication
            model: Model identifier (e.g., 'gpt-4o')
            temperature: Default sampling temperature (0.0-2.0)
            max_tokens: Default maximum tokens in response
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Initialized LLM client: model={model}, base_url={base_url}")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text content

        Raises:
            openai.OpenAIError: If API call fails
        """
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

        content = response.choices[0].message.content or ""
        logger.debug(f"Generated {len(content)} characters")

        return content

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Async version of generate().

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text content
        """
        import asyncio

        return await asyncio.to_thread(
            self.generate,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

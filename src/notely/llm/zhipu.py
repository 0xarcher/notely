"""
Zhipu LLM backend using Zhipu AI API.

This module provides LLM functionality using Zhipu AI's GLM models.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator

from notely.config import LLMConfig

logger = logging.getLogger(__name__)


class ZhipuLLMBackend:
    """
    LLM backend using Zhipu AI GLM models.

    This backend uses Zhipu AI's GLM models for text generation
    via the Zhipu AI SDK.

    Attributes:
        config: LLM configuration instance.
        model: Model identifier (e.g., 'glm-4-flash').
        temperature: Default sampling temperature.
        max_tokens: Default maximum tokens in response.
        client: ZhipuAI client instance.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize Zhipu LLM backend.

        Args:
            config: LLM configuration instance.

        Raises:
            ValueError: If API key is not provided in config.
            ImportError: If zhipuai package is not installed.
        """
        if not config.api_key:
            raise ValueError("API key required for Zhipu LLM backend")

        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Lazy import to avoid dependency issues
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError(
                "zhipuai package is required for Zhipu LLM. Install it with: uv add zhipuai"
            ) from e

        self.client = ZhipuAI(api_key=config.api_key, base_url=config.base_url)
        logger.info(f"Initialized Zhipu LLM backend with model: {self.model}")

    def _build_messages(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """
        Build message list for API call.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            List of message dictionaries.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            Generated text content.

        Raises:
            RuntimeError: If API call fails.
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )

            content = response.choices[0].message.content or ""
            logger.debug(f"Generated {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"Zhipu LLM API call failed: {e}")
            raise RuntimeError(f"Zhipu LLM API call failed: {e}") from e

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        Stream text generation from prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Yields:
            Text chunks as they are generated.

        Raises:
            RuntimeError: If API call fails.
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                stream=True,
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content

        except Exception as e:
            logger.error(f"Zhipu LLM API call failed: {e}")
            raise RuntimeError(f"Zhipu LLM API call failed: {e}") from e

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
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            Generated text content.
        """
        return await asyncio.to_thread(
            self.generate,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def astream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        Async version of stream().

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Yields:
            Text chunks as they are generated.
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                stream=True,
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content

        except Exception as e:
            logger.error(f"Zhipu LLM API call failed: {e}")
            raise RuntimeError(f"Zhipu LLM API call failed: {e}") from e

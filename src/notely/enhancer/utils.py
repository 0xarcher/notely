"""
Utility functions for the enhancer module.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def parallel_with_limit(
    items: list[Any],
    process_func: Callable[..., Any],
    max_concurrent: int = 5,
    max_retries: int = 3,
) -> list[T]:
    """
    Process items in parallel with concurrency limit and retry logic.

    Args:
        items: List of items to process
        process_func: Async function to process each item
        max_concurrent: Maximum number of concurrent tasks (default: 5)
        max_retries: Maximum number of retries on failure (default: 3)

    Returns:
        List of processed results

    Raises:
        Exception: If all retries fail for any item
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_retry(item: Any, index: int) -> T:
        """Process single item with retry logic."""
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    result = await process_func(item)
                    if attempt > 0:
                        logger.info(f"Item {index} succeeded on attempt {attempt + 1}")
                    return result  # type: ignore[no-any-return]
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Item {index} failed after {max_retries} attempts: {e}")
                        raise
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    logger.warning(
                        f"Item {index} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
        raise RuntimeError("Should never reach here")  # For mypy

    tasks = [process_with_retry(item, i) for i, item in enumerate(items)]
    return await asyncio.gather(*tasks)

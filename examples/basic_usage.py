#!/usr/bin/env python3
"""
Basic usage example for Notely.
"""

import asyncio

from notely import ASRConfig, EnhancerConfig, LLMConfig, Notely, NotelyConfig, OCRConfig


async def main() -> None:
    """Basic usage example."""
    # Create configuration
    config = NotelyConfig(
        asr=ASRConfig(backend="funasr", device="cpu"),
        ocr=OCRConfig(backend="paddleocr"),
        enhancer=EnhancerConfig(
            llm=LLMConfig(
                api_key="sk-xxx",  # Replace with your API key
                model="gpt-4o",
            )
        ),
    )

    # Initialize Notely
    notely = Notely(config)

    # Process audio/video file
    result = await notely.process(
        input_path="lecture.mp4",
        metadata={"title": "Machine Learning Basics"},
    )

    # Save result
    result.save("output/notes.md")
    print(f"✓ Notes saved: {len(result.markdown)} characters")


if __name__ == "__main__":
    asyncio.run(main())
